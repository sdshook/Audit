# Microsoft Azure AD Authentication Report
# Shane Shook (c)2025

# Requires PSv7 and DotNetv8
# Modules must be manually installed before use:
## Install-Module -Name PackageManagement
## Install-Module -Name Microsoft.Graph -Scope AllUsers
## Install-Module -Name Microsoft.Graph.Beta
## Install-Module Microsoft.Graph.Identity.DirectoryManagement -Scope CurrentUser

# Update lines 45, 46, 49 as required - defaults to 360 day all users in PWD

# runas from commandline pwsh.exe <script> and authenticate
## Connect-MgGraph -Scopes "AuditLog.Read.All,Directory.Read.All,IdentityRiskEvent.Read.All,Application.Read.All" -ContextScope Process -NoWelcome

$null = @'

REPORTS INCLUDE THE FOLLOWING

• UALDF.csv
  - Structured Azure AD sign-in report
  - Includes user, app, device, IP, location, and client metadata

• UALDR.json
  - Raw JSON output of all Azure AD sign-in events
  - Suitable for forensics or SIEM import

'@

# Ensure PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "This script requires PowerShell 7+. Please run it from pwsh.exe."
    return
}

# Verbose output toggle
$verbose = $true
function Log {
    param([string]$msg)
    if ($verbose) { Write-Host "[+] $msg" -ForegroundColor Cyan }
}

###### Edit this section #####
# "<user>@domain", comma-separated users or APPiD(s), or "ALL"
$userInput = "ALL" 
$days = 360
$startDate = (Get-Date).AddDays(-$days)
$endDate = Get-Date
$outputPath = ".\output\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
###### Edit this section #####

# Module validation & import
$requiredModules = @(
    "Microsoft.Graph",
    "Microsoft.Graph.Beta.Reports",
    "Microsoft.Graph.Authentication",
    "Microsoft.Graph.Identity.DirectoryManagement"
)

$missingModules = $requiredModules | Where-Object { -not (Get-Module -ListAvailable -Name $_) }
if ($missingModules) {
    foreach ($mod in $missingModules) {
        Log "Installing missing module: $mod"
        try { Install-Module -Name $mod -Scope CurrentUser -Force -Confirm:$false -ErrorAction Stop }
        catch { Write-Error "Failed to install module '$mod': $_"; return }
    }
}
foreach ($mod in $requiredModules) {
    if (-not (Get-Module -Name $mod)) {
        try { Import-Module -Name $mod -ErrorAction Stop }
        catch { Write-Error "Failed to import module '$mod': $_"; return }
    }
}

function Invoke-WithThrottling {
    param ([scriptblock]$ScriptBlock, [int]$MaxRetries = 5)
    $retry = 0
    while ($retry -lt $MaxRetries) {
        try { return & $ScriptBlock }
        catch {
            if ($_.Exception.Message -match "TooManyRequests|429") {
                $wait = 2 * ($retry + 1)
                Write-Warning "Throttled. Waiting $wait seconds..."
                Start-Sleep -Seconds $wait
                $retry++
            } else { throw $_ }
        }
    }
    throw "Max retry attempts reached after throttling."
}

if (-not (Get-MgContext)) {
    Connect-MgGraph -Scopes "AuditLog.Read.All,Directory.Read.All,IdentityRiskEvent.Read.All,Application.Read.All" -ContextScope Process -NoWelcome
}

Log "Retrieving Azure AD Sign-In logs..."
$azureADAuthHistory = @()
if ($userInput -eq "All") {
    $azureADAuthHistory = Invoke-WithThrottling { Get-MgBetaAuditLogSignIn -All | Where-Object { $_.CreatedDateTime -ge $startDate } }
} else {
    $userList = $userInput -split "," | ForEach-Object { $_.Trim() }
    foreach ($user in $userList) {
        $userResults = Invoke-WithThrottling { Get-MgBetaAuditLogSignIn -All -Filter "userPrincipalName eq '$user'" | Where-Object { $_.CreatedDateTime -ge $startDate } }
        $azureADAuthHistory += $userResults
    }
}

Log "Exporting sign-in reports..."
$results = foreach ($signIn in $azureADAuthHistory) {
    [PSCustomObject]@{
        UserPrincipalName = $signIn.UserPrincipalName
        UserDisplayName   = $signIn.UserDisplayName
        cDate             = $signIn.CreatedDateTime
        AppName           = $signIn.AppDisplayName
        AppId             = $signIn.AppId
        AppUsed           = $signIn.ClientAppUsed
        Resource          = $signIn.ResourceDisplayName
        IPAddress         = $signIn.IpAddress
        City              = $signIn.Location.City
        Country           = $signIn.Location.CountryOrRegion
        DeviceId          = $signIn.DeviceDetail.DeviceId
        DeviceDetail      = $signIn.DeviceDetail.DisplayName
        DeviceOS          = $signIn.DeviceDetail.OperatingSystem
        DeviceBrowser     = $signIn.DeviceDetail.Browser
        UserAgent         = $signIn.UserAgent
    }
}
$results | Export-Csv -Path "$outputPath\UALDF.csv" -NoTypeInformation -Encoding UTF8
$azureADAuthHistory | ConvertTo-Json -Depth 10 | Out-File "$outputPath\UALDR.json" -Encoding UTF8

