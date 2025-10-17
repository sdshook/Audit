# Microsoft Azure and O365 User Activity Report
# Note this is very slow (limited to 30 days by API access)
# Shane Shook (c) 2025

# Ensure PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "This script requires PowerShell 7+."
    return
}

# Verbose logging
$verbose = $true
function Log {
    param([string]$msg)
    if ($verbose) {
        $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "[$timestamp] [+] $msg" -ForegroundColor Cyan
    }
}

# === CONFIGURATION ===
$userInput = "ALL"  # Or comma-separated UPNs or "ALL"
$days = 7           # Change to desired number of days
$startDate = (Get-Date).AddDays(-$days).ToUniversalTime().ToString("o")
$outputPath = ".\output\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

# === Device Auth via MSAL.PS ===
$clientId = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"   # Microsoft Azure PowerShell Public CLI

if (-not (Get-Module -ListAvailable -Name MSAL.PS)) {
    Install-Module MSAL.PS -Scope CurrentUser -Force
}
Import-Module MSAL.PS

Log "Authenticating with Microsoft Graph via device code..."
$graphToken = Get-MsalToken -ClientId $clientId -Scopes "https://graph.microsoft.com/.default" -DeviceCode
$graphHeaders = @{ Authorization = "Bearer $($graphToken.AccessToken)" }

# === Sign-in Audit Output Setup ===
$csvPath = "$outputPath\SignInAudit.csv"
$jsonPath = "$outputPath\SignInAudit.json"
$recordCount = 0

function ExportRecord {
    param ([object]$record)
    if (-not (Test-Path $csvPath)) {
        $record | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
    } else {
        $record | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8 -Append
    }
    ($record | ConvertTo-Json -Depth 10) | Add-Content -Path $jsonPath -Encoding UTF8
}

function ProcessSignIn {
    param ([object]$signIn)
    $recordCount++
    if ($recordCount % 100 -eq 0) { Log "Processed $recordCount records..." }

    $initiatorType = if ($signIn.AppId -and -not $signIn.UserPrincipalName) {
        "App"
    } elseif ($signIn.UserPrincipalName) {
        "User"
    } else {
        "Unknown"
    }

    $obj = [PSCustomObject]@{
        UserPrincipalName = $signIn.UserPrincipalName
        UserDisplayName   = $signIn.UserDisplayName
        InitiatorType     = $initiatorType
        AppName           = $signIn.AppDisplayName
        AppId             = $signIn.AppId
        ClientAppUsed     = $signIn.ClientAppUsed
        CreatedDateTime   = $signIn.CreatedDateTime
        ResourceDisplay   = $signIn.ResourceDisplayName
        IPAddress         = $signIn.IpAddress
        City              = $signIn.Location.City
        Country           = $signIn.Location.CountryOrRegion
        DeviceId          = $signIn.DeviceDetail.DeviceId
        DeviceOS          = $signIn.DeviceDetail.OperatingSystem
        Browser           = $signIn.DeviceDetail.Browser
        RiskState         = $signIn.RiskState
        RiskLevel         = $signIn.RiskLevelAggregated
        RiskDetail        = $signIn.RiskDetail
        ConditionalAccess = $signIn.ConditionalAccessStatus
        UserAgent         = $signIn.UserAgent
    }

    ExportRecord $obj
}

Log "Fetching sign-in audit logs from Microsoft Graph..."
$signInEndpoint = "https://graph.microsoft.com/beta/auditLogs/signIns?`$filter=createdDateTime ge $startDate&`$top=100"

$userFilter = @()
if ($userInput -ne "ALL") {
    $userFilter = $userInput -split "," | ForEach-Object { $_.Trim().ToLower() }
}

$t0 = Get-Date

do {
    $retryCount = 0
    $maxRetries = 5
    do {
        try {
            $signIns = Invoke-RestMethod -Method GET -Uri $signInEndpoint -Headers $graphHeaders -ErrorAction Stop
            break
        } catch {
            if ($_.Exception.Response.StatusCode.value__ -eq 429 -and $retryCount -lt $maxRetries) {
                $wait = 2 * ($retryCount + 1)
                Log "Throttled. Waiting $wait seconds..."
                Start-Sleep -Seconds $wait
                $retryCount++
            } elseif ($_.Exception.Response.StatusCode.value__ -eq 401 -and $_.Exception.Message -like "*InvalidAuthenticationToken*") {
                Log "Access token expired. Renewing..."
                $graphToken = Get-MsalToken -ClientId $clientId -Scopes "https://graph.microsoft.com/.default" -DeviceCode
                $graphHeaders = @{ Authorization = "Bearer $($graphToken.AccessToken)" }
                $retryCount++
                Start-Sleep -Seconds 2
            } else {
                throw $_
            }
        }
    } while ($true)

    foreach ($record in $signIns.value) {
        if ($userInput -ne "ALL") {
            $recordUPN = $record.UserPrincipalName.ToLower()
            if (-not ($userFilter -contains $recordUPN)) { continue }
        }
        ProcessSignIn $record
    }
    $signInEndpoint = $signIns."@odata.nextLink"
} while ($signInEndpoint)

$t1 = Get-Date
$elapsed = [math]::Round(($t1 - $t0).TotalSeconds, 2)
Log "Sign-in audit complete. Total records processed: $recordCount in $elapsed seconds."
Log "CSV Report: $csvPath"
Log "JSON Report: $jsonPath"
