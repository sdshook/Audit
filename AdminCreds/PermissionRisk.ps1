# Microsoft Azure and Office 365 Permissions Audit Risk Report
# Shane Shook (c) 2025

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

# Configuration
$userInput = "ALL"  # or UPN or comma-separated UPN's
$days = 7           # update to desired days
$startDate = (Get-Date).AddDays(-$days)
$outputPath = ".\output\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Log "Writing reports to $outputPath"

# Required modules
$requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Beta.Reports", "Microsoft.Graph.Authentication", "Microsoft.Graph.Identity.DirectoryManagement")
foreach ($mod in $requiredModules) {
    if (-not (Get-Module -ListAvailable -Name $mod)) {
        Install-Module -Name $mod -Scope CurrentUser -Force -Confirm:$false
    }
    Import-Module -Name $mod -Force
}

# Graph connection
if (-not (Get-MgContext)) {
    Connect-MgGraph -Scopes "AuditLog.Read.All,Directory.Read.All,IdentityRiskEvent.Read.All,Application.Read.All" -ContextScope Process -NoWelcome
}

# IP and risk config
$trustedIPs = @("192.168.0.0/24", "10.0.0.0/8")
$sensitiveResourceTypes = @("Application", "ServicePrincipal")
$sensitiveResourceIds = @("00000003-0000-0000-c000-000000000000")   # MSGraph API for visibility of possible abuse, replace or comma-separate with others

# IP utilities
function ConvertTo-IpBytes { param ([string]$ip) return [System.Net.IPAddress]::Parse($ip).GetAddressBytes() }
function IsIpInCidrRange {
    param ([string]$ip, [string]$cidr)
    $parts = $cidr -split "/"; $subnet = $parts[0]; $prefix = [int]$parts[1]
    $ipBytes = ConvertTo-IpBytes $ip; $subnetBytes = ConvertTo-IpBytes $subnet
    $bitsToCheck = $prefix
    for ($i = 0; $i -lt $ipBytes.Length -and $bitsToCheck -gt 0; $i++) {
        $bits = [math]::Min(8, $bitsToCheck)
        $mask = [byte](0xFF -bxor ((1 -shl (8 - $bits)) -band 0xFF))
        if (($ipBytes[$i] -band $mask) -ne ($subnetBytes[$i] -band $mask)) { return $false }
        $bitsToCheck -= $bits
    }
    return $true
}
function IsUnfamiliarIP {
    param ([string]$ipAddress, [string[]]$trustedRanges)
    if (-not [System.Net.IPAddress]::TryParse($ipAddress, [ref]$null)) { return $true }
    foreach ($range in $trustedRanges) {
        if (IsIpInCidrRange -ip $ipAddress -cidr $range) { return $false }
    }
    return $true
}

# Throttling wrapper
function Invoke-WithThrottling {
    param ([scriptblock]$ScriptBlock, [int]$MaxRetries = 5, [int]$InitialWaitSeconds = 2)
    $retry = 0; $wait = $InitialWaitSeconds
    while ($retry -lt $MaxRetries) {
        try { return & $ScriptBlock }
        catch {
            if ($_.Exception.Message -match "TooManyRequests|429") {
                Write-Warning "Throttled. Waiting $wait seconds..."
                Start-Sleep -Seconds $wait; $retry++; $wait *= 2
            } else { throw $_ }
        }
    }
    throw "Max retry attempts reached."
}

# Output paths
$csvPath = "$outputPath\PermissionChanges.csv"
$jsonPath = "$outputPath\PermissionChanges.json"
$script:recordCount = 0
$script:evaluatedCount = 0

# Processing logic
function ProcessChange {
    param ($change)
    $script:recordCount++
    if ($recordCount % 100 -eq 0) { Write-Host "Processed $recordCount records..." -ForegroundColor Yellow }

    $localTime = [DateTime]$change.ActivityDateTime.ToLocalTime()
    $isOffHours = ($localTime.Hour -lt 6 -or $localTime.Hour -gt 20)
    $isHighRiskActivity = $change.ActivityDisplayName -match '(?i)add.*permission|consent.*application|app.*role.*assignment'
    $targetResourceType = $change.TargetResources[0].Type
    $targetResourceId = $change.TargetResources[0].Id
    $isSensitiveResource = $sensitiveResourceTypes -contains $targetResourceType -or $sensitiveResourceIds -contains $targetResourceId

    $initiatorIP = $change.InitiatedBy.User.IpAddress
    $isUnfamiliarIP = IsUnfamiliarIP $initiatorIP $trustedIPs

    $initiatorDisplay = $change.InitiatedBy.User.DisplayName
    $initiatorUPN = $change.InitiatedBy.User.UserPrincipalName
    if (-not $initiatorDisplay -and $change.InitiatedBy.App.DisplayName) { $initiatorDisplay = $change.InitiatedBy.App.DisplayName + " (App)" }
    if (-not $initiatorUPN -and $change.InitiatedBy.App.AppId) { $initiatorUPN = $change.InitiatedBy.App.AppId }

    $RiskFlags = @()
    if ($isOffHours) { $RiskFlags += "OffHours" }
    if ($isHighRiskActivity) { $RiskFlags += "HighRiskActivity" }
    if ($isSensitiveResource) { $RiskFlags += "SensitiveResource" }
    if ($isUnfamiliarIP) { $RiskFlags += "UnfamiliarIP" }
    $RiskFlag = if ($RiskFlags) { $RiskFlags -join "; " } else { "None" }

    $record = [PSCustomObject]@{
        ActivityDateTime    = $localTime
        InitiatedBy         = $initiatorDisplay
        InitiatorUPN        = $initiatorUPN
        ActivityDisplayName = $change.ActivityDisplayName
        TargetResource      = $change.TargetResources[0].DisplayName
        TargetResourceType  = $targetResourceType
        TargetResourceId    = $targetResourceId
        OperationResult     = $change.Result
        IPAddress           = $initiatorIP
        Location            = ($change.InitiatedBy.User.Location.City + ", " + $change.InitiatedBy.User.Location.CountryOrRegion).Trim(", ")
        RiskFlag            = $RiskFlag
        AdditionalDetails   = ($change.AdditionalDetails | ConvertTo-Json -Compress)
    }

    if (-not (Test-Path $csvPath)) {
        $record | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8
    } else {
        $record | Export-Csv $csvPath -NoTypeInformation -Encoding UTF8 -Append
    }

    ($record | ConvertTo-Json -Depth 10) | Add-Content -Path $jsonPath -Encoding UTF8
}

# Main data pull
Log "Retrieving directory permission changes..."

$filterDate = $startDate.ToUniversalTime().ToString("o")
Log "Querying records since $filterDate..."
$filterDate = $startDate.ToUniversalTime().ToString("o")
if ($userInput -eq "ALL") {
    Log "Fetching and streaming directory audit records using -All..."
    Invoke-WithThrottling {
        Get-MgBetaAuditLogDirectoryAudit -All -Filter "activityDateTime ge $filterDate"
    } | ForEach-Object {
        $script:evaluatedCount++
        if ($evaluatedCount -le 25) {
            Log "ActivityDisplayName: $($_.ActivityDisplayName)"
        }
        if ($evaluatedCount % 100 -eq 0) {
            Log "Evaluated $evaluatedCount records..."
        }
        if ($_.ActivityDisplayName -match '(?i)add.*permission|consent.*application|app.*role.*assignment') {
            ProcessChange $_
        }
    }
} else {
    $userList = $userInput -split "," | ForEach-Object { $_.Trim() }
    foreach ($user in $userList) {
        Log "Fetching directory audit records for user $user..."
        Invoke-WithThrottling {
            Get-MgBetaAuditLogDirectoryAudit -All -Filter "initiatedBy/user/userPrincipalName eq '$user' and activityDateTime ge $filterDate"
        } | ForEach-Object {
            $script:evaluatedCount++
            if ($evaluatedCount -le 25) {
                Log "ActivityDisplayName: $($_.ActivityDisplayName)"
            }
            if ($evaluatedCount % 100 -eq 0) {
                Log "Evaluated $evaluatedCount records..."
            }
            if ($_.ActivityDisplayName -match '(?i)add.*permission|consent.*application|app.*role.*assignment') {
                ProcessChange $_
            }
        }
    }
}
Log "Audit complete. Total records processed: $recordCount"
Log "CSV Report: $csvPath"
Log "JSON Report: $jsonPath"
