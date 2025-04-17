# Microsoft Azure and O365 App Change Audit Report (Graph API, 30-day limit)
# Note that this is very slow 
# Shane Shook (c) 2025

# Ensure PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "This script requires PowerShell 7+."
    return
}

$verbose = $true
function Log {
    param([string]$msg)
    if ($verbose) {
        $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "[$timestamp] [+] $msg" -ForegroundColor Cyan
    }
}

# === CONFIGURATION ===
$userInput = "ALL"  # Or comma-separated AppIds or AppDisplayNames
$days = 30 # Change this to however many days the script will be rerun
$startTime = (Get-Date).AddDays(-$days).ToString("yyyy-MM-ddTHH:mm:ssZ")
$outputPath = ".\output\AppChanges_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

# === Graph Auth via Device Code ===
$clientId = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"  # Microsoft Azure Public CLI
if (-not (Get-Module -ListAvailable -Name MSAL.PS)) {
    Install-Module -Name MSAL.PS -Scope CurrentUser -Force
}
Import-Module MSAL.PS

Log "Authenticating with Microsoft Graph via device code..."
$graphToken = Get-MsalToken -ClientId $clientId -Scopes "AuditLog.Read.All" -DeviceCode
$headers = @{ Authorization = "Bearer $($graphToken.AccessToken)" }

# === Setup Output ===
$unmatchedOps = @{}
$appCsvPath = "$outputPath\AppChanges.csv"
$appJsonPath = "$outputPath\AppChanges.json"
$appChangesCount = 0
$appSummary = @{}
$appMatchCache = @{}
$t0 = Get-Date

$appPatterns = @(
    "add application",
    "update application",
    "certificates and secrets",
    "remove app role assignment",
    "add app role assignment",
    "add service principal",
    "add service principal credentials",
    "add owner to service principal",
    "update service principal",
    "add password credential",
    "update password credential",
    "remove password credential",
    "add key credential",
    "remove key credential",
    "update key credential",
    "add delegated permission",
    "grant application permission",
    "remove delegated permission",
    "consent to application",
    "grant admin consent",
    "add app role",
    "remove app role",
    "update app role"
)

function TallyAppOp($op) {
    $key = $op.Trim()
    if ($appSummary.ContainsKey($key)) { $appSummary[$key]++ } else { $appSummary[$key] = 1 }
}

# === Query Graph Directory Audit Logs ===
$graphUrl = "https://graph.microsoft.com/beta/auditLogs/directoryAudits?`$filter=activityDateTime ge $startTime&`$top=100"
do {
    $retryCount = 0
    $maxRetries = 5
    do {
        try {
            $response = Invoke-RestMethod -Uri $graphUrl -Headers $headers -Method Get -ErrorAction Stop
            break
        } catch {
            if ($_.Exception.Response.StatusCode.value__ -eq 429 -and $retryCount -lt $maxRetries) {
                $retryAfter = 5 * ($retryCount + 1)
                Log "Throttled. Retrying in $retryAfter seconds..."
                Start-Sleep -Seconds $retryAfter
                $retryCount++
            } elseif ($_.Exception.Response.StatusCode.value__ -eq 401 -and $_.Exception.Message -like "*InvalidAuthenticationToken*") {
                Log "Access token expired. Renewing..."
                $graphToken = Get-MsalToken -ClientId $clientId -Scopes "AuditLog.Read.All" -DeviceCode
                $headers = @{ Authorization = "Bearer $($graphToken.AccessToken)" }
                $retryCount++
                Start-Sleep -Seconds 2
            } else {
                throw $_
            }
        }
    } while ($true)
    foreach ($entry in $response.value) {
        $appChangesCount++
        if ($appChangesCount % 100 -eq 0) {
            $elapsedSoFar = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
            Log "Processed $appChangesCount entries... Elapsed time: $elapsedSoFar sec â€” Cache: $($appMatchCache.Count)"
        }

        $userFilterMatch = $false
        if ($userInput -eq "ALL") {
            $userFilterMatch = $true
        } else {
            $entryKey = ($entry.initiatedBy.app.id + '|' + $entry.initiatedBy.app.displayName).ToLower()
            if ($appMatchCache.ContainsKey($entryKey)) {
                $userFilterMatch = $appMatchCache[$entryKey]
            } else {
                $filters = $userInput -split "," | ForEach-Object { $_.Trim().ToLower() }
                $entryAppId = ($entry.initiatedBy.app.id).ToLower()
                $entryAppName = ($entry.initiatedBy.app.displayName).ToLower()
                foreach ($f in $filters) {
                    if ($entryAppId -like "*$f*" -or $entryAppName -like "*$f*") {
                        $userFilterMatch = $true
                        break
                    }
                }
                $appMatchCache[$entryKey] = $userFilterMatch
            }
        }
        if (-not $userFilterMatch) { continue }

        $matchFound = $false
        foreach ($pattern in $appPatterns) {
            if ($entry.activityDisplayName -and $entry.activityDisplayName.ToLower() -like "*$pattern*") {
                Log "Matched Operation: $($entry.activityDisplayName)"
                $matchFound = $true
                break
            }
        }
        if (-not $matchFound) {
    $label = $entry.activityDisplayName
    if ($label) {
        if ($unmatchedOps.ContainsKey($label)) {
            $unmatchedOps[$label]++
        } else {
            $unmatchedOps[$label] = 1
        }
    }
    continue
}

        TallyAppOp $entry.activityDisplayName

        $modifiedProps = @()
foreach ($target in $entry.targetResources) {
    if ($target.modifiedProperties) {
        $modifiedProps += $target.modifiedProperties
    }
}

$appSource = ""
if ($entry.initiatedBy.app.id) {
    $appId = $entry.initiatedBy.app.id
    $appName = $entry.initiatedBy.app.displayName
    $appSource = "app"
} elseif ($entry.initiatedBy.application.appId) {
    $appId = $entry.initiatedBy.application.appId
    $appName = $entry.initiatedBy.application.displayName
    $appSource = "application"
} elseif ($entry.initiatedBy.servicePrincipal.id) {
    $appId = $entry.initiatedBy.servicePrincipal.id
    $appName = $entry.initiatedBy.servicePrincipal.displayName
    $appSource = "servicePrincipal"
} elseif ($entry.targetResources -and $entry.targetResources[0].id) {
    $appId = $entry.targetResources[0].id
    $appName = $entry.targetResources[0].displayName
    $appSource = "targetResource"
} else {
    $appId = ""
    $appName = ""
    $appSource = "unattributed"
}

$flat = [PSCustomObject]@{
    CreationDate          = $entry.activityDateTime
    Operation             = $entry.activityDisplayName
    Actor                 = $entry.initiatedBy.user.userPrincipalName
    AppId                = $appId
    AppDisplayName       = $appName
    AppAttributionSource = $appSource
    TargetResources       = ($entry.targetResources | ConvertTo-Json -Depth 5 -Compress)
    ModifiedProperties    = ($modifiedProps | ForEach-Object { $_.displayName + ':' + $_.newValue }) -join "; "
    OldValue              = ($modifiedProps | ForEach-Object { $_.oldValue }) -join "; "
    NewValue              = ($modifiedProps | ForEach-Object { $_.newValue }) -join "; "
    ModifiedPropertyNames = ($modifiedProps | ForEach-Object { $_.displayName }) -join "; "
    RawRecord             = ($entry | ConvertTo-Json -Depth 10 -Compress)
}


        if (-not (Test-Path $appCsvPath)) {
            $flat | Export-Csv -Path $appCsvPath -NoTypeInformation -Encoding UTF8
        } else {
            $flat | Export-Csv -Path $appCsvPath -NoTypeInformation -Encoding UTF8 -Append
        }
        ($flat | ConvertTo-Json -Depth 10) | Add-Content -Path $appJsonPath -Encoding UTF8
    }
    $graphUrl = $response."@odata.nextLink"
} while ($graphUrl)

$t1 = Get-Date
$elapsed = [math]::Round(($t1 - $t0).TotalSeconds, 2)
Log "App change audit complete. Records processed: $appChangesCount in $elapsed seconds. Cached matches: $($appMatchCache.Count)"
Log "App CSV Report: $appCsvPath"
Log "App JSON Report: $appJsonPath"
Log "App Change Summary:"
Log "Unmatched Operation Summary (Top 20):"
$unmatchedOps.GetEnumerator() | Sort-Object -Property Value -Descending | Select-Object -First 20 | ForEach-Object {
    Log "$($_.Key): $($_.Value)"
}
$appSummary.GetEnumerator() | Sort-Object -Property Name | ForEach-Object {
    Log "$($_.Name): $($_.Value)"
}
