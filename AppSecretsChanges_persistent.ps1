# Microsoft Azure and O365 App Change Audit Report 
# Shane Shook (c) 2025
# Requires an app registration and related config file

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
$days = 365  # modify to desired days
$startDate = (Get-Date).AddDays(-$days).ToUniversalTime().ToString("o")
$endDate = (Get-Date).ToUniversalTime().ToString("o")
$outputPath = ".\output\AppChanges_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

# === Auth via MSAL.PS (Encrypted Config) ===
function Show-TokenForValidation {
    param ([string]$Scope = "https://manage.office.com/.default")
    Log "Generating token for manual validation..."
    $tokenObj = Get-MsalToken -ClientId $clientId -TenantId $tenantId -ClientSecret $clientSecret -Scopes $Scope
    $accessToken = $tokenObj.AccessToken
    Set-Clipboard -Value $accessToken
    Log "Access token copied to clipboard. Visit https://jwt.ms and paste it to inspect scopes/roles."
}

$configPath = "./appsecrets.secure.xml"
if (-not (Test-Path $configPath)) {
    Write-Error "Encrypted config not found: $configPath"
    return
}
$secureConfig = Import-Clixml -Path $configPath
$clientId = $secureConfig.clientId
$tenantId = $secureConfig.tenantId
$clientSecret = $secureConfig.clientSecret

if (-not (Get-Module -ListAvailable -Name MSAL.PS)) {
    Install-Module -Name MSAL.PS -Scope CurrentUser -Force
}
Import-Module MSAL.PS

Log "Authenticating with Office 365 Management API via client credentials..."
$manageToken = Get-MsalToken -ClientId $clientId -TenantId $tenantId -ClientSecret $clientSecret -Scopes "https://manage.office.com/.default"
$manageHeaders = @{ Authorization = "Bearer $($manageToken.AccessToken)" }

# === Start Subscription ===
$subscriptionUri = "https://manage.office.com/api/v1.0/$tenantId/activity/feed/subscriptions/start?contentType=Audit.AzureActiveDirectory"
try {
    Invoke-RestMethod -Method POST -Uri $subscriptionUri -Headers $manageHeaders -ErrorAction Stop | Out-Null
    Log "Started (or confirmed) subscription to Audit.AzureActiveDirectory feed."
} catch {
    Log "âš ï¸ Failed to start subscription: $($_.Exception.Message)"
}

# === Setup Output ===
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

# === Query UAL ===
$manageUri = "https://manage.office.com/api/v1.0/$tenantId/activity/feed/audit?contentType=Audit.AzureActiveDirectory&startTime=$startDate&endTime=$endDate&status=enabled"
$contentUris = Invoke-RestMethod -Headers $manageHeaders -Uri $manageUri -Method GET
Log "Total content blobs received: $($contentUris.Count)"

foreach ($content in $contentUris) {
    $auditUrl = $content.contentUri
    try {
        $auditJson = Invoke-RestMethod -Headers $manageHeaders -Uri $auditUrl -Method GET
        Log "Blob contains $($auditJson.Count) entries"
    } catch {
        Log "âš ï¸ Failed to fetch blob: $auditUrl â€” $($_.Exception.Message)"
        continue
    }

    $entryCount = 0
    foreach ($entry in $auditJson) {
        $entryCount++
        if ($entryCount % 100 -eq 0) {
            $elapsedSoFar = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
            Log "Processed $entryCount entries so far... Elapsed time: $elapsedSoFar sec â€” Matches: $appChangesCount / Cache: $($appMatchCache.Count)"
        }

        $userFilterMatch = $false
        if ($userInput -eq "ALL") {
            $userFilterMatch = $true
        } else {
            $entryKey = (($entry.AppId -join "") + '|' + ($entry.AppDisplayName -join "")).ToLower()
            if ($appMatchCache.ContainsKey($entryKey)) {
                $userFilterMatch = $appMatchCache[$entryKey]
            } else {
                $filters = $userInput -split "," | ForEach-Object { $_.Trim().ToLower() }
                $entryAppId = ($entry.AppId -join "").ToLower()
                $entryAppName = ($entry.AppDisplayName -join "").ToLower()
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
            if ($entry.Operation -and $entry.Operation.ToLower() -like "*$pattern*") {
                Log "Matched Operation: $($entry.Operation)"
                $matchFound = $true
                break
            }
        }
        if (-not $matchFound) { continue }

        $appChangesCount++
        TallyAppOp $entry.Operation

        $flat = [PSCustomObject]@{
            CreationDate         = $entry.CreationTime
            Operation            = $entry.Operation
            AppId                = $entry.AppId
            AppDisplayName       = $entry.AppDisplayName
            ModifiedProperties   = ($entry.ModifiedProperties -join "; ")
            OldValue             = ($entry.ModifiedProperties | ForEach-Object { $_.OldValue } -join "; ")
            NewValue             = ($entry.ModifiedProperties | ForEach-Object { $_.NewValue } -join "; ")
            ModifiedPropertyNames = ($entry.ModifiedProperties | ForEach-Object { $_.DisplayName } -join "; ")
            Actor                = $entry.UserId
            TargetResources      = ($entry.TargetResources | ConvertTo-Json -Depth 5 -Compress)
            RawRecord            = ($entry | ConvertTo-Json -Depth 10 -Compress)
        }

        if (-not (Test-Path $appCsvPath)) {
            $flat | Export-Csv -Path $appCsvPath -NoTypeInformation -Encoding UTF8
        } else {
            $flat | Export-Csv -Path $appCsvPath -NoTypeInformation -Encoding UTF8 -Append
        }
        ($flat | ConvertTo-Json -Depth 10) | Add-Content -Path $appJsonPath -Encoding UTF8
    }
}

$t1 = Get-Date
$elapsed = [math]::Round(($t1 - $t0).TotalSeconds, 2)
Log "App change audit complete. Records processed: $appChangesCount in $elapsed seconds. Cached matches: $($appMatchCache.Count)"
Log "App CSV Report: $appCsvPath"
Log "App JSON Report: $appJsonPath"
Log "App Change Summary:"
$appSummary.GetEnumerator() | Sort-Object -Property Name | ForEach-Object {
    Log "$($_.Name): $($_.Value)"
}
