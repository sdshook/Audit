# Microsoft Azure AD Role Assignment Change Report 
# Shane Shook (c) 2025

# === CONFIGURATION ===
$userInput = "ALL"    # e.g. "ALL" or "user1@domain.com,user2@domain.com" or AppId(s)
$days = 30            # modify to desired number of days
$startDate = (Get-Date).AddDays(-$days).ToUniversalTime().ToString("o")
$outputPath = ".\output\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

# === VERBOSE LOGGING ===
$Verbose = $true
function Log { param([string]$msg) if ($Verbose) { Write-Host "[+] $msg" -ForegroundColor Cyan } }

# === MODULES & AUTH ===
$requiredModules = @("Microsoft.Graph")
foreach ($mod in $requiredModules) {
    if (-not (Get-Module -ListAvailable -Name $mod)) {
        Install-Module -Name $mod -Scope CurrentUser -Force -Confirm:$false
    }
    Import-Module -Name $mod -Force
}

# === LOGIN ===
Log "Authenticating to Microsoft Graph..."
Disconnect-MgGraph -ErrorAction SilentlyContinue
try {
    Connect-MgGraph -Scopes "AuditLog.Read.All,Directory.Read.All" -ContextScope Process -NoWelcome -ErrorAction Stop
    $context = Get-MgContext
    Log "Connected as: $($context.Account)"
} catch {
    Write-Error "Authentication failed: $_"
    exit
}

# === FUNCTION: THROTTLE WRAPPER ===
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
    throw "Max retry attempts reached."
}

# === RESOLVE USER INPUTS ===
$userList = @()
if ($userInput.ToUpper() -ne "ALL") {
    $items = $userInput -split "," | ForEach-Object { $_.Trim() }
    foreach ($item in $items) {
        if ($item -match "@") {
            $user = Invoke-WithThrottling { Get-MgUser -UserId $item }
            if ($user) { $userList += $user.Id }
        } elseif ($item -match "^[0-9a-f\-]{36}$") {
            $sp = Invoke-WithThrottling { Get-MgServicePrincipal -Filter "AppId eq '$item'" }
            if ($sp) { $userList += $sp.Id } else { $userList += $item }
        }
    }
}

# === QUERY ROLE ASSIGNMENT CHANGES ===
Log "Querying role assignment changes (Directory Audit Logs)..."
$roleAuditChanges = @()
$auditUri = "https://graph.microsoft.com/v1.0/auditLogs/directoryAudits?$filter=" +
    [System.Web.HttpUtility]::UrlEncode("(activityDisplayName eq 'Add member to role' or activityDisplayName eq 'Remove member from role') and activityDateTime ge " + $startDate)

do {
    $auditResponse = Invoke-WithThrottling {
        Invoke-MgGraphRequest -Uri $auditUri -Method GET -OutputType PSObject
    }
    $roleAuditChanges += $auditResponse.value
    $auditUri = $auditResponse.'@odata.nextLink'
} while ($auditUri)

# === FILTER BY USER INPUT (if applicable) ===
if ($userInput.ToUpper() -ne "ALL" -and $userList.Count -gt 0) {
    $roleAuditChanges = $roleAuditChanges | Where-Object {
        $userList -contains $_.targetResources[0].id
    }
}

# === PARSE AND EXPORT ===
$roleAuditParsed = foreach ($entry in $roleAuditChanges) {
    [PSCustomObject]@{
        Category         = $entry.category
        Result          = $entry.result
        ModifiedProps   = ($entry.targetResources[0].modifiedProperties | ForEach-Object { "$($_.displayName): $($_.oldValue) => $($_.newValue)" }) -join "; "
        Timestamp        = $entry.activityDateTime
        Action           = $entry.activityDisplayName
        PerformedBy      = $entry.initiatedBy.user.displayName
        PerformedByUPN   = $entry.initiatedBy.user.userPrincipalName
        TargetDisplay    = $entry.targetResources[0].displayName
        TargetType       = $entry.targetResources[0].type
        TargetId         = $entry.targetResources[0].id
        RoleName         = ($entry.targetResources | Where-Object { $_.type -eq 'Role' }).displayName
        CorrelationId    = $entry.correlationId
    }
}

$auditCsvOut = "$outputPath\RoleAssignmentChanges.csv"
$roleAuditParsed | Export-Csv -Path $auditCsvOut -NoTypeInformation -Encoding UTF8
$auditJsonOut = "$outputPath\RoleAssignmentChanges.json"
$roleAuditParsed | ForEach-Object { $_ | ConvertTo-Json -Depth 10 -Compress } | Out-File -FilePath $auditJsonOut -Encoding UTF8

Log "Audit log export complete. Found $($roleAuditParsed.Count) role assignment change events."
