# Microsoft Azure AD Audit & Access Report Summary
# Updated by Shane Shook (c)2025

# Update lines 69, 72, 77 as required - defaults to 360 day all users in PWD
# runas from commandline pwsh.exe <script>
# Requires PSv7 and DotNetv8
# Modules must be manually installed before use:
## Install-Module -Name PackageManagement
## Install-Module -Name Microsoft.Graph -Scope AllUsers
## Install-Module -Name Microsoft.Graph.Beta

$null = @'

REPORTS INCLUDE THE FOLLOWING

• UALDF.csv
  - Structured Azure AD sign-in report
  - Includes user, app, device, IP, location, and client metadata

• UALDR.json
  - Raw JSON output of all Azure AD sign-in events
  - Suitable for forensics or SIEM import

• PermissionChanges.json
  - Shows permission grants, app consents, and delegated permissions
  - Tracks changes in user/application access rights

• RoleAssignments.json
  - Lists all Microsoft Entra ID role assignments
  - Includes role name, description, scope, and assigned principal

• RiskySignIns.json
  - Identifies users and sessions flagged as risky by Microsoft Entra
  - Based on risk detection signals

• AppSecretChanges.csv
  - Tracks secret creation or updates across apps
  - Includes AppId, AppName, SecretId, OldSecretId, and change timestamp

• ReusedAppSecrets.csv
  - Detects secrets reused across multiple applications
  - Indicates potential shared credentials or security hygiene issues

• CrossAccessResourceAudit.csv
  - Object-level resource access from Unified Audit Log
  - Includes mail/file/team access details with identifiers

• CrossAccessResourceAudit.json
  - Raw JSON of audit log entries
  - Full record of resource access actions and metadata

• JoinedCrossAccessReport.csv
  - Combines resource access with sign-in and app usage
  - Includes:
    - AccessType (AppOnly or Delegated)
    - ReusedSecret and HighVolumeReuse indicators
    - IP, City, Country
    - Suspect flag for large-scale access (>20 items)
'@

# Ensure PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "This script requires PowerShell 7+. Please run it from pwsh.exe."
    return
}

###### Edit this section #####
# Provide a single username, comma-separated usernames, or "All"
$userInput = "All"  # e.g., "user1@domain.com,user2@domain.com" or "All"

# Define number of days for the date range
$days = 360

# Calculate start and end dates
$startDate = (Get-Date).AddDays(-$days)
$endDate = Get-Date
$outputPath = ".\output"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
###### Edit this section #####

# Validate required modules
$requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Beta.Reports", "Microsoft.Graph.Authentication")
$missingModules = $requiredModules | Where-Object { -not (Get-Module -ListAvailable -Name $_) }
if ($missingModules) {
    Write-Error "Missing required modules: $($missingModules -join ', '). Please install them before proceeding."
    return
}

# Connect to Microsoft Graph
Import-Module Microsoft.Graph.Authentication
Import-Module Microsoft.Graph.Beta.Reports
if (-not (Get-MgContext)) {
    Connect-MgGraph -Scopes "AuditLog.Read.All,Directory.Read.All,IdentityRiskEvent.Read.All,Compliance.Read.All,Application.Read.All" -ContextScope Process -NoWelcome
}

# Retrieve core logs
$azureADAuthHistory = @()
if ($userInput -eq "All") {
    $azureADAuthHistory = Get-MgBetaAuditLogSignIn -All | Where-Object { $_.CreatedDateTime -ge $startDate }
} else {
    $userList = $userInput -split "," | ForEach-Object { $_.Trim() }
    foreach ($user in $userList) {
        $userResults = Get-MgBetaAuditLogSignIn -All -Filter "userPrincipalName eq '$user'" | Where-Object { $_.CreatedDateTime -ge $startDate }
        $azureADAuthHistory += $userResults
    }
}

# Export sign-ins (formatted and raw)
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

# Permission grants and role assignments
$permissionChanges = Get-MgBetaAuditLogDirectoryAudit -All | Where-Object {
    $_.ActivityDisplayName -match 'Add delegated permission grant|Add app role assignment|Consent to application' -and $_.ActivityDateTime -ge $startDate
}
$permissionChanges | ConvertTo-Json -Depth 10 | Out-File "$outputPath\PermissionChanges.json" -Encoding UTF8

$roleAssignments = Get-MgDirectoryRoleAssignment -All | Where-Object { $_.CreatedDateTime -ge $startDate }
$roleDefinitions = Get-MgDirectoryRole -All
$roleAssignmentsDetailed = foreach ($assignment in $roleAssignments) {
    $role = $roleDefinitions | Where-Object { $_.Id -eq $assignment.RoleDefinitionId }
    [PSCustomObject]@{
        RoleName       = $role.DisplayName
        Description    = $role.Description
        AssignedTo     = $assignment.PrincipalId
        RoleId         = $assignment.RoleDefinitionId
        DirectoryScope = $assignment.DirectoryScopeId
        CreatedDate    = $assignment.CreatedDateTime
    }
}
$roleAssignmentsDetailed | ConvertTo-Json -Depth 10 | Out-File "$outputPath\RoleAssignments.json" -Encoding UTF8

# Risky sign-ins
$riskySignIns = Get-MgBetaRiskyUser -All | Where-Object { $_.RiskLastUpdatedDateTime -ge $startDate }
$riskySignIns | ConvertTo-Json -Depth 10 | Out-File "$outputPath\RiskySignIns.json" -Encoding UTF8

# App secret tracking and reuse
$appSecretChanges = Get-MgBetaAuditLogDirectoryAudit -All | Where-Object {
    $_.ActivityDisplayName -match 'Add password|Update password' -and $_.ActivityDateTime -ge $startDate
}
$appSecretDetails = foreach ($entry in $appSecretChanges) {
    $entry.TargetResources | ForEach-Object {
        [PSCustomObject]@{
            Date        = $entry.ActivityDateTime
            Operation   = $entry.ActivityDisplayName
            AppId       = $_.Id
            AppName     = $_.DisplayName
            SecretId    = $_.ModifiedProperties | Where-Object { $_.DisplayName -eq 'KeyId' } | Select-Object -ExpandProperty NewValue
            OldSecretId = $_.ModifiedProperties | Where-Object { $_.DisplayName -eq 'KeyId' } | Select-Object -ExpandProperty OldValue
        }
    }
}
$appSecretDetails | Export-Csv -Path "$outputPath\AppSecretChanges.csv" -NoTypeInformation -Encoding UTF8

$reusedSecrets = $appSecretDetails | Group-Object SecretId | Where-Object { $_.Count -gt 1 } | Select-Object -ExpandProperty Group
$reusedSecrets | Export-Csv -Path "$outputPath\ReusedAppSecrets.csv" -NoTypeInformation -Encoding UTF8
$reusedSecretIds = $reusedSecrets | Select-Object -ExpandProperty SecretId -Unique

# Resource-level access logs and joined access report
$ualEvents = Search-UnifiedAuditLog -StartDate $startDate -EndDate $endDate -RecordType Mailbox,SharePointFileOperation,OneDriveFileOperation,MicrosoftTeams -ResultSize 5000
$ualFiltered = $ualEvents | Where-Object {
    $_.Operations -match 'Bind|Sync' -and ($_.UserIds -ne $_.AuditData.ObjectId)
}
$ualDetailed = foreach ($entry in $ualFiltered) {
    $data = ($entry.AuditData | ConvertFrom-Json)
    [PSCustomObject]@{
        Date        = $entry.CreationDate
        RecordType  = $entry.RecordType
        Operation   = $entry.Operation
        Workload    = $entry.Workload
        UserId      = $data.UserId
        AppId       = $data.AppId
        ObjectId    = $data.ObjectId
        ItemName    = $data.ItemName
        ItemId      = $data.ItemId
        MessageId   = $data.MessageId
        AttachmentId= $data.AttachmentId
        IsSync      = if ($entry.Operation -match 'Sync') { $true } else { $false }
        AccessType  = if ($data.UserId -and $data.AppId) { "Delegated" } elseif ($data.AppId) { "AppOnly" } else { "Unknown" }
    }
}
$ualDetailed | Export-Csv -Path "$outputPath\CrossAccessResourceAudit.csv" -NoTypeInformation -Encoding UTF8
$ualFiltered | ConvertTo-Json -Depth 10 | Out-File "$outputPath\CrossAccessResourceAudit.json" -Encoding UTF8

# High-volume access detection per user and secret
$suspectThreshold = 20
$groupedCounts = $ualDetailed | Group-Object UserId | Where-Object { $_.Count -gt $suspectThreshold }
$suspectUsers = $groupedCounts.Name

$secretUsageCount = @{}
foreach ($entry in $ualDetailed) {
    $secret = $appSecretDetails | Where-Object { $_.AppId -eq $entry.AppId } | Sort-Object Date -Descending | Select-Object -First 1
    if ($secret.SecretId) {
        if ($secretUsageCount.ContainsKey($secret.SecretId)) {
            $secretUsageCount[$secret.SecretId]++
        } else {
            $secretUsageCount[$secret.SecretId] = 1
        }
    }
}

# Join resource access and sign-in info
$crossAccess = foreach ($entry in $ualDetailed) {
    $signInMatch = $azureADAuthHistory | Where-Object {
        ($_.UserPrincipalName -eq $entry.UserId -or $_.AppId -eq $entry.AppId) -and $_.CreatedDateTime -le $entry.Date
    } | Sort-Object CreatedDateTime -Descending | Select-Object -First 1

    $secretUsed = $appSecretDetails | Where-Object { $_.AppId -eq $entry.AppId } | Sort-Object Date -Descending | Select-Object -First 1
    $isReusedSecret = if ($reusedSecretIds -contains $secretUsed.SecretId) { "Y" } else { "N" }
    $isHighVolumeReuse = if ($isReusedSecret -eq "Y" -and $secretUsageCount[$secretUsed.SecretId] -gt $suspectThreshold) { "Y" } else { "N" }

    [PSCustomObject]@{
        UserPrincipalName   = $entry.UserId
        AppId               = $entry.AppId
        Resource            = $entry.Workload
        ObjectId            = $entry.ObjectId
        ItemId              = $entry.ItemId
        MessageId           = $entry.MessageId
        AttachmentId        = $entry.AttachmentId
        Operation           = $entry.Operation
        IsSync              = $entry.IsSync
        AccessType          = $entry.AccessType
        SecretId            = $secretUsed.SecretId
        OldSecretId         = $secretUsed.OldSecretId
        ReusedSecret        = $isReusedSecret
        HighVolumeReuse     = $isHighVolumeReuse
        IPAddress           = $signInMatch.IpAddress
        City                = $signInMatch.Location.City
        Country             = $signInMatch.Location.CountryOrRegion
        Suspect             = if ($suspectUsers -contains $entry.UserId) { "Y" } else { "N" }
    }
}
$crossAccess | Export-Csv -Path "$outputPath\JoinedCrossAccessReport.csv" -NoTypeInformation -Encoding UTF8
