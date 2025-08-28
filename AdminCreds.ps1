<#
.SYNOPSIS
  Export privileged admins from On-Prem AD and Azure AD (PIM-aware), plus
  app/service principal credentials and their recent changes.

  (c) 2025 Shane D. Shook, All Rights Reserved.

.DESCRIPTION
  Outputs:
    - OnPrem_AD_Admins.csv                (expanded on-prem privileged groups)
    - AzureAD_Admins_Active.csv           (PIM active assignments; users/SPs; group expansion)
    - AzureAD_Admins_Eligible.csv         (PIM eligible assignments; users/SPs; group expansion)
    - AzureAD_AppSecrets.csv              (flattened secrets/certs for SPs holding privileged roles)
    - AzureAD_AppSecretChanges.csv        (audit log events for secret/cert adds/removes/updates)
    - All_Admins.csv                      (merged view across on-prem + Azure)

  Notes:
    - Uses Microsoft Graph (v1.0 profile) for Azure pulls.
    - Uses RSAT ActiveDirectory module for on-prem group membership.
    - Group-based role assignments are flattened; each user/SP appears with InheritedFromGroup set.
    - Date filtering (see PARAMETERS) controls the *audit log* timeframe. If a timeframe is provided,
      AzureAD_AppSecrets.csv is additionally filtered to credentials whose validity overlaps the window.

.PARAMETERS
  -Mode   OP | AZ | BOTH     (default: BOTH)
  -OutDir Output directory   (default: current folder)
  -TenantId Optional AAD tenant ID for Connect-MgGraph

  Date filtering (choose one set; default is -All):
  -All                   -> no time filter
  -Start MMDDYYYY
  -End   MMDDYYYY        -> inclusive range, e.g. -Start 07012025 -End 08312025
  -Last  N               -> last N days (e.g., -Last 90)

.REQUIREMENTS
  On-Prem:
    - Windows joined to the domain, RSAT AD PowerShell module installed.
  Azure:
    - PowerShell modules: Microsoft.Graph
    - Scopes: RoleManagement.Read.Directory, Directory.Read.All, Group.Read.All,
              Application.Read.All, AuditLog.Read.All
    - Sufficient permissions to read role assignments and audit logs.

.HOW TO RUN

  # One-time installs (as needed)
  Add-WindowsCapability -Online -Name "Rsat.ActiveDirectory.DS-LDS.Tools~~~~0.0.1.0"
  Install-Module Microsoft.Graph -Scope CurrentUser

  # Everything, no time filter:
  .\AdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -All

  # Azure-only, last 120 days of app secret changes:
  .\AdminCreds.ps1 -Mode AZ -OutDir "C:\Exports" -Last 120

  # Both, explicit range (Jul 1–Aug 31, 2025):
  .\AdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -Start 07012025 -End 08312025

  # Azure-only with tenant ID:
  .\AdminCreds.ps1 -Mode AZ -OutDir "C:\Exports" -All -TenantId "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  
#>

[CmdletBinding(DefaultParameterSetName='All')]
param(
  [ValidateSet('OP','AZ','BOTH')]
  [string]$Mode = 'BOTH',

  [string]$OutDir = ".",
  [string]$TenantId,

  # --- Date filters ---
  [Parameter(ParameterSetName='All')]
  [switch]$All,

  [Parameter(ParameterSetName='Range')]
  [string]$Start,

  [Parameter(ParameterSetName='Range')]
  [string]$End,

  [Parameter(ParameterSetName='Last')]
  [int]$Last
)

# =========================
# Utilities & Validation
# =========================
$null = New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Ensure-Module {
  param([string]$Name)
  if (-not (Get-Module -ListAvailable -Name $Name)) {
    Write-Host "Installing module $Name ..." -ForegroundColor Yellow
    Install-Module $Name -Scope CurrentUser -Force -AllowClobber
  }
  Import-Module $Name -ErrorAction Stop
}

function Safe-String($v) { if ($null -eq $v -or $v -eq "") { $null } else { "$v" } }

# Parse date parameters -> returns @{ Start=<DateTimeNullable>; End=<DateTimeNullable> }
function Get-TimeWindow {
  param([switch]$All, [string]$Start, [string]$End, [int]$Last)

  if ($All -or ($PSCmdlet.ParameterSetName -eq 'All')) {
    return @{ Start = $null; End = $null }
  }

  if ($PSCmdlet.ParameterSetName -eq 'Last') {
    if ($Last -le 0) { throw "Parameter -Last must be a positive integer (days)." }
    $end = Get-Date
    $start = $end.AddDays(-[math]::Abs($Last))
    return @{ Start = $start; End = $end }
  }

  # Range parsing (MMDDYYYY)
  $parse = {
    param($s)
    if (-not $s) { return $null }
    if ($s -notmatch '^\d{8}$') { throw "Date '$s' must be MMDDYYYY." }
    [datetime]::ParseExact($s, 'MMddyyyy', $null)
  }
  $s = & $parse $Start
  $e = & $parse $End
  if ($s -and $e -and $e -lt $s) { throw "-End cannot be before -Start." }
  return @{ Start = $s; End = $e }
}

$Window = Get-TimeWindow -All:$All -Start:$Start -End:$End -Last:$Last

# Helper: does [start,end] overlap with time window?
function In-Window {
  param([datetime]$ItemStart, [datetime]$ItemEnd, $Window)
  if (-not $Window.Start -and -not $Window.End) { return $true } # no filter
  $wStart = $Window.Start
  $wEnd   = $Window.End
  # Treat null start/end as open-ended
  if (-not $ItemStart) { $ItemStart = [datetime]::MinValue }
  if (-not $ItemEnd)   { $ItemEnd   = [datetime]::MaxValue }
  # Overlap test
  return ($ItemStart -le $wEnd) -and ($ItemEnd -ge $wStart)
}

# =========================
# On-Prem AD
# =========================
function Get-OPAdmins {
  Write-Host "Collecting On-Prem AD privileged groups..." -ForegroundColor Cyan
  Ensure-Module -Name ActiveDirectory

  $PrivGroups = @(
    'Administrators','Domain Admins','Enterprise Admins','Schema Admins',
    'Account Operators','Backup Operators','Server Operators','Print Operators',
    'DnsAdmins','Group Policy Creator Owners','Key Admins','Enterprise Key Admins'
  )

  $rows = foreach ($g in $PrivGroups) {
    try {
      $members = Get-ADGroupMember -Identity $g -Recursive -ErrorAction Stop
    } catch {
      Write-Warning "Failed reading group '$g' : $($_.Exception.Message)"
      continue
    }
    foreach ($m in $members | Where-Object {$_.objectClass -eq 'user'}) {
      $u = Get-ADUser -Identity $m.DistinguishedName -Properties PasswordLastSet, LastLogonDate, Enabled, DoesNotRequirePreAuth, UseDESKeyOnly, PasswordNeverExpires, SamAccountName
      [pscustomobject]@{
        Source                 = 'OnPrem'
        Scope                  = 'OnPrem'
        RoleOrGroup            = $g
        ObjectType             = 'user'
        DisplayName            = $null
        UserPrincipalName      = $null
        AccountEnabled         = $null
        ObjectId               = $null
        InheritedFromGroup     = $null
        SamAccountName         = $u.SamAccountName
        Enabled                = $u.Enabled
        LastLogonDate          = $u.LastLogonDate
        PasswordLastSet        = $u.PasswordLastSet
        DoesNotRequirePreAuth  = $u.DoesNotRequirePreAuth
        UseDESKeyOnly          = $u.UseDESKeyOnly
        PasswordNeverExpires   = $u.PasswordNeverExpires
      }
    }
  }

  $opCsv = Join-Path $OutDir "OnPrem_AD_Admins.csv"
  $rows | Sort-Object RoleOrGroup, SamAccountName | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $opCsv
  Write-Host "On-Prem export: $opCsv" -ForegroundColor Green
  return $rows
}

# =========================
# Azure AD (Microsoft Graph)
# =========================
function Connect-GraphIfNeeded {
  $scopes = @(
    'RoleManagement.Read.Directory',
    'Directory.Read.All',
    'Group.Read.All',
    'Application.Read.All',
    'AuditLog.Read.All'
  )
  if (-not (Get-Module -ListAvailable -Name Microsoft.Graph)) {
    Ensure-Module -Name Microsoft.Graph
  }
  if (-not (Get-MgContext)) {
    if ($TenantId) { Connect-MgGraph -Scopes $scopes -TenantId $TenantId }
    else           { Connect-MgGraph -Scopes $scopes }
  }
  Select-MgProfile -Name v1.0
}

# Caches
$script:RoleDefs  = $null
$script:UserCache = @{}
$script:SpCache   = @{}
$script:GroupCache= @{}
$script:AppCache  = @{}

function Get-RoleDefinitions {
  if ($null -eq $script:RoleDefs) {
    $script:RoleDefs = @{}
    Get-MgRoleManagementDirectoryRoleDefinition -All | ForEach-Object {
      $script:RoleDefs[$_.Id] = $_
    }
  }
  return $script:RoleDefs
}

function Get-UserById($id) {
  if ($script:UserCache.ContainsKey($id)) { return $script:UserCache[$id] }
  $u = Get-MgUser -UserId $id -ErrorAction SilentlyContinue
  $script:UserCache[$id] = $u
  return $u
}

function Get-SpById($id) {
  if ($script:SpCache.ContainsKey($id)) { return $script:SpCache[$id] }
  $sp = Get-MgServicePrincipal -ServicePrincipalId $id -ErrorAction SilentlyContinue
  $script:SpCache[$id] = $sp
  return $sp
}

function Get-AppByAppId($appId) {
  if (-not $appId) { return $null }
  if ($script:AppCache.ContainsKey($appId)) { return $script:AppCache[$appId] }
  $app = Get-MgApplication -Filter "appId eq '$appId'" -ErrorAction SilentlyContinue | Select-Object -First 1
  $script:AppCache[$appId] = $app
  return $app
}

function Get-GroupById($id) {
  if ($script:GroupCache.ContainsKey($id)) { return $script:GroupCache[$id] }
  $g = Get-MgGroup -GroupId $id -ErrorAction SilentlyContinue
  $script:GroupCache[$id] = $g
  return $g
}

function Expand-GroupMembersFlatten {
  param([Parameter(Mandatory)] [string] $GroupId)
  $members = Get-MgGroupTransitiveMember -GroupId $GroupId -All -ErrorAction SilentlyContinue
  foreach ($m in $members) {
    if ($m.'@odata.type' -eq '#microsoft.graph.user') {
      $u = Get-UserById $m.Id
      [pscustomobject]@{
        ObjectType        = 'user'
        DisplayName       = Safe-String $u.DisplayName
        UserPrincipalName = Safe-String $u.UserPrincipalName
        AccountEnabled    = $u.AccountEnabled
        ObjectId          = $u.Id
      }
    } elseif ($m.'@odata.type' -eq '#microsoft.graph.servicePrincipal') {
      $sp = Get-SpById $m.Id
      [pscustomobject]@{
        ObjectType        = 'servicePrincipal'
        DisplayName       = Safe-String $sp.DisplayName
        UserPrincipalName = $null
        AccountEnabled    = $null
        ObjectId          = $sp.Id
      }
    }
  }
}

function Convert-PrincipalToRow {
  param(
    [Parameter(Mandatory)] $principal,
    [Parameter(Mandatory)] [string] $roleName,
    [Parameter(Mandatory)] [string] $scopeLabel # 'Active' | 'Eligible'
  )

  $principalId = if ($principal.PrincipalId) { $principal.PrincipalId } else { $principal.Id }
  $principalType = if ($principal.PrincipalType) { $principal.PrincipalType } elseif ($principal.AdditionalProperties['principal']['@odata.type']) {
    $principal.AdditionalProperties['principal']['@odata.type'] -replace '^#microsoft\.graph\.', ''
  } else { $principal.AdditionalProperties['principalType'] }

  switch -Regex ($principalType) {
    '^user$' {
      $u = Get-UserById $principalId
      return ,([pscustomobject]@{
        Source                 = 'AzureAD'
        Scope                  = $scopeLabel
        RoleOrGroup            = $roleName
        ObjectType             = 'user'
        DisplayName            = Safe-String $u.DisplayName
        UserPrincipalName      = Safe-String $u.UserPrincipalName
        AccountEnabled         = $u.AccountEnabled
        ObjectId               = $u.Id
        InheritedFromGroup     = $null
        SamAccountName         = $null
        Enabled                = $null
        LastLogonDate          = $null
        PasswordLastSet        = $null
        DoesNotRequirePreAuth  = $null
        UseDESKeyOnly          = $null
        PasswordNeverExpires   = $null
      })
    }
    '^servicePrincipal$' {
      $sp = Get-SpById $principalId
      return ,([pscustomobject]@{
        Source                 = 'AzureAD'
        Scope                  = $scopeLabel
        RoleOrGroup            = $roleName
        ObjectType             = 'servicePrincipal'
        DisplayName            = Safe-String $sp.DisplayName
        UserPrincipalName      = $null
        AccountEnabled         = $sp.AccountEnabled
        ObjectId               = $sp.Id
        InheritedFromGroup     = $null
        SamAccountName         = $null
        Enabled                = $null
        LastLogonDate          = $null
        PasswordLastSet        = $null
        DoesNotRequirePreAuth  = $null
        UseDESKeyOnly          = $null
        PasswordNeverExpires   = $null
      })
    }
    '^group$' {
      $g = Get-GroupById $principalId
      $gMembers = Expand-GroupMembersFlatten -GroupId $g.Id
      $out = @()
      foreach ($gm in $gMembers) {
        $out += [pscustomobject]@{
          Source                 = 'AzureAD'
          Scope                  = $scopeLabel
          RoleOrGroup            = $roleName
          ObjectType             = $gm.ObjectType
          DisplayName            = Safe-String $gm.DisplayName
          UserPrincipalName      = Safe-String $gm.UserPrincipalName
          AccountEnabled         = $gm.AccountEnabled
          ObjectId               = $gm.ObjectId
          InheritedFromGroup     = $g.DisplayName
          SamAccountName         = $null
          Enabled                = $null
          LastLogonDate          = $null
          PasswordLastSet        = $null
          DoesNotRequirePreAuth  = $null
          UseDESKeyOnly          = $null
          PasswordNeverExpires   = $null
        }
      }
      return ,$out
    }
    default {
      return ,([pscustomobject]@{
        Source                 = 'AzureAD'
        Scope                  = $scopeLabel
        RoleOrGroup            = $roleName
        ObjectType             = Safe-String $principalType
        DisplayName            = $null
        UserPrincipalName      = $null
        AccountEnabled         = $null
        ObjectId               = $principalId
        InheritedFromGroup     = $null
        SamAccountName         = $null
        Enabled                = $null
        LastLogonDate          = $null
        PasswordLastSet        = $null
        DoesNotRequirePreAuth  = $null
        UseDESKeyOnly          = $null
        PasswordNeverExpires   = $null
      })
    }
  }
}

function Get-AZAdmins {
  Write-Host "Collecting Azure AD PIM (active + eligible)..." -ForegroundColor Cyan
  Connect-GraphIfNeeded
  $roleDefs = Get-RoleDefinitions

  # ACTIVE
  $active = @()
  Get-MgRoleManagementDirectoryRoleAssignmentScheduleInstance -All |
    ForEach-Object {
      $roleName = ($roleDefs[$_.RoleDefinitionId]).DisplayName
      $active += Convert-PrincipalToRow -principal $_ -roleName $roleName -scopeLabel 'Active'
    }

  # ELIGIBLE
  $eligible = @()
  Get-MgRoleManagementDirectoryRoleEligibilityScheduleInstance -All |
    ForEach-Object {
      $roleName = ($roleDefs[$_.RoleDefinitionId]).DisplayName
      $eligible += Convert-PrincipalToRow -principal $_ -roleName $roleName -scopeLabel 'Eligible'
    }

  # Write CSVs
  $azActiveCsv   = Join-Path $OutDir "AzureAD_Admins_Active.csv"
  $azEligibleCsv = Join-Path $OutDir "AzureAD_Admins_Eligible.csv"

  $active   | Sort-Object RoleOrGroup, ObjectType, UserPrincipalName, DisplayName | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $azActiveCsv
  $eligible | Sort-Object RoleOrGroup, ObjectType, UserPrincipalName, DisplayName | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $azEligibleCsv

  Write-Host "Azure AD exports:`n  $azActiveCsv`n  $azEligibleCsv" -ForegroundColor Green
  return @{ Active = $active; Eligible = $eligible }
}

# ---- Secrets/certs for SPs holding roles; window-applied if provided ----
function Get-AZAppSecretsFromRoleHolders {
  param(
    [Parameter(Mandatory)] [System.Collections.IEnumerable] $ActiveRows,
    [Parameter(Mandatory)] [System.Collections.IEnumerable] $EligibleRows,
    $Window
  )

  Connect-GraphIfNeeded

  $spIds = @(
    $ActiveRows  | Where-Object {$_.ObjectType -eq 'servicePrincipal'} | Select-Object -Expand ObjectId
    $EligibleRows| Where-Object {$_.ObjectType -eq 'servicePrincipal'} | Select-Object -Expand ObjectId
  ) | Sort-Object -Unique

  $out = @()
  foreach ($spId in $spIds) {
    $sp = Get-SpById $spId
    if (-not $sp) { continue }
    $app = Get-AppByAppId $sp.AppId

    $appId   = $app.AppId
    $appName = Safe-String $app.DisplayName
    $spName  = Safe-String $sp.DisplayName

    foreach ($p in ($app.PasswordCredentials | ForEach-Object { $_ })) {
      if (-not (In-Window -ItemStart $p.StartDateTime -ItemEnd $p.EndDateTime -Window $Window)) {
        if ($Window.Start -or $Window.End) { continue }
      }
      $out += [pscustomobject]@{
        AppId                        = $appId
        ApplicationDisplayName       = $appName
        ServicePrincipalId           = $sp.Id
        ServicePrincipalDisplayName  = $spName
        CredentialType               = 'Secret'
        CredentialDisplayName        = Safe-String $p.DisplayName
        StartDateTime                = $p.StartDateTime
        EndDateTime                  = $p.EndDateTime
        KeyId                        = $p.KeyId
        Hint                         = $null
      }
    }
    foreach ($k in ($app.KeyCredentials | ForEach-Object { $_ })) {
      if (-not (In-Window -ItemStart $k.StartDateTime -ItemEnd $k.EndDateTime -Window $Window)) {
        if ($Window.Start -or $Window.End) { continue }
      }
      $out += [pscustomobject]@{
        AppId                        = $appId
        ApplicationDisplayName       = $appName
        ServicePrincipalId           = $sp.Id
        ServicePrincipalDisplayName  = $spName
        CredentialType               = 'Certificate'
        CredentialDisplayName        = Safe-String $k.DisplayName
        StartDateTime                = $k.StartDateTime
        EndDateTime                  = $k.EndDateTime
        KeyId                        = $k.KeyId
        Hint                         = if ($k.CustomKeyIdentifier) { [System.Convert]::ToBase64String($k.CustomKeyIdentifier) } else { $null }
      }
    }
  }

  $csv = Join-Path $OutDir "AzureAD_AppSecrets.csv"
  $out | Sort-Object ApplicationDisplayName, CredentialType, EndDateTime |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
  Write-Host "App secrets/certs export: $csv" -ForegroundColor Green
  return $out
}

# ---- Audit log: app secret/cert changes within window ----
function Get-AZAppSecretChanges {
  param($Window)

  Connect-GraphIfNeeded

  $filter = $null
  if ($Window.Start -and $Window.End) {
    $filter = "activityDateTime ge $($Window.Start.ToString('o')) and activityDateTime le $($Window.End.ToString('o'))"
  } elseif ($Window.Start) {
    $filter = "activityDateTime ge $($Window.Start.ToString('o'))"
  } elseif ($Window.End) {
    $filter = "activityDateTime le $($Window.End.ToString('o'))"
  }

  $events = if ($filter) {
    try { Get-MgAuditLogDirectoryAudit -All -Filter $filter } catch { Get-MgAuditLogDirectoryAudit -All }
  } else {
    Get-MgAuditLogDirectoryAudit -All
  }

  $wanted = $events | Where-Object {
    ($_.ActivityDisplayName -match 'servicePrincipal' -and $_.ActivityDisplayName -match '(password|key|certificate|credential)') `
      -or ($_.ActivityDisplayName -match 'application' -and $_.ActivityDisplayName -match '(password|key|certificate|credential)') `
      -or ($_.ActivityDisplayName -match 'Certificates and secrets')
  }

  $rows = foreach ($e in $wanted) {
    $target = $e.TargetResources | Select-Object -First 1
    [pscustomobject]@{
      ActivityDateTime     = $e.ActivityDateTime
      ActivityDisplayName  = $e.ActivityDisplayName
      InitiatedBy          = $e.InitiatedBy.User.DisplayName
      Category             = $e.Category
      TargetType           = if ($target) { $target.Type } else { $null }
      TargetId             = if ($target) { $target.Id } else { $null }
      AdditionalDetails    = ($e.AdditionalDetails | ConvertTo-Json -Compress -Depth 3)
    }
  }

  # If a window exists, apply client-side filter too (defensive)
  if ($Window.Start -or $Window.End) {
    $rows = $rows | Where-Object {
      ($null -eq $Window.Start -or $_.ActivityDateTime -ge $Window.Start) -and
      ($null -eq $Window.End   -or $_.ActivityDateTime -le $Window.End)
    }
  }

  $csv = Join-Path $OutDir "AzureAD_AppSecretChanges.csv"
  $rows | Sort-Object ActivityDateTime | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
  Write-Host "App secret/cert changes export: $csv" -ForegroundColor Green
  return $rows
}

# =========================
# Run
# =========================
$opRows = @()
$az = $null

switch ($Mode) {
  'OP'   { $opRows = Get-OPAdmins }
  'AZ'   { $az = Get-AZAdmins }
  'BOTH' { $opRows = Get-OPAdmins; $az = Get-AZAdmins }
}

# Azure extras (secrets + changes) with time window
if ($az) {
  $null = Get-AZAppSecretsFromRoleHolders -ActiveRows $az.Active -EligibleRows $az.Eligible -Window $Window
  $null = Get-AZAppSecretChanges -Window $Window
}

# Merge outputs
$all = @()
if ($opRows) { $all += $opRows }
if ($az)     { $all += $az.Active; $all += $az.Eligible }

$allCsv = Join-Path $OutDir "All_Admins.csv"
$all |
  Sort-Object Source, Scope, RoleOrGroup, ObjectType, UserPrincipalName, DisplayName |
  Export-Csv -NoTypeInformation -Encoding UTF8 -Path $allCsv

Write-Host "Merged export: $allCsv" -ForegroundColor Green
