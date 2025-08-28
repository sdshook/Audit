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

.PARAMETERS
  -Mode   OP | AZ | BOTH     (default: BOTH)
  -OutDir Output directory   (default: current folder)
  -TenantId Optional AAD tenant ID (or domain) for Connect-MgGraph

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

.HOW TO RUN

  # One-time installs (as needed, run elevated for RSAT)
  Add-WindowsCapability -Online -Name "Rsat.ActiveDirectory.DS-LDS.Tools~~~~0.0.1.0"
  Install-Module Microsoft.Graph -Scope CurrentUser

  # Everything, no time filter:
  .\AdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -All

  # Azure-only, last 120 days of app secret changes:
  .\AdminCreds.ps1 -Mode AZ -OutDir "C:\Exports" -Last 120

  # Both, explicit range (Jul 1–Aug 31, 2025):
  .\AdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -Start 07012025 -End 08312025

#>

param(
  [ValidateSet('OP','AZ','BOTH')] [string] $Mode = 'BOTH',
  [string] $OutDir = '.',
  [string] $TenantId,

  [Parameter(ParameterSetName='All')] [switch] $All,
  [Parameter(ParameterSetName='Range')] [string] $Start,
  [Parameter(ParameterSetName='Range')] [string] $End,
  [Parameter(ParameterSetName='Last')] [int] $Last
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Ensure-Module {
  param([Parameter(Mandatory)][string]$Name)
  if (-not (Get-Module -ListAvailable -Name $Name)) {
    Write-Host "Installing module $Name ..." -ForegroundColor Yellow
    Install-Module $Name -Scope CurrentUser -Force -AllowClobber
  }
  Import-Module $Name -ErrorAction Stop | Out-Null
}

function Safe-String($v) { if ($null -eq $v -or $v -eq '') { $null } else { "$v" } }

# Parse date parameters
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

function In-Window {
  param(
    [datetime] $ItemStart,
    [datetime] $ItemEnd,
    [Parameter(Mandatory)] [hashtable] $Window
  )
  if (-not $Window.Start -and -not $Window.End) { return $true }
  $ItemStart = if ($ItemStart) { $ItemStart } else { [datetime]::MinValue }
  $ItemEnd   = if ($ItemEnd)   { $ItemEnd   } else { [datetime]::MaxValue }
  $wStart    = if ($Window.Start) { $Window.Start } else { [datetime]::MinValue }
  $wEnd      = if ($Window.End)   { $Window.End   } else { [datetime]::MaxValue }
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
      $u = Get-ADUser -Identity $m.DistinguishedName -Properties PasswordLastSet, LastLogonDate, Enabled, DoesNotRequirePreAuth, UseDESKeyOnly, PasswordNeverExpires, SamAccountName, DisplayName, UserPrincipalName
      [pscustomobject]@{
        Source                 = 'OnPrem'
        Scope                  = 'OnPrem'
        RoleOrGroup            = $g
        ObjectType             = 'user'
        DisplayName            = $u.DisplayName
        UserPrincipalName      = $u.UserPrincipalName
        AccountEnabled         = $u.Enabled
        ObjectId               = $null
        InheritedFromGroup     = $g
        SamAccountName         = $u.SamAccountName
        Enabled                = $u.Enabled
        LastLogonDate          = $u.LastLogonDate
        PasswordLastSet        = $u.PasswordLastSet
        DoesNotRequirePreAuth  = $u.DoesNotRequirePreAuth
        UseDESKeyOnly          = $u.UseDESKeyOnly
        PasswordNeverExpires   = $u.PasswordNeverExpires
        DirectoryScopeId       = $null
        AssignmentStart        = $null
        AssignmentEnd          = $null
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
  Select-MgProfile -Name 'v1.0'
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
  $app = Get-MgApplication -Filter "appId eq '$appId'" -All -Property "AppId,DisplayName,PasswordCredentials,KeyCredentials" | Select-Object -First 1
  $script:AppCache[$appId] = $app
  return $app
}

function Get-GroupById($id) {
  if ($script:GroupCache.ContainsKey($id)) { return $script:GroupCache[$id] }
  $g = Get-MgGroup -GroupId $id -Property "Id,DisplayName" -ErrorAction SilentlyContinue
  $script:GroupCache[$id] = $g
  return $g
}

function Expand-GroupMembersFlatten {
  param([Parameter(Mandatory)] [string] $GroupId)
  # Robust: don't rely solely on @odata.type; try user first, then SP
  $members = Get-MgGroupTransitiveMember -GroupId $GroupId -All -ErrorAction SilentlyContinue
  foreach ($m in $members) {
    $u = Get-UserById $m.Id
    if ($u) {
      [pscustomobject]@{
        ObjectType        = 'user'
        DisplayName       = Safe-String $u.DisplayName
        UserPrincipalName = Safe-String $u.UserPrincipalName
        AccountEnabled    = $u.AccountEnabled
        ObjectId          = $u.Id
      }
      continue
    }
    $sp = Get-SpById $m.Id
    if ($sp) {
      [pscustomobject]@{
        ObjectType        = 'servicePrincipal'
        DisplayName       = Safe-String $sp.DisplayName
        UserPrincipalName = $null
        AccountEnabled    = $sp.AccountEnabled
        ObjectId          = $sp.Id
      }
      continue
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
  $ptype = $principal.PrincipalType
  if (-not $ptype -and $principal.AdditionalProperties) {
    $ptype = $principal.AdditionalProperties['principalType']
    if (-not $ptype -and $principal.AdditionalProperties['principal']) {
      $t = $principal.AdditionalProperties['principal']['@odata.type']
      if ($t) { $ptype = ($t -replace '^#microsoft\.graph\.', '') }
    }
  }

  $scopeId   = $principal.DirectoryScopeId
  $startTime = $principal.StartDateTime
  $endTime   = $principal.EndDateTime

  switch -Regex ($ptype) {
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
        DirectoryScopeId       = $scopeId
        AssignmentStart        = $startTime
        AssignmentEnd          = $endTime
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
        DirectoryScopeId       = $scopeId
        AssignmentStart        = $startTime
        AssignmentEnd          = $endTime
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
          DirectoryScopeId       = $scopeId
          AssignmentStart        = $startTime
          AssignmentEnd          = $endTime
        }
      }
      return ,$out
    }
    default {
      return ,([pscustomobject]@{
        Source                 = 'AzureAD'
        Scope                  = $scopeLabel
        RoleOrGroup            = $roleName
        ObjectType             = $ptype
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
        DirectoryScopeId       = $scopeId
        AssignmentStart        = $startTime
        AssignmentEnd          = $endTime
      })
    }
  }
}

function Get-AZAdmins {
  param([hashtable]$Window)

  Write-Host "Collecting Azure AD PIM (active + eligible)..." -ForegroundColor Cyan
  Connect-GraphIfNeeded | Out-Null
  $null = Get-RoleDefinitions

  $active = @()
  foreach ($it in (Get-MgRoleManagementDirectoryRoleAssignmentScheduleInstance -All)) {
    $roleName = if ($script:RoleDefs.ContainsKey($it.RoleDefinitionId)) { $script:RoleDefs[$it.RoleDefinitionId].DisplayName } else { $it.RoleDefinitionId }
    $active += Convert-PrincipalToRow -principal $it -roleName $roleName -scopeLabel 'Active'
  }

  $eligible = @()
  foreach ($it in (Get-MgRoleManagementDirectoryRoleEligibilityScheduleInstance -All)) {
    $roleName = if ($script:RoleDefs.ContainsKey($it.RoleDefinitionId)) { $script:RoleDefs[$it.RoleDefinitionId].DisplayName } else { $it.RoleDefinitionId }
    $eligible += Convert-PrincipalToRow -principal $it -roleName $roleName -scopeLabel 'Eligible'
  }

  $azActiveCsv = Join-Path $OutDir "AzureAD_Admins_Active.csv"
  $azEligibleCsv = Join-Path $OutDir "AzureAD_Admins_Eligible.csv"
  $active   | Sort-Object RoleOrGroup, ObjectType, UserPrincipalName, DisplayName | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $azActiveCsv
  $eligible | Sort-Object RoleOrGroup, ObjectType, UserPrincipalName, DisplayName | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $azEligibleCsv
  Write-Host "Azure Active export   : $azActiveCsv" -ForegroundColor Green
  Write-Host "Azure Eligible export : $azEligibleCsv" -ForegroundColor Green

  # Return both sets for downstream
  [pscustomobject]@{ Active = $active; Eligible = $eligible }
}

function Get-AZAppSecrets {
  param([hashtable]$Window, [array]$ActiveRows, [array]$EligibleRows)

  Write-Host "Collecting app/service principal credentials..." -ForegroundColor Cyan
  Connect-GraphIfNeeded | Out-Null

  $spIds = @(
    $ActiveRows  | Where-Object {$_.ObjectType -eq 'servicePrincipal'} | Select-Object -Expand ObjectId
    $EligibleRows| Where-Object {$_.ObjectType -eq 'servicePrincipal'} | Select-Object -Expand ObjectId
  ) | Sort-Object -Unique

  $out = @()
  foreach ($spId in $spIds) {
    $sp = Get-SpById $spId
    if (-not $sp) { continue }
    $app = Get-AppByAppId $sp.AppId
    if (-not $app) { continue }

    $appId   = $app.AppId
    $appName = Safe-String $app.DisplayName
    $spName  = Safe-String $sp.DisplayName

    foreach ($p in ($app.PasswordCredentials | ForEach-Object { $_ })) {
      $sd = $p.StartDateTime; $ed = $p.EndDateTime
      if (-not (In-Window -ItemStart $sd -ItemEnd $ed -Window $Window)) {
        if ($Window.Start -or $Window.End) { continue }
      }
      $out += [pscustomobject]@{
        AppId                        = $appId
        ApplicationDisplayName       = $appName
        ServicePrincipalId           = $sp.Id
        ServicePrincipalDisplayName  = $spName
        CredentialType               = 'Secret'
        CredentialDisplayName        = Safe-String $p.DisplayName
        StartDateTime                = $sd
        EndDateTime                  = $ed
        KeyId                        = $p.KeyId
        Hint                         = $null
      }
    }

    foreach ($k in ($app.KeyCredentials | ForEach-Object { $_ })) {
      $sd = $k.StartDateTime; $ed = $k.EndDateTime
      if (-not (In-Window -ItemStart $sd -ItemEnd $ed -Window $Window)) {
        if ($Window.Start -or $Window.End) { continue }
      }
      $hint = if ($k.CustomKeyIdentifier) { [System.Convert]::ToBase64String($k.CustomKeyIdentifier) } else { $null }
      $out += [pscustomobject]@{
        AppId                        = $appId
        ApplicationDisplayName       = $appName
        ServicePrincipalId           = $sp.Id
        ServicePrincipalDisplayName  = $spName
        CredentialType               = 'Certificate'
        CredentialDisplayName        = Safe-String $k.DisplayName
        StartDateTime                = $sd
        EndDateTime                  = $ed
        KeyId                        = $k.KeyId
        Hint                         = $hint
      }
    }
  }

  $csv = Join-Path $OutDir "AzureAD_AppSecrets.csv"
  $out | Sort-Object ApplicationDisplayName, CredentialType, EndDateTime | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
  Write-Host "App secrets export: $csv" -ForegroundColor Green
  return $out
}

function Get-AZAppSecretChanges {
  param([hashtable]$Window)

  Write-Host "Collecting app secret/cert change audit logs..." -ForegroundColor Cyan
  Connect-GraphIfNeeded | Out-Null

  $filter = $null
  if ($Window.Start -and $Window.End) {
    $filter = "activityDateTime ge {0:s} and activityDateTime le {1:s}" -f ($Window.Start.ToUniversalTime().ToString("s") + "Z"), ($Window.End.ToUniversalTime().ToString("s") + "Z")
  } elseif ($Window.Start) {
    $filter = "activityDateTime ge {0:s}" -f ($Window.Start.ToUniversalTime().ToString("s") + "Z")
  } elseif ($Window.End) {
    $filter = "activityDateTime le {0:s}" -f ($Window.End.ToUniversalTime().ToString("s") + "Z")
  }

  $args = @{ All = $true }
  if ($filter) { $args['Filter'] = $filter }

  $rows = @()
  foreach ($e in (Get-MgAuditLogDirectoryAudit @args)) {
    $name = $e.ActivityDisplayName
    if ($null -eq $name) { $name = '' }
    $name = $name.ToLower()

    if ($name -match 'serviceprincipal.*(password|key|certificate|credential)' -or
        $name -match 'application.*(password|key|certificate|credential)' -or
        $name -match 'certificates and secrets') {

      $target = $e.TargetResources | Select-Object -First 1
      $user   = $e.InitiatedBy.User

      $initiator = $null
      if ($user) {
        if ($user.UserPrincipalName) { $initiator = $user.UserPrincipalName }
        elseif ($user.DisplayName)   { $initiator = $user.DisplayName }
      }

      $rows += [pscustomobject]@{
        ActivityDateTime     = $e.ActivityDateTime
        ActivityDisplayName  = $e.ActivityDisplayName
        InitiatedBy          = $initiator
        Category             = $e.Category
        TargetType           = if ($target) { $target.Type } else { $null }
        TargetId             = if ($target) { $target.Id } else { $null }
        AdditionalDetails    = ($e.AdditionalDetails | ConvertTo-Json -Compress -Depth 3)
      }
    }
  }

  # Client-side guard in case server filter is quirky
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

$Window = Get-TimeWindow -All:$All -Start:$Start -End:$End -Last:$Last

# Ensure outdir
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

switch ($Mode) {
  'OP'   { $opRows = Get-OPAdmins }
  'AZ'   { $az = Get-AZAdmins -Window $Window }
  'BOTH' { $opRows = Get-OPAdmins; $az = Get-AZAdmins -Window $Window }
}

if ($Mode -in @('AZ','BOTH')) {
  $secrets = Get-AZAppSecrets -Window $Window -ActiveRows $az.Active -EligibleRows $az.Eligible
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
