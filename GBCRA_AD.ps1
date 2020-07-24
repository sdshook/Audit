# Audit Cyber Risk Active Directory Assessment (ADA) Copyright @2017 All Rights Reserved 
# Updated by Shane D. Shook 
# Version 201709221400
# Runas:  PowerShell.exe -ExecutionPolicy bypass -WindowStyle hidden (path to script) 
$ErrorActionPreference = 'Continue'
# Variables declared here
#$exportPath = "\\<path>\Out\AD"
$exportPath = "C:\Windows\Temp"

Add-WindowsFeature RSAT-AD-PowerShell
#if AD is not installed on host that audit is being performed from it adds powershell 

Import-Module ActiveDirectory
#Add-WindowsFeature RSAT-AD-PowerShell
#Change the "SearchBase to target the certain OU you wish you pull from or use a wildcard
# example:  -Filter * -SearchBase "ou=Herriard Lab,dc=room101,dc=local"
# example2:  -Filter *

#AD Users details
Get-ADUser -Filter * -Properties * | 
select @{Name='AuditDate';Expression={ Get-Date -Uformat %s }},
canonicalname,  
samaccountname, 
displayname,  
isdeleted, 
distinguishedname, 
cn, 
@{ Label = "created"; Expression = { $_.created | Get-Date -Uformat %s }},
@{ Label = "modified"; Expression = { $_.modified | Get-Date -Uformat %s }},
department, 
@{ Label = "passwordlastset"; Expression = { $_.PasswordLastSet | Get-Date -Uformat %s }},
Enabled, 
PasswordNotRequired,  
PasswordNeverExpires, 
PasswordExpired, 
lastlogontimestamp,
@{ Label = "accountexpires"; Expression = { $_.accountexpires | Get-Date -Uformat %s }} |  
Export-Csv "$Exportpath\AD_Usersview.csv" -Encoding UTF8 -NoTypeInformation

#AD Group membership details
$Groups = (Get-AdGroup -filter * | Where {$_.name -like "**"} | select name -expandproperty name)
$Table = @()
$Record = [ordered]@{
"Group Name" = ""
"Name" = ""
"Username" = ""
}
Foreach ($Group in $Groups)
{
$Arrayofmembers = Get-ADGroupMember -identity $Group | select name,samaccountname
foreach ($Member in $Arrayofmembers)
{
$Record."Group Name" = $Group
$Record."Name" = $Member.name
$Record."UserName" = $Member.samaccountname
$objRecord = New-Object PSObject -property $Record
$Table += $objrecord
}
}
$Table | select @{Name='AuditDate';Expression={ Get-Date -Uformat %s }}, "Group Name", Name, UserName |
Export-Csv "$Exportpath\AD_Groupmembersview.csv" -Encoding UTF8 -NoTypeInformation


#AD Computers details
Get-ADComputer -Filter * -Properties * | 
select @{Name='AuditDate';Expression={ Get-Date -Uformat %s }},
canonicalname,  
samaccountname, 
displayname,  
isdeleted, 
distinguishedname, 
cn, 
@{ Label = "whencreated"; Expression = { $_.whencreated | Get-Date -Uformat %s }},
@{ Label = "whenchanged"; Expression = { $_.whenchanged | Get-Date -Uformat %s }},
operatingsystem, 
operatingsystemversion, 
operatingsystemservicepack, 
lastlogontimestamp | 
Export-Csv "$Exportpath\AD_Computersview.csv" -Encoding UTF8 -NoTypeInformation