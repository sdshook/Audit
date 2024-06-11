# Microsoft Azure Active Directory Signins Report
# Created by Shane Shook (c)2024
# Note: adjust username and days as required

# Requires PSv7 and DotNetv8
# It is necessary to first have the following installed (Warning these need to be manually installed)
## Install-Module -Name PackageManagement
## Install-Module -Name Microsoft.Graph -Scope AllUsers
## Install-Module -Name Microsoft.Graph.Beta 

Function Load-Powershell_7{

    function New-OutOfProcRunspace {
        param($ProcessId)

        $connectionInfo = New-Object -TypeName System.Management.Automation.Runspaces.NamedPipeConnectionInfo -ArgumentList @($ProcessId)

        $TypeTable = [System.Management.Automation.Runspaces.TypeTable]::LoadDefaultTypeFiles()

        #$Runspace = [System.Management.Automation.Runspaces.RunspaceFactory]::CreateOutOfProcessRunspace($connectionInfo,$Host,$TypeTable)
        $Runspace = [System.Management.Automation.Runspaces.RunspaceFactory]::CreateRunspace($connectionInfo,$Host,$TypeTable)

        $Runspace.Open()
        $Runspace
    }

    $Process = Start-Process PWSH -ArgumentList @("-NoExit") -PassThru -WindowStyle Hidden

    $Runspace = New-OutOfProcRunspace -ProcessId $Process.Id

    $Host.PushRunspace($Runspace)
}
Load-Powershell_7


###### Edit this section #####
# Define the user
$username = "<user@domain.com>"

# Define the number of days for the date range
$days = 360

# Calculate start and end dates
$startDate = (Get-Date).AddDays(-$days)
$endDate = Get-Date
###### Edit this section #####


# Get User Signin Activities from Azure AD via Microsoft Graph API (will prompt for web AuthN)
# Important Note: AzAdSigninLogs is deprecated, and Get-MgAuditLogSignIn is missing data
# Must use Get-MgBetaAuditLogSignIn...
Import-Module Microsoft.Graph.Authentication
Import-Module Microsoft.Graph.Beta.Reports
Connect-MgGraph -Scopes AuditLog.Read.All -ContextScope Process -NoWelcome 

# Select Formatted Events for Review
$azureADAuthHistory = Get-MgBetaAuditLogSignIn -All -Filter "userPrincipalName eq '$username'" | where-object {$_.createddatetime -ge $startDate}
$results = foreach ($signIn in $azureADAuthHistory) {
	[PSCustomObject]@{
		UserPrincipalName = $signIn.UserPrincipalName
		UserDisplayName = $signIn.UserDisplayName
		cDate = $signIn.CreatedDateTime
		AppName = $signIn.AppDisplayName
		AppUsed = $signIn.ClientAppUsed
		Resource = $signIn.ResourceDisplayName
		IPAddress = $signIn.IpAddress
		City = $signIn.Location.City
		LocST = $signIn.Location.State
		Country = $signIn.Location.CountryOrRegion
		DeviceId = $signIn.DeviceDetail.DeviceId
		DeviceDetail = $signIn.DeviceDetail.DisplayName
		DeviceOS = $signIn.DeviceDetail.OperatingSystem
		DeviceBrowser = $signIn.DeviceDetail.Browser
        UserAgent = $signIn.UserAgent
	}
}
$results | Export-Csv -Path .\UALDF.csv -NoTypeInformation

# Raw Events for System of Record Maintenance
$azureADAuthHistoryRaw = Get-MgBetaAuditLogSignIn -All -Filter "userPrincipalName eq '$username'" | convertto-json -depth 10
$azureADAuthHistoryRaw | Out-File .\UALDR.json

