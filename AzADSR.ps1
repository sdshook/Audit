# Microsoft Azure Active Directory Signins Report
# Updated by Shane Shook (c)2025
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
        $Runspace = [System.Management.Automation.Runspaces.RunspaceFactory]::CreateRunspace($connectionInfo,$Host,$TypeTable)
        $Runspace.Open()
        $Runspace
    }

    if ($PSVersionTable.PSVersion.Major -lt 7) {
        $Process = Start-Process PWSH -ArgumentList @("-NoExit") -PassThru -WindowStyle Hidden
        $Runspace = New-OutOfProcRunspace -ProcessId $Process.Id
        $Host.PushRunspace($Runspace)
    }
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
$outputPath = ".\output"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
###### Edit this section #####

# Validate required modules
$requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Beta.Reports", "Microsoft.Graph.Authentication")
foreach ($mod in $requiredModules) {
    if (-not (Get-Module -ListAvailable -Name $mod)) {
        Write-Warning "$mod is not installed. Please install it before running this script."
    }
}

# Get User Signin Activities from Azure AD via Microsoft Graph API (will prompt for web AuthN)
Import-Module Microsoft.Graph.Authentication
Import-Module Microsoft.Graph.Beta.Reports
Connect-MgGraph -Scopes AuditLog.Read.All -ContextScope Process -NoWelcome 

# Collect sign-in logs once
$azureADAuthHistory = Get-MgBetaAuditLogSignIn -All -Filter "userPrincipalName eq '$username'" | Where-Object { $_.CreatedDateTime -ge $startDate }

# Export formatted events for review
$results = foreach ($signIn in $azureADAuthHistory) {
    [PSCustomObject]@{
        UserPrincipalName = $signIn.UserPrincipalName
        UserDisplayName   = $signIn.UserDisplayName
        cDate             = $signIn.CreatedDateTime
        AppName           = $signIn.AppDisplayName
        AppUsed           = $signIn.ClientAppUsed
        Resource          = $signIn.ResourceDisplayName
        IPAddress         = $signIn.IpAddress
        City              = $signIn.Location.City
        LocST             = $signIn.Location.State
        Country           = $signIn.Location.CountryOrRegion
        DeviceId          = $signIn.DeviceDetail.DeviceId
        DeviceDetail      = $signIn.DeviceDetail.DisplayName
        DeviceOS          = $signIn.DeviceDetail.OperatingSystem
        DeviceBrowser     = $signIn.DeviceDetail.Browser
        UserAgent         = $signIn.UserAgent
    }
}
$results | Export-Csv -Path "$outputPath\UALDF.csv" -NoTypeInformation -Encoding UTF8

# Export raw data for system-of-record
$azureADAuthHistory | ConvertTo-Json -Depth 10 | Out-File "$outputPath\UALDR.json" -Encoding UTF8
