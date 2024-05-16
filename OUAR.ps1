# Microsoft User Activity Report
# Created by Shane Shook (c)2024
# Note: adjust username and days as required

# Define the user
$username = "username@domain.com"

# Define the number of days for the date range
$days = 30

# Calculate start and end dates
$startDate = (Get-Date).AddDays(-$days)
$endDate = Get-Date

# Function to perform an NSLookup for an IP address
function Get-HostnameFromIP {
    param (
        [string]$IPAddress
    )

    $hostnames = @()

    try {
        # Perform NSLookup (DNS reverse lookup)
        $result = [System.Net.Dns]::GetHostEntry($IPAddress)

        foreach ($address in $result.AddressList) {
            $hostnames += @{
                "IPAddress" = $address.IPAddressToString
                "Hostname" = $result.HostName
            }
        }
    }
    catch {
        # If NSLookup fails, return the original IP address
        $hostnames += @{
            "IPAddress" = $IPAddress
            "Hostname" = "Unknown"
        }
    }

    return $hostnames
}

# Connect to Office 365 Security & Compliance Center
Connect-IPPSSession

# Get tenant identifier
$tenantIdentifier = (Get-OrganizationConfig).TenantGuid

# Get Office 365 tenant time zone
$tenantTimeZone = (Get-OrganizationConfig).TimeZone

# Connect to Azure Active Directory
Connect-AzureAD

# Get Azure AD tenant identifier
$azureADTenant = (Get-AzureADTenantDetail).ObjectId

# Get user activities from Office 365
$userActivities = Search-UnifiedAuditLog -UserIds $username -StartDate $startDate -EndDate $endDate |
    Select-Object @{Name="TenantIdentifier"; Expression={$tenantIdentifier}}, @{Name="TenantTimeZone"; Expression={$tenantTimeZone}}, CreationDate, Operation, UserIds, RecordType, @{Name="Details"; Expression={$_.AuditData | ConvertTo-Json -Depth 10}}

# Get device usage activities from Office 365 Management Activity API
$deviceUsageActivities = Search-UnifiedAuditLog -StartDate $startDate -EndDate $endDate -Operations "AzureActiveDirectory" |
    Where-Object { $_.ResultStatus -eq "success" -and $_.ClientIP -ne $null -and $_.UserIds -contains $username } |
    Select-Object @{Name="TenantIdentifier"; Expression={$tenantIdentifier}}, @{Name="TenantTimeZone"; Expression={$tenantTimeZone}}, CreationDate, Operation, UserIds, ClientIP, @{Name="Details"; Expression={$_.AuditData | ConvertTo-Json -Depth 10}}

# Get authentication activities from Office 365 Management Activity API
$authActivities = Search-UnifiedAuditLog -StartDate $startDate -EndDate $endDate -Operations "UserLoggedIn" |
    Where-Object { $_.ResultStatus -eq "success" -and $_.UserIds -contains $username } |
    Select-Object @{Name="TenantIdentifier"; Expression={$tenantIdentifier}}, @{Name="TenantTimeZone"; Expression={$tenantTimeZone}}, CreationDate, Operation, UserIds, ClientIP, @{Name="Details"; Expression={$_.AuditData | ConvertTo-Json -Depth 10}}

# Get Azure AD authentication history
$azureADAuthHistory = Get-AzureADAuditSignInLogs -Filter "userPrincipalName eq '$username'" -Top 1000 |
    Select-Object @{Name="TenantIdentifier"; Expression={$azureADTenant}}, CreationTime, UserPrincipalName, IPAddress, @{Name="OperatingSystem"; Expression={$_.OperatingSystem}},
    @{Name="BrowserType"; Expression={$_.Browser}}, @{Name="DeviceName"; Expression={$_.DeviceDetail | ConvertTo-Json -Depth 10}}

# Get hostname for IP addresses
$hostnames = @()
foreach ($activity in $deviceUsageActivities) {
    $hostnameInfo = Get-HostnameFromIP -IPAddress $activity.ClientIP
    $hostnames += $hostnameInfo
}

foreach ($activity in $authActivities) {
    $hostnameInfo = Get-HostnameFromIP -IPAddress $activity.ClientIP
    $hostnames += $hostnameInfo
}

# Combine all activities with hostname information
$allActivities = $userActivities + $deviceUsageActivities + $authActivities + $azureADAuthHistory + $hostnames

# Sort activities by time
$sortedActivities = $allActivities | Sort-Object CreationDate

# Export to CSV
$sortedActivities | Export-Csv -Path "UserActivities.csv" -NoTypeInformation
