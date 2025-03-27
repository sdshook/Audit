# Microsoft Unified Audit Log Search Report - Application Permission Changes
# Allows querying a specific user, multiple users, or all users
# Exports to CSV with raw JSON per record
# (c) 2025 Shane Shook

# Configuration
$logFile = ".\AuditLogSearchLog.txt"
$outputFile = ".\O365_App_Permissions_Changes.csv"

# Specify a username, multiple usernames (comma-separated), or "ALL" for all users
$username = "<user@domain.com>"  # Example: "admin@domain.com" or "user1@domain.com, user2@domain.com"
[DateTime]$start = [DateTime]::UtcNow.AddDays(-360)
[DateTime]$end = [DateTime]::UtcNow
$record = "AzureActiveDirectory"
$resultSize = 5000
$intervalMinutes = 10080

# Handle user input
if ($username -eq "ALL") {
    $userFilter = $null  # No filtering; retrieves logs for all users
} else {
    $userFilter = $username -split "," | ForEach-Object { $_.Trim() }
}

# Start script
[DateTime]$currentStart = $start
[DateTime]$currentEnd = $end

Function Write-LogFile ([String]$Message) {
    $final = [DateTime]::Now.ToUniversalTime().ToString("s") + ":" + $Message
    $final | Out-File $logFile -Append
}

Write-LogFile "BEGIN: Retrieving application permission changes for $($username) between $($start) and $($end), RecordType=$record, PageSize=$resultSize."
Write-Host "Retrieving application permission changes for $($username) between $($start) and $($end), RecordType=$record, ResultsSize=$resultSize"

# Connect to Exchange Online
Import-Module ExchangeOnlineManagement
Connect-ExchangeOnline 

$totalCount = 0
$AuditRecords = @()

while ($true) {
    $currentEnd = $currentStart.AddMinutes($intervalMinutes)
    if ($currentEnd -gt $end) { $currentEnd = $end }
    if ($currentStart -eq $currentEnd) { break }

    $sessionID = [Guid]::NewGuid().ToString() + "_" +  "ExtractLogs" + (Get-Date).ToString("yyyyMMddHHmmssfff")
    Write-LogFile "INFO: Retrieving audit records from $($currentStart) to $($currentEnd)"
    Write-Host "Retrieving records from $($currentStart) to $($currentEnd)"

    do {
        # Retrieve audit log events including app role assignments
        $operations = @("Consent to application", "Update application", "Remove application", "Add app role assignment to service principal")

        if ($userFilter) {
            $results = Search-UnifiedAuditLog -UserIds $userFilter -StartDate $currentStart -EndDate $currentEnd -RecordType $record -Operations $operations -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize 
        } else {
            $results = Search-UnifiedAuditLog -StartDate $currentStart -EndDate $currentEnd -RecordType $record -Operations $operations -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize 
        }

        if ($results.Count -ne 0) {
            foreach ($entry in $results) {
                $AuditData = $entry.AuditData | ConvertFrom-Json

                # Determine if context is User or App based on UserId
                $contextType = if ($AuditData.UserId -match '@') { 'User' } else { 'App' }

                # Extract old and new values if available
                $Changes = @()
                if ($AuditData.ModifiedProperties) {
                    foreach ($change in $AuditData.ModifiedProperties) {
                        $Changes += "Property: $($change.Name); Old: $($change.OldValue -join ', '); New: $($change.NewValue -join ', ')"
                    }
                }

                # Store extracted data
                $AuditRecords += [PSCustomObject]@{
                    Date      = $entry.CreationDate
                    Operation = $entry.Operations
                    AppId     = $AuditData.AppId
                    AppName   = $AuditData.AppName
                    UPN       = $AuditData.UserId
                    Context   = $contextType
                    Actions   = $AuditData.Operation
                    Changes   = $Changes -join " | "
                    RawRecord = ($AuditData | ConvertTo-Json -Depth 3 -Compress)
                }
            }

            $totalCount += $results.Count
            Write-LogFile "INFO: Retrieved $($results.Count) records. Total so far: $totalCount"

            if ($results[$results.Count - 1].ResultIndex -eq $results[0].ResultCount) {
                Write-LogFile "INFO: Reached the end of the result set for this time range."
                break
            }
        }
    } while ($results.Count -ne 0)

    $currentStart = $currentEnd
}

# Export results to CSV
$AuditRecords | Export-Csv -Path $outputFile -NoTypeInformation -Encoding UTF8
Write-Host "Audit data saved to $outputFile" -ForegroundColor Green
Write-LogFile "END: Retrieved $totalCount application permission change records."

# Disconnect session
Disconnect-ExchangeOnline -Confirm:$false
