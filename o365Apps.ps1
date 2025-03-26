# Microsoft Unified Audit Log Search Report - Application Permission Changes
# Modified to include operation, actions performed, old and new values
# (c) 2025 Shane Shook

# Configuration
$logFile = ".\AuditLogSearchLog.txt"
$outputFile = ".\O365_App_Permissions_Changes.json"
$username = "<user@domain.com>"
[DateTime]$start = [DateTime]::UtcNow.AddDays(-360)
[DateTime]$end = [DateTime]::UtcNow
$record = "AzureActiveDirectory"
$resultSize = 5000
$intervalMinutes = 10080

# Start script
[DateTime]$currentStart = $start
[DateTime]$currentEnd = $end

Function Write-LogFile ([String]$Message)
{
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

while ($true)
{
    $currentEnd = $currentStart.AddMinutes($intervalMinutes)
    if ($currentEnd -gt $end) { $currentEnd = $end }
    if ($currentStart -eq $currentEnd) { break }

    $sessionID = [Guid]::NewGuid().ToString() + "_" +  "ExtractLogs" + (Get-Date).ToString("yyyyMMddHHmmssfff")
    Write-LogFile "INFO: Retrieving $($username) audit records from $($currentStart) to $($currentEnd)"
    Write-Host "Retrieving records from $($currentStart) to $($currentEnd)"

    do
    {
        # Retrieve audit log events for application permission changes
        $results = Search-UnifiedAuditLog -UserIds $username -StartDate $currentStart -EndDate $currentEnd -RecordType $record -Operations "Consent to application", "Update application permissions" -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize 

        if ($results.Count -ne 0)
        {
            foreach ($entry in $results)
            {
                $AuditData = $entry.AuditData | ConvertFrom-Json

                $oldValue = if ($AuditData.OldValue) { $AuditData.OldValue | ConvertTo-Json -Depth 3 } else { $null }
                $newValue = if ($AuditData.NewValue) { $AuditData.NewValue | ConvertTo-Json -Depth 3 } else { $null }

                $AuditRecords += [PSCustomObject]@{
                    Date        = $entry.CreationDate
                    Operation   = $entry.Operation
                    AppId       = $AuditData.AppId
                    AppName     = $AuditData.AppName
                    UPN         = $AuditData.UserId
                    Actions     = $AuditData.ModifiedProperties.Name -join ", "
                    OldValue    = $oldValue
                    NewValue    = $newValue
                    RawRecord   = $AuditData | ConvertTo-Json -Depth 3
                }
            }

            $totalCount += $results.Count
            Write-LogFile "INFO: Retrieved $($results.Count) records. Total so far: $totalCount"

            if ($results[$results.Count - 1].ResultIndex -eq $results[0].ResultCount)
            {
                Write-LogFile "INFO: Reached the end of the result set for this time range."
                break
            }
        }
    }
    while ($results.Count -ne 0)

    $currentStart = $currentEnd
}

# Export results to JSON
$AuditRecords | ConvertTo-Json -Depth 3 | Out-File $outputFile
Write-Host "Audit data saved to $outputFile" -ForegroundColor Green
Write-LogFile "END: Retrieved $totalCount application permission change records."

# Disconnect session
Disconnect-ExchangeOnline -Confirm:$false
