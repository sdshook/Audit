# Microsoft  Unified Audit Log Search Report
# Updated by Shane Shook (c)2025
# Reference https://learn.microsoft.com/en-us/purview/audit-log-search-script 
# Modifications based on Reference https://www.invictus-ir.com/news/what-dfir-experts-need-to-know-about-the-current-state-of-the-unified-audit-log

# Change the values for the following variables to configure the audit log search.
$logFile = ".\AuditLogSearchLog.txt"
$outputFile = ".\AuditLogRecords.csv"
$identityFilter = "<user@domain.com>"  # Can be UPNs or App IDs, comma-separated, or "ALL"
[DateTime]$start = [DateTime]::UtcNow.AddDays(-360)
[DateTime]$end = [DateTime]::UtcNow
# Note, to specify record types change the following
# $record = "AzureActiveDirectory" 
$record = $null
# Do not change the result size or you will suffer API problems
$resultSize = 5000
# Note, change the following according to expected size of org, in minutes - caps at 5000 events so it is important to assess
# whether smaller window is needed (i.e. incidents involving mass downloads or deletions etc.)
$intervalMinutes = 10080

# Handle identity input (usernames or app IDs)
if ($identityFilter -eq "ALL") {
    $userFilter = $null
} else {
    $userFilter = $identityFilter -split "," | ForEach-Object { $_.Trim() }
}

#Start script
[DateTime]$currentStart = $start
[DateTime]$currentEnd = $end

Function Write-LogFile ([String]$Message)
{
    $final = [DateTime]::Now.ToUniversalTime().ToString("s") + ":" + $Message
    $final | Out-File $logFile -Append
}

Write-LogFile "BEGIN: Retrieving audit records for $($identityFilter) between $($start) and $($end), RecordType=$record, PageSize=$resultSize."
Write-Host "Retrieving audit $($identityFilter) records for the date range between $($start) and $($end), RecordType=$record, ResultsSize=$resultSize"

# Note requires ExchangeOnlineManagement Module and Powershell v5.1+
Import-Module ExchangeOnlineManagement
Connect-ExchangeOnline 

$totalCount = 0
$firstExport = $true

while ($true)
{
    $currentEnd = $currentStart.AddMinutes($intervalMinutes)
    if ($currentEnd -gt $end) {
        $currentEnd = $end
    }
    if ($currentStart -eq $currentEnd) {
        break
    }

    $sessionID = [Guid]::NewGuid().ToString() + "_" +  "ExtractLogs" + (Get-Date).ToString("yyyyMMddHHmmssfff")
    Write-LogFile "INFO: Retrieving $($identityFilter) audit records for activities performed between $($currentStart) and $($currentEnd)"
    Write-Host "Retrieving $($identityFilter) audit records for activities performed between $($currentStart) and $($currentEnd)"
    $currentCount = 0

    $sw = [Diagnostics.StopWatch]::StartNew()
    do {
        if ($userFilter) {
            $results = Search-UnifiedAuditLog -UserIds $userFilter -StartDate $currentStart -EndDate $currentEnd -RecordType $record -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize
        } else {
            $results = Search-UnifiedAuditLog -StartDate $currentStart -EndDate $currentEnd -RecordType $record -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize
        }

        if ($results.Count -ne 0) {
            if ($firstExport) {
                $results | Export-Csv -Path $outputFile -NoTypeInformation -Encoding UTF8
                $firstExport = $false
            } else {
                $results | Export-Csv -Path $outputFile -Append -NoTypeInformation -Encoding UTF8
            }

            $currentTotal = $results[0].ResultCount
            $totalCount += $results.Count
            $currentCount += $results.Count
            Write-LogFile "INFO: Retrieved $($currentCount) audit records out of the total $($currentTotal)"

            if ($currentTotal -eq $results[$results.Count - 1].ResultIndex) {
                $message = "INFO: Successfully retrieved $($currentTotal) audit records for the current time range. Moving on!"
                Write-LogFile $message
                Write-Host "Successfully retrieved $($currentTotal) audit records for the current time range. Moving on to the next interval." -foregroundColor Yellow
                ""
                break
            }
        }
    } while ($results.Count -ne 0)

    $currentStart = $currentEnd
}

Write-LogFile "END: Retrieving $($identityFilter) audit records between $($start) and $($end), RecordType=$record, PageSize=$resultSize, total count: $totalCount."
Write-Host "Script complete! Finished retrieving $($identityFilter) audit records for the date range between $($start) and $($end). Total count: $totalCount" -foregroundColor Green
