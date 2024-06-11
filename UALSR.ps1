# Microsoft  Unified Audit Log Search Report
# Created by Shane Shook (c)2024
# Reference https://learn.microsoft.com/en-us/purview/audit-log-search-script 
# Modifications based on Reference https://www.invictus-ir.com/news/what-dfir-experts-need-to-know-about-the-current-state-of-the-unified-audit-log

# Change the values for the following variables to configure the audit log search.
$logFile = ".\AuditLogSearchLog.txt"
$outputFile = ".\AuditLogRecords.csv"
$username = "<user@domain.com>"
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

#Start script
[DateTime]$currentStart = $start
[DateTime]$currentEnd = $end

Function Write-LogFile ([String]$Message)
{
    $final = [DateTime]::Now.ToUniversalTime().ToString("s") + ":" + $Message
    $final | Out-File $logFile -Append
}

Write-LogFile "BEGIN: Retrieving audit records for $($username) between $($start) and $($end), RecordType=$record, PageSize=$resultSize."
Write-Host "Retrieving audit $($username) records for the date range between $($start) and $($end), RecordType=$record, ResultsSize=$resultSize"

# Note requires ExchangeOnlineManagement Module and Powershell v5.1+
Import-Module ExchangeOnlineManagement
Connect-ExchangeOnline 

$totalCount = 0
while ($true)
{
    $currentEnd = $currentStart.AddMinutes($intervalMinutes)
    if ($currentEnd -gt $end)
    {
        $currentEnd = $end
    }

if ($currentStart -eq $currentEnd)
    {
        break
    }

$sessionID = [Guid]::NewGuid().ToString() + "_" +  "ExtractLogs" + (Get-Date).ToString("yyyyMMddHHmmssfff")
    Write-LogFile "INFO: Retrieving $($username) audit records for activities performed between $($currentStart) and $($currentEnd)"
    Write-Host "Retrieving $($username) audit records for activities performed between $($currentStart) and $($currentEnd)"
    $currentCount = 0

$sw = [Diagnostics.StopWatch]::StartNew()
    do
    {
        $results = Search-UnifiedAuditLog -UserIds $username -StartDate $currentStart -EndDate $currentEnd -RecordType $record -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize

if (($results | Measure-Object).Count -ne 0)
        {
            $results | export-csv -Path $outputFile -Append -NoTypeInformation

$currentTotal = $results[0].ResultCount
            $totalCount += $results.Count
            $currentCount += $results.Count
            Write-LogFile "INFO: Retrieved $($currentCount) audit records out of the total $($currentTotal)"

if ($currentTotal -eq $results[$results.Count - 1].ResultIndex)
            {
                $message = "INFO: Successfully retrieved $($currentTotal) audit records for the current time range. Moving on!"
                Write-LogFile $message
                Write-Host "Successfully retrieved $($currentTotal) audit records for the current time range. Moving on to the next interval." -foregroundColor Yellow
                ""
                break
            }
        }
    }
    while (($results | Measure-Object).Count -ne 0)

$currentStart = $currentEnd
}

Write-LogFile "END: Retrieving $($username) audit records between $($start) and $($end), RecordType=$record, PageSize=$resultSize, total count: $totalCount."
Write-Host "Script complete! Finished retrieving $($username) audit records for the date range between $($start) and $($end). Total count: $totalCount" -foregroundColor Green

