# Microsoft Unified Audit Log Search Report - Application Permission Changes
# (c) 2025 Shane Shook
# ref https://learn.microsoft.com/en-us/purview/audit-log-activities 
# Note that this script can be simply modified for any types of operations report by user or App

# Configuration
$logFile = ".\AuditLogSearchLog.txt"
$outputFile = ".\O365_App_Permissions_Changes.csv"

# Specify usernames or app IDs (comma-separated), or "ALL" for all entries
# Example: "admin@domain.com", "user1@domain.com, user2@domain.com", or app IDs like "00000000-0000-0000-0000-000000000000"
$identityFilter = "ALL"
[DateTime]$start = [DateTime]::UtcNow.AddDays(-360)
[DateTime]$end = [DateTime]::UtcNow
$record = "AzureActiveDirectory"
$resultSize = 5000
$intervalMinutes = 10080

# Handle input
if ($identityFilter -eq "ALL") {
    $userFilter = $null
} else {
    $userFilter = $identityFilter -split "," | ForEach-Object { $_.Trim() }
}

# Start script
[DateTime]$currentStart = $start
[DateTime]$currentEnd = $end

Function Write-LogFile ([String]$Message) {
    $final = [DateTime]::Now.ToUniversalTime().ToString("s") + ":" + $Message
    $final | Out-File $logFile -Append
}

Write-LogFile "BEGIN: Retrieving application permission changes for $($identityFilter) between $($start) and $($end), RecordType=$record, PageSize=$resultSize."
Write-Host "Retrieving application permission changes for $($identityFilter) between $($start) and $($end), RecordType=$record, ResultsSize=$resultSize"

# Connect to Exchange Online
Import-Module ExchangeOnlineManagement
Connect-ExchangeOnline 

$totalCount = 0
$AuditRecords = @()

while ($true) {
    $currentEnd = $currentStart.AddMinutes($intervalMinutes)
    if ($currentEnd -gt $end) { $currentEnd = $end }
    if ($currentStart -eq $currentEnd) { break }

    $sessionID = [Guid]::NewGuid().ToString() + "_" + "ExtractLogs" + (Get-Date).ToString("yyyyMMddHHmmssfff")
    Write-LogFile "INFO: Retrieving audit records from $($currentStart) to $($currentEnd)"
    Write-Host "Retrieving records from $($currentStart) to $($currentEnd)"

    #Note: Modify list of operations below according to your interests
    do {
        $operations = @(
            "Add application.",
            "Update application.",
            "Update application – Certificates and secrets management ",
            "Remove app role assignment from service principal.",
            "Add app role assignment to service principal.",
            "Add service principal.",
            "Add service principal credentials.",
            "Add owner to service principal.",
            "Update service principal."
        )

        try {
            if ($userFilter) {
                $results = Search-UnifiedAuditLog -UserIds $userFilter -StartDate $currentStart -EndDate $currentEnd -RecordType $record -Operations $operations -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize 
            } else {
                $results = Search-UnifiedAuditLog -StartDate $currentStart -EndDate $currentEnd -RecordType $record -Operations $operations -SessionId $sessionID -SessionCommand ReturnLargeSet -ResultSize $resultSize 
            }
        } catch {
            Write-LogFile "ERROR: Failed retrieving logs for $currentStart to $currentEnd - $_"
            break
        }

        if ($results.Count -ne 0) {
            foreach ($entry in $results) {
                $AuditData = $entry.AuditData | ConvertFrom-Json
                $contextType = if ($AuditData.UserId -match '@') { 'User' } else { 'App' }

                # Resolve AppName
                $appName = $AuditData.AppName
                if (-not $appName -or $appName -eq "") {
                    $displayNameProp = ($AuditData.ModifiedProperties | Where-Object { $_.Name -eq "DisplayName" }).NewValue
                    if ($displayNameProp) {
                        try { $appName = $displayNameProp | ConvertFrom-Json } catch { $appName = $displayNameProp }
                    }
                }
                if (-not $appName -or $appName -eq "") {
                    $appName = ($AuditData.Target | Where-Object { $_.Type -eq 1 } | Select-Object -First 1).ID
                }

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
                    AppName   = $appName
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
