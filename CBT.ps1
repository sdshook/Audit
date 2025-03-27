# Cyber Breach Triage Script
# Created by Shane Shook (c) 2025
# runAs administrator 
# PowerShell.exe -ExecutionPolicy bypass -WindowStyle hidden -File (path to script) 

# This script produces useful information to identify Hosts of Interest for examination after suspected breach.
# Match SHA1 signatures of binaries, DNS addresses, or IP addresses to known bad or suspicious threat information.
# Correlate low frequency process commands to suspicious activities; and improper users by host to activities.

Clear-Host
$localpath = '.' #update path to a preferred output location 
# Cache process and service data for performance
$WP = @{}
$Processes = Get-Process | Group-Object Id -AsHashTable -AsString
$WMIProcesses = Get-WmiObject Win32_Process | ForEach-Object { $WP[$_.ProcessID] = $_ }
$AuditDate = [int][double]::Parse((Get-Date -UFormat %s))
$Services = Get-CimInstance -Class Win32_Service | Group-Object ProcessId -AsHashTable -AsString

function Get-ConnectionReport($connections, $protocol) {
    $connections | Select-Object -Property LocalAddress, LocalPort, RemoteAddress, RemotePort, State,
        @{Name='Computername';Expression={$env:COMPUTERNAME}},
        @{Name='AuditDate';Expression={$AuditDate}},
        @{Name='Protocol';Expression={ $protocol }},
        @{Name='PID';Expression={$_.OwningProcess}},
        @{Name='Process';Expression={if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Name } else { "" } }},
        @{Name='UserName';Expression={try { (Get-Process -IncludeUserName -Id $_.OwningProcess).UserName } catch { "N/A" }}},
        @{Name='UserSID';Expression={try { ($WP[[UInt32]$_.OwningProcess]).GetOwnerSid().Sid.Value } catch { "N/A" }}},
        @{Name='ParentPID';Expression={try { ($WP[[UInt32]$_.OwningProcess]).ParentProcessId } catch { "" }}},
        @{Name='ParentProcess';Expression={
            try {
                $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Name } else { "" }
            } catch { "" }
        }},
        @{Name='ParentPath';Expression={
            try {
                $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
            } catch { "" }
        }},
        @{Name='ParentSHA1';Expression={
            try {
                $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                $ppath = if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
                if ($ppath) { (Get-FileHash -Path $ppath -Algorithm SHA1 -ErrorAction Stop).Hash } else { "" }
            } catch { "" }
        }},
        @{Name='ParentMD5';Expression={
            try {
                $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                $ppath = if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
                if ($ppath) { (Get-FileHash -Path $ppath -Algorithm MD5 -ErrorAction Stop).Hash } else { "" }
            } catch { "" }
        }},
        @{Name='ParentCommandLine';Expression={
            try {
                $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                if ($WP.ContainsKey($ppid)) { $WP[$ppid].CommandLine } else { "" }
            } catch { "" }
        }},
        @{Name='ServiceName';Expression={if ($Services.ContainsKey("$($_.OwningProcess)")) { $Services["$($_.OwningProcess)"].Name } else { "" }}},
        @{Name='ServiceStartType';Expression={if ($Services.ContainsKey("$($_.OwningProcess)")) { $Services["$($_.OwningProcess)"].StartMode } else { "" }}},
        @{Name='Path';Expression={if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Path } else { "" }}},
        @{Name='ProcessSHA1';Expression={
            try {
                $path = $Processes["$($_.OwningProcess)"].Path
                if ($path) { (Get-FileHash -Path $path -Algorithm SHA1 -ErrorAction Stop).Hash } else { "" }
            } catch { "" }
        }},
        @{Name='ProcessMD5';Expression={
            try {
                $path = $Processes["$($_.OwningProcess)"].Path
                if ($path) { (Get-FileHash -Path $path -Algorithm MD5 -ErrorAction Stop).Hash } else { "" }
            } catch { "" }
        }},
        @{Name='CommandLine';Expression={$WP[[UInt32]$_.OwningProcess].CommandLine}},
        @{Name='Connected';Expression={try { [int][double]::Parse((Get-Date $_.CreationTime -UFormat %s)) } catch { 0 }}}
}

$tcpReport = Get-ConnectionReport (Get-NetTCPConnection) "TCP"
$udpReport = Get-ConnectionReport (Get-NetUDPEndpoint) "UDP"

$combinedReport = $tcpReport + $udpReport

$combinedReport |
    Where-Object { $_.Process -ne 'Idle' } |
    Select-Object Computername, AuditDate, Protocol, UserName, UserSID, PID, ParentPID, ParentProcess, ParentPath, ParentSHA1, ParentMD5, ParentCommandLine, Process, ServiceName, Path, 
        ServiceStartType, ProcessSHA1, ProcessMD5, CommandLine, Connected, State, LocalAddress, LocalPort, RemoteAddress, RemotePort |
    Export-Csv -Path "$localpath\$env:COMPUTERNAME-activecomms.csv" -Append -Encoding UTF8 -NoTypeInformation -ErrorAction $ErrorActionPreference
