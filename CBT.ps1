# Cyber Breach Triage Script
# Created by Shane Shook (c) 2025
# runAs administrator 
# PowerShell.exe -ExecutionPolicy bypass -WindowStyle hidden -File (path to script) 

# This script produces useful information to identify Hosts of Interest for examination after suspected breach.
# Match SHA1 signatures of binaries, DNS addresses, or IP addresses to known bad or suspicious threat information.
# Correlate low frequency process commands to suspicious activities; and improper users by host to activities.

# COMPATIBILITY: This script supports Windows 2000 through modern Windows versions
# - Uses Get-CimInstance when available (PowerShell 3.0+/Windows Vista+), falls back to Get-WmiObject
# - Uses Get-NetTCPConnection/Get-NetUDPEndpoint when available (PowerShell 3.0+/Windows 8+), falls back to netstat parsing
# - Uses Get-FileHash when available (PowerShell 4.0+), falls back to .NET Framework cryptography classes
# - Handles PowerShell version differences for Export-Csv parameters and process username retrieval

Clear-Host
$localpath = '.' #update path to a preferred output location 

# Detect PowerShell and Windows versions for compatibility
try {
    $PSVersion = if ($PSVersionTable -and $PSVersionTable.PSVersion) { $PSVersionTable.PSVersion.Major } else { 1 }
} catch {
    $PSVersion = 1  # Assume PowerShell 1.0 if PSVersionTable doesn't exist
}

try {
    $WinVersion = [System.Environment]::OSVersion.Version
} catch {
    # Fallback for very old systems
    $WinVersion = New-Object System.Version(5, 0)  # Assume Windows 2000
}

# Backwards-compatible wrapper functions
function Get-CompatibleProcess {
    try {
        if ($PSVersion -ge 4) {
            Get-Process -IncludeUserName -ErrorAction SilentlyContinue
        } else {
            Get-Process -ErrorAction SilentlyContinue
        }
    } catch {
        Get-Process -ErrorAction SilentlyContinue
    }
}

function Get-CompatibleWMIProcess {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            Get-CimInstance -Class Win32_Process -ErrorAction SilentlyContinue
        } else {
            Get-WmiObject Win32_Process -ErrorAction SilentlyContinue
        }
    } catch {
        Get-WmiObject Win32_Process -ErrorAction SilentlyContinue
    }
}

function Get-CompatibleService {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            Get-CimInstance -Class Win32_Service -ErrorAction SilentlyContinue
        } else {
            Get-WmiObject Win32_Service -ErrorAction SilentlyContinue
        }
    } catch {
        Get-WmiObject Win32_Service -ErrorAction SilentlyContinue
    }
}

function Get-CompatibleFileHash {
    param(
        [string]$Path,
        [string]$Algorithm = "SHA1"
    )
    
    if (-not (Test-Path $Path)) { return "" }
    
    try {
        # Try modern Get-FileHash first (PowerShell 4.0+)
        if ($PSVersion -ge 4) {
            return (Get-FileHash -Path $Path -Algorithm $Algorithm -ErrorAction Stop).Hash
        }
    } catch { }
    
    try {
        # Fallback to .NET Framework methods for older systems
        if ($Algorithm -eq "SHA1") {
            $hasher = [System.Security.Cryptography.SHA1]::Create()
        } elseif ($Algorithm -eq "MD5") {
            $hasher = [System.Security.Cryptography.MD5]::Create()
        } else {
            return ""
        }
        
        $fileStream = [System.IO.File]::OpenRead($Path)
        $hashBytes = $hasher.ComputeHash($fileStream)
        $fileStream.Close()
        $hasher.Dispose()
        
        return [System.BitConverter]::ToString($hashBytes).Replace("-", "")
    } catch {
        return ""
    }
}

function Get-CompatibleNetworkConnections {
    $connections = @()
    
    try {
        # Try modern network cmdlets first (PowerShell 3.0+ / Windows 8+)
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6 -and $WinVersion.Minor -ge 2) {
            $tcpConnections = Get-NetTCPConnection -ErrorAction SilentlyContinue | ForEach-Object {
                $_ | Add-Member -NotePropertyName "Protocol" -NotePropertyValue "TCP" -PassThru
            }
            $udpConnections = Get-NetUDPEndpoint -ErrorAction SilentlyContinue | ForEach-Object {
                $_ | Add-Member -NotePropertyName "Protocol" -NotePropertyValue "UDP" -PassThru
                $_ | Add-Member -NotePropertyName "State" -NotePropertyValue "N/A" -PassThru
                $_ | Add-Member -NotePropertyName "RemoteAddress" -NotePropertyValue "0.0.0.0" -PassThru
                $_ | Add-Member -NotePropertyName "RemotePort" -NotePropertyValue 0 -PassThru
            }
            return $tcpConnections + $udpConnections
        }
    } catch { }
    
    try {
        # Fallback to netstat parsing for older systems
        $netstatOutput = netstat -ano 2>$null
        if ($netstatOutput) {
            foreach ($line in $netstatOutput) {
                if ($line -match '^\s*(TCP|UDP)\s+([^\s]+):(\d+)\s+([^\s]+):(\d+)\s+(\w+)?\s*(\d+)?\s*$') {
                    $protocol = $matches[1]
                    $localAddr = $matches[2]
                    $localPort = [int]$matches[3]
                    $remoteAddr = $matches[4]
                    $remotePort = [int]$matches[5]
                    $state = if ($matches[6]) { $matches[6] } else { "N/A" }
                    $pid = if ($matches[7]) { [int]$matches[7] } else { 0 }
                    
                    $conn = New-Object PSObject -Property @{
                        Protocol = $protocol
                        LocalAddress = $localAddr
                        LocalPort = $localPort
                        RemoteAddress = $remoteAddr
                        RemotePort = $remotePort
                        State = $state
                        OwningProcess = $pid
                        CreationTime = Get-Date
                    }
                    $connections += $conn
                }
            }
        }
    } catch { }
    
    return $connections
}

function Get-ProcessUserName {
    param([int]$ProcessId)
    
    try {
        if ($PSVersion -ge 4) {
            $proc = Get-Process -Id $ProcessId -IncludeUserName -ErrorAction SilentlyContinue
            if ($proc) { return $proc.UserName }
        }
        
        # Fallback using WMI
        $wmiProc = Get-WmiObject Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
        if ($wmiProc) {
            $owner = $wmiProc.GetOwner()
            if ($owner.Domain -and $owner.User) {
                return "$($owner.Domain)\$($owner.User)"
            }
        }
    } catch { }
    
    return "N/A"
}

# Cache process and service data for performance
$WP = @{}
$Processes = Get-CompatibleProcess | Group-Object Id -AsHashTable -AsString
$WMIProcesses = Get-CompatibleWMIProcess | ForEach-Object { $WP[$_.ProcessID] = $_ }
$AuditDate = [int][double]::Parse((Get-Date -UFormat %s))
$Services = Get-CompatibleService | Group-Object ProcessId -AsHashTable -AsString

function Get-ConnectionReport($connections) {
    $connections | Select-Object -Property LocalAddress, LocalPort, RemoteAddress, RemotePort, State,
        @{Name='Computername';Expression={$env:COMPUTERNAME}},
        @{Name='AuditDate';Expression={$AuditDate}},
        @{Name='Protocol';Expression={$_.Protocol}},
        @{Name='PID';Expression={$_.OwningProcess}},
        @{Name='Process';Expression={if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Name } else { "" } }},
        @{Name='UserName';Expression={Get-ProcessUserName -ProcessId $_.OwningProcess}},
        @{Name='UserSID';Expression={
            try { 
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    if ($PSVersion -ge 3) {
                        ($WP[[UInt32]$_.OwningProcess]).GetOwnerSid().Sid.Value
                    } else {
                        ($WP[[UInt32]$_.OwningProcess]).GetOwnerSid().Sid
                    }
                } else { "N/A" }
            } catch { "N/A" }
        }},
        @{Name='ParentPID';Expression={try { if ($WP.ContainsKey([UInt32]$_.OwningProcess)) { ($WP[[UInt32]$_.OwningProcess]).ParentProcessId } else { "" } } catch { "" }}},
        @{Name='ParentProcess';Expression={
            try {
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                    if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Name } else { "" }
                } else { "" }
            } catch { "" }
        }},
        @{Name='ParentPath';Expression={
            try {
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                    if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
                } else { "" }
            } catch { "" }
        }},
        @{Name='ParentSHA1';Expression={
            try {
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                    $ppath = if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
                    if ($ppath) { Get-CompatibleFileHash -Path $ppath -Algorithm "SHA1" } else { "" }
                } else { "" }
            } catch { "" }
        }},
        @{Name='ParentMD5';Expression={
            try {
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                    $ppath = if ($Processes.ContainsKey("$ppid")) { $Processes["$ppid"].Path } else { "" }
                    if ($ppath) { Get-CompatibleFileHash -Path $ppath -Algorithm "MD5" } else { "" }
                } else { "" }
            } catch { "" }
        }},
        @{Name='ParentCommandLine';Expression={
            try {
                if ($WP.ContainsKey([UInt32]$_.OwningProcess)) {
                    $ppid = ($WP[[UInt32]$_.OwningProcess]).ParentProcessId
                    if ($WP.ContainsKey($ppid)) { $WP[$ppid].CommandLine } else { "" }
                } else { "" }
            } catch { "" }
        }},
        @{Name='ServiceName';Expression={
            if ($Services.ContainsKey("$($_.OwningProcess)")) { 
                $svc = $Services["$($_.OwningProcess)"]
                if ($svc -is [Array]) { $svc[0].Name } else { $svc.Name }
            } else { "" }
        }},
        @{Name='ServiceStartType';Expression={
            if ($Services.ContainsKey("$($_.OwningProcess)")) { 
                $svc = $Services["$($_.OwningProcess)"]
                if ($svc -is [Array]) { $svc[0].StartMode } else { $svc.StartMode }
            } else { "" }
        }},
        @{Name='Path';Expression={if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Path } else { "" }}},
        @{Name='ProcessSHA1';Expression={
            try {
                $path = if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Path } else { "" }
                if ($path) { Get-CompatibleFileHash -Path $path -Algorithm "SHA1" } else { "" }
            } catch { "" }
        }},
        @{Name='ProcessMD5';Expression={
            try {
                $path = if ($Processes.ContainsKey("$($_.OwningProcess)")) { $Processes["$($_.OwningProcess)"].Path } else { "" }
                if ($path) { Get-CompatibleFileHash -Path $path -Algorithm "MD5" } else { "" }
            } catch { "" }
        }},
        @{Name='CommandLine';Expression={try { if ($WP.ContainsKey([UInt32]$_.OwningProcess)) { $WP[[UInt32]$_.OwningProcess].CommandLine } else { "" } } catch { "" }}},
        @{Name='Connected';Expression={try { [int][double]::Parse((Get-Date $_.CreationTime -UFormat %s)) } catch { 0 }}}
}

# Get network connections using compatibility function
$allConnections = Get-CompatibleNetworkConnections

# Generate the report
$combinedReport = Get-ConnectionReport $allConnections

# Export results with backwards compatibility
$filteredReport = $combinedReport |
    Where-Object { $_.Process -ne 'Idle' -and $_.Process -ne '' } |
    Select-Object Computername, AuditDate, Protocol, UserName, UserSID, PID, ParentPID, ParentProcess, ParentPath, ParentSHA1, ParentMD5, ParentCommandLine, Process, ServiceName, Path, 
        ServiceStartType, ProcessSHA1, ProcessMD5, CommandLine, Connected, State, LocalAddress, LocalPort, RemoteAddress, RemotePort

# Use compatible Export-Csv parameters based on PowerShell version
$csvPath = "$localpath\$env:COMPUTERNAME-activecomms.csv"
try {
    if ($PSVersion -ge 3) {
        $filteredReport | Export-Csv -Path $csvPath -Append -NoTypeInformation -Encoding UTF8 -ErrorAction Continue
    } else {
        # PowerShell 2.0 doesn't support -Append or -Encoding UTF8
        if (Test-Path $csvPath) {
            $existingData = Import-Csv $csvPath -ErrorAction SilentlyContinue
            $combinedData = $existingData + $filteredReport
            $combinedData | Export-Csv -Path $csvPath -NoTypeInformation -ErrorAction Continue
        } else {
            $filteredReport | Export-Csv -Path $csvPath -NoTypeInformation -ErrorAction Continue
        }
    }
} catch {
    Write-Warning "Failed to export CSV: $($_.Exception.Message)"
    # Fallback: try basic export without advanced parameters
    try {
        $filteredReport | Export-Csv -Path $csvPath -NoTypeInformation
    } catch {
        Write-Error "Critical error: Unable to export results to CSV file."
    }
}
