# Cyber Risk Assessment (CRA) FULL - Refactored Version
# Original Copyright @2017 All Rights Reserved
# Refactored by Shane Shook (c) 2025
$version="20250106"
# Runas: PowerShell.exe -ExecutionPolicy bypass -WindowStyle hidden -File (path to script)

# COMPATIBILITY: This script supports Windows 2000 through modern Windows versions
# - Uses Get-CimInstance when available (PowerShell 3.0+/Windows Vista+), falls back to Get-WmiObject
# - Uses Get-NetTCPConnection when available (PowerShell 3.0+/Windows 8+), falls back to netstat parsing
# - Uses Get-FileHash when available (PowerShell 4.0+), falls back to .NET Framework cryptography classes
# - Optimized for performance with caching and batched operations
# - Comprehensive error handling and logging

Clear-Host

# Variables declared here - adjust to suit the environment
$localpath = "C:\secaudit" # This is the location where the output files will drop at runtime

# This is the network share where the script will drop off the zip files
#$networkshare = "\\ADDC\CRA\" 
#$outputfile = "\\ADDC\CRA\$env:computername*.zip"

# To use local storage on host just comment out the above two lines and uncomment the following two
$networkshare = "c:\windows\temp"
$outputfile = "c:\windows\temp\$env:computername*.zip"

$logtime = (Get-Date -Uformat %s)
$ErrorActionPreference = 'SilentlyContinue'

# ============================================================================
# COMPATIBILITY FRAMEWORK
# ============================================================================

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

# Global cache for performance optimization
$Global:ProcessCache = @{}
$Global:ServiceCache = @{}
$Global:WMIProcessCache = @{}
$Global:NetworkConnectionCache = @{}

# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

function Write-CompatibleLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    try {
        Write-Host $logMessage
        Add-Content -Path "$localpath\audit.log" -Value $logMessage -ErrorAction SilentlyContinue
    } catch {
        # Fallback to console only if file write fails
        Write-Host $logMessage
    }
}

function Get-CompatibleProcess {
    if ($Global:ProcessCache.Count -eq 0) {
        Write-CompatibleLog "Caching process information..."
        try {
            if ($PSVersion -ge 4) {
                $processes = Get-Process -IncludeUserName -ErrorAction SilentlyContinue
            } else {
                $processes = Get-Process -ErrorAction SilentlyContinue
            }
            $processes | ForEach-Object { $Global:ProcessCache[$_.Id] = $_ }
        } catch {
            Get-Process -ErrorAction SilentlyContinue | ForEach-Object { $Global:ProcessCache[$_.Id] = $_ }
        }
    }
    return $Global:ProcessCache
}

function Get-CompatibleWMIProcess {
    if ($Global:WMIProcessCache.Count -eq 0) {
        Write-CompatibleLog "Caching WMI process information..."
        try {
            if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
                Get-CimInstance -Class Win32_Process -ErrorAction SilentlyContinue | ForEach-Object { $Global:WMIProcessCache[$_.ProcessID] = $_ }
            } else {
                Get-WmiObject Win32_Process -ErrorAction SilentlyContinue | ForEach-Object { $Global:WMIProcessCache[$_.ProcessID] = $_ }
            }
        } catch {
            Get-WmiObject Win32_Process -ErrorAction SilentlyContinue | ForEach-Object { $Global:WMIProcessCache[$_.ProcessID] = $_ }
        }
    }
    return $Global:WMIProcessCache
}

function Get-CompatibleService {
    if ($Global:ServiceCache.Count -eq 0) {
        Write-CompatibleLog "Caching service information..."
        try {
            if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
                Get-CimInstance -Class Win32_Service -ErrorAction SilentlyContinue | ForEach-Object { 
                    if ($_.ProcessId -and $_.ProcessId -ne 0) {
                        if (-not $Global:ServiceCache.ContainsKey($_.ProcessId)) {
                            $Global:ServiceCache[$_.ProcessId] = @()
                        }
                        $Global:ServiceCache[$_.ProcessId] += $_
                    }
                }
            } else {
                Get-WmiObject Win32_Service -ErrorAction SilentlyContinue | ForEach-Object { 
                    if ($_.ProcessId -and $_.ProcessId -ne 0) {
                        if (-not $Global:ServiceCache.ContainsKey($_.ProcessId)) {
                            $Global:ServiceCache[$_.ProcessId] = @()
                        }
                        $Global:ServiceCache[$_.ProcessId] += $_
                    }
                }
            }
        } catch {
            Get-WmiObject Win32_Service -ErrorAction SilentlyContinue | ForEach-Object { 
                if ($_.ProcessId -and $_.ProcessId -ne 0) {
                    if (-not $Global:ServiceCache.ContainsKey($_.ProcessId)) {
                        $Global:ServiceCache[$_.ProcessId] = @()
                    }
                    $Global:ServiceCache[$_.ProcessId] += $_
                }
            }
        }
    }
    return $Global:ServiceCache
}

function Get-CompatibleFileHash {
    param(
        [string]$Path,
        [string]$Algorithm = "SHA1"
    )
    
    if (-not $Path -or -not (Test-Path $Path)) { return "" }
    
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
    if ($Global:NetworkConnectionCache.Count -eq 0) {
        Write-CompatibleLog "Caching network connection information..."
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
                $connections = $tcpConnections + $udpConnections
            }
        } catch { }
        
        if ($connections.Count -eq 0) {
            try {
                # Fallback to netstat parsing for older systems
                Write-CompatibleLog "Using netstat fallback for network connections..."
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
            } catch {
                Write-CompatibleLog "Failed to get network connections: $($_.Exception.Message)" "ERROR"
            }
        }
        
        $Global:NetworkConnectionCache = $connections
    }
    return $Global:NetworkConnectionCache
}

function Get-ProcessUserName {
    param([int]$ProcessId)
    
    try {
        if ($PSVersion -ge 4) {
            $proc = Get-Process -Id $ProcessId -IncludeUserName -ErrorAction SilentlyContinue
            if ($proc) { return $proc.UserName }
        }
        
        # Fallback using WMI
        $wmiProc = $Global:WMIProcessCache[$ProcessId]
        if ($wmiProc) {
            $owner = $wmiProc.GetOwner()
            if ($owner.Domain -and $owner.User) {
                return "$($owner.Domain)\$($owner.User)"
            }
        }
    } catch { }
    
    return "N/A"
}

function Get-CompatibleUserAccounts {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            return Get-CimInstance Win32_UserAccount -Filter "LocalAccount='True'" -ErrorAction SilentlyContinue
        } else {
            return Get-WmiObject Win32_UserAccount -Filter "LocalAccount='True'" -ErrorAction SilentlyContinue
        }
    } catch {
        return Get-WmiObject Win32_UserAccount -Filter "LocalAccount='True'" -ErrorAction SilentlyContinue
    }
}

function Get-CompatibleNetworkAdapter {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            return Get-CimInstance -Class Win32_NetworkAdapterConfiguration -ErrorAction SilentlyContinue
        } else {
            return Get-WmiObject -Class Win32_NetworkAdapterConfiguration -ErrorAction SilentlyContinue
        }
    } catch {
        return Get-WmiObject -Class Win32_NetworkAdapterConfiguration -ErrorAction SilentlyContinue
    }
}

function Get-CompatibleStartupCommands {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            return Get-CimInstance -Class Win32_StartupCommand -ErrorAction SilentlyContinue
        } else {
            return Get-WmiObject -Class Win32_StartupCommand -ErrorAction SilentlyContinue
        }
    } catch {
        return Get-WmiObject -Class Win32_StartupCommand -ErrorAction SilentlyContinue
    }
}

function Export-CompatibleCSV {
    param(
        [Parameter(ValueFromPipeline=$true)]
        $InputObject,
        [string]$Path,
        [switch]$Append
    )
    
    begin {
        $data = @()
    }
    
    process {
        $data += $InputObject
    }
    
    end {
        try {
            if ($PSVersion -ge 3) {
                if ($Append) {
                    $data | Export-Csv -Path $Path -Append -NoTypeInformation -Encoding UTF8 -ErrorAction Continue
                } else {
                    $data | Export-Csv -Path $Path -NoTypeInformation -Encoding UTF8 -ErrorAction Continue
                }
            } else {
                # PowerShell 2.0 doesn't support -Append or -Encoding UTF8
                if ($Append -and (Test-Path $Path)) {
                    $existingData = Import-Csv $Path -ErrorAction SilentlyContinue
                    $combinedData = $existingData + $data
                    $combinedData | Export-Csv -Path $Path -NoTypeInformation -ErrorAction Continue
                } else {
                    $data | Export-Csv -Path $Path -NoTypeInformation -ErrorAction Continue
                }
            }
        } catch {
            Write-CompatibleLog "Failed to export CSV to $Path: $($_.Exception.Message)" "ERROR"
            # Fallback: try basic export without advanced parameters
            try {
                $data | Export-Csv -Path $Path -NoTypeInformation
            } catch {
                Write-CompatibleLog "Critical error: Unable to export results to CSV file $Path" "ERROR"
            }
        }
    }
}

function Get-CompatibleFQDN {
    try {
        if ($PSVersion -ge 3 -and $WinVersion.Major -ge 6) {
            $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
        } else {
            $cs = Get-WmiObject Win32_ComputerSystem -ErrorAction SilentlyContinue
        }
        
        if ($cs) {
            return "$($cs.DNSHostName).$($cs.Domain)"
        }
    } catch { }
    
    return $env:COMPUTERNAME
}

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

function Get-ActiveCommunications {
    Write-CompatibleLog "Collecting active communications..."
    
    $processCache = Get-CompatibleProcess
    $wmiProcessCache = Get-CompatibleWMIProcess
    $serviceCache = Get-CompatibleService
    $connections = Get-CompatibleNetworkConnections
    
    $results = @()
    foreach ($conn in $connections) {
        $pid = $conn.OwningProcess
        $process = $processCache[$pid]
        $wmiProcess = $wmiProcessCache[$pid]
        $services = $serviceCache[$pid]
        
        $result = [PSCustomObject]@{
            Computername = $env:COMPUTERNAME
            AuditDate = Get-Date -Uformat %s
            UserName = Get-ProcessUserName -ProcessId $pid
            PID = $pid
            Process = if ($process) { $process.Name } else { "" }
            ServiceName = if ($services) { ($services | Select-Object -First 1).Name } else { "" }
            Path = if ($process) { $process.Path } else { "" }
            ServiceStartType = if ($services) { ($services | Select-Object -First 1).StartMode } else { "" }
            SHA1 = if ($process -and $process.Path) { Get-CompatibleFileHash -Path $process.Path -Algorithm "SHA1" } else { "" }
            MD5 = if ($process -and $process.Path) { Get-CompatibleFileHash -Path $process.Path -Algorithm "MD5" } else { "" }
            CommandLine = if ($wmiProcess) { $wmiProcess.CommandLine } else { "" }
            Connected = try { [int][double]::Parse((Get-Date $conn.CreationTime -UFormat %s)) } catch { 0 }
            State = $conn.State
            LocalAddress = $conn.LocalAddress
            LocalPort = $conn.LocalPort
            RemoteAddress = $conn.RemoteAddress
            RemotePort = $conn.RemotePort
        }
        
        if ($result.Process -ne 'Idle' -and $result.Process -ne '') {
            $results += $result
        }
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-activecomms.csv" -Append
    Write-CompatibleLog "Active communications collection completed. Found $($results.Count) connections."
}

function Get-ServiceBinaries {
    Write-CompatibleLog "Collecting service binaries..."
    
    $results = @()
    Get-Service | ForEach-Object {
        $binaryPath = $_.BinaryPathName
        if (-not $binaryPath) {
            $binaryPath = try { Get-ItemPropertyValue -EA Ignore "HKLM:\SYSTEM\CurrentControlSet\Services\$($_.Name)" ImagePath } catch { }
        }
        if ($binaryPath -like '*\svchost.exe *') {
            foreach ($keyName in $_.Name, ($_.Name -split '_')[0]) {
                foreach ($subKeyName in "$keyName\Parameters", $keyName) {
                    $binaryPath = try { Get-ItemPropertyValue -EA Ignore "HKLM:\SYSTEM\CurrentControlSet\Services\$subKeyName" ServiceDLL } catch { }
                    if ($binaryPath) { break }
                }
            }
        }
        $binaryPath = if ($binaryPath -like '"*') {
            ($binaryPath -split '"')[1]
        } else {
            (-split $binaryPath)[0]
        }
        
        $FileVersionInfo = if ($binaryPath -and (Test-Path $binaryPath)) { (Get-Item -LiteralPath $binaryPath).VersionInfo }
        
        $result = [PSCustomObject]@{
            Computername = $env:COMPUTERNAME
            AuditDate = Get-Date -Uformat %s
            Name = $_.Name
            BinaryPath = if ($binaryPath) { $binaryPath } else { '(n/a)' }
            ProductName = if ($FileVersionInfo) { $FileVersionInfo.ProductName } else { "" }
            FileDescription = if ($FileVersionInfo) { $FileVersionInfo.FileDescription } else { "" }
            CompanyName = if ($FileVersionInfo) { $FileVersionInfo.CompanyName } else { "" }
            FileVersion = if ($FileVersionInfo) { $FileVersionInfo.FileVersion } else { "" }
            ProductVersion = if ($FileVersionInfo) { $FileVersionInfo.ProductVersion } else { "" }
            SHA1 = if ($binaryPath) { Get-CompatibleFileHash -Path $binaryPath -Algorithm "SHA1" } else { "" }
            MD5 = if ($binaryPath) { Get-CompatibleFileHash -Path $binaryPath -Algorithm "MD5" } else { "" }
        }
        
        $results += $result
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-servicebinaries.csv" -Append
    Write-CompatibleLog "Service binaries collection completed. Found $($results.Count) services."
}

function Get-LocalUsers {
    Write-CompatibleLog "Collecting local users..."
    
    # Method 1: ADSI Users and Groups
    $results1 = @()
    try {
        $adsi = [ADSI]"WinNT://$env:COMPUTERNAME"
        $adsi.Children | Where-Object { $_.SchemaClassName -eq 'user' } | ForEach-Object {
            $groups = $_.Groups() | ForEach-Object {
                $_.GetType().InvokeMember('Name', 'GetProperty', $null, $_, $null)
            }
            
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                UserName = $_.Name
                LastLogin = try { $_.LastLogin | Get-Date -Uformat %s } catch { 0 }
                Enabled = if ($_.psbase.properties.item("userflags").value -band $ADS_UF_ACCOUNTDISABLE) { $False } else { $True }
                Groups = $groups -join ';'
            }
            
            $results1 += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect ADSI users: $($_.Exception.Message)" "ERROR"
    }
    
    $results1 | Export-CompatibleCSV -Path "$localpath\$env:computername-allusers.csv"
    
    # Method 2: Registry Users
    $results2 = @()
    try {
        $userAccounts = Get-CompatibleUserAccounts
        foreach ($user in $userAccounts) {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                AccountType = $user.AccountType
                Caption = $user.Caption
                Domain = $user.Domain
                SID = $user.SID
                FullName = $user.FullName
                Name = $user.Name
            }
            $results2 += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect registry users: $($_.Exception.Message)" "ERROR"
    }
    
    $results2 | Export-CompatibleCSV -Path "$localpath\$env:computername-allusers_reg.csv"
    Write-CompatibleLog "Local users collection completed. ADSI: $($results1.Count), Registry: $($results2.Count) users."
}

function Get-UserProfiles {
    Write-CompatibleLog "Collecting user profiles..."
    
    # Method 1: Profile Files
    $results1 = @()
    try {
        Get-ChildItem 'HKLM:\Software\Microsoft\Windows NT\CurrentVersion\ProfileList' | ForEach-Object {      
            $profilePath = $_.GetValue('ProfileImagePath')    
            if ($profilePath -and (Test-Path $profilePath)) {
                Get-ChildItem -Path "$profilePath\*" -Force -Include NTUSER.DAT -ErrorAction SilentlyContinue | ForEach-Object {
                    $result = [PSCustomObject]@{
                        Computername = $env:COMPUTERNAME
                        AuditDate = Get-Date -Uformat %s
                        Name = $_.Name
                        Length = $_.Length
                        DirectoryName = $_.DirectoryName
                        CreationTime = $_.CreationTime | Get-Date -Uformat %s
                        LastWriteTime = $_.LastWriteTime | Get-Date -Uformat %s
                        ProductVersion = if ($_.VersionInfo) { $_.VersionInfo.ProductVersion } else { "" }
                        FileVersion = if ($_.VersionInfo) { $_.VersionInfo.FileVersion } else { "" }
                        Description = if ($_.VersionInfo) { $_.VersionInfo.FileDescription } else { "" }
                    }
                    $results1 += $result
                }
            }
        }
    } catch {
        Write-CompatibleLog "Failed to collect profile files: $($_.Exception.Message)" "ERROR"
    }
    
    $results1 | Export-CompatibleCSV -Path "$localpath\$env:computername-allprofiles.csv"
    
    # Method 2: Registry Profiles
    $results2 = @()
    try {
        Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\*" | ForEach-Object {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                Sid = $_.Sid
                PSChildName = $_.PSChildName
                ProfileImagePath = $_.ProfileImagePath
            }
            $results2 += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect registry profiles: $($_.Exception.Message)" "ERROR"
    }
    
    $results2 | Export-CompatibleCSV -Path "$localpath\$env:computername-allprofiles_reg.csv"
    Write-CompatibleLog "User profiles collection completed. Files: $($results1.Count), Registry: $($results2.Count) profiles."
}

function Get-OperatingSystemInfo {
    Write-CompatibleLog "Collecting operating system information..."
    
    $results = @()
    try {
        Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion" | ForEach-Object {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                ProductName = $_.ProductName
                CSDVersion = $_.CSDVersion
                CurrentVersion = $_.CurrentVersion
                CurrentBuild = $_.CurrentBuild
                BuildLabEx = $_.BuildLabEx
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect OS info: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-osinfo.csv"
    Write-CompatibleLog "Operating system information collection completed."
}

function Get-NetworkConfiguration {
    Write-CompatibleLog "Collecting network configuration..."
    
    $results = @()
    try {
        $adapters = Get-CompatibleNetworkAdapter
        foreach ($adapter in $adapters) {
            $result = [PSCustomObject]@{
                "Computer Name" = if ($adapter.__SERVER) { $adapter.__SERVER } else { $env:COMPUTERNAME }
                AuditDate = Get-Date -Uformat %s
                Description = $adapter.Description
                MACAddress = $adapter.MACAddress
                IPAddress = if ($adapter.IPAddress) { $adapter.IPAddress | Select-Object -First 1 } else { "" }
                IPSubnet = if ($adapter.IPSubnet) { $adapter.IPSubnet -join ";" } else { "" }
                DefaultIPGateway = if ($adapter.DefaultIPGateway) { $adapter.DefaultIPGateway -join ";" } else { "" }
                DHCPEnabled = $adapter.DHCPEnabled
                DHCPServer = $adapter.DHCPServer
                "DNS Server" = if ($adapter.DNSServerSearchOrder) { $adapter.DNSServerSearchOrder -join ";" } else { "" }
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect network configuration: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-nic.csv"
    Write-CompatibleLog "Network configuration collection completed. Found $($results.Count) adapters."
}

function Get-OpenPorts {
    Write-CompatibleLog "Collecting open ports (using netstat)..."
    
    $results = @()
    try {
        $connections = Get-CompatibleNetworkConnections
        $processCache = Get-CompatibleProcess
        
        foreach ($conn in $connections) {
            $process = $processCache[$conn.OwningProcess]
            
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                Protocol = $conn.Protocol
                LocalAddress = $conn.LocalAddress
                LocalPort = $conn.LocalPort
                RemoteAddress = $conn.RemoteAddress
                RemotePort = $conn.RemotePort
                State = $conn.State
                PID = $conn.OwningProcess
                ProcessName = if ($process) { $process.ProcessName } else { "" }
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect open ports: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-netstat.csv"
    Write-CompatibleLog "Open ports collection completed. Found $($results.Count) connections."
}

function Get-DNSCache {
    Write-CompatibleLog "Collecting DNS cache..."
    
    $results = @()
    try {
        $dnsOutput = ipconfig /displaydns 2>$null
        if ($dnsOutput) {
            $dnsOutput | ForEach-Object {
                $dnsEntry = $_.ToString().Split(' ')[-1]
                if ($dnsEntry -like "*.*") {
                    $result = [PSCustomObject]@{
                        Computername = $env:COMPUTERNAME
                        AuditDate = Get-Date -Uformat %s
                        DNS = $dnsEntry
                    }
                    $results += $result
                }
            }
        }
    } catch {
        Write-CompatibleLog "Failed to collect DNS cache: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Select-Object -Unique | Export-CompatibleCSV -Path "$localpath\$env:computername-dnscache.csv"
    Write-CompatibleLog "DNS cache collection completed. Found $($results.Count) unique entries."
}

function Get-ActiveProcesses {
    Write-CompatibleLog "Collecting active processes..."
    
    $results = @()
    try {
        $wmiProcesses = Get-CompatibleWMIProcess
        foreach ($process in $wmiProcesses.Values) {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                Name = $process.Name
                ProcessId = $process.ProcessId
                Path = $process.Path
                CommandLine = $process.CommandLine
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect active processes: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-processes.csv"
    Write-CompatibleLog "Active processes collection completed. Found $($results.Count) processes."
}

function Get-ScheduledTasks {
    Write-CompatibleLog "Collecting scheduled tasks..."
    
    $results = @()
    try {
        $sched = New-Object -Com "Schedule.Service"
        $sched.Connect()
        
        $sched.GetFolder("\").GetTasks(0) | ForEach-Object {
            $xml = [xml]$_.xml
            $result = [PSCustomObject]@{
                ComputerName = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                Name = $_.Name
                Status = switch($_.State) {0 {"Unknown"} 1 {"Disabled"} 2 {"Queued"} 3 {"Ready"} 4 {"Running"}}
                LastRunTime = try { $_.LastRunTime | Get-Date -Uformat %s } catch { 0 }
                NextRunTime = try { $_.NextRunTime | Get-Date -Uformat %s } catch { 0 }
                Actions = ($xml.Task.Actions.Exec | ForEach-Object { "$($_.Command) $($_.Arguments)" }) -join "`n"
                Enabled = $xml.task.settings.enabled
                Author = $xml.task.principals.Principal.UserID
                Description = $xml.task.registrationInfo.Description
                RunAs = $xml.task.principals.principal.userid
                Created = try { $xml.Task.RegistrationInfo.Date | Get-Date -Uformat %s } catch { 0 }
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect scheduled tasks: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-tasks.csv"
    Write-CompatibleLog "Scheduled tasks collection completed. Found $($results.Count) tasks."
}

function Get-RegistryServices {
    Write-CompatibleLog "Collecting registry services..."
    
    $results = @()
    try {
        Get-ItemProperty -Path HKLM:\SYSTEM\*\Services\* | ForEach-Object {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                ControlSet = ($_.PSPath -split "\\")[-3]
                ServiceName = ($_.PSPath -split "\\")[-1]
                ImagePath = $_.ImagePath
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect registry services: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-services.csv"
    Write-CompatibleLog "Registry services collection completed. Found $($results.Count) services."
}

function Get-ServiceDLLs {
    Write-CompatibleLog "Collecting service DLLs..."
    
    $results = @()
    try {
        Get-ItemProperty -Path HKLM:\SYSTEM\*\Services\*\Parameters | Where-Object {$_.ServiceDll -ne $null -and $_.ServiceDll -ne ''} | ForEach-Object {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                ServiceName = ($_.PSPath -split "\\")[-2]
                ControlSet = ($_.PSPath -split "\\")[-4]
                ServiceDll = $_.ServiceDll
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect service DLLs: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-servicedlls.csv"
    Write-CompatibleLog "Service DLLs collection completed. Found $($results.Count) DLLs."
}

function Get-StartupCommands {
    Write-CompatibleLog "Collecting startup commands..."
    
    $results = @()
    try {
        $startupCommands = Get-CompatibleStartupCommands
        foreach ($startup in $startupCommands) {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                Name = $startup.Name
                Command = $startup.Command
                Location = $startup.Location
                User = $startup.User
            }
            $results += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect startup commands: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-startups.csv"
    Write-CompatibleLog "Startup commands collection completed. Found $($results.Count) commands."
}

function Get-USBHistory {
    Write-CompatibleLog "Collecting USB history..."
    
    # USB Storage device history
    $results1 = @()
    try {
        Get-ItemProperty -EA SilentlyContinue HKLM:\system\currentcontrolset\enum\usbstor\*\* | Where-Object {$_.HardwareID -notlike '%vid%'} | ForEach-Object {
            $result = [PSCustomObject]@{
                Computername = $env:COMPUTERNAME
                AuditDate = Get-Date -Uformat %s
                HardwareID = ($_.HardwareID -split ",")[1]
                SerialNo = $_.PSChildName
                Class = $_.Class
                Service = $_.Service
            }
            $results1 += $result
        }
    } catch {
        Write-CompatibleLog "Failed to collect USB device history: $($_.Exception.Message)" "ERROR"
    }
    
    $results1 | Export-CompatibleCSV -Path "$localpath\$env:computername-usbdev.csv"
    
    # USB Storage devices with serial number
    $results2 = @()
    try {
        Get-ItemProperty -Path HKLM:\system\currentcontrolset\enum\usbstor\*\* | ForEach-Object {
            $P = $_.PSChildName
            Get-ItemProperty HKLM:\SOFTWARE\Microsoft\"Windows Portable Devices"\*\* | Where-Object {$_.PSChildName -like "*$P*"} | ForEach-Object {
                $result = [PSCustomObject]@{
                    Computername = $env:COMPUTERNAME
                    AuditDate = Get-Date -Uformat %s
                    FriendlyName = $_.FriendlyName
                    ProductName = ($_.PSChildName -split "&")[5]
                    SerialNo = ($_.PSChildName -split "#")[6]
                }
                $results2 += $result
            }
        }
    } catch {
        Write-CompatibleLog "Failed to collect USB serial numbers: $($_.Exception.Message)" "ERROR"
    }
    
    $results2 | Export-CompatibleCSV -Path "$localpath\$env:computername-usbsn.csv"
    Write-CompatibleLog "USB history collection completed. Devices: $($results1.Count), Serial Numbers: $($results2.Count)."
}

function Get-BinaryFiles {
    Write-CompatibleLog "Collecting binary files (this may take a while)..."
    
    $results = @()
    try {
        $localdrives = ([System.IO.DriveInfo]::getdrives() | Where-Object {$_.DriveType -eq 'Fixed'} | Select-Object -ExpandProperty Name)
        $excludePaths = @('*common*', '*\IME\*', '*onedrive*', '*csc*', '*.old\*', '*recycle*', '*migration*', '*install*', '*setup*', '*migwiz*', '*driverstore*', '*sxs*', '*cache*', '*kb*', '*update*', '*assembly*', '*.NET*')
        
        foreach ($drive in $localdrives) {
            Write-CompatibleLog "Scanning drive $drive for binary files..."
            Get-ChildItem -Path "$drive\*" -Force -Include *.dll, *.exe, *.sys, *.asp, *.aspx, *.jsp, *.jar, *.iso -Recurse -ErrorAction SilentlyContinue | Where-Object {
                $exclude = $false
                foreach ($excludePath in $excludePaths) {
                    if ($_.DirectoryName -like $excludePath) {
                        $exclude = $true
                        break
                    }
                }
                -not $exclude
            } | ForEach-Object {
                $result = [PSCustomObject]@{
                    Computername = $env:COMPUTERNAME
                    AuditDate = Get-Date -Uformat %s
                    Name = $_.Name
                    Length = $_.Length
                    DirectoryName = $_.DirectoryName
                    CreationTime = $_.CreationTime | Get-Date -Uformat %s
                    LastWriteTime = $_.LastWriteTime | Get-Date -Uformat %s
                    ProductVersion = if ($_.VersionInfo) { ("{0}.{1}.{2}.{3}" -f $_.VersionInfo.FileMajorPart, $_.VersionInfo.FileMinorPart, $_.VersionInfo.FileBuildPart, $_.VersionInfo.FilePrivatePart) } else { "" }
                    FileVersion = if ($_.VersionInfo) { $_.VersionInfo.FileVersion } else { "" }
                    Description = if ($_.VersionInfo) { $_.VersionInfo.FileDescription } else { "" }
                    SHA1 = Get-CompatibleFileHash -Path $_.FullName -Algorithm "SHA1"
                    MD5 = Get-CompatibleFileHash -Path $_.FullName -Algorithm "MD5"
                }
                $results += $result
                
                # Export in batches to avoid memory issues
                if ($results.Count -ge 1000) {
                    $results | Export-CompatibleCSV -Path "$localpath\$env:computername-allfiles.csv" -Append
                    $results = @()
                }
            }
        }
        
        # Export remaining results
        if ($results.Count -gt 0) {
            $results | Export-CompatibleCSV -Path "$localpath\$env:computername-allfiles.csv" -Append
        }
    } catch {
        Write-CompatibleLog "Failed to collect binary files: $($_.Exception.Message)" "ERROR"
    }
    
    Write-CompatibleLog "Binary files collection completed."
}

function Get-PrefetchFiles {
    Write-CompatibleLog "Collecting prefetch files..."
    
    $results = @()
    try {
        if (Test-Path "C:\Windows\Prefetch") {
            Get-ChildItem -Path "C:\Windows\Prefetch" -Force -Include *.pf -Recurse | ForEach-Object {
                $result = [PSCustomObject]@{
                    Computername = $env:COMPUTERNAME
                    AuditDate = Get-Date -Uformat %s
                    Name = $_.Name
                    Length = $_.Length
                    DirectoryName = $_.DirectoryName
                    CreationTime = $_.CreationTime | Get-Date -Uformat %s
                    LastWriteTime = $_.LastWriteTime | Get-Date -Uformat %s
                    ProductVersion = if ($_.VersionInfo) { $_.VersionInfo.ProductVersion } else { "" }
                    FileVersion = if ($_.VersionInfo) { $_.VersionInfo.FileVersion } else { "" }
                    Description = if ($_.VersionInfo) { $_.VersionInfo.FileDescription } else { "" }
                }
                $results += $result
            }
        }
    } catch {
        Write-CompatibleLog "Failed to collect prefetch files: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:Computername-prefetch.csv"
    Write-CompatibleLog "Prefetch files collection completed. Found $($results.Count) files."
}

function Get-AMCacheHistory {
    Write-CompatibleLog "Collecting AMCACHE history..."
    
    $results = @()
    try {
        # Ensure HKLM drive is available
        if (!(Get-PSDrive -Name HKLM -PSProvider Registry -ErrorAction SilentlyContinue)) {
            try {
                New-PSDrive -Name HKLM -PSProvider Registry -Root HKEY_LOCAL_MACHINE -ErrorAction SilentlyContinue | Out-Null
            } catch {
                Write-CompatibleLog "Failed to create HKLM registry drive" "ERROR"
                return
            }
        }
        
        # Try to get AppCompatCache from registry
        $AppCompatCache = $null
        try {
            $AppCompatCache = Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Session Manager\AppCompatCache\' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty AppCompatCache
        } catch { }
        
        if (-not $AppCompatCache) {
            try {
                $AppCompatCache = Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Session Manager\AppCompatibility\AppCompatCache' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty AppCompatCache
            } catch { }
        }
        
        if ($AppCompatCache -ne $null) {
            Write-CompatibleLog "Found AppCompatCache data, parsing..."
            
            $MemoryStream = New-Object System.IO.MemoryStream(,$AppCompatCache)
            $BinReader = New-Object System.IO.BinaryReader $MemoryStream
            $UnicodeEncoding = New-Object System.Text.UnicodeEncoding
            $ASCIIEncoding = New-Object System.Text.ASCIIEncoding
            
            try {
                $Header = ([System.BitConverter]::ToString($BinReader.ReadBytes(4))) -replace "-",""
                Write-CompatibleLog "AMCACHE header detected: $Header"
                
                switch ($Header) {
                    "30000000" {
                        Write-CompatibleLog "Processing AMCACHE format 30000000"
                        $BinReader.ReadBytes(32) | Out-Null
                        $NumberOfEntries = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $BinReader.ReadBytes(8) | Out-Null
                        
                        for ($i=0; $i -lt $NumberOfEntries; $i++) {
                            try {
                                $TempObject = New-Object PSObject -Property @{
                                    FileName = ""
                                    LastModifiedTime = ""
                                    Data = ""
                                }
                                $TempObject | Add-Member -MemberType NoteProperty -Name "Tag" -Value ($ASCIIEncoding.GetString($BinReader.ReadBytes(4)))
                                $BinReader.ReadBytes(4) | Out-Null
                                $CacheEntrySize = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                $NameLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                $TempObject.FileName = $UnicodeEncoding.GetString($BinReader.ReadBytes($NameLength))
                                $TempObject.LastModifiedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                $DataLength = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                $TempObject.Data = $ASCIIEncoding.GetString($BinReader.ReadBytes($DataLength))
                                
                                $result = [PSCustomObject]@{
                                    Computername = $env:COMPUTERNAME
                                    AuditDate = Get-Date -Uformat %s
                                    Command = ($TempObject.FileName -split "\\")[-1]
                                    Path = $TempObject.FileName
                                    LastMod = try { $TempObject.LastModifiedTime | Get-Date -Uformat %s } catch { 0 }
                                }
                                $results += $result
                            } catch {
                                Write-CompatibleLog "Error processing AMCACHE entry $i`: $($_.Exception.Message)" "ERROR"
                                break
                            }
                        }
                    }
                    
                    "34000000" {
                        Write-CompatibleLog "Processing AMCACHE format 34000000"
                        $BinReader.ReadBytes(36) | Out-Null
                        $NumberOfEntries = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $BinReader.ReadBytes(8) | Out-Null
                        
                        for ($i=0; $i -lt $NumberOfEntries; $i++) {
                            try {
                                $TempObject = New-Object PSObject -Property @{
                                    FileName = ""
                                    LastModifiedTime = ""
                                    Data = ""
                                }
                                $TempObject | Add-Member -MemberType NoteProperty -Name "Tag" -Value ($ASCIIEncoding.GetString($BinReader.ReadBytes(4)))
                                $BinReader.ReadBytes(4) | Out-Null
                                $CacheEntrySize = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                $NameLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                $TempObject.FileName = $UnicodeEncoding.GetString($BinReader.ReadBytes($NameLength))
                                $TempObject.LastModifiedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                $DataLength = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                $TempObject.Data = $ASCIIEncoding.GetString($BinReader.ReadBytes($DataLength))
                                
                                $result = [PSCustomObject]@{
                                    Computername = $env:COMPUTERNAME
                                    AuditDate = Get-Date -Uformat %s
                                    Command = ($TempObject.FileName -split "\\")[-1]
                                    Path = $TempObject.FileName
                                    LastMod = try { $TempObject.LastModifiedTime | Get-Date -Uformat %s } catch { 0 }
                                }
                                $results += $result
                            } catch {
                                Write-CompatibleLog "Error processing AMCACHE entry $i`: $($_.Exception.Message)" "ERROR"
                                break
                            }
                        }
                    }
                    
                    "80000000" {
                        Write-CompatibleLog "Processing AMCACHE format 80000000"
                        $Offset = [System.BitConverter]::ToUInt32($AppCompatCache[0..3],0)
                        $Tag = [System.BitConverter]::ToString($AppCompatCache[$Offset..($Offset+3)],0) -replace "-",""
                        
                        if ($Tag -eq "30307473" -or $Tag -eq "31307473") {
                            $MemoryStream.Position = ($Offset)
                            while ($MemoryStream.Position -lt $MemoryStream.Length) {
                                try {
                                    $EntryTag = [System.BitConverter]::ToString($BinReader.ReadBytes(4),0) -replace "-",""
                                    if ($EntryTag -eq "30307473" -or $EntryTag -eq "31307473") {
                                        $BinReader.ReadBytes(4) | Out-Null
                                        $TempObject = New-Object PSObject -Property @{
                                            Name = ""
                                            Time = ""
                                        }
                                        $JMP = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $SZ = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                        $TempObject.Name = $UnicodeEncoding.GetString($BinReader.ReadBytes($SZ + 2))
                                        $BinReader.ReadBytes(8) | Out-Null
                                        $TempObject.Time = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                        $BinReader.ReadBytes(4) | Out-Null
                                        
                                        $result = [PSCustomObject]@{
                                            Computername = $env:COMPUTERNAME
                                            AuditDate = Get-Date -Uformat %s
                                            Command = ($TempObject.Name -split "\\")[-1]
                                            Path = $TempObject.Name
                                            LastMod = try { $TempObject.Time | Get-Date -Uformat %s } catch { 0 }
                                        }
                                        $results += $result
                                    } else {
                                        # Scan for next valid entry
                                        $Exit = $False
                                        while ($Exit -ne $true -and $MemoryStream.Position -lt $MemoryStream.Length) {
                                            $Byte1 = [System.BitConverter]::ToString($BinReader.ReadBytes(1),0) -replace "-",""
                                            if ($Byte1 -eq "30" -or $Byte1 -eq "31") {
                                                $Byte2 = [System.BitConverter]::ToString($BinReader.ReadBytes(1),0) -replace "-",""
                                                if ($Byte2 -eq "30") {
                                                    $Byte3 = [System.BitConverter]::ToString($BinReader.ReadBytes(1),0) -replace "-",""
                                                    if ($Byte3 -eq "74") {
                                                        $Byte4 = [System.BitConverter]::ToString($BinReader.ReadBytes(1),0) -replace "-",""
                                                        if ($Byte4 -eq "73") {
                                                            $MemoryStream.Position = ($MemoryStream.Position - 4)
                                                            $Exit = $True
                                                        } else {
                                                            $MemoryStream.Position = ($MemoryStream.Position - 3)
                                                        }
                                                    } else {
                                                        $MemoryStream.Position = ($MemoryStream.Position - 2)
                                                    }
                                                } else {
                                                    $MemoryStream.Position = ($MemoryStream.Position - 1)
                                                }
                                            }
                                        }
                                    }
                                } catch {
                                    Write-CompatibleLog "Error in AMCACHE 80000000 parsing: $($_.Exception.Message)" "ERROR"
                                    break
                                }
                            }
                        } elseif ($Tag -eq "726F7473") {
                            $MemoryStream.Position = ($Offset + 8)
                            while ($MemoryStream.Position -lt $MemoryStream.Length) {
                                try {
                                    $TempObject = New-Object PSObject -Property @{
                                        Name = ""
                                        Time = ""
                                    }
                                    $JMP = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                    $TempObject.Time = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                    $SZ = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                    $TempObject.Name = $UnicodeEncoding.GetString($BinReader.ReadBytes($SZ))
                                    
                                    $result = [PSCustomObject]@{
                                        Computername = $env:COMPUTERNAME
                                        AuditDate = Get-Date -Uformat %s
                                        Command = ($TempObject.Name -split "\\")[-1]
                                        Path = $TempObject.Name
                                        LastMod = try { $TempObject.Time | Get-Date -Uformat %s } catch { 0 }
                                    }
                                    $results += $result
                                } catch {
                                    Write-CompatibleLog "Error in AMCACHE 726F7473 parsing: $($_.Exception.Message)" "ERROR"
                                    break
                                }
                            }
                        }
                    }
                    
                    "EE0FDCBA" {
                        Write-CompatibleLog "Processing AMCACHE format EE0FDCBA"
                        $NumberOfEntries = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $MemoryStream.Position=128
                        $Length = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                        $MaxLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                        $Padding = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $MemoryStream.Position=128
                        
                        if (($MaxLength - $Length) -eq 2) {
                            if ($Padding -eq 0) {
                                for ($i=0; $i -lt $NumberOfEntries; $i++) {
                                    try {
                                        $TempObject = New-Object PSObject -Property @{
                                            Name = ""
                                            Length = 0
                                            MaxLength = 0
                                            Padding = 0
                                            Offset0 = 0
                                            Offset1 = 0
                                            Time = ""
                                            Flag0 = 0
                                            Flag1 = 0
                                        }
                                        $TempObject.Length = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                        $TempObject.MaxLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                        $TempObject.Padding = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Offset0 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Offset1 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Time = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                        $TempObject.Flag0 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Flag1 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Name = ($UnicodeEncoding.GetString($AppCompatCache[$TempObject.Offset0..($TempObject.Offset0+$TempObject.Length-1)])) -replace "\\\?\?\\",""
                                        $BinReader.ReadBytes(16) | Out-Null
                                        
                                        $result = [PSCustomObject]@{
                                            Computername = $env:COMPUTERNAME
                                            AuditDate = Get-Date -Uformat %s
                                            Command = ($TempObject.Name -split "\\")[-1]
                                            Path = $TempObject.Name
                                            LastMod = try { $TempObject.Time | Get-Date -Uformat %s } catch { 0 }
                                        }
                                        $results += $result
                                    } catch {
                                        Write-CompatibleLog "Error processing AMCACHE EE0FDCBA entry $i`: $($_.Exception.Message)" "ERROR"
                                        break
                                    }
                                }
                            } else {
                                for ($i=0; $i -lt $NumberOfEntries; $i++) {
                                    try {
                                        $TempObject = New-Object PSObject -Property @{
                                            Name = ""
                                            Length = 0
                                            MaxLength = 0
                                            Offset = 0
                                            Time = ""
                                            Flag0 = 0
                                            Flag1 = 0
                                        }
                                        $TempObject.Length = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                        $TempObject.MaxLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                        $TempObject.Offset = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Time = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                        $TempObject.Flag0 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Flag1 = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                        $TempObject.Name = ($UnicodeEncoding.GetString($AppCompatCache[$TempObject.Offset..($TempObject.Offset+$TempObject.Length-1)])) -replace "\\\?\?\\",""
                                        $BinReader.ReadBytes(16) | Out-Null
                                        
                                        $result = [PSCustomObject]@{
                                            Computername = $env:COMPUTERNAME
                                            AuditDate = Get-Date -Uformat %s
                                            Command = ($TempObject.Name -split "\\")[-1]
                                            Path = $TempObject.Name
                                            LastMod = try { $TempObject.Time | Get-Date -Uformat %s } catch { 0 }
                                        }
                                        $results += $result
                                    } catch {
                                        Write-CompatibleLog "Error processing AMCACHE EE0FDCBA entry $i`: $($_.Exception.Message)" "ERROR"
                                        break
                                    }
                                }
                            }
                        }
                    }
                    
                    "FE0FDCBA" {
                        Write-CompatibleLog "Processing AMCACHE format FE0FDCBA"
                        $NumberOfEntries = [System.BitConverter]::ToUInt32($AppCompatCache[4..7],0)
                        $Padding = [System.BitConverter]::ToUInt32($AppCompatCache[12..15],0)
                        $MemoryStream.Position=8
                        
                        if ($Padding -eq 0) {
                            for ($i=0; $i -lt $NumberOfEntries; $i++) {
                                try {
                                    $TempObject = New-Object PSObject -Property @{
                                        Name = ""
                                        ModifiedTime = ""
                                        FileSize = 0
                                        Executed = $false
                                    }
                                    $Length = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                    $MaxLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                    $Padding = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                    $Offset = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                    $BinReader.ReadBytes(4) | Out-Null
                                    $TempObject.ModifiedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                    $TempObject.FileSize = [System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)
                                    $TempObject.Name = $UnicodeEncoding.GetString($AppCompatCache[$Offset..($Offset + $Length)])
                                    $TempObject.Executed = $TempObject.FileSize -gt 0
                                    
                                    $result = [PSCustomObject]@{
                                        Computername = $env:COMPUTERNAME
                                        AuditDate = Get-Date -Uformat %s
                                        Command = ($TempObject.Name -split "\\")[-1]
                                        Path = $TempObject.Name
                                        LastMod = try { $TempObject.ModifiedTime | Get-Date -Uformat %s } catch { 0 }
                                    }
                                    $results += $result
                                } catch {
                                    Write-CompatibleLog "Error processing AMCACHE FE0FDCBA entry $i`: $($_.Exception.Message)" "ERROR"
                                    break
                                }
                            }
                        } else {
                            for ($i=0; $i -lt $NumberOfEntries; $i++) {
                                try {
                                    $TempObject = New-Object PSObject -Property @{
                                        FileName = ""
                                        ModifiedTime = ""
                                        FileSize = 0
                                    }
                                    $Length = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                    $MaxLength = [System.BitConverter]::ToUInt16($BinReader.ReadBytes(2),0)
                                    $Offset = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                                    $TempObject.ModifiedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                    $TempObject.FileSize = [System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)
                                    $TempObject.FileName = $UnicodeEncoding.GetString($AppCompatCache[$Offset..($Offset + $Length)])
                                    
                                    $result = [PSCustomObject]@{
                                        Computername = $env:COMPUTERNAME
                                        AuditDate = Get-Date -Uformat %s
                                        Command = ($TempObject.FileName -split "\\")[-1]
                                        Path = $TempObject.FileName
                                        LastMod = try { $TempObject.ModifiedTime | Get-Date -Uformat %s } catch { 0 }
                                    }
                                    $results += $result
                                } catch {
                                    Write-CompatibleLog "Error processing AMCACHE FE0FDCBA entry $i`: $($_.Exception.Message)" "ERROR"
                                    break
                                }
                            }
                        }
                    }
                    
                    "EFBEADDE" {
                        Write-CompatibleLog "Processing AMCACHE format EFBEADDE"
                        $NumberOfEntries = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $NumberOfLRUEntries = [System.BitConverter]::ToUInt32($BinReader.ReadBytes(4),0)
                        $BinReader.ReadBytes(4) | Out-Null
                        
                        # Skip LRU entries
                        for ($i=0; $i -lt $NumberOfLRUEntries; $i++) {
                            $BinReader.ReadBytes(8) | Out-Null
                        }
                        
                        $MemoryStream.Position=400
                        for ($i=0; $i -lt $NumberOfEntries; $i++) {
                            try {
                                $TempObject = New-Object PSObject -Property @{
                                    FileName = ""
                                    LastModifiedTime = ""
                                    FileSize = 0
                                    LastUpdatedTime = ""
                                }
                                $TempObject.FileName = ($UnicodeEncoding.GetString($BinReader.ReadBytes(528))) -replace "\\\?\?\\",""
                                $TempObject.LastModifiedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                $TempObject.FileSize = [System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)
                                $TempObject.LastUpdatedTime = [DateTime]::FromFileTime([System.BitConverter]::ToUInt64($BinReader.ReadBytes(8),0)).ToString("G")
                                
                                $result = [PSCustomObject]@{
                                    Computername = $env:COMPUTERNAME
                                    AuditDate = Get-Date -Uformat %s
                                    Command = ($TempObject.FileName -split "\\")[-1]
                                    Path = $TempObject.FileName
                                    LastMod = try { $TempObject.LastModifiedTime | Get-Date -Uformat %s } catch { 0 }
                                }
                                $results += $result
                            } catch {
                                Write-CompatibleLog "Error processing AMCACHE EFBEADDE entry $i`: $($_.Exception.Message)" "ERROR"
                                break
                            }
                        }
                    }
                    
                    default {
                        Write-CompatibleLog "Unknown AMCACHE format: $Header" "WARNING"
                    }
                }
            } catch {
                Write-CompatibleLog "Error parsing AMCACHE header: $($_.Exception.Message)" "ERROR"
            } finally {
                # Clean up resources
                if ($BinReader) { $BinReader.Dispose() }
                if ($MemoryStream) { $MemoryStream.Dispose() }
            }
        } else {
            Write-CompatibleLog "No AppCompatCache data found in registry" "WARNING"
        }
    } catch {
        Write-CompatibleLog "Failed to collect AMCACHE history: $($_.Exception.Message)" "ERROR"
    }
    
    $results | Export-CompatibleCSV -Path "$localpath\$env:Computername-amcache.csv"
    Write-CompatibleLog "AMCACHE history collection completed. Found $($results.Count) entries."
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Get FQDN using compatible method
$myFQDN = Get-CompatibleFQDN

# Check if marker file exists, used to control script execution when linked to GPO/Login 
If (-Not (Test-Path $outputfile.trim())) {
    Write-CompatibleLog "Starting Cyber Risk Assessment on $myFQDN"
    Write-CompatibleLog "PowerShell Version: $PSVersion, Windows Version: $($WinVersion.ToString())"
    
    # PREPARATION
    try {
        if (-not (Test-Path $localpath)) {
            New-Item -ItemType Directory -Path $localpath -Force | Out-Null
        }
        Write-CompatibleLog "Created output directory: $localpath"
    } catch {
        Write-CompatibleLog "Failed to create output directory: $($_.Exception.Message)" "ERROR"
        exit 1
    }
    
    # Initialize caches
    Write-CompatibleLog "Initializing data caches for performance optimization..."
    Get-CompatibleProcess | Out-Null
    Get-CompatibleWMIProcess | Out-Null
    Get-CompatibleService | Out-Null
    
    # Execute data collection functions
    try {
        Get-ActiveCommunications
        Get-ServiceBinaries
        Get-LocalUsers
        Get-UserProfiles
        Get-OperatingSystemInfo
        Get-NetworkConfiguration
        Get-OpenPorts
        Get-DNSCache
        Get-ActiveProcesses
        Get-ScheduledTasks
        Get-RegistryServices
        Get-ServiceDLLs
        Get-StartupCommands
        Get-USBHistory
        Get-BinaryFiles
        Get-PrefetchFiles
        Get-AMCacheHistory
        
        Write-CompatibleLog "All data collection completed successfully."
        
        # Create ZIP archive
        Write-CompatibleLog "Creating ZIP archive..."
        $zipPath = "$localpath\$env:computername-CRA-$version.zip"
        
        try {
            # Use .NET compression if available (PowerShell 3.0+)
            if ($PSVersion -ge 3) {
                Add-Type -AssemblyName System.IO.Compression.FileSystem
                $files = Get-ChildItem -Path $localpath\* -Exclude "*.zip" -Recurse
                $zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')
                foreach ($file in $files) {
                    $relativePath = $file.FullName.Substring($localpath.Length + 1)
                    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $file.FullName, $relativePath) | Out-Null
                }
                $zip.Dispose()
            } else {
                # Fallback to Shell.Application for older systems
                $shellApplication = New-Object -ComObject Shell.Application
                $zip = $shellApplication.NameSpace($zipPath)
                $files = Get-ChildItem -Path $localpath\* -Exclude "*.zip" -Recurse
                
                foreach ($file in $files) {
                    $zip.CopyHere($file.FullName)
                    Start-Sleep -Milliseconds 500
                }
            }
            
            Write-CompatibleLog "ZIP archive created successfully: $zipPath"
        } catch {
            Write-CompatibleLog "Failed to create ZIP archive: $($_.Exception.Message)" "ERROR"
        }
        
        # MOVE zip file to the network share
        try {
            Move-Item $localpath\*.zip $networkshare -Force
            Write-CompatibleLog "ZIP file moved to network share: $networkshare"
        } catch {
            Write-CompatibleLog "Failed to move ZIP file to network share: $($_.Exception.Message)" "ERROR"
        }
        
        # REMOVE Files and Folder
        try {
            Remove-Item $localpath -Recurse -Force
            Write-CompatibleLog "Temporary files cleaned up successfully."
        } catch {
            Write-CompatibleLog "Failed to clean up temporary files: $($_.Exception.Message)" "ERROR"
        }
        
        # Write success to logfile
        Add-Content $networkshare\CRA_Collection.log "$logtime - SUCCESS : $myFQDN has been audited. The collection archive was moved to $networkshare"
        Write-CompatibleLog "Cyber Risk Assessment completed successfully for $myFQDN"
        
    } catch {
        Write-CompatibleLog "Critical error during data collection: $($_.Exception.Message)" "ERROR"
        Add-Content $networkshare\CRA_Collection.log "$logtime - ERROR : Critical error during audit of $myFQDN - $($_.Exception.Message)"
        exit 1
    }
    
} else {
    # Write failure to logfile
    Add-Content $networkshare\CRA_Collection.log "$logtime - FAILURE : A previous collection for $myFQDN was found at the networkshare. Script terminated on $myFQDN"
    Write-CompatibleLog "Previous collection found. Script terminated." "WARNING"
}

Write-CompatibleLog "Script execution completed."