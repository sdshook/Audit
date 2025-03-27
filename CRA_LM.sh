#!/bin/sh

# Cyber Risk Assessment (CRA) MAC/Linux Version
# Originally created by Shane Shook and Brandon Pimentel updated (c) 2025
# Usage: sudo sh script.sh

trap 'echo "Error on line $LINENO"; exit 1' ERR

HOST=$(hostname)
DATE=$(date +%Y-%m-%d)
PREFIX="$HOST,$(date +%s)"
EPOC=$(date +%s)
FOLDERPREFIX="./${HOST}_${DATE}"
FILEPREFIX="$FOLDERPREFIX/${HOST}_${DATE}-"
DEBUG=1
ISMAC=2

# Detect OS
detect_os() {
  if [ "$(uname)" = "Darwin" ]; then
    ISMAC=1
  fi
  [ $DEBUG -eq 1 ] && echo "ISMAC: $ISMAC"
}

# Create folder
create_output_folder() {
  [ ! -d "$FOLDERPREFIX" ] && mkdir "$FOLDERPREFIX"
}

capture_mac_services() {
  echo "Computername,AuditDate,Service,EnableTransactions,LimitLoadType,Program,Timeout,OnDemand,MachServices,ProgramArguments" > "$FILEPREFIX"launchctl.csv
  launchctl list | awk 'NR>1 {print $3}' | while read -r s; do
    dump=$(launchctl list "$s" 2>/dev/null)
    enable_transactions=$(echo "$dump" | grep -iq 'EnableTransactions = true' && echo true || echo false)
    on_demand=$(echo "$dump" | grep -iq 'OnDemand = true' && echo true || echo false)
    program=$(echo "$dump" | grep '"Program"' | awk -F '"' '{print $4}')
    timeout=$(echo "$dump" | grep 'TimeOut' | awk -F '=|;' '{print $2}' | xargs)
    limit_load=$(echo "$dump" | grep 'LimitLoad' | awk -F '"' '{print $4}')
    mach=$(echo "$dump" | awk '/MachServices/,/};/' | tr -d '";(){}' | xargs)
    program_args=$(echo "$dump" | awk '/ProgramArguments/,/};/' | tr -d '";(){}' | xargs)
    echo "$PREFIX,$s,$enable_transactions,$limit_load,$program,$timeout,$on_demand,$mach,$program_args" >> "$FILEPREFIX"launchctl.csv
  done
}

capture_command_history() {
  echo "Computername,AuditDate,Command" > "$FILEPREFIX"commandshistory.csv
  for path in /Users /var/root /root /home; do
    find "$path" -name '*sh_history' 2>/dev/null | while read -r hist; do
      grep -E "install|sudo|sh|su|ifconfig|tcpdump|/etc/bin|/usr/bin|/usr/sbin" "$hist" 2>/dev/null | sed 's/:/,/' | while read -r line; do
        echo "$PREFIX,$line" >> "$FILEPREFIX"commandshistory.csv
      done
    done
  done
}

capture_dns() {
  echo "Computername,AuditDate,Type,Locations" > "$FILEPREFIX"dnsresolvers.csv
  awk '!/^#/ && NF > 0 {print ENVIRON["HOST"],ENVIRON["EPOC"],$1,$2}' /etc/resolv.conf | sed 's/ /,/g' >> "$FILEPREFIX"dnsresolvers.csv
}

capture_ipconfig() {
  echo "Computername,AuditDate,Interface,IPv4,IPv6,Network Mask" > "$FILEPREFIX"IPConfig.csv
  if [ $ISMAC -eq 1 ]; then
    ifconfig | awk '/^[a-z]/ {iface=$1} /inet / {print iface,$2} /inet6 / {print iface,$2}' | while read -r iface ip; do
      echo "$PREFIX,$iface,$ip" >> "$FILEPREFIX"IPConfig.csv
    done
  else
    ip -o -a addr | awk '{sub (/\/.*$/, _, $4); print $2,$3,$4}' | sed "s/ /,/g" | while read -r line; do
      echo "$PREFIX,$line" >> "$FILEPREFIX"IPConfig.csv
    done
  fi
}

capture_os_info() {
  echo "Computername,AuditDate,KernelType,Version,ReleaseInformation" > "$FILEPREFIX"os_data.csv
  uname -a | awk -v pre="$PREFIX" '{print pre, $1, $3, $14}' | sed "s/ /,/g" >> "$FILEPREFIX"os_data.csv
}

capture_processes() {
  echo "Computername,AuditDate,User,PPID,PID,Comm,Args" > "$FILEPREFIX"processes.csv
  ps -eo user,ppid,pid,comm,args | awk -v prefix="$PREFIX" 'NR>1 {printf "%s,%s,%s,%s,%s,", prefix, $1, $2, $3, $4; for (i=5; i<=NF; i++) printf "%s ", $i; print ""}' | sed 's/ /+/g' >> "$FILEPREFIX"processes.csv
}

capture_passwd() {
  echo "Computername,AuditDate,User,UID,GID,HomeDir,Shell" > "$FILEPREFIX"etc_password.csv
  awk -F ':' -v prefix="$PREFIX" '{print prefix "," $1 "," $3 "," $4 "," $6 "," $7}' /etc/passwd >> "$FILEPREFIX"etc_password.csv
}

capture_logon_events() {
  echo "Computername,AuditDate,User,LogonType,Date,Time,Duration" > "$FILEPREFIX"LogonEvents.csv
  last | while read -r line; do
    type=$(echo "$line" | awk '{print $2}' | grep -ic 'tty')
    if [ "$type" -eq 1 ]; then
      FIRST=$(echo "$line" | awk -v pre="$PREFIX" '{print pre, $1, $2, $5"#"$4, $6}' | sed "s/ /,/g" | sed "s/#/ /g")
      if echo "$line" | awk '{print $7}' | grep -q '-'; then
        LAST=$(echo "$line" | awk '{for (i=8; i<=NF; i++) printf $i" "; print ""}')
      else
        LAST=$(echo "$line" | awk '{for (i=7; i<=NF; i++) printf $i" "; print ""}')
      fi
      echo "$FIRST,$LAST" >> "$FILEPREFIX"LogonEvents.csv
    fi
  done
}

capture_netstat() {
  echo "Computername,AuditDate,Protocol,LocalAddress,RemoteAddress,State,PID" > "$FILEPREFIX"netstat.csv
  if [ $ISMAC -eq 1 ]; then
    netstat -vanp tcp | awk -v prefix="$PREFIX" 'NR>2 {print prefix ",tcp," $4 "," $5 "," $6 "," $9}' >> "$FILEPREFIX"netstat.csv
    netstat -vanp udp | awk -v prefix="$PREFIX" 'NR>2 {print prefix ",udp," $4 "," $5 ",," $8}' >> "$FILEPREFIX"netstat.csv
  else
    netstat -tunap | awk -v prefix="$PREFIX" 'NR>2 {print prefix "," $1 "," $4 "," $5 "," $6 "," $7}' >> "$FILEPREFIX"netstat.csv
  fi
}

capture_ss() {
  echo "Computername,AuditDate,State,Protocol,LocalAddress,RemoteAddress,PID" > "$FILEPREFIX"ss.csv
  ss -aneptH | awk -v prefix="$PREFIX" '{print prefix ",tcp," $1 "," $4 "," $5 "," $6}' >> "$FILEPREFIX"ss.csv
  ss -anepuH | awk -v prefix="$PREFIX" '{print prefix ",udp," $1 "," $4 "," $5 "," $6}' >> "$FILEPREFIX"ss.csv
}

capture_linux_services() {
  if [ $ISMAC -eq 2 ]; then
    echo "Computername,AuditDate,Service,ServiceStatus,ServiceLoaded,ServiceActive" > "$FILEPREFIX"services.csv
    systemctl list-unit-files --type=service | grep '.service' | while read -r line; do
      servicename=$(echo "$line" | awk '{print $1}')
      servicestatus=$(echo "$line" | awk '{print $2}')
      loaded=$(systemctl show "$servicename" --no-page 2>/dev/null | grep LoadState | cut -d= -f2)
      active=$(systemctl show "$servicename" --no-page 2>/dev/null | grep ActiveState | cut -d= -f2)
      echo "$PREFIX,$servicename,$servicestatus,$loaded,$active" >> "$FILEPREFIX"services.csv
    done
  fi
}

capture_initd_services() {
  if [ $ISMAC -eq 2 ]; then
    echo "Computername,AuditDate,User,DateModified,Size,Service" > "$FILEPREFIX"StartupService.csv
    ls -la /etc/init.d/ | grep '^-' | while read -r line; do
      echo "$PREFIX,$line" | awk -v prefix="$PREFIX" '{print prefix","$3","$6"-"$7"-"$8","$5","$9}' >> "$FILEPREFIX"StartupService.csv
    done
  fi
}

capture_user_groups() {
  echo "Computername,AuditDate,User,UID,GID,GroupName,GroupDescription,LoginShell" > "$FILEPREFIX"UserGroups.csv
  getent passwd | while IFS=: read -r username password uid gid comment home shell; do
    GROUPNAME=$(getent group "$gid" | cut -d: -f1)
    GROUPDESC=$(getent group "$GROUPNAME" | cut -d: -f5)
    echo "$PREFIX,$username,$uid,$gid,$GROUPNAME,$GROUPDESC,$shell" >> "$FILEPREFIX"UserGroups.csv

    id -Gn "$username" 2>/dev/null | tr ' ' '\n' | while read -r grp; do
      if [ "$grp" != "$GROUPNAME" ]; then
        ALTID=$(getent group "$grp" | cut -d: -f3)
        ALTDESC=$(getent group "$grp" | cut -d: -f5)
        echo "$PREFIX,$username,$uid,$ALTID,$grp,$ALTDESC,$shell" >> "$FILEPREFIX"UserGroups.csv
      fi
    done
  done
}

capture_active_sessions() {
  echo "Computername,AuditDate,User,Terminal,Host,LoginTime" > "$FILEPREFIX"ActiveSessions.csv
  who | while read -r user terminal date time rest; do
    echo "$PREFIX,$user,$terminal,$rest,$date $time" >> "$FILEPREFIX"ActiveSessions.csv
  done
}

capture_sudo_access() {
  echo "Computername,AuditDate,User,HasSudoersEntry,Shell" > "$FILEPREFIX"SudoAccess.csv
  getent passwd | while IFS=: read -r username password uid gid comment home shell; do
    SUDOFLAG=$(grep -E "^$username[[:space:]]|ALL" /etc/sudoers 2>/dev/null | wc -l)
    [ "$SUDOFLAG" -gt 0 ] && SUDOFLAG="yes" || SUDOFLAG="no"
    echo "$PREFIX,$username,$SUDOFLAG,$shell" >> "$FILEPREFIX"SudoAccess.csv
  done
}

capture_cron_jobs() {
  echo "Computername,AuditDate,Min,Hour,Day,Month,DayOfWeek,Command" > "$FILEPREFIX"UserCron.csv
  cut -d: -f1 /etc/passwd | while read -r user; do
    crontab -u "$user" -l 2>/dev/null | grep -v '^#' | awk -v prefix="$PREFIX" '{print prefix","$1","$2","$3","$4","$5",\"" substr($0, index($0,$6)) "\""}' >> "$FILEPREFIX"UserCron.csv
  done

  echo "Computername,AuditDate,DateModified,Command,Interval" > "$FILEPREFIX"Cron.csv
  for dir in daily hourly monthly weekly; do
    if [ -d "/etc/cron.$dir" ]; then
      ls -l /etc/cron.$dir | grep '^-' | awk -v prefix="$PREFIX" -v label="$dir" '{print prefix","$6"-"$7"-"$8","$9","label}' >> "$FILEPREFIX"Cron.csv
    fi
  done
}

capture_lsof() {
  echo "Computername,AuditDate,Command,PID,User,FileDescriptor,FileType,Size,Node" > "$FILEPREFIX"tasklist.csv
  lsof -nP | awk 'NR>1 {print}' | while read -r line; do
    printf "%s,%s\n" "$PREFIX" "$line" | awk '{print $1","$2","$3","$4","$5","$6","$7","$8","$9}' >> "$FILEPREFIX"tasklist.csv
  done
}

bundle_output() {
  tar cvfz "${FOLDERPREFIX}.tgz" -C "$FOLDERPREFIX" .
  if [ -f "${FOLDERPREFIX}.tgz" ]; then
    echo "The file ${FOLDERPREFIX}.tgz has been created"
    rm -rf "$FOLDERPREFIX"
  fi
}

main() {
  detect_os
  create_output_folder
  [ $ISMAC -eq 1 ] && capture_mac_services

  capture_command_history
  capture_dns
  capture_ipconfig
  capture_os_info
  capture_processes
  capture_passwd
  capture_logon_events
  capture_netstat
  capture_ss
  capture_linux_services
  capture_initd_services
  capture_user_groups
  capture_active_sessions
  capture_sudo_access
  capture_cron_jobs
  capture_lsof
  bundle_output

  echo "Audit complete. Output archived at ${FOLDERPREFIX}.tgz"
}

main
exit 0
