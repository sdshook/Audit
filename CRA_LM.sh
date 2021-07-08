#!/bin/sh

#!/bin/ksh
# Cyber Risk Assessment (CRA) MAC/Linux Version - Copyright @2018 All Rights Reserved
# Updated by Brandon M. Pimentel & Shane Shook
# Version 20210708
# usage: sudo sh (Path to Script)

#variables for the script usage

HOST=$(hostname)
DATE=$(date +%Y-%m-%d)
PREFIX=$HOST","$(date +"%s")
EPOC=$(date +%s)
FILEPREFIX=$(echo ./$HOST"_"$DATE/$HOST"_"$DATE"-")
FOLDERPREFIX=$(echo ./$HOST"_"$DATE)
DEBUG=1
ISMAC=2

#determine if the system is a Mac or Linux
if [ $(uname -a | awk '{print($1)}'| grep  -i -c "darwin") -eq 1 ]; then 
	ISMAC=1 

fi

#create the storage folder
if [ ! -d "$FOLDERPREFIX" ]; then
  mkdir "$FOLDERPREFIX";
fi

#mac launch control
echo "ISMAC: "$ISMAC
if [ $ISMAC -eq 1 ]; then
	echo "Host,Epoch,Service,EnableTransactions,LimitLoadType,Program,Timeout,OnDemand,MachServices,ProgramArguments" >> $FILEPREFIX"launchctl.csv"

	servicename=$(launchctl list | awk 'NR>1 {print $3}')
	for s in $servicename
	do
    	    enable_transactions="none"
	    limit_load="none"
	    time_out=-100
	    program="none"
	    on_demand="none"
	    mach="none"
	    program_args="none"

    	if [ $(launchctl list "$s" | grep -i -c 'TimeOut') -eq 1 ]; then
        	time_out=$(launchctl list "$s" | grep 'TimeOut' | sed "s/=/ /" | awk '{ print $2}' | sed 's/;/ /g' | awk '{print $1}')        
	    else
    	    time_out="none"
	    fi

    	if [ $(launchctl list "$s" | grep -i -c '"Program"') -eq 1 ]; then
        	program=$(launchctl list "$s" | grep '"Program"' | sed "s/=/ /"| awk '{ print $2}' | sed 's/"/ /g' | awk '{print $1}')
	    fi

    	if [ $(launchctl list "$s" | grep -i -c "LimitLoad") -eq 1 ]; then
        	limit_load=$(launchctl list "$s" | grep "LimitLoad" | sed "s/=/ /"| awk '{ print $2}' | sed 's/"/ /g' | awk '{print $1}')
	    fi

    	if [ $(launchctl list "$s" | grep -i -c '"EnableTransactions" = true;') -eq 1 ]; then
        	enable_transactions="true"
	    elif [ $(launchctl list "$s" | grep -i -c '"EnableTransactions"') -eq 1 ]; then
    	    enable_transactions="false"
	    fi

    	if [ $(launchctl list "$s" | grep -i -c '"OnDemand" = true') -eq 1 ]; then
        	on_demand="true"
	    elif [ $(launchctl list "$s" | grep -i -c '"OnDemand"') -eq 1 ]; then
    	    on_demand="false"
	    fi

    	if [ $( launchctl list "$s" | grep -i -c "MachServices" ) -gt 0 ]; then 
        	servicestat=$(launchctl list "$s" | sed -n '/MachServices/,/};/p' | sed 's/"//g' | sed 's/\;//g' | sed 's/(//g' | sed 's/)//g' | sed 's/{//g' | sed 's/}//g' | awk 'BEGIN { ORS = " " } {print}')
	        mach=$(echo "$servicestat" | awk '{ s = ""; for (i = 1; i <= NF; i++) s = s $i " "; print s }')
    	fi

	    if [ $( launchctl list "$s" | grep -i -c "ProgramArguments" ) -gt 0 ]; then 
    	    servicestat=$(launchctl list "$s" | sed -n '/ProgramArguments/,/};/p' | sed 's/"//g' | sed 's/\;//g' | sed 's/(//g' | sed 's/)//g' | sed 's/{//g' | sed 's/}//g' | awk 'BEGIN { ORS = " " } {print}')
        	program_args=$(echo "$servicestat" | awk '{ s = ""; for (i = 1; i <= NF; i++) s = s $i " "; print s }' )
	    fi

    	echo "$PREFIX,$s,$enable_transactions,$limit_load,$program,$time_out,$on_demand,$mach,$program_args" >> $FILEPREFIX"launchctl.csv"
	done
fi 

	echo "Host,Epoch,Command\n" > $FILEPREFIX"commandshistory.csv"

	find /Users | grep .*sh_history | xargs grep -E "install|sudo|sh|su|ifconfig|tcpdump|/etc/bin|/etc/sbin|/usr/bin|/usr/sbin" | sed "s/:/,/g" | while IFS= read -r line; do
        echo "$PREFIX","$line" >> $FILEPREFIX"commandshistory.csv"; done

	find /var/root/.*sh_history | xargs grep -E "install|sudo|sh|su|ifconfig|tcpdump|/etc/bin|/etc/sbin|/usr/bin|/usr/sbin|/usr/share" | sed "s/:/|/g" | sed "s/,/+/g" | while IFS= read -r line; do 
    	echo "$PREFIX","$line" >> $FILEPREFIX"commandshistory.csv"; done

	find /root/.*sh_history | xargs strings | grep -E "install|sudo|sh|su|ifconfig|tcpdump|/etc/bin|/etc/sbin|/usr/bin|/usr/sbin|/usr/share" | sed "s/,/+/g" | sed "s/:/,/g" | while read line; do
 	echo "$PREFIX","$line"  >> $FILEPREFIX"commandshistory.csv"; done

	find /home | grep .*sh_history | xargs strings | grep -E "install|sudo|sh|su|ifconfig|tcpdump|/etc/bin|/etc/sbin|/usr/bin|/usr/sbin|/usr/share" | sed "s/,/+/g" | sed "s/:/,/g" | while read line; do
 	echo "$PREFIX","$line"  >> $FILEPREFIX"commandshistory.csv"; done

if [ $ISMAC -eq 1 ]; then
	#DNS
	echo "Host,Epoch,Type,Locations" > $FILEPREFIX"dnsresolvers.csv"
	if [ $DEBUG -eq 1 ]; then echo "Getting the DNS data"; fi
	cat /etc/resolv.conf | while IFS= read -r line; do printf "$line\n"; done | awk -v host=$(hostname) -v date=$(date +%s) '!/^[ \t]*#/{print host,date,$1,$2}' | sed "s/ /,/g" >> $FILEPREFIX"dnsresolvers.csv"
else
	if [ $DEBUG -eq 1 ]; then echo "Getting the DNS data"; fi
	echo "Hostname,Epoch,DNSType,Address" > $FILEPREFIX"dnsresolvers.csv"
	cat /etc/resolv.conf | while IFS= read -r line; do printf "$PREFIX","$line","\n"; done | awk 'NR>1{print $1,$2}' | sed "s/ /,/g" >> $FILEPREFIX"dnsresolvers.csv"
fi

#IPConfig Start
if [ $ISMAC -eq 1 ]; then
	echo "Hostname,Epoch,Interface,IPv4,IPv6,Network Mask" > $FILEPREFIX"IPConfig.csv"

	interface=""
	ipv4=""
	ipv6=""
	netmask=""
	resetnote=0

	ifconfig |  while IFS= read -r line; do
	
		#interface name
		if [ $(echo $line | awk '{print ($1) }' | grep -i -c ":") -eq 1 ]; then
			if [ $(echo $line | awk '{print ($1) }' | grep -i -c "status:") -eq 0 ]; then
				if [ $(echo $line | awk '{print ($1) }' | grep -i -c "media:") -eq 0 ]; then
					if [ $(echo $line | awk '{print ($1) }' | grep -i -c "member:") -eq 0 ]; then
						if [ $(echo $line | awk '{print ($1) }' | grep -i -c "Configuration:") -eq 0 ]; then
							if [ $resetnote -eq 1 ]; then
								echo $PREFIX","$interface","$ipv4","$netmask","$ipv6 >> $FILEPREFIX"IPConfig.csv"
								interface=""
								ipv4=""
								ipv6=""
								netmask=""
							fi
							resetnote=1
							interface=$(echo $line | awk '{print ($1) }')
						fi #end of configuration
					fi # end of member
				fi # end of media:
			fi # end of status :
		fi # end of looking for :
	
		#ipv4 / ipv6	
		if [  $(echo $line | awk '{print ($1) }' | grep -i -c "inet6") -eq 1 ]; then
			ipv6=$(echo $line | awk '{print ($2) }')
		elif [  $(echo $line | awk '{print ($1) }' | grep -i -c "inet") -eq 1 ]; then
			ipv4=$(echo $line | awk '{print ($2) }')
		fi
	
		#broadcast
		if [  $(echo $line | awk '{print ($5) }' | grep -i -c "broadcast") -eq 1 ]; then
			netmask=$(echo $line | awk '{print ($6) }')
		fi	 
	done
	#IP Config End
else
	interface_number=1
	line_number=1
	interface_name=""; mac=""; ipv4=""; ipv6=""

	echo "Hostname,Epoch,Interface,Link,Address" > $FILEPREFIX"IPConfig.csv"
	ip -o -a addr | awk '{sub (/\/.*$/, _, $4); print $2,$3,$4}' | sed "s/ /,/g" | while IFS= read -r line;
	do
	 echo "$PREFIX", "$line" >> $FILEPREFIX"IPConfig.csv"
	done
	ip -o -a link | awk '{sub (/\/.*$/, _, $4); print $2,$16,$17}' | sed "s/ /,/g" | while IFS= read -r line;	
	do
	 echo "$PREFIX", "$line" >> $FILEPREFIX"IPConfig.csv"
	done

fi
#IPConfig Done

#Logon Events
echo "Host,Epoch,User,LogonType,Date,Time,Duration" > $FILEPREFIX"LogonEvents.csv";
if [$ISMAC -eq 1 ]; then
  last |  while read line; do

    type=$(echo $line | awk '{print $2}' | grep -ic 'tty')

    if [ $type -eq 1 ]; then
      FIRST=$(echo $line | awk -v pre=$PREFIX '{print pre,$1,$2,$5"#"$4,$6}' | sed "s/ /,/g" | sed "s/#/ /g")
      if [ $(echo $line | awk '{print $7}' | grep -ic '-') -eq 1 ]; then
        LAST=$(echo $line| awk '{ s = ""; for (i = 8; i <= NF; i++) s = s $i " "; print s }')
      else
        LAST=$(echo $line| awk '{ s = ""; for (i = 7; i <= NF; i++) s = s $i " "; print s }')
      fi
      echo $FIRST","$LAST >> $FILEPREFIX"LogonEvents.csv";
    fi
  done;
else
  last |  while read line; do

    type=$(echo $line | awk '{print $2}' | grep -ic 'tty')

    if [ $type -eq 1 ]; then
      FIRST=$(echo $line | awk -v pre=$PREFIX '{print pre,$1,$2,$5"#"$4,$6}' | sed "s/ /,/g" | sed "s/#/ /g")
      if [ $(echo $line | awk '{print $7}' | grep -ic '-') -eq 1 ]; then
        LAST=$(echo $line| awk '{ s = ""; for (i = 8; i <= NF; i++) s = s $i " "; print s }')
      else
        LAST=$(echo $line| awk '{ s = ""; for (i = 7; i <= NF; i++) s = s $i " "; print s }')
      fi
      echo $FIRST","$LAST >> $FILEPREFIX"LogonEvents.csv";
    fi
  done;
fi

if [ $ISMAC -eq 2 ]; then
	echo "Host,Epoch,Date,User,HomeDir,Method,Command\n"> $FILEPREFIX'authlog.csv'
	cat /var/log/auth.log | grep "TTY=pts" |while IFS= read -r line; do 
          printf "$PREFIX","$line","\n" | awk '{print $1"-"$2"-"$3,$12,$10,$8,$14}' | sed "s|/bin/su,|/bin/su|g" | sed "s/ /,/g";
        done >> $FILEPREFIX"authlog.csv"; 

	cat /var/log/secure | grep "TTY=pts" |while IFS= read -r line; do 
          printf "$PREFIX","$line","\n" | awk '{print $1"-"$2"-"$3,$12,$10,$8,$14}' | sed "s|/bin/su,|/bin/su|g" | sed "s/ /,/g"; 
        done >> $FILEPREFIX"authlog.csv"; 
fi
#End Logon Events

#OS Info
echo "Host,Epoch,KernelType,Version,ReleaseInformation" > $FILEPREFIX"os_data.csv"
uname -a | awk  -v pre=$PREFIX '{print pre,$1,$3,$14}' | sed "s/ /,/g" >> $FILEPREFIX"os_data.csv"

#netstat
if [ $ISMAC -eq 1 ]; then
	if [ $DEBUG -eq 1 ]; then echo "Creating Netstat Headers"; fi
	echo "Host,Epoch,Protocol,LocalAddress,RemoteAddress,State,PID" > $FILEPREFIX'netstat.csv'
	##tcp
	if [ $DEBUG -eq 1 ]; then echo "Getting the TCP"; fi
	netstat -vanp tcp | while IFS= read -r line; do
                printf "$PREFIX","$line\n" | awk '{print $1,$4,$5,$6,$9}' | sed "s/ /,/g"; 
        done >> $FILEPREFIX'netstat.csv'

	##udp
	if [ $DEBUG -eq 1 ]; then echo "Getting the UDP"; fi

	netstat -vanp udp | while IFS= read -r line; do
	  	printf "$PREFIX","$line\n" | awk '{print $1,$4,$5,"",$8}' | sed "s/ /,/g";
	done >> $FILEPREFIX'netstat.csv'

	if [ $DEBUG -eq 1 ]; then echo "End of UDP"; fi
	if [ $DEBUG -eq 1 ]; then echo "End of Netstat Section"; fi

else
	if [ $DEBUG -eq 1 ]; then echo "Creating Netstat Headers"; fi
	echo "Host,Epoch,Protocol,LocalAddress,LocalPort,RemoteAddress,RemotePort,State,PID" > $FILEPREFIX'netstat.csv'
	##tcp
	if [ $DEBUG -eq 1 ]; then echo "Getting the TCP"; fi
	count=0
	netstat -vanp -t | while IFS= read -r line; do 
		if [ $count -gt 1 ]; then
			printf "$PREFIX","$line","\n" | awk '{print $1,$4,"",$5,"",$6,$7}' | sed "s/ /,/g"; 
		fi 
		count=$((1 + $count))
	done >> $FILEPREFIX'netstat.csv'
fi
#end of netstat

#ss
if [ $ISMAC -eq 1 ]; then
	if [ $DEBUG -eq 1 ]; then echo "Creating SS Headers"; fi
	echo "Host,Epoch,Protocol,LocalAddress,RemoteAddress,State,PID" > $FILEPREFIX'ss.csv'
	##tcp
	if [ $DEBUG -eq 1 ]; then echo "Getting the TCP"; fi
	ss -aneptH | while IFS= read -r line; do
                printf "$PREFIX","$line\n" | awk '{print "tcp",$4,$5,$1,$6}' | sed "s/ /,/g"; 
        done >> $FILEPREFIX'ss.csv'

	##udp
	if [ $DEBUG -eq 1 ]; then echo "Getting the UDP"; fi

	ss -anepuH | while IFS= read -r line; do
	  	printf "$PREFIX","$line\n" | awk '{print "udp",$4,$5,$1,$6}' | sed "s/ /,/g";
	done >> $FILEPREFIX'ss.csv'

	if [ $DEBUG -eq 1 ]; then echo "End of UDP"; fi
	if [ $DEBUG -eq 1 ]; then echo "End of SS Section"; fi

else
	if [ $DEBUG -eq 1 ]; then echo "Creating SS Headers"; fi
	echo "Host,Epoch,Protocol,LocalAddress,RemoteAddress,State,PID" > $FILEPREFIX'ss.csv'
	##tcp
	if [ $DEBUG -eq 1 ]; then echo "Getting the TCP"; fi
	count=0
	ss -aneptH | while IFS= read -r line; do 
		if [ $count -gt 1 ]; then
			printf "$PREFIX","$line","\n" | awk '{print "tcp",$4,$5,$1,$6}' | sed "s/ /,/g"; 
		fi 
		count=$((1 + $count))
	done >> $FILEPREFIX'ss.csv'
fi
#end of ss

#Processes
echo "Host,Epoch,User,PPID,PID,Comm,Args" > $FILEPREFIX"processes.csv"
count=0
sudo ps -exo "user,ppid,pid,comm,args" | while read line; do
  if [ $count -eq 0 ]; then
  	echo ""
  else
	  FIRST=$(echo $line| awk -v prefix=$PREFIX '{print prefix,$1,$2,$3,$4}' | sed "s/ /,/g")
	  LAST=$( echo $line |  awk '{ s = ""; for (i = 5; i <= NF; i++) s = s $i " "; print s }' | sed "s/,/+/g" )
	  echo "$FIRST,$LAST" >> $FILEPREFIX"processes.csv"
  fi
  count=$(($count+1))
done;

#linux services
if [ $ISMAC -eq 2 ]; then
	echo "Host,Epoch,Service,ServiceStatus,ServiceLoaded,ServiceActive" > $FILEPREFIX"services.csv"

	systemctl -all list-unit-files | while read line
	do  # start of outer loop
    	servicename=$(echo $line | awk '{print $1}')
    	servicestatus=$(echo $line | awk '{print $2}')
    	rest=$( echo "$line" |  awk 'NR<1{ s = ""; for (i = 3; i <= NF; i++) s = s $i " "; print s }' )
    	if [ ${#servicename} -gt 0 ]; then 
        	if [ $(echo $line | grep -i -c "unit files listed.") -eq 0  ]; then
            	if [ $(echo $line | grep -i -c "UNIT FILE STATE") -eq 0 ]; then
                	loaded=""; active=""
                	systemctl status $servicename | awk '{print $1,$2}' | while read ctl_line
                	do 
                    	if [ $( echo $ctl_line | grep -i -c "Loaded: ") -gt 0 ]; then 
                        	loaded=$(echo $ctl_line | awk '{print $2}')
                    	elif [ $( echo $ctl_line | grep -i -c "Active: ") -gt 0 ]; then 
                        	active=$(echo $ctl_line | awk '{print $2}')
                    	fi #search headers

                    	if [ ${#loaded} -gt 0 ] && [ ${#active} -gt 0 ]; then
                        	echo "$PREFIX,$servicename,$servicestatus,$loaded,$active" >> $FILEPREFIX"services.csv"
                        	loaded=""
                        	active=""
                    	fi # print the data to the file
                	done # inner service loop
            	fi # remove the inital header
        	fi # outer service xxx unit files listed
    	fi #outer service names not empty
	done # end of outer loop
fi
#end of linux services

#linux /etc/init.d
if [ $ISMAC -eq 2 ]; then

	echo "Host,Epoch,User,DateModified,Size,Service\n"> $FILEPREFIX'StartupService.csv'
		ls -la /etc/init.d/ | grep "-" | while IFS= read -r line; do echo "$PREFIX","$line" | awk -v prefix=$PREFIX '{print prefix,$3,$6"-"$7"-"$8,$5,$9}' |sed "s/ /,/g"; done >> $FILEPREFIX"StartupService.csv"; 
fi
#end of linux /etc/init.d

#lsof 
echo "Host,Epoch,Command,PID,User,FileDescriptor,FileType,Size,Node" > $FILEPREFIX"tasklist.csv"
lsof | awk -v pre=$PREFIX 'NR>1&&!_[$2]++{print pre","$1","$2","$3","$4","$5","$7","$8}'  | sed "s/ /+/g" >> $FILEPREFIX"tasklist.csv"
#end of lsof

#Etc Password
if [ $ISMAC -eq 1 ]; then 
	echo "Host,Epoch,User,UID,GID,HomeDir,Shell" > $FILEPREFIX"etc_password.csv"
	cat /etc/passwd | sed "s/ /#/g" | sed "s/:/ /g" | awk -v prefix=$PREFIX 'NR>10{print prefix","$1","$3","$4","$6","$7}' >> $FILEPREFIX"etc_password.csv"
else
	echo "Host,Epoch,User,UID,GID,HomeDir,Shell" > $FILEPREFIX"etc_password.csv"
	cat /etc/passwd | sed "s/ /#/g" | sed "s/:/ /g" | awk -v prefix=$PREFIX '{print prefix","$1","$3","$4","$6","$7}' >> $FILEPREFIX"etc_password.csv"
fi
#end of etc passwrd


#user paths
if [ $ISMAC -eq 1 ]; then
	echo "Host,Epoch,User,DateCreated,HomeDir" > $FILEPREFIX"UserHomePaths.csv"
	ls -l /Users | awk '{print $9}'  | while read user;
	do
  	if [ -n "$user" ]; then
    	unamePREFIX="$PREFIX,$user"
    	stat -r "/Users/$user" | awk -v prefix=$unamePREFIX '{print prefix,$11,$16}' | sed "s/ /,/g" >> $FILEPREFIX"UserHomePaths.csv"
  	fi
	done
else
	echo "Host,Epoch,User,DateModified,HomeDir" > $FILEPREFIX"UserHomePaths.csv"
	ls -l /home | awk '{print $9}'  | while read user;
	do
    	#echo $user
    	if  [ ${#user} -gt 0 ]; then 
        	#echo $user 
        	unamePREFIX="$PREFIX,$user"
        	mod=$(stat -c %Y /home/$user)
        	echo "$unamePREFIX,$mod,/home/$user" >> $FILEPREFIX"UserHomePaths.csv"
    	fi
	done
fi
#end of user paths

#cron data
if [ $ISMAC -eq 1 ]; then
	echo "Host,Epoch,Min,Hour,Day,Month,DayOfWeek,Command" > $FILEPREFIX"UserCron.csv"
	for user in $(cut -d: -f1 /etc/passwd)
	do
		crondata=$(sudo crontab -u $user -l) 

		if [ ${#crondata} -gt 0 ] 
		then
			echo "$crondata" | while read line; do

				ishash=$(echo "$line" | sed "s/\*/STAR/g" | awk '{print $1 }' | grep -i -c "#" )           
				if [ $ishash -eq 0 ]; then

					first=$(echo "$line" | sed "s/,/\&/g" | sed "s/\*/STAR/g" | awk '{print $1,$2,$3,$4,$5 }' | sed "s/ /,/g"  )            
				last=$( echo "$line" |  awk '{ s = ""; for (i = 6; i <= NF; i++) s = s $i " "; print s }' )
				echo "$PREFIX,$first,$last" >> $FILEPREFIX"UserCron.csv"
		fi
			done
		fi
	done 

	echo "Host,Epoch,DateModified,Command,Time" > $FILEPREFIX"Cron.csv"
	ls -l /etc/cron.daily | sed -n '1!p' | awk '{print $6,$7,$8",",$9}' | sed "s/,\s/,/g" | sed "s/$/,Daily/g" | sed "s/^/$(hostname),$(date +%s),/g" >> $FILEPREFIX"Cron.csv"
	ls -l /etc/cron.hourly | sed -n '1!p' | awk '{print $6,$7,$8",",$9}' | sed "s/,\s/,/g" | sed "s/$/,Hourly/g" | sed "s/^/$(hostname),$(date +%s),/g" >> $FILEPREFIX"Cron.csv"
	ls -l /etc/cron.monthly | sed -n '1!p' | awk '{print $6,$7,$8",",$9}' | sed "s/,\s/,/g" | sed "s/$/,Monthly/g" | sed "s/^/$(hostname),$(date +%s),/g" >> $FILEPREFIX"Cron.csv"
	ls -l /etc/cron.weekly | sed -n '1!p' | awk '{print $6,$7,$8",",$9}' | sed "s/,\s/,/g" | sed "s/$/,Weekly/g" | sed "s/^/$(hostname),$(date +%s),/g" >> $FILEPREFIX"Cron.csv"
fi

# UserAllGroups & MainUserGroups
if [ $ISMAC -eq 1 ]; then

	echo "Host,Epoch,User,UID,GID,GName" >  $FILEPREFIX"MainUserGroups.csv"
	echo "Host,Epoch,User,UID,GID,GName" >  $FILEPREFIX"UserAllGroups.csv"

	dscl . list /users | while read line; do
		#Main users and groups section
		USERNAME=$(id $line | awk '{print $1}' | sed "s/=/ /" | sed "s/(/ /" | sed "s/)/ /" | awk '{print $3}')
		USERID=$(id $line | awk '{print $1}' | sed "s/=/ /" | sed "s/(/ /" | sed "s/)/ /" | awk '{print $2}') 
		GROUPNAME=$(id $line | awk '{print $2}' | sed "s/=/ /" | sed "s/(/ /" | sed "s/)/ /" | awk '{print $3}')
		GROUPID=$(id $line | awk '{print $2}' | sed "s/=/ /" | sed "s/(/ /" | sed "s/)/ /" | awk '{print $2}')

		echo $PREFIX","$USERNAME","$USERID","$GROUPID","$GROUPNAME >>  $FILEPREFIX"MainUserGroups.csv"

		ALTGROUPS=$(id $line | awk '{print $3}' | sed "s/=/ /"| awk '{print $2}' | sed "s/,/ /g" )
		for altgroup in $ALTGROUPS
		do
			suffix=$( echo $altgroup | sed "s/(/,/" | sed "s/)//" | awk '{print $1,$2}')
			echo $PREFIX","$USERNAME","$USERID","$suffix >> $FILEPREFIX"UserAllGroups.csv"
		done
	done
	if [ $DEBUG -eq 1 ]; then echo "End of Test Users."; fi
fi
#/UserAllGroups & MainUserGroups

#Compression

tar cvfz $(echo $FOLDERPREFIX".tgz") $(echo $FOLDERPREFIX"/*")

if [ -f $(echo $FOLDERPREFIX".tgz") ]; then
    echo "The File $FOLDERPREFIX'.tgz' has been created"
    rm -rf "$FOLDERPREFIX"
fi
exit 0
