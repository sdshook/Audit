# Audit stuff from Shane Shook (c) 2025

Various security auditing tools

* CSR = Cloud Services Reports scripts to automate collection of cloud services activities for posture and triage

  * AzAuditFull.ps1 = Powershell Script to pull Azure Active Directory and o365 activities by user(s) #requires POSH 7 and DotNet 8
 
  * AzUAR.ps1 = Powershell Script to pull Azure Active Directory authentication activities by user(s) #requires POSH 7 and DotNet 8

  * o365UAR.ps1 = Powershell Script to pull Unified Audit Logs from o365 by user(s)
 
  * o365AppChanges.ps1 = Powershell Script to pull changes made to applications by user(s)

  * GUAR.py = Python Script to pull Unified Activity Logs from Google Workspace by user(s)

* CRA = Cyber Risk Assessments scripts to automate collection of security posture information 

  * Win = Windows (run on each host)
  
  * LM = Linux & Mac (run on each host)
  
  * AD = Active Directory (for on-premise AD, run only once from any domain-connected host)

* CBA = Cyber Breach Assessment script to automate collection of security posture information for incident triage

  * CBT - Cyber Breach Triage script to quickly collect active communicating processes for incident triage

Kudos to Brandon Pimentel for his help also.
