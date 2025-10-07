# Security Audit Toolkit

A comprehensive collection of security auditing and forensic analysis tools developed by Shane Shook (c) 2025.

## Overview

This repository contains various PowerShell and Python scripts designed to automate security assessments, incident response, and forensic analysis across different platforms and cloud services.

## Tool Categories

### Software Intellectual Property Analysis
Advanced forensic code similarity analysis for IP theft detection:

* **[SIPCompare.py](SIPCompare.py)** - SIPCompare v2.0: Advanced Forensic Code Similarity Analysis Tool
  - Multi-dimensional analysis combining token-based, semantic, structural, and control-flow similarity
  - Obfuscation resistance with advanced normalization techniques
  - Industry-standard Type 1-4 clone detection
  - Cross-language support for 15+ programming languages
  - Statistical significance testing with confidence intervals
  - Forensic-quality reporting with evidence packages
  - Parallel processing for large repositories

### Cloud Services Reports (CSR)
Scripts to automate collection of cloud services activities for security posture assessment and incident triage:

* **[AzUAR.ps1](AzUAR.ps1)** - PowerShell script to pull Azure Active Directory authentication activities by user(s)  
  *Requires PowerShell 7 and .NET 8*

* **[o365UAR.ps1](o365UAR.ps1)** - PowerShell script to pull Unified Audit Logs from Office 365 by user(s)

* **[o365AppsChanges.ps1](o365AppsChanges.ps1)** - PowerShell script to pull changes made to applications by user(s)

* **[GUAR.py](GUAR.py)** - Python script to pull Unified Activity Logs from Google Workspace by user(s)

### Cyber Risk Assessments (CRA)
Scripts to automate collection of security posture information:

* **[CRA_Win.ps1](CRA_Win.ps1)** - Windows assessment (run on each host)
* **[CRA_LM.sh](CRA_LM.sh)** - Linux & Mac assessment (run on each host)  
* **[CRA_AD.ps1](CRA_AD.ps1)** - Active Directory assessment (for on-premise AD, run only once from any domain-connected host)

### Cyber Breach Assessment (CBA)
Scripts to automate collection of security posture information for incident triage:

* **[CBA_Win.ps1](CBA_Win.ps1)** - Cyber Breach Assessment script for Windows systems
* **[CBT.ps1](CBT.ps1)** - Cyber Breach Triage script to quickly collect active communicating processes for incident triage

### Forensic Analysis Tools
Advanced forensic analysis and AI-assisted investigation tools:

* **[FORAI.py](FORAI.py)** - AI-assisted forensic analysis tool for Windows systems
* **[FORAIR.py](FORAIR.py)** - Related forensic analysis tool
* **[FORAI Workflow Documentation](FORAI_Workflow.md)** - Detailed workflow and usage guide for FORAI.py

### Additional Security Tools

* **[UserRisk.ps1](UserRisk.ps1)** - User risk assessment script
* **[UserRisk_JITA.ps1](UserRisk_JITA.ps1)** - Just-in-time access user risk assessment
* **[PermissionRisk.ps1](PermissionRisk.ps1)** - Permission risk analysis script
* **[AzRoleChanges.ps1](AzRoleChanges.ps1)** - Azure role changes monitoring script
* **[AppSecretsChanges_JITA.ps1](AppSecretsChanges_JITA.ps1)** - Application secrets changes monitoring (JITA)
* **[AppSecretsChanges_persistent.ps1](AppSecretsChanges_persistent.ps1)** - Application secrets changes monitoring (persistent)
* **[GBCRA_Win.ps1](GBCRA_Win.ps1)** - Group-based cyber risk assessment for Windows
* **[AdminCreds/](AdminCreds/)** - Administrative credentials management tools

## Acknowledgments

Special thanks to Brandon Pimentel for his valuable contributions to this project.
