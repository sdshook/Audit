# New_FORAI.py CLI Intelligence Enhancement Guide

## Overview

New_FORAI.py now supports adding custom domain names, tools history, and IOCs (Indicators of Compromise) directly via CLI arguments or files. This enhancement allows forensic analysts to inject contextual intelligence data during evidence processing, improving the accuracy and relevance of forensic analysis.

## New CLI Arguments

### Domain Intelligence
- `--domains DOMAIN1 DOMAIN2 ...` - Add suspicious domains via command line
- `--domains-file FILE` - Load domains from text file (one per line)

### Tool Intelligence  
- `--tools TOOL1 TOOL2 ...` - Add suspicious tools/executables via command line
- `--tools-file FILE` - Load tools from JSON file

### IOC Intelligence
- `--iocs-file FILE` - Load IOCs from JSON file

## Usage Examples

### 1. Basic Domain and Tool Intelligence via CLI

```bash
python New_FORAI.py --case-id CASE001 \
    --domains malicious.com evil.net c2-server.org \
    --tools mimikatz.exe netcat.exe psexec.exe \
    --search "malicious"
```

### 2. Load Intelligence from Files

```bash
python New_FORAI.py --case-id CASE001 \
    --domains-file suspicious_domains.txt \
    --tools-file attack_tools.json \
    --iocs-file threat_indicators.json \
    --report json
```

### 3. Combined CLI and File Intelligence

```bash
python New_FORAI.py --case-id CASE001 \
    --domains additional-threat.com \
    --domains-file known_bad_domains.txt \
    --tools custom-backdoor.exe \
    --tools-file standard_attack_tools.json \
    --full-analysis --target-drive C:
```

### 4. Intelligence with Forensic Questions

```bash
python New_FORAI.py --case-id CASE001 \
    --domains-file apt_domains.txt \
    --tools-file apt_tools.json \
    --question "What evidence of APT activity was found?"
```

## File Formats

### Domain File Format (Text)
```
# Suspicious domains - lines starting with # are comments
malicious-site.com
evil-domain.net
phishing-site.org
c2-server.info
data-exfil.biz
```

### Tools File Format (JSON)
```json
{
  "tools": [
    "mimikatz.exe",
    "psexec.exe",
    "netcat.exe"
  ],
  "executables": [
    "powershell_empire.exe",
    "cobalt_strike.exe"
  ],
  "processes": [
    "cmd.exe",
    "powershell.exe",
    "wmic.exe"
  ]
}
```

### IOCs File Format (JSON)
```json
{
  "indicators": [
    "192.168.1.100",
    "suspicious.exe",
    "malware.dll"
  ],
  "domains": [
    "additional-bad-domain.com",
    "another-c2.net"
  ],
  "files": [
    "backdoor.exe",
    "keylogger.dll"
  ],
  "hashes": [
    "d41d8cd98f00b204e9800998ecf8427e",
    "5d41402abc4b2a76b9719d911017c592"
  ],
  "registry_keys": [
    "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware"
  ]
}
```

## Integration Points

### 1. Evidence Database Integration
- Custom intelligence is injected directly into the evidence database
- Each indicator becomes a searchable evidence record
- Maintains chain of custody logging for intelligence sources

### 2. Search Enhancement
- Intelligence indicators are immediately searchable
- Search results include custom intelligence matches
- Supports full-text search across all intelligence data

### 3. Forensic Question Analysis
- Custom intelligence enhances forensic question answering
- Provides context for threat attribution and analysis
- Improves accuracy of automated analysis

### 4. Report Generation
- Intelligence data is included in comprehensive reports
- Shows custom indicators alongside discovered evidence
- Maintains source attribution for intelligence data

### 5. Chain of Custody
- All intelligence loading is logged with timestamps
- Source files and CLI arguments are recorded
- Maintains forensic integrity of custom data

## Workflow Integration

### Full Analysis with Intelligence
```bash
python New_FORAI.py --case-id APT_INVESTIGATION \
    --full-analysis --target-drive C: \
    --domains-file apt29_domains.txt \
    --tools-file apt29_tools.json \
    --iocs-file apt29_indicators.json \
    --report pdf --chain-of-custody
```

### Artifact Collection with Intelligence
```bash
python New_FORAI.py --case-id INCIDENT_001 \
    --collect-artifacts --target-drive C: \
    --domains malware-c2.com data-exfil.net \
    --tools backdoor.exe keylogger.dll
```

### Analysis with Intelligence
```bash
python New_FORAI.py --case-id INCIDENT_001 \
    --parse-artifacts \
    --domains-file threat_domains.txt \
    --tools-file malware_tools.json \
    --question "What malicious activity was detected?"
```

## Benefits

### 1. Enhanced Context
- Provides threat intelligence context to forensic analysis
- Improves detection of known bad indicators
- Enables threat attribution and campaign tracking

### 2. Improved Accuracy
- Reduces false negatives by flagging known threats
- Provides immediate context for suspicious artifacts
- Enhances automated analysis capabilities

### 3. Operational Efficiency
- Streamlines intelligence integration workflow
- Eliminates manual correlation steps
- Provides immediate searchable intelligence database

### 4. Forensic Integrity
- Maintains chain of custody for intelligence sources
- Preserves source attribution and timestamps
- Ensures audit trail for intelligence usage

## Best Practices

### 1. Intelligence Preparation
- Maintain curated lists of known bad domains and tools
- Use standardized JSON formats for complex IOCs
- Include source attribution in intelligence files

### 2. Case-Specific Intelligence
- Tailor intelligence to specific threat scenarios
- Combine generic and case-specific indicators
- Update intelligence based on investigation findings

### 3. Documentation
- Document intelligence sources and rationale
- Maintain version control for intelligence files
- Include intelligence loading in case documentation

### 4. Validation
- Verify intelligence accuracy before injection
- Test intelligence files for format compliance
- Validate search results include expected indicators

## Example Intelligence Files

The following example files are provided:
- `example_domains.txt` - Sample suspicious domains
- `example_tools.json` - Sample attack tools and executables
- `example_iocs.json` - Sample IOCs in multiple categories

## Testing

Use the provided test script to validate functionality:
```bash
python test_cli_intelligence.py
```

This comprehensive enhancement transforms New_FORAI.py into a threat intelligence-aware forensic analysis platform, significantly improving its capability to detect and analyze sophisticated threats.