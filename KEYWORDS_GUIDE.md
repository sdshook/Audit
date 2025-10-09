# New_FORAI.py Keywords Enhancement Guide

## Overview

New_FORAI.py now supports adding custom keywords via the `--keywords-file` argument. This simplified approach allows forensic analysts to flag any terms of interest during evidence processing with case-insensitive matching.

## Usage

### Basic Keyword Loading
```bash
python New_FORAI.py --case-id CASE001 --keywords-file suspicious_terms.txt --search "malware"
```

### Keywords with Forensic Questions
```bash
python New_FORAI.py --case-id CASE001 --keywords-file keywords.txt --question "What suspicious activity was detected?"
```

### Keywords with Report Generation
```bash
python New_FORAI.py --case-id CASE001 --keywords-file keywords.txt --report json
```

## Keywords File Format

Simple text file with one keyword per line:
```
# Comments start with #
mimikatz
powershell
netcat
malicious.com
backdoor
trojan
ransomware
```

## Features

- **Case-Insensitive**: Keywords match regardless of case (mimikatz matches MIMIKATZ, Mimikatz, etc.)
- **Simple Format**: Plain text file, one keyword per line
- **Comments Supported**: Lines starting with # are ignored
- **Duplicate Removal**: Automatically removes duplicate keywords
- **Search Integration**: Keywords are immediately searchable in evidence database
- **Report Integration**: Keywords appear in comprehensive reports
- **Chain of Custody**: Keyword loading is logged with timestamps

## Benefits

- **Unified Approach**: Single argument replaces complex domain/tool/IOC separation
- **Flexible**: Works with any type of indicator (domains, files, processes, etc.)
- **Efficient**: Simple text format, easy to maintain and version control
- **Case-Insensitive**: Matches terms regardless of how they appear in evidence

## Example Keywords File

See `example_keywords.txt` for a comprehensive example including:
- Suspicious domains
- Attack tools and executables  
- Suspicious processes
- File extensions
- Registry keys
- Network indicators
- Common malware terms

## Testing

Use the provided test script:
```bash
python test_keywords.py
```

This simplified approach makes New_FORAI.py more efficient while maintaining full forensic capability.