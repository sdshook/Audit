#!/usr/bin/env python3
"""
Test script to demonstrate the new CLI intelligence capabilities of New_FORAI.py
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result"""
    print(f"\n🔍 Running: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def test_cli_intelligence():
    """Test all CLI intelligence features"""
    
    print("🚀 Testing New_FORAI.py CLI Intelligence Capabilities")
    print("=" * 80)
    
    # Test 1: Initialize database
    print("\n📊 Test 1: Initialize Database")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--init-db", 
        "--verbose"
    ])
    
    # Test 2: Add domains via CLI arguments
    print("\n🌐 Test 2: Add Domains via CLI Arguments")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--domains", "malicious.com", "evil.net", "phishing.org",
        "--search", "malicious",
        "--verbose"
    ])
    
    # Test 3: Add tools via CLI arguments
    print("\n🔧 Test 3: Add Tools via CLI Arguments")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--tools", "mimikatz.exe", "netcat.exe", "psexec.exe",
        "--search", "mimikatz",
        "--verbose"
    ])
    
    # Test 4: Load from files
    print("\n📁 Test 4: Load Intelligence from Files")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--domains-file", "example_domains.txt",
        "--tools-file", "example_tools.json",
        "--iocs-file", "example_iocs.json",
        "--search", "suspicious",
        "--verbose"
    ])
    
    # Test 5: Combined CLI and file intelligence
    print("\n🔄 Test 5: Combined CLI and File Intelligence")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--domains", "additional-bad.com",
        "--domains-file", "example_domains.txt",
        "--tools", "custom-tool.exe",
        "--tools-file", "example_tools.json",
        "--search", "tool",
        "--verbose"
    ])
    
    # Test 6: Forensic question with intelligence
    print("\n❓ Test 6: Forensic Question with Custom Intelligence")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--domains", "c2-server.com",
        "--tools", "backdoor.exe",
        "--question", "What suspicious network activity was detected?",
        "--verbose"
    ])
    
    # Test 7: Generate report with intelligence
    print("\n📋 Test 7: Generate Report with Custom Intelligence")
    success = run_command([
        "python", "New_FORAI.py", 
        "--case-id", "CLI_TEST", 
        "--domains-file", "example_domains.txt",
        "--tools-file", "example_tools.json",
        "--report", "json",
        "--verbose"
    ])
    
    print("\n✅ CLI Intelligence Testing Complete!")
    print("=" * 80)
    print("🎯 Key Features Demonstrated:")
    print("  • Domain intelligence via --domains and --domains-file")
    print("  • Tool intelligence via --tools and --tools-file") 
    print("  • IOC intelligence via --iocs-file")
    print("  • Combined CLI arguments and file loading")
    print("  • Intelligence integration with search functionality")
    print("  • Intelligence integration with forensic questions")
    print("  • Intelligence integration with report generation")
    print("  • Chain of custody logging for intelligence loading")

if __name__ == "__main__":
    test_cli_intelligence()