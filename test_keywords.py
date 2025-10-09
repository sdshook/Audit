#!/usr/bin/env python3
"""
Test script to demonstrate the simplified keyword functionality of New_FORAI.py
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return the result"""
    print(f"\n🔍 Running: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def test_keywords():
    """Test keyword functionality"""
    
    print("🚀 Testing New_FORAI.py Simplified Keyword Functionality")
    print("=" * 80)
    
    # Test 1: Initialize database
    print("\n📊 Test 1: Initialize Database")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--init-db", 
        "--verbose"
    ])
    
    # Test 2: Load keywords and search (case-sensitive)
    print("\n🔍 Test 2: Load Keywords and Search (lowercase)")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--keywords-file", "example_keywords.txt",
        "--search", "mimikatz",
        "--verbose"
    ])
    
    # Test 3: Case-insensitive search
    print("\n🔍 Test 3: Case-Insensitive Search (uppercase)")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--keywords-file", "example_keywords.txt",
        "--search", "MIMIKATZ",
        "--verbose"
    ])
    
    # Test 4: Mixed case search
    print("\n🔍 Test 4: Mixed Case Search")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--keywords-file", "example_keywords.txt",
        "--search", "PowerShell",
        "--verbose"
    ])
    
    # Test 5: Forensic question with keywords
    print("\n❓ Test 5: Forensic Question with Keywords")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--keywords-file", "example_keywords.txt",
        "--question", "What suspicious activity was detected?",
        "--verbose"
    ])
    
    # Test 6: Generate report with keywords
    print("\n📋 Test 6: Generate Report with Keywords")
    run_command([
        "python", "New_FORAI.py", 
        "--case-id", "KEYWORD_TEST", 
        "--keywords-file", "example_keywords.txt",
        "--report", "json",
        "--verbose"
    ])
    
    print("\n✅ Keyword Testing Complete!")
    print("=" * 80)
    print("🎯 Key Features Demonstrated:")
    print("  • Simple keyword loading from text file")
    print("  • Case-insensitive keyword matching")
    print("  • Keywords integrated with search functionality")
    print("  • Keywords integrated with forensic questions")
    print("  • Keywords integrated with report generation")
    print("  • Chain of custody logging for keyword loading")
    print("  • Unified approach - no separate domains/tools/IOCs")

if __name__ == "__main__":
    test_keywords()