#!/usr/bin/env python3
"""
Simple test script to diagnose log2timeline issues
"""

import subprocess
import sys
import os
from pathlib import Path

def test_log2timeline():
    """Test log2timeline with progressively simpler commands"""
    
    # Test 1: Basic version check
    print("=== Test 1: log2timeline version ===")
    try:
        result = subprocess.run(['log2timeline', '--version'], capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Help command
    print("\n=== Test 2: log2timeline help ===")
    try:
        result = subprocess.run(['log2timeline', '--help'], capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        print("Help command successful" if result.returncode == 0 else "Help command failed")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: List parsers
    print("\n=== Test 3: Available parsers ===")
    try:
        result = subprocess.run(['log2timeline', '--parsers', 'list'], capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        if result.returncode == 0:
            print("Available parsers:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"STDERR: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Simple command with reduced parameters
    print("\n=== Test 4: Simple log2timeline test ===")
    test_storage = "D:\\FORAI\\extracts\\test_timeline.plaso"
    artifacts_dir = "D:\\FORAI\\artifacts\\CASE001_artifacts"
    
    # Remove test file if it exists
    if os.path.exists(test_storage):
        os.remove(test_storage)
    
    simple_cmd = [
        'log2timeline',
        '--storage-file', test_storage,
        '--parsers', 'filestat',  # Just one simple parser
        '--workers', '2',  # Reduced workers
        '--worker_memory_limit', '1024',  # Reduced memory
        artifacts_dir
    ]
    
    print(f"Command: {' '.join(simple_cmd)}")
    try:
        result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if os.path.exists(test_storage):
            size = os.path.getsize(test_storage)
            print(f"Created storage file: {test_storage} ({size} bytes)")
            # Clean up
            os.remove(test_storage)
        else:
            print("Storage file was not created")
            
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_log2timeline()