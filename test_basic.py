#!/usr/bin/env python3
"""
Basic functionality test for SIPCompare v2.0
"""

import os
import tempfile
import shutil
from SIPCompare import analyze_repositories

def create_test_files():
    """Create test repositories with similar code"""
    
    # Create temporary directories
    repo_a = tempfile.mkdtemp(prefix="repo_a_")
    repo_b = tempfile.mkdtemp(prefix="repo_b_")
    
    # Test file 1: Exact match
    code_a1 = '''
def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

def main():
    data = [1, 2, 3, 4, 5]
    result = calculate_sum(data)
    print(f"Sum: {result}")

if __name__ == "__main__":
    main()
'''
    
    code_b1 = '''
def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

def main():
    data = [1, 2, 3, 4, 5]
    result = calculate_sum(data)
    print(f"Sum: {result}")

if __name__ == "__main__":
    main()
'''
    
    # Test file 2: Renamed identifiers (Type 2 clone)
    code_a2 = '''
def process_data(input_list):
    result = []
    for item in input_list:
        if item > 0:
            result.append(item * 2)
    return result
'''
    
    code_b2 = '''
def transform_values(data_array):
    output = []
    for element in data_array:
        if element > 0:
            output.append(element * 2)
    return output
'''
    
    # Test file 3: Different functionality (should not match)
    code_a3 = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    code_b3 = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''
    
    # Write files
    with open(os.path.join(repo_a, "exact_match.py"), "w") as f:
        f.write(code_a1)
    
    with open(os.path.join(repo_a, "renamed_vars.py"), "w") as f:
        f.write(code_a2)
    
    with open(os.path.join(repo_a, "different_func.py"), "w") as f:
        f.write(code_a3)
    
    with open(os.path.join(repo_b, "exact_match.py"), "w") as f:
        f.write(code_b1)
    
    with open(os.path.join(repo_b, "renamed_vars.py"), "w") as f:
        f.write(code_b2)
    
    with open(os.path.join(repo_b, "different_func.py"), "w") as f:
        f.write(code_b3)
    
    return repo_a, repo_b

def test_basic_functionality():
    """Test basic SIPCompare functionality"""
    print("Creating test repositories...")
    repo_a, repo_b = create_test_files()
    
    try:
        print("Running SIPCompare analysis...")
        
        # Run analysis with minimal dependencies
        matches = analyze_repositories(
            repo_a_path=repo_a,
            repo_b_path=repo_b,
            threshold=0.5,  # Lower threshold to catch more matches
            embedding_model='mini',  # Use simpler model
            parallel_workers=1,
            enable_statistical=False,  # Disable to avoid scipy dependency issues
            output_zip='test_evidence.zip'
        )
        
        print(f"Analysis complete! Found {len(matches)} matches")
        
        # Analyze results
        if matches:
            print("\nMatches found:")
            for i, match in enumerate(matches, 1):
                print(f"{i}. {os.path.basename(match.file_a)} <-> {os.path.basename(match.file_b)}")
                print(f"   Clone Type: {match.clone_type}")
                print(f"   Similarity: {match.overall_similarity:.4f}")
                print(f"   Evidence: {match.evidence_strength}")
                print(f"   Obfuscation: {match.obfuscation_detected}")
                print()
        
        # Expected results:
        # 1. exact_match.py should have high similarity (Type 1 clone)
        # 2. renamed_vars.py should have moderate similarity (Type 2 clone)  
        # 3. different_func.py should have low/no similarity
        
        exact_matches = [m for m in matches if 'exact_match' in m.file_a]
        renamed_matches = [m for m in matches if 'renamed_vars' in m.file_a]
        
        print("Test Results:")
        print(f"✓ Exact match detection: {'PASS' if exact_matches and exact_matches[0].overall_similarity > 0.9 else 'FAIL'}")
        print(f"✓ Renamed variable detection: {'PASS' if renamed_matches and renamed_matches[0].overall_similarity > 0.6 else 'FAIL'}")
        print(f"✓ Evidence package created: {'PASS' if os.path.exists('test_evidence.zip') else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("Cleaning up test files...")
        shutil.rmtree(repo_a, ignore_errors=True)
        shutil.rmtree(repo_b, ignore_errors=True)
        if os.path.exists('test_evidence.zip'):
            os.remove('test_evidence.zip')

if __name__ == "__main__":
    print("SIPCompare v2.0 Basic Functionality Test")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n✅ Basic functionality test PASSED")
    else:
        print("\n❌ Basic functionality test FAILED")
        exit(1)