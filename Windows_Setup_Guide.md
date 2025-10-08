# SIPCompare Windows 11 Setup Guide

Complete step-by-step instructions for configuring your Windows 11 IBM ThinkPad to run SIPCompare.py successfully.

## Prerequisites

- Windows 11 operating system
- Administrator access
- Stable internet connection
- At least 4GB free disk space

## Step 1: Install Python 3.8+

### Download and Install Python

1. **Download Python**:
   - Visit: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Click "Download Python 3.12.x" (latest stable version)
   - Select "Windows installer (64-bit)" for your ThinkPad

2. **Install Python**:
   - Run the downloaded installer (`python-3.12.x-amd64.exe`)
   - ⚠️ **CRITICAL**: Check "Add Python to PATH" at the bottom of the installer
   - Click "Install Now"
   - Wait for installation to complete (2-5 minutes)
   - Click "Close" when finished

3. **Verify Installation**:
   - Press `Win + R`, type `cmd`, press Enter
   - In Command Prompt, type:
   ```cmd
   python --version
   ```
   - Expected output: `Python 3.12.x`
   - Also verify pip:
   ```cmd
   pip --version
   ```

## Step 2: Install Git for Windows

### Download and Install Git

1. **Download Git**:
   - Visit: [https://git-scm.com/download/win](https://git-scm.com/download/win)
   - Click "64-bit Git for Windows Setup"

2. **Install Git**:
   - Run the installer (`Git-2.x.x-64-bit.exe`)
   - Use default settings for most options
   - **Important**: Choose "Git from the command line and also from 3rd-party software"
   - Complete installation

3. **Verify Git Installation**:
   ```cmd
   git --version
   ```
   - Expected output: `git version 2.x.x.windows.x`

## Step 3: Install Visual Studio Build Tools

### Required for Python Package Compilation

1. **Download Build Tools**:
   - Visit: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Click "Download Build Tools"

2. **Install Build Tools**:
   - Run `vs_buildtools.exe`
   - Select "C++ build tools" workload
   - Ensure "Windows 10/11 SDK" is selected
   - Click "Install" (this may take 15-30 minutes)
   - Restart your computer when prompted

## Step 4: Set Up SIPCompare Environment

### Create Project Directory and Virtual Environment

1. **Open Command Prompt as Administrator**:
   - Press `Win + X`, select "Windows Terminal (Admin)" or "Command Prompt (Admin)"

2. **Create Project Directory**:
   ```cmd
   mkdir C:\SIPCompare
   cd C:\SIPCompare
   ```

3. **Download SIPCompare** (choose one method):

   **Method A: Clone from Repository** (if you have access):
   ```cmd
   git clone https://github.com/sdshook/Audit.git
   cd Audit
   copy SIPCompare.py C:\SIPCompare\
   cd C:\SIPCompare
   ```

   **Method B: Manual Download**:
   - Download `SIPCompare.py` directly to `C:\SIPCompare\`

4. **Create Virtual Environment** (Recommended):
   ```cmd
   python -m venv sipcompare_env
   ```

5. **Activate Virtual Environment**:
   ```cmd
   sipcompare_env\Scripts\activate
   ```
   - Your prompt should change to show `(sipcompare_env)`

## Step 5: Install Python Dependencies

### Core Dependencies

1. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

2. **Install Core Scientific Libraries**:
   ```cmd
   pip install numpy scipy tqdm
   ```

3. **Install AI/ML Dependencies**:
   ```cmd
   pip install sentence-transformers
   ```

4. **Install PyTorch (CPU Version)**:
   ```cmd
   pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install Tree-sitter for Enhanced Analysis**:
   ```cmd
   pip install tree-sitter==0.20.4
   pip install tree-sitter-languages==1.9.1
   ```

### Optional: GPU Support (NVIDIA GPUs only)

If you have an NVIDIA GPU and want better performance:

1. **Check GPU Compatibility**:
   - Visit: [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
   - Verify your GPU supports CUDA

2. **Install CUDA Toolkit**:
   - Visit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Download and install CUDA 11.8 or 12.1

3. **Install GPU-enabled PyTorch**:
   ```cmd
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Step 6: Test Installation

### Create and Run Test Script

1. **Create Test File**:
   - Open Notepad or your preferred text editor
   - Copy and paste the following code:

```python
# test_setup.py - SIPCompare Dependency Test
print("Testing SIPCompare dependencies...")
print("=" * 50)

dependencies = [
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("tqdm", "tqdm"),
    ("Sentence Transformers", "sentence_transformers"),
    ("Transformers", "transformers"),
    ("PyTorch", "torch"),
    ("Tree-sitter", "tree_sitter"),
    ("Tree-sitter Languages", "tree_sitter_languages")
]

success_count = 0
for name, module in dependencies:
    try:
        __import__(module)
        print(f"✓ {name} - OK")
        success_count += 1
    except ImportError as e:
        print(f"✗ {name} - MISSING ({e})")

print("=" * 50)
print(f"Dependencies installed: {success_count}/{len(dependencies)}")

# Test PyTorch GPU availability
try:
    import torch
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU mode (this is normal for most setups)")
except:
    print("PyTorch not available")

print("\nSetup test complete!")
input("Press Enter to continue...")
```

2. **Save the file** as `C:\SIPCompare\test_setup.py`

3. **Run the test**:
   ```cmd
   cd C:\SIPCompare
   sipcompare_env\Scripts\activate
   python test_setup.py
   ```

4. **Expected Output**:
   - All dependencies should show "✓ OK"
   - If any show "✗ MISSING", reinstall that specific package

## Step 7: First Run of SIPCompare

### Create Test Repositories

1. **Create test directories**:
   ```cmd
   mkdir test_repo_a test_repo_b
   ```

2. **Create sample Python files**:

   **File: `test_repo_a\sample.py`**:
   ```python
   def calculate_sum(a, b):
       return a + b
   
   def main():
       result = calculate_sum(5, 3)
       print(f"Result: {result}")
   
   if __name__ == "__main__":
       main()
   ```

   **File: `test_repo_b\sample.py`**:
   ```python
   def add_numbers(x, y):
       return x + y
   
   def run():
       answer = add_numbers(5, 3)
       print(f"Answer: {answer}")
   
   if __name__ == "__main__":
       run()
   ```

3. **Run SIPCompare Test**:
   ```cmd
   python SIPCompare.py --repoA test_repo_a --repoB test_repo_b --verbose
   ```

## Step 8: Usage Examples

### Basic Commands

1. **Activate Environment** (run this each time you open a new terminal):
   ```cmd
   cd C:\SIPCompare
   sipcompare_env\Scripts\activate
   ```

2. **Basic Analysis**:
   ```cmd
   python SIPCompare.py --repoA "C:\path\to\repo1" --repoB "C:\path\to\repo2"
   ```

3. **High-Accuracy Forensic Analysis**:
   ```cmd
   python SIPCompare.py --repoA "C:\path\to\suspected" --repoB "C:\path\to\original" --threshold 0.6 --embedding-model graphcodebert --parallel 4 --verbose --output evidence.zip
   ```

4. **Cross-Language Detection**:
   ```cmd
   python SIPCompare.py --repoA "C:\path\to\python_repo" --repoB "C:\path\to\java_repo" --cross-language --embedding-model codet5
   ```

5. **Large Repository Analysis** (optimized for speed):
   ```cmd
   python SIPCompare.py --repoA "C:\large_repo1" --repoB "C:\large_repo2" --parallel 8 --embedding-model mini --threshold 0.8
   ```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Python is not recognized as an internal or external command"
**Solution**:
- Reinstall Python and ensure "Add Python to PATH" is checked
- Manually add Python to PATH:
  1. Press `Win + X`, select "System"
  2. Click "Advanced system settings"
  3. Click "Environment Variables"
  4. Under "System variables", find "Path", click "Edit"
  5. Click "New" and add: `C:\Users\[YourUsername]\AppData\Local\Programs\Python\Python312`

#### Issue: "Microsoft Visual C++ 14.0 is required"
**Solution**:
- Install Visual Studio Build Tools (Step 3)
- Alternative: Download "Microsoft C++ Build Tools" from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Issue: "Out of memory" errors during analysis
**Solutions**:
```cmd
# Reduce parallel workers
python SIPCompare.py --repoA repo1 --repoB repo2 --parallel 2

# Use lightweight model
python SIPCompare.py --repoA repo1 --repoB repo2 --embedding-model mini

# Increase virtual memory (Windows setting)
```

#### Issue: Tree-sitter installation fails
**Solutions**:
```cmd
# Method 1: Upgrade build tools
pip install --upgrade setuptools wheel

# Method 2: Install without cache
pip install tree-sitter==0.20.4 --no-cache-dir

# Method 3: Use pre-compiled wheel
pip install --only-binary=all tree-sitter==0.20.4
```

#### Issue: "No processable files found"
**Solutions**:
- Verify repository paths are correct
- Check that repositories contain supported file types (.py, .java, .cpp, etc.)
- Use absolute paths: `C:\full\path\to\repository`

#### Issue: Slow performance on first run
**Expected Behavior**:
- First run downloads AI models (1-3 GB)
- Subsequent runs are much faster
- Models are cached in: `C:\Users\[YourUsername]\.cache\huggingface`

### Performance Optimization

#### For Better Performance:
1. **Use SSD storage** for repositories and cache
2. **Close unnecessary applications** during analysis
3. **Use appropriate model** for your use case:
   - `graphcodebert`: Best accuracy, moderate speed
   - `codet5`: Good for cross-language, slower
   - `mini`: Fastest, good accuracy for large repos

#### Memory Usage Guidelines:
- **8GB RAM**: Use `--parallel 2` and `--embedding-model mini`
- **16GB RAM**: Use `--parallel 4` and `--embedding-model graphcodebert`
- **32GB+ RAM**: Use `--parallel 8` and any model

## Quick Reference

### Daily Usage Workflow
```cmd
# 1. Open Command Prompt
# 2. Navigate to SIPCompare directory
cd C:\SIPCompare

# 3. Activate virtual environment
sipcompare_env\Scripts\activate

# 4. Run analysis
python SIPCompare.py --repoA "path1" --repoB "path2" --verbose

# 5. Deactivate when done
deactivate
```

### Useful File Paths
- **SIPCompare Directory**: `C:\SIPCompare\`
- **Virtual Environment**: `C:\SIPCompare\sipcompare_env\`
- **Model Cache**: `C:\Users\[YourUsername]\.cache\huggingface\`
- **Output Files**: `C:\SIPCompare\evidence_package.zip`

### Support Links
- **Python Downloads**: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **Git for Windows**: [https://git-scm.com/download/win](https://git-scm.com/download/win)
- **Visual Studio Build Tools**: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **PyTorch Installation**: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- **CUDA Toolkit**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

## System Requirements Summary

### Minimum Requirements
- **OS**: Windows 11
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Intel i5 or AMD Ryzen 5 (or equivalent)
- **Internet**: Required for initial model downloads

### Recommended Configuration
- **RAM**: 16GB+
- **Storage**: SSD with 20GB+ free space
- **CPU**: Intel i7 or AMD Ryzen 7 (or better)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for better performance)

---

**Setup Complete!** Your Windows 11 ThinkPad is now configured to run SIPCompare.py successfully. For additional support or advanced configuration, refer to the main SIPCompare documentation.