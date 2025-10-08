# SIPCompare Multi-Platform Setup Guide

Complete step-by-step instructions for configuring Windows 11, macOS, and Ubuntu Linux to run SIPCompare.py successfully.

## Table of Contents

- [Windows 11 Setup](#windows-11-setup)
- [macOS Setup](#macos-setup)
- [Ubuntu Linux Setup](#ubuntu-linux-setup)
- [Common Usage Examples](#common-usage-examples)
- [Troubleshooting](#troubleshooting)

---

# Windows 11 Setup

Complete step-by-step instructions for configuring your Windows 11 system to run SIPCompare.py successfully.

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
   - Select "Windows installer (64-bit)" 

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

## Step 4: Set Up SIPCompare Environment (Windows)

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

## Step 5: Install Python Dependencies (Windows)

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

---

# macOS Setup

Complete step-by-step instructions for configuring your macOS system to run SIPCompare.py successfully.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Administrator access
- Stable internet connection
- At least 4GB free disk space

## Step 1: Install Homebrew (Package Manager)

### Install Homebrew

1. **Open Terminal**:
   - Press `Cmd + Space`, type "Terminal", press Enter

2. **Install Homebrew**:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Add Homebrew to PATH** (for Apple Silicon Macs):
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Verify Installation**:
   ```bash
   brew --version
   ```

## Step 2: Install Python 3.8+

### Install Python via Homebrew

1. **Install Python**:
   ```bash
   brew install python@3.12
   ```

2. **Create symlinks** (if needed):
   ```bash
   brew link python@3.12
   ```

3. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

### Alternative: Download from Python.org

1. **Download Python**:
   - Visit: [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/)
   - Download "macOS 64-bit universal2 installer"

2. **Install Python**:
   - Run the downloaded `.pkg` file
   - Follow installation wizard

## Step 3: Install Git

### Install Git via Homebrew

1. **Install Git**:
   ```bash
   brew install git
   ```

2. **Verify Installation**:
   ```bash
   git --version
   ```

### Alternative: Install Xcode Command Line Tools

```bash
xcode-select --install
```

## Step 4: Set Up SIPCompare Environment (macOS)

### Create Project Directory and Virtual Environment

1. **Create Project Directory**:
   ```bash
   mkdir ~/SIPCompare
   cd ~/SIPCompare
   ```

2. **Download SIPCompare** (choose one method):

   **Method A: Clone from Repository**:
   ```bash
   git clone https://github.com/sdshook/Audit.git
   cp Audit/SIPCompare.py ~/SIPCompare/
   cd ~/SIPCompare
   ```

   **Method B: Manual Download**:
   - Download `SIPCompare.py` to `~/SIPCompare/`

3. **Create Virtual Environment**:
   ```bash
   python3 -m venv sipcompare_env
   ```

4. **Activate Virtual Environment**:
   ```bash
   source sipcompare_env/bin/activate
   ```

## Step 5: Install Python Dependencies (macOS)

### Core Dependencies

1. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install Core Libraries**:
   ```bash
   pip install numpy scipy tqdm
   ```

3. **Install AI/ML Dependencies**:
   ```bash
   pip install sentence-transformers transformers torch torchvision torchaudio
   ```

4. **Install Tree-sitter**:
   ```bash
   pip install tree-sitter==0.20.4
   pip install tree-sitter-languages==1.9.1
   ```

### Optional: Apple Silicon Optimization

For M1/M2/M3 Macs, you can use optimized versions:

1. **Install PyTorch with Metal Performance Shaders**:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Verify Metal Support**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

---

# Ubuntu Linux Setup

Complete step-by-step instructions for configuring your Ubuntu Linux system to run SIPCompare.py successfully.

## Prerequisites

- Ubuntu 20.04 LTS or later
- sudo access
- Stable internet connection
- At least 4GB free disk space

## Step 1: Update System and Install Dependencies

### Update Package Lists

1. **Update system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install essential build tools**:
   ```bash
   sudo apt install -y build-essential curl wget git
   ```

## Step 2: Install Python 3.8+

### Install Python and pip

1. **Install Python** (usually pre-installed on Ubuntu):
   ```bash
   sudo apt install -y python3 python3-pip python3-venv python3-dev
   ```

2. **Install additional Python dependencies**:
   ```bash
   sudo apt install -y python3-setuptools python3-wheel
   ```

3. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

### Alternative: Install Latest Python from Source

If you need a newer Python version:

1. **Install build dependencies**:
   ```bash
   sudo apt install -y libssl-dev libffi-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6-dev libgdm-dev libdb4o-cil-dev libpcap-dev
   ```

2. **Download and compile Python**:
   ```bash
   cd /tmp
   wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
   tar -xf Python-3.12.0.tgz
   cd Python-3.12.0
   ./configure --enable-optimizations
   make -j 8
   sudo make altinstall
   ```

## Step 3: Install Git (if not already installed)

```bash
sudo apt install -y git
git --version
```

## Step 4: Set Up SIPCompare Environment (Ubuntu)

### Create Project Directory and Virtual Environment

1. **Create Project Directory**:
   ```bash
   mkdir ~/SIPCompare
   cd ~/SIPCompare
   ```

2. **Download SIPCompare** (choose one method):

   **Method A: Clone from Repository**:
   ```bash
   git clone https://github.com/sdshook/Audit.git
   cp Audit/SIPCompare.py ~/SIPCompare/
   cd ~/SIPCompare
   ```

   **Method B: Manual Download**:
   - Download `SIPCompare.py` to `~/SIPCompare/`

3. **Create Virtual Environment**:
   ```bash
   python3 -m venv sipcompare_env
   ```

4. **Activate Virtual Environment**:
   ```bash
   source sipcompare_env/bin/activate
   ```

## Step 5: Install Python Dependencies (Ubuntu)

### Core Dependencies

1. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install system dependencies for scientific computing**:
   ```bash
   sudo apt install -y python3-numpy python3-scipy
   ```

3. **Install Core Libraries**:
   ```bash
   pip install numpy scipy tqdm
   ```

4. **Install AI/ML Dependencies**:
   ```bash
   pip install sentence-transformers transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install Tree-sitter**:
   ```bash
   pip install tree-sitter==0.20.4
   pip install tree-sitter-languages==1.9.1
   ```

### Optional: GPU Support (NVIDIA GPUs)

For NVIDIA GPU acceleration:

1. **Install NVIDIA drivers**:
   ```bash
   sudo apt install -y nvidia-driver-535
   ```

2. **Install CUDA Toolkit**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

3. **Install GPU-enabled PyTorch**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Reboot system**:
   ```bash
   sudo reboot
   ```

---

# Common Usage Examples

These examples work across all platforms (adjust paths as needed):

## Basic Commands

### Activate Environment

**Windows**:
```cmd
cd C:\SIPCompare
sipcompare_env\Scripts\activate
```

**macOS/Linux**:
```bash
cd ~/SIPCompare
source sipcompare_env/bin/activate
```

### Basic Analysis
```bash
python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2
```

### High-Accuracy Forensic Analysis
```bash
python SIPCompare.py --repoA /path/to/suspected --repoB /path/to/original --threshold 0.6 --embedding-model graphcodebert --parallel 4 --verbose --output evidence.zip
```

### Cross-Language Detection
```bash
python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo --cross-language --embedding-model codet5
```

### Large Repository Analysis (optimized for speed)
```bash
python SIPCompare.py --repoA /path/to/large_repo1 --repoB /path/to/large_repo2 --parallel 8 --embedding-model mini --threshold 0.8
```

## Test Installation Script

Create this test script on any platform to verify your installation:

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

# Test PyTorch device availability
try:
    import torch
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Metal Performance Shaders")
    else:
        print("Using CPU mode")
except:
    print("PyTorch not available")

print("\nSetup test complete!")
```

Run with:
```bash
python test_setup.py
```

---

# Troubleshooting

## Common Issues Across All Platforms

### Issue: "Python is not recognized" or "command not found"
**Solutions**:
- **Windows**: Reinstall Python with "Add to PATH" checked
- **macOS**: Use `python3` instead of `python`
- **Linux**: Install python3: `sudo apt install python3`

### Issue: "Microsoft Visual C++ 14.0 is required" (Windows only)
**Solution**: Install Visual Studio Build Tools

### Issue: "Out of memory" errors
**Solutions**:
```bash
# Reduce parallel workers
python SIPCompare.py --repoA repo1 --repoB repo2 --parallel 2

# Use lightweight model
python SIPCompare.py --repoA repo1 --repoB repo2 --embedding-model mini
```

### Issue: Tree-sitter installation fails
**Solutions**:
```bash
# Method 1: Upgrade build tools
pip install --upgrade setuptools wheel

# Method 2: Install without cache
pip install tree-sitter==0.20.4 --no-cache-dir

# Method 3: Install system dependencies (Linux)
sudo apt install -y python3-dev
```

### Issue: "No processable files found"
**Solutions**:
- Verify repository paths are correct
- Check that repositories contain supported file types
- Use absolute paths

### Issue: Slow performance on first run
**Expected Behavior**:
- First run downloads AI models (1-3 GB)
- Models are cached for subsequent runs
- **Cache locations**:
  - **Windows**: `C:\Users\[Username]\.cache\huggingface`
  - **macOS**: `~/.cache/huggingface`
  - **Linux**: `~/.cache/huggingface`

## Platform-Specific Issues

### Windows-Specific

#### Issue: PowerShell execution policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue: Long path support
Enable in Windows settings or registry for paths > 260 characters

### macOS-Specific

#### Issue: "xcrun: error: invalid active developer path"
```bash
xcode-select --install
```

#### Issue: Permission denied on /usr/local
```bash
sudo chown -R $(whoami) /usr/local
```

### Linux-Specific

#### Issue: "Failed building wheel for [package]"
```bash
sudo apt install -y python3-dev build-essential
```

#### Issue: NVIDIA driver conflicts
```bash
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install -y nvidia-driver-535
```

## Performance Optimization by Platform

### Windows
- Use SSD storage for better I/O performance
- Disable Windows Defender real-time scanning for SIPCompare directory
- Use Windows Terminal instead of Command Prompt

### macOS
- For Apple Silicon: Ensure you're using ARM64 versions of packages
- Use Activity Monitor to check memory usage
- Consider using `caffeinate` to prevent sleep during long analyses

### Linux
- Use `htop` to monitor system resources
- Consider using `nice` to adjust process priority:
  ```bash
  nice -n 10 python SIPCompare.py --repoA repo1 --repoB repo2
  ```
- For servers: Use `screen` or `tmux` for long-running analyses

## System Requirements Summary

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Dual-core 2.5GHz (Quad-core recommended)
- **Internet**: Required for initial model downloads

### Recommended Configuration
- **RAM**: 16GB+
- **Storage**: SSD with 20GB+ free space
- **CPU**: Quad-core 3.0GHz or better
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for better performance)

---

**Setup Complete!** Your system is now configured to run SIPCompare.py successfully across Windows 11, macOS, and Ubuntu Linux. For additional support or advanced configuration, refer to the main SIPCompare documentation.