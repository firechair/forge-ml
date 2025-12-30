# Windows Setup Guide

Complete guide for running ForgeML on Windows.

## Recommended Approach: WSL2

WSL2 (Windows Subsystem for Linux) provides the best experience.

### Install WSL2

```powershell
# Run in PowerShell (Admin)
wsl --install

# Restart computer

# After restart, set up Ubuntu
# Follow prompts to create username/password
```

### Install ForgeML in WSL2

```bash
# Inside WSL terminal:
sudo apt update
sudo apt install python3.10 python3.10-venv git

# Clone and install
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

python3.10 -m venv venv
source venv/bin/activate
pip install -e .

# Verify
mlfactory --help
```

### Benefits of WSL2

- ✅ Same commands as Linux/Mac
- ✅ Better performance
- ✅ Docker Desktop integration
- ✅ Native Git experience
- ✅ All documentation examples work as-is

## Native Windows

If you prefer native Windows (PowerShell):

### Prerequisites

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation

2. **Git for Windows**
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Visual C++ Build Tools** (for some dependencies)
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"

### Installation

```powershell
# Open PowerShell
cd C:\Projects  # Or your preferred directory

# Clone repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install ForgeML
pip install -e .

# Verify
mlfactory --help
```

### Command Prompt Alternative

```cmd
# If using CMD instead of PowerShell:
venv\Scripts\activate.bat

# Rest is the same
pip install -e .
```

## Path Differences

### Forward vs Backslash

```powershell
# Windows uses backslashes
C:\Users\YourName\forge-ml

# In Python/config files, use forward slashes OR escape backslashes
data_path = "C:/Users/YourName/data"  # ✅ Works
data_path = "C:\\Users\\YourName\\data"  # ✅ Works
data_path = "C:\Users\YourName\data"  # ❌ Fails
```

### Environment Variables

```powershell
# Set temporarily (current session)
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

# Set permanently
[System.Environment]::SetEnvironmentVariable("MLFLOW_TRACKING_URI", "http://localhost:5000", "User")

# View
$env:MLFLOW_TRACKING_URI
```

## Docker on Windows

### Docker Desktop

1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install and restart
3. Start Docker Desktop

### WSL2 Integration

```powershell
# In Docker Desktop settings:
# ✅ Enable "Use the WSL 2 based engine"
# ✅ Enable WSL Integration for your Ubuntu distro
```

### Start MLflow

```powershell
# In PowerShell (or WSL):
cd infra
docker-compose up -d

# Access MLflow
# http://localhost:5000
```

## PowerShell vs Bash Commands

| Task | Bash (WSL/Mac/Linux) | PowerShell (Windows) |
|------|---------------------|----------------------|
| Activate venv | `source venv/bin/activate` | `venv\Scripts\Activate.ps1` |
| List files | `ls` | `dir` or `ls` (alias) |
| Change directory | `cd folder` | `cd folder` (same) |
| Create directory | `mkdir folder` | `mkdir folder` (same) |
| Remove file | `rm file` | `del file` or `Remove-Item` |
| Environment variable | `export VAR=value` | `$env:VAR = "value"` |
| View file | `cat file.txt` | `Get-Content file.txt` or `type` |

## GPU Support (NVIDIA)

### Requirements

- NVIDIA GPU (GTX 10xx series or newer)
- Latest NVIDIA drivers

### Install CUDA PyTorch

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Troubleshooting GPU

```powershell
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# If CUDA not found, install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
```

## Common Windows Issues

### Issue 1: "Command not found"

```powershell
# Make sure PATH is set
$env:PATH

# Reinstall Python, check "Add to PATH"
```

### Issue 2: Long Path Names

Windows has 260 character path limit.

**Solution:**

```powershell
# Enable long paths (Admin PowerShell)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or clone to shorter path
cd C:\ml
git clone ...
```

### Issue 3: Antivirus Blocking

Some antivirus software blocks Python scripts.

**Solution:**
- Add `forge-ml` folder to antivirus exclusions
- Or temporarily disable during installation

### Issue 4: SSL/Certificate Errors

```powershell
# Try with trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .

# Or upgrade certifi
pip install --upgrade certifi
```

### Issue 5: Permission Denied

```powershell
# Run PowerShell as Administrator
# Or change folder permissions
icacls C:\Projects\forge-ml /grant Users:F /T
```

## File Line Endings

Git may convert line endings (LF ↔ CRLF).

**Configure Git:**

```powershell
# Auto-convert to Windows endings
git config --global core.autocrlf true

# Or keep Linux endings (if using WSL mostly)
git config --global core.autocrlf input
```

## Best Practices for Windows

1. **Use WSL2** when possible (best compatibility)
2. **Short paths**: Install to `C:\forge-ml` instead of deep nested folders
3. **PowerShell 7+**: Install from [Microsoft Store](https://aka.ms/PSWindows)
4. **Windows Terminal**: Better terminal experience
5. **VS Code**: Excellent WSL integration

## Editor Setup

### VS Code (Recommended)

```powershell
# Install VS Code
# Download from: https://code.visualstudio.com/

# Install WSL extension
# In VS Code: Extensions → "WSL"

# Open project in WSL
wsl
cd ~/forge-ml
code .
```

### PyCharm

- Set interpreter to WSL Python: `\\wsl$\Ubuntu\home\user\forge-ml\venv\bin\python`
- Or use native Windows Python

## Quick Reference

```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Create project
mlfactory init sentiment --name myproject

# Navigate (use Tab completion)
cd myproject

# Install dependencies
pip install -r requirements.txt

# Train
python train.py

# Serve
python serve.py
```

## Getting Help

**WSL Issues:**
- WSL docs: https://aka.ms/wsl
- Check: `wsl --status`

**Python Issues:**
- Check version: `python --version`
- Check PATH: `where python`

**ForgeML Issues:**
- Check installation: `pip show forge-ml`
- GitHub Issues: Report Windows-specific problems
