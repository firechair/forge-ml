# Installation Guide

Complete installation instructions for all platforms.

## Prerequisites

- **Python**: 3.10 or higher
- **Git**: For version control
- **Docker**: Optional, for MLflow tracking server

Check versions:
```bash
python --version  # Should be 3.10+
git --version
docker --version
```

## Quick Install (All Platforms)

```bash
# 1. Clone repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install ForgeML
pip install -e .

# 5. Verify
mlfactory --help
```

## Platform-Specific Instructions

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv git

# Fedora/RHEL
sudo dnf install python3.10 python3-pip git

# Install ForgeML
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install ForgeML
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

**Apple Silicon (M1/M2):**
```bash
# PyTorch will use Metal Performance Shaders automatically
# Verify with:
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Windows

**Option 1: Native Windows**

```powershell
# Install Python from python.org
# Then:
python -m venv venv
venv\Scripts\activate
pip install -e .
```

**Option 2: WSL2 (Recommended)**

```bash
# Install WSL2
wsl --install

# Inside WSL:
sudo apt update
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

## GPU Support

### NVIDIA GPU (CUDA)

```bash
# Linux
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD GPU (ROCm)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

### Mac M1/M2 (MPS)

PyTorch automatically uses Metal - no extra steps needed!

## Docker Setup

```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Verify Docker is running
docker --version
docker ps
```

## MLflow Infrastructure

MLflow provides experiment tracking, model registry, and visualization for your ML projects.

### Local MLflow Server (Single User)

```bash
# From the forge-ml root directory
cd infra
docker-compose up -d

# Verify it's running
curl http://localhost:5000/health
# Or open in browser
open http://localhost:5000
```

**What this provides:**
- MLflow UI at `http://localhost:5000`
- Experiment tracking for all your projects
- Model registry
- Metric visualization

### Stop MLflow

```bash
cd infra
docker-compose down
```

### Team MLflow Server (Multi-User)

For team collaboration with persistent storage:

```bash
cd infra
docker-compose -f docker-compose-team.yml up -d
```

This uses PostgreSQL as the backend store for better performance and multi-user support.

**→ For team setup details, see the [Team Collaboration Guide](team-collaboration.md)**

### Using MLflow in Your Projects

Your generated projects are pre-configured to use MLflow. The `config.yaml` file contains:

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"  # Points to local server
  experiment_name: "my-project"
```

**Alternative: File-based tracking (no Docker needed)**

Edit your project's `config.yaml`:
```yaml
mlflow:
  tracking_uri: "file:./mlruns"  # Local directory instead of server
```

## Development Install

For contributors:

```bash
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Conda Alternative

```bash
# Create environment
conda create -n forgeml python=3.10
conda activate forgeml

# Install
pip install -e .
```

## Troubleshooting

**Python version too old:**
```bash
# Use pyenv to install newer Python
curl https://pyenv.run | bash
pyenv install 3.10.12
pyenv global 3.10.12
```

**Permission denied (Linux/Mac):**
```bash
# Don't use sudo with pip
# Instead, use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

**SSL certificate error:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .
```

## Verification

```bash
# Test CLI
mlfactory --help

# Create test project
mlfactory init sentiment --name test
cd test
pip install -r requirements.txt

# Quick test (won't train, just validates)
python -c "import torch; import transformers; print('✓ All imports work!')"
```

## Updating

```bash
cd forge-ml
git pull
pip install -e . --upgrade
```

## Uninstall

```bash
pip uninstall forge-ml
rm -rf venv  # Remove virtual environment
```
