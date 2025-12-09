# ForgeML Setup Script for Windows (PowerShell)
# One-command setup for development environment

Write-Host "🔧 ForgeML Setup Script" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "📋 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1 | Select-String -Pattern "\d+\.\d+" | ForEach-Object { $_.Matches.Value }

    if (-not $pythonVersion) {
        Write-Host "❌ Python not found. Please install Python 3.10+ first." -ForegroundColor Red
        exit 1
    }

    $major, $minor = $pythonVersion.Split('.')
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
        Write-Host "❌ Python $pythonVersion found. Requires 3.10+." -ForegroundColor Red
        exit 1
    }

    Write-Host "✓ Python $pythonVersion (OK)" -ForegroundColor Green
}
catch {
    Write-Host "❌ Error checking Python version: $_" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Skipping..." -ForegroundColor Yellow
}
else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host ""
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel --quiet

# Install ForgeML
Write-Host ""
Write-Host "📥 Installing ForgeML..." -ForegroundColor Yellow
pip install -e . --quiet
Write-Host "✓ ForgeML installed" -ForegroundColor Green

# Check Docker
Write-Host ""
Write-Host "🐳 Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($dockerVersion -match "Docker version") {
        Write-Host "✓ Docker is installed" -ForegroundColor Green

        # Check if Docker is running
        $dockerRunning = docker ps 2>&1
        if ($dockerRunning -match "CONTAINER") {
            Write-Host "✓ Docker is running" -ForegroundColor Green

            # Ask to start infrastructure
            $response = Read-Host "Start MLflow infrastructure? (y/n)"
            if ($response -eq "y" -or $response -eq "Y") {
                Set-Location infra
                docker-compose up -d
                Set-Location ..
                Write-Host "✓ MLflow started at http://localhost:5000" -ForegroundColor Green
            }
        }
        else {
            Write-Host "⚠️  Docker installed but not running" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "⚠️  Docker not found (optional)" -ForegroundColor Yellow
}

# Run verification
Write-Host ""
Write-Host "✅ Running verification tests..." -ForegroundColor Yellow
python scripts\verify.py

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate environment: venv\Scripts\Activate.ps1"
Write-Host "  2. Create project: mlfactory init sentiment --name my-project"
Write-Host "  3. Start coding!"
Write-Host ""
