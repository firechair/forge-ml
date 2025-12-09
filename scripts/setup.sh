#!/bin/bash
# ForgeML Setup Script for Linux/Mac
# One-command setup for development environment

set -e  # Exit on error

echo "🔧 ForgeML Setup Script"
echo "======================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found. Requires 3.10+."
    exit 1
fi

echo "✓ Python $PYTHON_VERSION (OK)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# Install ForgeML
echo ""
echo "📥 Installing ForgeML..."
pip install -e . --quiet
echo "✓ ForgeML installed"

# Check Docker (optional)
echo ""
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo "✓ Docker is running"

        # Start infrastructure
        read -p "Start MLflow infrastructure? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd infra
            docker-compose up -d
            cd ..
            echo "✓ MLflow started at http://localhost:5000"
        fi
    else
        echo "⚠️  Docker installed but not running"
    fi
else
    echo "⚠️  Docker not found (optional)"
fi

# Run verification
echo ""
echo "✅ Running verification tests..."
python scripts/verify.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Create project: mlfactory init sentiment --name my-project"
echo "  3. Start coding!"
echo ""
