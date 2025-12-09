# Quick Start Verification Guide

**Use this checklist to verify ForgeML works perfectly on your computer.**

---

## Prerequisites Check

Run these commands to verify prerequisites:

```bash
# Check Python version (need 3.10+)
python --version  # or python3 --version

# Check Git
git --version

# Check Docker (optional)
docker --version
docker-compose --version
```

**Expected:** Python 3.10 or higher, Git installed, Docker optional

---

## Installation Test

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# Run setup script
# On Linux/Mac:
./scripts/setup.sh

# On Windows PowerShell:
.\scripts\setup.ps1
```

**Expected Output:**
- ✅ Python version check passes
- ✅ Virtual environment created
- ✅ Dependencies installed
- ✅ Pre-commit hooks installed
- ✅ Docker infrastructure started
- ✅ Verification tests pass

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install ForgeML
pip install -e .

# Verify installation
mlfactory --help
```

**Expected Output:** Help message showing available commands

---

## CLI Verification

Test all CLI commands:

```bash
# Test help command
mlfactory --help

# Test init command (don't worry, we'll delete this test project)
mlfactory init sentiment --name verification-test
cd verification-test
ls -la

# Expected files:
# - config.yaml
# - train.py
# - model.py
# - serve.py
# - requirements.txt
# - Dockerfile
# - README.md
# - tests/

# Clean up test project
cd ..
rm -rf verification-test
```

**Expected:** Project created successfully with all files present

---

## Sentiment Template Test

Test the sentiment analysis template end-to-end:

```bash
# 1. Create project
mlfactory init sentiment --name test-sentiment
cd test-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# Expected: All tests pass

# 4. Quick training test (1 epoch)
python train.py --epochs 1

# Expected:
# - Training starts
# - Progress bar shows
# - Model saves to best_model/
# - MLflow logs experiment (if Docker running)

# 5. Test serving
python serve.py &
sleep 5

# Test health endpoint
curl http://localhost:8000/health

# Expected: {"status": "healthy"}

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'

# Expected: {"label": 1, "confidence": 0.XX, ...}

# Stop server
pkill -f "python serve.py"

# 6. Test CLI serve command
mlfactory serve --port 8000 &
sleep 5
curl http://localhost:8000/health
pkill -f "mlfactory serve"

# Clean up
cd ..
rm -rf test-sentiment
```

**All steps should complete without errors**

---

## Time Series Template Test

Test the time series forecasting template:

```bash
# 1. Create project
mlfactory init timeseries --name test-timeseries
cd test-timeseries

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# Expected: All tests pass

# 4. Quick training test (1 epoch)
python train.py --epochs 1

# Expected:
# - Training starts with sequence data
# - Model saves successfully
# - MSE/MAE metrics logged

# 5. Test serving
python serve.py &
sleep 5

# Test health endpoint
curl http://localhost:8000/health

# Test forecast endpoint
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]], "normalize": true}'

# Expected: {"predictions": [...], "input_length": 8, ...}

# Stop server
pkill -f "python serve.py"

# Clean up
cd ..
rm -rf test-timeseries
```

**All steps should complete without errors**

---

## MLflow UI Test

If Docker is running, verify MLflow UI:

```bash
# Start MLflow infrastructure
cd forge-ml/infra
docker-compose up -d

# Wait for startup
sleep 10

# Check MLflow is accessible
curl http://localhost:5000

# Open in browser
open http://localhost:5000  # Mac
# or
xdg-open http://localhost:5000  # Linux
# or
start http://localhost:5000  # Windows
```

**Expected:** MLflow UI loads showing experiments

---

## Documentation Verification

Verify all documentation exists and is accessible:

```bash
cd forge-ml

# Main documentation
ls -la README.md CONTRIBUTING.md LICENSE

# Docs directory
ls -la docs/
# Expected:
# - installation.md
# - quickstart.md
# - cli-reference.md
# - team-collaboration.md
# - faq.md
# - windows-guide.md

# Examples
ls -la examples/
# Expected:
# - basic-sentiment/
# - custom-data-sentiment/
# - model-comparison/
# - team-workflow/

# GitHub templates
ls -la .github/ISSUE_TEMPLATE/
ls -la .github/pull_request_template.md
```

**Expected:** All files present

---

## Integration Test

Run the full integration test suite:

```bash
cd forge-ml

# Activate virtual environment if not already active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run integration tests
pytest tests/test_integration.py -v

# Expected: All integration tests pass
```

---

## Common Issues & Solutions

### Issue: `python: command not found`
**Solution:** Use `python3` instead of `python` on Linux/Mac

### Issue: `mlfactory: command not found`
**Solution:** Activate virtual environment first: `source venv/bin/activate`

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: Port 8000 already in use
**Solution:** Kill existing process: `pkill -f "python serve.py"` or use different port

### Issue: Docker containers won't start
**Solution:**
- Check Docker is running: `docker ps`
- Try: `docker-compose down && docker-compose up -d`
- Check ports 5000 and 9000 aren't in use

### Issue: Tests fail
**Solution:**
- Ensure all dependencies installed: `pip install -e ".[dev]"`
- Check Python version: `python --version` (need 3.10+)
- Try in fresh virtual environment

---

## Success Indicators

You've successfully set up ForgeML if:

- ✅ CLI responds to `mlfactory --help`
- ✅ Can create projects with `mlfactory init`
- ✅ Templates have all required files
- ✅ Dependencies install without errors
- ✅ Training script runs successfully
- ✅ Tests pass
- ✅ Model serving works
- ✅ API endpoints respond correctly
- ✅ MLflow UI accessible (if Docker running)
- ✅ Documentation is complete and readable

---

## Quick Verification Script

Save this as `quick_verify.sh` for automated verification:

```bash
#!/bin/bash
set -e

echo "🔍 ForgeML Quick Verification"
echo "=============================="
echo ""

# Check Python
echo "✓ Checking Python..."
python --version || python3 --version

# Check CLI
echo "✓ Checking CLI..."
mlfactory --help > /dev/null
echo "  CLI working!"

# Create test project
echo "✓ Creating test project..."
mlfactory init sentiment --name verify-test
cd verify-test

# Check files
echo "✓ Checking files..."
test -f config.yaml && echo "  config.yaml ✓"
test -f train.py && echo "  train.py ✓"
test -f serve.py && echo "  serve.py ✓"
test -f requirements.txt && echo "  requirements.txt ✓"
test -d tests && echo "  tests/ ✓"

# Install deps
echo "✓ Installing dependencies..."
pip install -r requirements.txt > /dev/null

# Run tests
echo "✓ Running tests..."
pytest tests/ -q

# Cleanup
echo "✓ Cleaning up..."
cd ..
rm -rf verify-test

echo ""
echo "=============================="
echo "✅ All checks passed!"
echo "ForgeML is working correctly."
```

Run with: `bash quick_verify.sh`

---

## Next Steps After Verification

Once everything works, you're ready to:

1. **Create your first real project:**
   ```bash
   mlfactory init sentiment --name my-real-project
   cd my-real-project
   ```

2. **Customize the configuration:**
   - Edit `config.yaml` to choose your model and dataset
   - See [README.md](README.md) for available options

3. **Train your model:**
   ```bash
   python train.py
   ```

4. **Monitor in MLflow:**
   - Open http://localhost:5000
   - View experiments, compare runs, select best model

5. **Deploy your model:**
   - Use `python serve.py` or `mlfactory serve`
   - Build Docker image: `docker build -t my-model .`

6. **Learn more:**
   - Read [docs/quickstart.md](docs/quickstart.md)
   - Check [examples/](examples/) for common workflows
   - Join discussions on GitHub

---

**Happy Machine Learning! 🚀**

If you encounter any issues not covered here, please open an issue on GitHub.
