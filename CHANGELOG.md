# Changelog

All notable changes to ForgeML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-09

### 🎉 Initial Release

ForgeML's first public release! A comprehensive ML project scaffolding tool for fine-tuning language models and time series forecasting.

### ✨ Features

#### Core Functionality
- **CLI Tool**: Complete command-line interface with `init`, `train`, and `serve` commands
- **Project Templates**: Two production-ready templates (sentiment analysis and time series forecasting)
- **MLflow Integration**: Automatic experiment tracking and model registry
- **Model Serving**: FastAPI-based REST API for model inference
- **Data Versioning**: DVC integration for dataset management

#### Templates

**Sentiment Analysis**
- Pre-configured training with HuggingFace Transformers
- Support for BERT, DistilBERT, RoBERTa, and more
- Automatic model and dataset download from HuggingFace Hub
- MLflow experiment tracking
- FastAPI serving endpoint with health checks
- Docker deployment setup
- Comprehensive test suite (25+ tests)

**Time Series Forecasting**
- LSTM-based sequence-to-sequence modeling
- Support for univariate and multivariate time series
- Configurable sequence length and prediction horizons
- MLflow experiment tracking
- FastAPI serving with batch prediction support
- Visualization and evaluation tools
- Comprehensive test suite (8+ tests)

#### Documentation
- Complete installation guide for Linux, macOS, and Windows
- Quick start tutorial
- CLI reference
- Team collaboration guide
- FAQ with 40+ common questions
- Windows-specific setup guide
- 4 comprehensive example workflows

#### Developer Tools
- Automated setup scripts for all platforms (`setup.sh`, `setup.ps1`)
- Makefile with 30+ common development commands
- Pre-commit hooks (black, flake8, mypy, isort)
- GitHub Actions CI/CD pipelines
- Integration test suite
- Verification script to check installation

#### Team Collaboration
- Shared MLflow server setup (PostgreSQL backend)
- DVC remote storage configuration
- Team workflow documentation with examples
- Git workflow best practices for ML teams

### 📦 What's Included

```
forge-ml/
├── cli/                    # CLI implementation
├── templates/
│   ├── sentiment/         # NLP text classification
│   └── timeseries/        # LSTM time series forecasting
├── infra/
│   ├── docker-compose.yml          # Local MLflow
│   └── docker-compose-team.yml     # Team MLflow with PostgreSQL
├── examples/              # 4 complete example workflows
├── docs/                  # 6 comprehensive guides
├── scripts/              # Automated setup scripts
├── tests/                # Integration tests
└── .github/              # CI/CD + issue/PR templates
```

### 🎯 Key Highlights

- **Instant Setup**: One-command setup script for all platforms
- **Production Ready**: Complete with Docker, CI/CD, and testing
- **Cross-Platform**: Full support for Linux, macOS, and Windows
- **Well Tested**: 33+ test functions across templates
- **Comprehensive Docs**: 2,800+ lines of documentation
- **Team Ready**: Shared infrastructure and collaboration guides
- **Open Source**: Custom non-commercial license (free for personal/educational use)

### 🔧 Technical Details

- **Python**: 3.10+ required
- **ML Framework**: PyTorch with HuggingFace Transformers
- **Experiment Tracking**: MLflow 2.9.0+
- **API Framework**: FastAPI
- **Data Versioning**: DVC
- **Containerization**: Docker & Docker Compose
- **Testing**: pytest with integration tests
- **Code Quality**: Pre-commit hooks with black, flake8, mypy

### 📚 Example Usage

```bash
# Install ForgeML
git clone https://github.com/firechair/forge-ml.git
cd forge-ml
./scripts/setup.sh

# Create a sentiment analysis project
mlfactory init sentiment --name my-project
cd my-project

# Train the model
pip install -r requirements.txt
python train.py

# Serve the model
python serve.py
```

### 🙏 Acknowledgments

Built with ❤️ for the ML community. Special thanks to:
- HuggingFace for Transformers and Datasets
- MLflow for experiment tracking
- FastAPI for the web framework
- PyTorch for deep learning

### 🔗 Links

- **Repository**: https://github.com/firechair/forge-ml
- **Issues**: https://github.com/firechair/forge-ml/issues
- **Discussions**: https://github.com/firechair/forge-ml/discussions

---

## Future Releases

### Planned for v0.2.0
- Image classification template (ResNet, EfficientNet)
- Named Entity Recognition (NER) template
- Enhanced model monitoring dashboard
- Cloud deployment automation (AWS, GCP, Azure)

### Planned for v0.3.0
- Object detection template
- AutoML integration
- Web-based project dashboard
- Template marketplace

---

**Note**: This is the initial release. We welcome contributions, bug reports, and feature requests!
