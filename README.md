# 🔧 ForgeML - ML Project Factory

> **Create production-ready ML projects in minutes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## What is ForgeML?

**ForgeML** is a developer toolkit that scaffolds production-ready machine learning projects with all the best practices baked in. Think of it as "create-react-app" for machine learning.

### Key Features

- 🤖 **Pre-configured Templates** - Sentiment Analysis & Time Series Forecasting
- 📊 **MLflow Integration** - Automatic experiment tracking
- 🚀 **FastAPI Serving** - Production-ready REST APIs
- 🐳 **Docker Ready** - One-command infrastructure setup
- 📦 **DVC Support** - Built-in data versioning
- ✅ **Testing Framework** - Comprehensive test suites included
- 🔄 **CI/CD Pipelines** - GitHub Actions pre-configured
- 💻 **Cross-Platform** - Linux, macOS, and Windows support

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/firechair/forge-ml.git
cd forge-ml
./scripts/setup.sh  # Or setup.ps1 on Windows

# Create your first project
mlfactory init sentiment --name my-project
cd my-project
pip install -r requirements.txt

# Train and serve
python train.py
python serve.py
```

**→ For detailed instructions, see the [Installation Guide](docs/installation.md)**

---

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Complete setup instructions for all platforms
- **[Quick Start Tutorial](docs/quickstart.md)** - Step-by-step first project guide
- **[Architecture Documentation](docs/architecture.md)** - System design and component architecture
- **[CLI Reference](docs/cli-reference.md)** - Full command documentation
- **[FAQ](docs/faq.md)** - Common questions and troubleshooting
- **[Team Collaboration](docs/team-collaboration.md)** - Shared workflows and infrastructure
- **[Windows Guide](docs/windows-guide.md)** - Windows-specific setup
- **[Verification Guide](docs/verification-guide.md)** - Validate your installation

---

## 🎨 Available Templates

### 1. Sentiment Analysis
Fine-tune transformer models (BERT, DistilBERT, RoBERTa) for text classification.

```bash
mlfactory init sentiment --name my-sentiment-project
```

**Use cases:** Customer review analysis, social media sentiment, feedback classification

**[→ Template Documentation](templates/sentiment/README.md)**

### 2. Time Series Forecasting
LSTM-based models for univariate and multivariate time series prediction.

```bash
mlfactory init timeseries --name my-forecast-project
```

**Use cases:** Energy demand forecasting, stock price prediction, IoT sensor predictions

**[→ Template Documentation](templates/timeseries/README.md)**

---

## 💻 CLI Commands

ForgeML provides three main commands:

- `mlfactory init <template> --name <project>` - Create a new project
- `mlfactory train` - Train a model (run inside project directory)
- `mlfactory serve` - Serve a trained model

**→ See [CLI Reference](docs/cli-reference.md) for full documentation**

---

## 🏗️ What You Get

When you create a project with ForgeML, you get:

```
my-project/
├── config.yaml           # Model and training configuration
├── train.py             # Training script with MLflow tracking
├── model.py             # Model architecture definition
├── serve.py             # FastAPI serving endpoint
├── requirements.txt     # All dependencies
├── Dockerfile          # Container configuration
├── tests/              # Test suite with pytest
└── README.md           # Template-specific documentation
```

- ✅ Organized project structure
- ✅ MLflow experiment tracking
- ✅ FastAPI REST API
- ✅ Docker deployment
- ✅ Comprehensive tests
- ✅ DVC data versioning
- ✅ Pre-commit hooks

---

## 📖 Examples

Check out our example workflows:

- **[Basic Sentiment](examples/basic-sentiment/)** - Simple end-to-end example
- **[Custom Data](examples/custom-data-sentiment/)** - Using your own CSV data
- **[Model Comparison](examples/model-comparison/)** - Comparing BERT vs DistilBERT vs RoBERTa
- **[Team Workflow](examples/team-workflow/)** - Multi-developer collaboration example

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up your development environment
- Running tests
- Creating new templates
- Code style and commit guidelines

---

## 📄 License

This project is licensed under a custom non-commercial license. See [LICENSE](LICENSE) for details.

**In simple terms:**
- ✅ Use it personally, modify it, contribute improvements
- ❌ Don't sell it, don't rebrand it, don't use it commercially
- 🔗 Always credit the original project

For commercial use, please contact the author.

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/firechair/forge-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/firechair/forge-ml/discussions)

---

## 🙏 Acknowledgments

Built with ❤️ for the ML community using:

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) for pre-trained models
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for model serving
- [PyTorch](https://pytorch.org/) for deep learning

---

<p align="center">
  <strong>Built with ❤️ for the ML community</strong>
</p>

<p align="center">
  Made by <a href="https://github.com/firechair">firechair</a>
</p>
