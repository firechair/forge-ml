# 🔧 ForgeML - ML Project Factory

> **Create, train, and deploy production-ready ML projects in minutes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 What is ForgeML?

**ForgeML** (also called ML Factory) is a developer toolkit that helps you build machine learning projects **the right way, from day one**. Think of it as a "create-react-app" for machine learning - it gives you a production-ready project structure with all the best practices baked in.

### What Does ForgeML Actually Do?

ForgeML helps you **fine-tune pre-trained language models (LLMs)** for specific tasks:

- **🤖 Fine-Tunes LLMs**: Takes pre-trained models (BERT, DistilBERT, RoBERTa) and fine-tunes them on your task
- **📥 Auto-Downloads Everything**: Models and datasets download automatically from HuggingFace
- **🎯 Task-Specific**: Currently supports sentiment analysis, more tasks coming soon
- **🚀 Production-Ready**: Creates deployable APIs, not just notebooks

**Example**: Fine-tune DistilBERT on IMDB movie reviews to classify sentiment in customer feedback.

### The Problem It Solves

Starting an ML project typically requires days or weeks of setup:
- ❌ Configuring experiment tracking (MLflow)
- ❌ Setting up data versioning (DVC)
- ❌ Building training pipelines
- ❌ Creating model serving infrastructure
- ❌ Writing deployment scripts (Docker, CI/CD)

**ForgeML solves this** by giving you everything pre-configured with a single command.

### What You Get

```bash
mlfactory init sentiment --name my-project
# ✅ Everything ready in 30 seconds
```

- ✅ **Organized project structure** - All files in the right places
- ✅ **Training script** - Configured with PyTorch and MLflow
- ✅ **Experiment tracking** - MLflow integration ready
- ✅ **Model serving** - FastAPI endpoint included
- ✅ **Docker setup** - For easy deployment
- ✅ **Documentation** - README and examples
- ✅ **Auto-downloads** - No manual model/data downloads needed!

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for MLflow tracking server)

### Installation

```bash
# Clone the repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# (Recommended) Create and activate a virtual environment
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the package
pip install -e .

# Verify installation
mlfactory --help
```

### Create Your First Project

```bash
# 1. Create a new sentiment analysis project
mlfactory init sentiment --name my-sentiment-project
cd my-sentiment-project

# 2. (Optional) Start MLflow tracking server
# Note: Path is relative to your project directory (my-sentiment-project)
# Points back to forge-ml/infra/docker-compose.yml
docker-compose -f ../infra/docker-compose.yml up -d

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Train your first model
python train.py

# 5. View results in MLflow
open http://localhost:5000

# 6. Serve the trained model
mlfactory serve --model-uri runs:/<run-id>/model
```

---

## ⚙️ How to Configure Models & Data

Users control **which model** and **which data** to use via the `config.yaml` file in each project:

### Choosing a Model

Edit `config.yaml`:
```yaml
model:
  name: "distilbert-base-uncased"  # Change this to any HuggingFace model!
  num_labels: 2
  max_length: 512
```

**Available models** (all auto-download from HuggingFace):
- `distilbert-base-uncased` - Fast, 66M params (recommended for testing)
- `bert-base-uncased` - Balanced, 110M params
- `roberta-base` - Strong performance, 125M params
- `albert-base-v2` - Efficient, 11M params
- Or **any model** from [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-classification)

### Choosing Training Data

Edit `config.yaml`:
```yaml
data:
  dataset: "imdb"           # Change this to any HuggingFace dataset!
  train_split: "train"      # Which split to use
  test_split: "test"
```

**Available datasets** (all auto-download from HuggingFace):
- `imdb` - Movie reviews (25k train, 25k test)
- `tweet_eval` - Tweet sentiment
- `amazon_polarity` - Product reviews (3.6M samples)
- `yelp_polarity` - Restaurant reviews (560k samples)
- Or **any dataset** from [HuggingFace Datasets](https://huggingface.co/datasets?task_categories=task_categories:text-classification)

### Using Your Own Data

Modify `train.py` to load custom data:
```python
import pandas as pd

# Load your CSV/JSON
df = pd.read_csv("my_customer_reviews.csv")
# Format: must have 'text' and 'label' columns
# label: 0=negative, 1=positive
```

**Everything downloads automatically on first run!** No manual setup required.

---

## 📚 Documentation

- **[Project Overview](ProjectOverview.md)** - Comprehensive vision and architecture
- **[CLI Reference](#cli-reference)** - All available commands
- **[Template Guide](#available-templates)** - Available project templates

---

## 🛠️ Available Templates

### 1. Sentiment Analysis
Text classification using transformer models (BERT, DistilBERT, RoBERTa)

```bash
mlfactory init sentiment --name my-sentiment-project
```

**Included:**
- Pre-configured training with HuggingFace Transformers
- MLflow experiment tracking
- FastAPI serving endpoint
- Docker deployment setup
- Comprehensive test suite

**Use cases:** Customer review analysis, social media sentiment, feedback classification

### 2. Time Series Forecasting
LSTM-based forecasting for univariate and multivariate time series

```bash
mlfactory init timeseries --name my-forecast-project
```

**Included:**
- LSTM model implementation (PyTorch)
- Sequence-to-sequence forecasting
- MLflow tracking integration
- FastAPI serving with batch prediction
- Visualization and evaluation tools
- Comprehensive test suite

**Use cases:** Energy demand forecasting, stock price prediction, weather forecasting, IoT sensor predictions

> **Note**: More templates (image classification, object detection, NER, etc.) coming soon!

---

## 💻 CLI Reference

### `mlfactory init`

Create a new ML project from a template.

```bash
mlfactory init <template> --name <project-name>

# Examples:
mlfactory init sentiment --name customer-reviews
mlfactory init timeseries --name energy-forecast
```

**Arguments:**
- `template` - Template name (`sentiment`, `timeseries`)
- `--name` - Project name (default: `my-project`)

### `mlfactory train`

Train a model using the configuration file (when inside a project directory).

```bash
mlfactory train --config config.yaml --experiment <name>

# Example:
mlfactory train --experiment baseline-v1
```

**Options:**
- `--config` - Path to config file (default: `config.yaml`)
- `--experiment` - Experiment name for tracking (default: `default`)

### `mlfactory serve`

Serve a trained model via FastAPI.

```bash
mlfactory serve --model-uri <uri> --port <port>

# Example:
mlfactory serve --model-uri runs:/abc123def/model --port 8000
```

**Options:**
- `--model-uri` - MLflow model URI
- `--port` - Server port (default: `8000`)

---

## 🏗️ Project Structure

When you create a project, you get this structure:

```
my-project/
├── config.yaml          # Training configuration
├── train.py             # Training script with MLflow tracking
├── model.py             # Model architecture definition
├── serve.py             # FastAPI serving application
├── requirements.txt     # Project dependencies
├── Dockerfile          # Docker configuration
├── README.md           # Project-specific documentation
└── tests/              # Unit tests
```

---

## 🔧 Technology Stack

ForgeML integrates industry-standard ML tools:

- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[MLflow](https://mlflow.org/)** - Experiment tracking and model registry
- **[DVC](https://dvc.org/)** - Data version control
- **[FastAPI](https://fastapi.tiangolo.com/)** - Model serving
- **[Docker](https://www.docker.com/)** - Containerization
- **[Typer](https://typer.tiangolo.com/)** - CLI framework

---

## 📊 Infrastructure Setup

ForgeML includes a Docker Compose setup for local development:

```bash
# Start MLflow and MinIO (S3-compatible storage)
cd forge-ml/infra
docker-compose up -d

# MLflow UI: http://localhost:5000
# MinIO Console: http://localhost:9001
```

**Services:**
- **MLflow** - Experiment tracking and model registry
- **MinIO** - Local object storage for artifacts and datasets

---

## 🎓 Example Workflow

Here's a complete example of building a sentiment analysis model:

```bash
# 1. Create project
mlfactory init sentiment --name movie-reviews
cd movie-reviews

# 2. Start infrastructure
docker-compose -f ../infra/docker-compose.yml up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Customize config (optional)
# Edit config.yaml to change model, hyperparameters, etc.

# 5. Train model
python train.py

# 6. Check MLflow for results
open http://localhost:5000

# 7. Test the model
python -c "
from model import SentimentClassifier
model = SentimentClassifier.load_from_checkpoint('path/to/model')
print(model.predict('This movie was amazing!'))
"

# 8. Serve via API
mlfactory serve --model-uri runs:/abc123/model

# 9. Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product, highly recommend!"}'
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs** - Open an issue describing the problem
2. **Suggest features** - Share your ideas for improvements
3. **Submit PRs** - Fix bugs or add features
4. **Create templates** - Share new project templates
5. **Improve docs** - Help make documentation clearer

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🗺️ Roadmap

### Current Status (v0.1.0)

- ✅ CLI framework with command-line tools
- ✅ Sentiment analysis template (BERT, DistilBERT, RoBERTa)
- ✅ Time series forecasting template (LSTM-based)
- ✅ MLflow experiment tracking integration
- ✅ Docker infrastructure (local and team setups)
- ✅ DVC data versioning integration
- ✅ GitHub Actions CI/CD pipelines
- ✅ FastAPI model serving
- ✅ Comprehensive test suites
- ✅ Team collaboration features
- ✅ Cross-platform setup scripts (Linux, macOS, Windows)

### Coming Soon

- ⏳ Image classification template (ResNet, EfficientNet)
- ⏳ Named Entity Recognition (NER) template
- ⏳ Object detection template
- ⏳ Model monitoring dashboard
- ⏳ Auto-deployment to cloud platforms (AWS, GCP, Azure)

### Future Plans

- 🔮 AutoML integration
- 🔮 Multi-cloud support
- 🔮 Template marketplace
- 🔮 Web-based dashboard

---

## 🔧 Troubleshooting

### Common Issues

**Q: `ImportError: No module named 'fastapi'` when running serve.py**
A: Make sure you've installed all dependencies in your project directory:
```bash
cd my-sentiment-project
pip install -r requirements.txt
```

**Q: Docker Compose file not found**
A: The path `../infra/docker-compose.yml` is relative to your project directory. Make sure you're inside the project folder created by `mlfactory init`.

**Q: MLflow tracking server not accessible**
A: Check if Docker is running and MLflow container is up:
```bash
docker ps  # Should show mlflow container
docker-compose -f infra/docker-compose.yml logs mlflow  # Check logs
```

**Q: Training is very slow**
A: By default, PyTorch installs CPU-only version. For GPU support:
- **NVIDIA GPU**: Install `torch` with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **Mac M1/M2**: PyTorch should auto-use MPS (Metal Performance Shaders)

**Q: `mlfactory` command not found**
A: Make sure you've installed the package and activated your virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
pip install -e .
```

**Q: Model URI loading fails**
A: Ensure MLflow tracking server is running and the run ID is correct:
```bash
# Check available runs in MLflow UI
open http://localhost:5000
# Or use: python serve.py (uses local best_model/ instead)
```

For more issues, see our [GitHub Issues](https://github.com/firechair/forge-ml/issues).

---

## 📖 Learn More

- **[Project Overview](ProjectOverview.md)** - Deep dive into architecture and design
- **[Examples](examples/)** - Complete example projects
- **[MLflow Documentation](https://mlflow.org/docs/)** - Learn about experiment tracking
- **[PyTorch Tutorials](https://pytorch.org/tutorials/)** - Deep learning fundamentals

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

ForgeML builds on the shoulders of giants:
- The amazing PyTorch team
- MLflow developers at Databricks
- FastAPI creator Sebastián Ramírez
- The entire Python ML community

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/firechair/forge-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/firechair/forge-ml/discussions)
- **Email**: your.email@example.com

---

<p align="center">
  <strong>Built with ❤️ for the ML community</strong>
</p>

<p align="center">
  Made by <a href="https://github.com/firechair">firechair</a>
</p>