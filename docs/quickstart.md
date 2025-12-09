# 🚀 ForgeML Quick Start Guide

Get up and running with ForgeML in minutes!

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Docker (optional, for MLflow tracking server)

---

## Installation

### 1. Install ForgeML

```bash
# Clone the repository
git clone https://github.com/firechair/forge-ml.git
cd forge-ml

# Install the package
pip install -e .
```

### 2. Verify Installation

```bash
mlfactory --help
```

You should see the available commands.

---

## Create Your First Project

### Step 1: Initialize a Project

```bash
# Create a new sentiment analysis project
mlfactory init sentiment --name my-sentiment-project

# Navigate to the project
cd my-sentiment-project
```

### Step 2: Explore the Project Structure

```
my-sentiment-project/
├── config.yaml          # Training configuration
├── train.py            # Training script
├── model.py            # Model definition
├── serve.py            # API endpoint
├── requirements.txt    # Dependencies
├── Dockerfile         # Container configuration
└── README.md          # Project documentation
```

### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Start MLflow (Optional but Recommended)

MLflow helps you track experiments and manage models.

```bash
# From the forge-ml directory
cd ../infra
docker-compose up -d

# Check it's running
open http://localhost:5000
```

---

## Train Your First Model

### Basic Training

```bash
# Run with default settings
python train.py
```

This will:
- Load the IMDB sentiment dataset
- Train a DistilBERT model
- Track metrics in MLflow
- Save the best model to `best_model/`

### Monitor Training

Watch the progress in your terminal:
```
Epoch 1: 100%|██████████| 313/313 [02:15<00:00, 2.31it/s, loss=0.234]

Epoch 1 Training Metrics:
  loss: 0.3421
  accuracy: 0.8654
  f1: 0.8644

Evaluating on test set...
Test Metrics:
  accuracy: 0.8912
  f1: 0.8904

New best F1 score: 0.8904 - Saving model...
```

### View Results in MLflow

Open `http://localhost:5000` to see:
- Training metrics over time
- Model parameters
- Saved models

---

## Customize Your Training

### Edit Configuration

Open `config.yaml` and modify settings:

```yaml
# Try a different model
model:
  name: "roberta-base"  # Instead of distilbert

# Adjust training
training:
  num_epochs: 5         # More epochs
  batch_size: 32        # Larger batches
  learning_rate: 0.00001  # Lower learning rate

# Use full dataset
data:
  train_split: "train"  # Remove [:5000]
  test_split: "test"
```

### Run New Experiment

```bash
python train.py --experiment "roberta-5epochs"
```

### Compare Experiments

Open MLflow UI to compare different experiments side-by-side.

---

## Use Your Trained Model

###Option 1: Python API

```python
from model import SentimentClassifier

# Load the model
classifier = SentimentClassifier.from_pretrained("best_model")

# Make a prediction
result = classifier.predict("This movie was amazing!")
print(result)
# Output: {'negative': 0.05, 'positive': 0.95, 'predicted_label': 1}
```

### Option 2: REST API

Start the serving endpoint:

```bash
mlfactory serve --port 8000
```

Then make requests:

```bash
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

Response:
```json
{
  "text": "Great product!",
  "sentiment": "positive",
  "confidence": 0.9823,
  "probabilities": {
    "negative": 0.0177,
    "positive": 0.9823
  }
}
```

### Test the API

Open the interactive API docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Deploy with Docker

### Build the Container

```bash
docker build -t my-sentiment-api .
```

### Run the Container

```bash
docker run -p 8000:8000 my-sentiment-api
```

Your API is now running at `http://localhost:8000`!

---

## Next Steps

### 1. Experiment with Different Settings

Try different:
- Models (BERT, RoBERTa, ALBERT)
- Learning rates
- Batch sizes
- Number of epochs

### 2. Use Your Own Data

Modify `train.py` to load custom data:

```python
# Replace the HuggingFace dataset with your own
import pandas as pd

df = pd.read_csv("my_data.csv")
# Process and use your data
```

### 3. Deploy to Production

- Deploy to cloud (AWS, GCP, Azure)
- Set up monitoring
- Add authentication
- Scale with Kubernetes

### 4. Create More Projects

```bash
# When more templates are available
mlfactory init image_classification --name my-image-project
mlfactory init time_series --name my-forecasting-project
```

---

## Troubleshooting

### Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

### CUDA Out of Memory

Edit `config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### MLflow Connection Issues

```bash
# Check if MLflow is running
curl http://localhost:5000

# Or use local file tracking
# Edit config.yaml:
mlflow:
  tracking_uri: "file:./mlruns"
```

### Port Already in Use

```bash
# Use a different port
mlfactory serve --port 8001
```

---

## Getting Help

- **Documentation**: Check `README.md` in your project
- **Examples**: Look in `examples/` directory
- **Issues**: Open an issue on GitHub
- **Discussions**: Join GitHub Discussions

---

## Summary

You've learned how to:
- ✅ Install ForgeML
- ✅ Create a project
- ✅ Train a model
- ✅ Track experiments with MLflow
- ✅ Serve models via API
- ✅ Deploy with Docker

**You're ready to build production ML projects!** 🎉

---

**Next**: Read the [CLI Reference](cli-reference.md) for all available commands
