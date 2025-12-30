# 🎭 Sentiment Analysis Project

Created with **ForgeML** - ML Project Factory

## 📖 Overview

This project **fine-tunes pre-trained language models** for sentiment analysis (positive/negative classification).

### What This Does

1. **Downloads a pre-trained LLM** automatically from HuggingFace (e.g., DistilBERT)
2. **Downloads training data** automatically (e.g., IMDB movie reviews)
3. **Fine-tunes the model** on the sentiment task
4. **Saves the fine-tuned model** that you can deploy

### Key Features

- ✅ **No manual downloads** - Models & data fetch automatically
- ✅ **Configurable** - Change model/data by editing config.yaml
- ✅ **MLflow tracking** - All experiments logged
- ✅ **Production-ready** - Deploy via FastAPI + Docker
- ✅ **Custom data support** - Use your own datasets

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server (Optional)

```bash
# From the forge-ml root directory
cd ../infra
docker-compose up -d
```

This starts MLflow tracking server at `http://localhost:5000`

### 3. Train Your Model

```bash
# Use default configuration
python train.py

# Or specify custom config and experiment name
python train.py --config config.yaml --experiment my-experiment
```

### 4. View Results in MLflow

Open your browser to `http://localhost:5000` to see:
- Training metrics (loss, accuracy, F1)
- Model parameters
- Logged models

### 5. Use the Trained Model

```python
from model import SentimentClassifier

# Load trained model
classifier = SentimentClassifier.from_pretrained("best_model")

# Make predictions
result = classifier.predict("This movie was absolutely amazing!")
print(result)
# Output: {'negative': 0.05, 'positive': 0.95, 'predicted_label': 1}

# Batch predictions
texts = [
    "I loved this product!",
    "Terrible service, very disappointed."
]
results = classifier.predict(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result}")
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize your project:

### 🎯 How Model Selection Works

The model is **automatically downloaded** from HuggingFace based on your config:

```yaml
model:
  name: "distilbert-base-uncased"  # ← This controls which model downloads
```

First time you train, ForgeML will:
1. Check HuggingFace Hub for the model
2. Download it (~250MB for DistilBERT)
3. Cache it locally (subsequent runs are instant)
4. Fine-tune it on your data

### 📊 How Data Selection Works

The dataset is **automatically downloaded** from HuggingFace:

```yaml
data:
  dataset: "imdb"  # ← This controls which dataset downloads
```

First time you train, ForgeML will:
1. Download IMDB dataset from HuggingFace
2. Cache it locally
3. Use it for training

**No manual downloads, no data preparation needed!**

---

### Model Settings

```yaml
model:
  name: "distilbert-base-uncased"  # Any HuggingFace model
  num_labels: 2
  max_length: 512
```

**Available models:**
- `distilbert-base-uncased` - Fast, 66M parameters (recommended for quick testing)
- `bert-base-uncased` - Balanced, 110M parameters
- `roberta-base` - Strong performance, 125M parameters
- `albert-base-v2` - Smaller, 11M parameters

### Training Settings

```yaml
training:
  batch_size: 16          # Reduce if running out of memory
  learning_rate: 0.00002  # Lower for fine-tuning
  num_epochs: 3           # More epochs = better performance (but longer training)
  seed: 42               # For reproducibility
```

### Data Settings

```yaml
data:
  dataset: "imdb"                    # HuggingFace dataset
  train_split: "train[:5000]"        # Use subset for quick testing
  test_split: "test[:1000]"
```

**Available datasets:**
- `imdb` - Movie reviews (25k train, 25k test)
- `tweet_eval` - Tweet sentiment
- `amazon_polarity` - Product reviews (3.6M train)
- `yelp_polarity` - Restaurant reviews (560k train)

To use your own data, modify the data loading in `train.py`.

---

## 📁 Project Structure

```
sentiment-project/
├── config.yaml         # Training configuration
├── train.py           # Training script with MLflow
├── model.py           # Model wrapper class
├── serve.py           # FastAPI serving endpoint
├── requirements.txt   # Python dependencies
├── Dockerfile        # Docker configuration
├── best_model/       # Saved best model (created during training)
└── README.md         # This file
```

---

## 🎯 How It Works

### 1. Data Loading

The training script loads data from HuggingFace Datasets:

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

### 2. Model Initialization

Uses pre-trained transformers from HuggingFace:

```python
classifier = SentimentClassifier(
    model_name="distilbert-base-uncased",
    num_labels=2
)
```

### 3. Training Loop

- Trains for specified number of epochs
- Tracks metrics with MLflow
- Evaluates on test set after each epoch
- Saves best model based on F1 score

### 4. MLflow Tracking

Automaticaly logs:
- **Parameters**: model name, learning rate, batch size, etc.
- **Metrics**: loss, accuracy, precision, recall, F1 score
- **Models**: Best performing model

---

## 📊 Monitoring Training

### View Progress in Terminal

Training shows real-time progress:

```
Epoch 1: 100%|████████████| 313/313 [02:15<00:00, 2.31it/s, loss=0.234]

Epoch 1 Training Metrics:
  loss: 0.3421
  accuracy: 0.8654
  precision: 0.8723
  recall: 0.8567
  f1: 0.8644

Evaluating on test set...
100%|████████████| 63/63 [00:18<00:00, 3.42it/s]

Epoch 1 Test Metrics:
  loss: 0.2891
  accuracy: 0.8912
  precision: 0.8967
  recall: 0.8843
  f1: 0.8904

New best F1 score: 0.8904 - Saving model...
```

### View in MLflow UI

1. Open `http://localhost:5000`
2. Find your experiment (e.g., "sentiment-analysis")
3. Click on runs to see:
   - Metrics over time
   - Parameter comparisons
   - Model artifacts

---

## 🧪 Experimentation Tips

### Try Different Models

```yaml
# In config.yaml
model:
  name: "roberta-base"  # Try different architectures
```

### Tune Hyperparameters

```yaml
training:
  learning_rate: 0.00001  # Lower = more stable
  batch_size: 32         # Higher = faster (needs more GPU memory)
  num_epochs: 5          # More = potentially better performance
```

### Use Full Dataset

```yaml
data:
  train_split: "train"  # Remove [:5000] to use full dataset
  test_split: "test"
```

### Compare Experiments in MLflow

Run multiple experiments with different settings:

```bash
python train.py --experiment "distilbert-baseline"
# Edit config.yaml to change model
python train.py --experiment "roberta-comparison"
# Edit config.yaml to change learning rate
python train.py --experiment "lr-tuning-001"
```

Then compare all experiments in MLflow UI.

---

## 🐛 Troubleshooting

### Out of Memory Error

**Problem**: `CUDA out of memory` or similar error

**Solutions**:
1. Reduce batch size in `config.yaml`:
   ```yaml
   training:
     batch_size: 8  # or even 4
   ```

2. Use a smaller model:
   ```yaml
   model:
     name: "distilbert-base-uncased"  # Smaller than bert-base
   ```

3. Reduce max sequence length:
   ```yaml
   model:
     max_length: 256  # Instead of 512
   ```

### MLflow Connection Error

**Problem**: Cannot connect to MLflow tracking server

**Solutions**:
1. Start the MLflow server:
   ```bash
   cd ../infra
   docker-compose up -d
   ```

2. Or use local file tracking:
   ```yaml
   mlflow:
     tracking_uri: "file:./mlruns"
   ```

### Dataset Download Issues

**Problem**: Dataset fails to download

**Solutions**:
1. Check internet connection
2. Try a different dataset
3. Download manually and load from local path

---

## 🚀 Next Steps

Once training is complete:

1. **Serve the Model**: Use `python serve.py` to start the FastAPI server
2. **Deploy with Docker**: Build and run with `docker build -t sentiment-api . && docker run -p 8000:8000 sentiment-api`
3. **Create Custom Dataset**: Modify `train.py` to use your own data
4. **Fine-tune on Specific Domain**: Use domain-specific data for better performance

---

## 📚 Learn More

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

---

## 📄 License

This project template is part of ForgeML and is licensed under a custom non-commercial license. See the main [LICENSE](../../LICENSE) file for details.

---

**Happy Training! 🎉**
