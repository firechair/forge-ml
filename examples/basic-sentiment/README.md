# Basic Sentiment Analysis Example

This example demonstrates a complete workflow using ForgeML's sentiment analysis template.

## Overview

Train a sentiment classifier on the IMDB movie reviews dataset and deploy it as an API.

## Steps

### 1. Create the Project

```bash
cd /tmp  # or your preferred workspace
mlfactory init sentiment --name basic-sentiment-example
cd basic-sentiment-example
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MLflow (Optional)

```bash
# From the forge-ml/infra directory
docker-compose up -d
```

### 4. Review Configuration

The default `config.yaml` uses:
- Model: `distilbert-base-uncased` (fast, good accuracy)
- Dataset: IMDB (movie reviews)
- Training: 3 epochs with small subset for quick testing

### 5. Train the Model

```bash
python train.py
```

Expected output:
```
Loading dataset: imdb
Train size: 5000, Test size: 1000
Initializing model: distilbert-base-uncased

Starting training for 3 epochs...
Epoch 1: 100%|██████| 313/313 [02:15<00:00, 2.31it/s, loss=0.234]

Epoch 1 Training Metrics:
  loss: 0.3421
  accuracy: 0.8654
  f1: 0.8644

✅ Training complete!
Best F1 Score: 0.8904
Model saved to: best_model/
View results at: http://localhost:5000
```

### 6. Test the Model

```python
from model import SentimentClassifier

# Load the trained model
classifier = SentimentClassifier.from_pretrained("best_model")

# Test predictions
texts = [
    "This movie was absolutely fantastic!",
    "Worst film I've ever seen.",
    "It was okay, nothing special."
]

for text in texts:
    result = classifier.predict(text)
    sentiment = "positive" if result["predicted_label"] == 1 else "negative"
    confidence = result[sentiment]
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} ({confidence:.2%} confidence)")
    print()
```

### 7. Serve via API

```bash
mlfactory serve --port 8000
```

### 8. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "Okay"]}'
```

### 9. View API Documentation

Open in your browser:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Results

After training, you should see:
- **Accuracy**: ~89%
- **F1 Score**: ~0.89

Note: These are results on a small subset. Using the full dataset (`train_split: "train"`) will give better results (~92-94%).

## Customization Ideas

### Use Full Dataset

Edit `config.yaml`:
```yaml
data:
  train_split: "train"      # Full 25k samples
  test_split: "test"        # Full 25k samples
```

### Try Different Model

Edit `config.yaml`:
```yaml
model:
  name: "roberta-base"  # More powerful
```

### Adjust Training

Edit `config.yaml`:
```yaml
training:
  num_epochs: 5
  learning_rate: 0.00001
  batch_size: 32
```

## Next Steps

1. **Compare Experiments**: Run multiple training runs with different settings
2. **Deploy**: Build Docker image and deploy to cloud
3. **Monitor**: Track API usage and model performance
4. **Iterate**: Improve based on results

## Cleanup

```bash
# Stop MLflow
cd ../forge-ml/infra
docker-compose down

# Delete project (optional)
cd /tmp
rm -rf basic-sentiment-example
```

---

**Time to Complete**: ~10-15 minutes (depending on hardware)

**Hardware Requirements**: Works on CPU, faster with GPU
