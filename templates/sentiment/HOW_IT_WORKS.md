# 🔍 How ForgeML Works - Behind the Scenes

This document explains **exactly what happens** when you use ForgeML to fine-tune a model.

---

## 🚀 What Happens When You Run `python train.py`

### Step 1: Configuration Loading

```yaml
# config.yaml controls everything
model:
  name: "distilbert-base-uncased"  # Which model to use
data:
  dataset: "imdb"                  # Which dataset to use
```

The training script reads this file and knows:
- **Which pre-trained model** to download
- **Which dataset** to download
- **How to train** (batch size, learning rate, epochs)

---

### Step 2: Automatic Model Download

```python
# This happens in model.py automatically
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"  # From config.yaml
)
```

**What happens:**
1. HuggingFace checks if model is cached locally (`~/.cache/huggingface/`)
2. If not found, downloads from HuggingFace Hub
3. Downloads model weights (~250MB for DistilBERT)
4. Caches locally for future use
5. Loads into memory

**You don't download anything manually!**

---

### Step 3: Automatic Dataset Download

```python
# This happens in train.py automatically
from datasets import load_dataset

dataset = load_dataset("imdb")  # From config.yaml
```

**What happens:**
1. HuggingFace Datasets checks cache (`~/.cache/huggingface/datasets/`)
2. If not found, downloads IMDB dataset
3. Downloads ~80MB of text data
4. Caches locally
5. Loads into memory

**Again, completely automatic!**

---

### Step 4: Fine-Tuning

The pre-trained model (e.g., DistilBERT) already knows English language patterns from training on millions of documents. Now we **fine-tune** it for sentiment analysis:

```python
# Simplified version of what happens
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch['text'])
        loss = calculate_loss(outputs, batch['labels'])
        
        # Backward pass (update model weights)
        loss.backward()
        optimizer.step()
```

**What fine-tuning means:**
- Start with pre-trained weights (general language understanding)
- Adjust weights slightly for sentiment classification
- Model learns: "amazing" → positive, "terrible" → negative
- Much faster than training from scratch (minutes vs. days)

---

### Step 5: Saving the Fine-Tuned Model

```python
# Saves to best_model/ directory
classifier.save_pretrained("best_model/")
```

Now you have a **custom model** fine-tuned for your specific task!

---

## 🎛️ How to Change What Gets Downloaded

### Change the Model

Edit `config.yaml`:
```yaml
model:
  name: "roberta-base"  # Different model, different download
```

Next time you run `python train.py`:
- Downloads RoBERTa instead of DistilBERT
- Uses RoBERTa architecture (125M params vs. 66M)
- Same training process, different model

**Available options:**
- `distilbert-base-uncased` - Smallest, fastest (66M params)
- `bert-base-uncased` - Medium (110M params)
- `roberta-base` - Largest, best accuracy (125M params)
- `albert-base-v2` - Efficient (11M params)
- Any HuggingFace model at https://huggingface.co/models

### Change the Dataset

Edit `config.yaml`:
```yaml
data:
  dataset: "tweet_eval"  # Different dataset, different download
```

Next time you train:
- Downloads tweet_eval instead of IMDB
- Uses Twitter sentiment data instead of movie reviews
- Same training code, different data

**Available options:**
- `imdb` - Movie reviews (25k train samples)
- `tweet_eval` - Tweets (sentiment subset)
- `amazon_polarity` - Amazon reviews (3.6M samples)
- `yelp_polarity` - Yelp reviews (560k samples)
- Any HuggingFace dataset at https://huggingface.co/datasets

---

## 📦 What Gets Stored Where

### Cache Locations

```bash
# Models cached here:
~/.cache/huggingface/hub/
├── models--distilbert-base-uncased/
├── models--roberta-base/
└── ...

# Datasets cached here:
~/.cache/huggingface/datasets/
├── imdb/
├── tweet_eval/
└── ...

# Your fine-tuned model saved here:
your-project/best_model/
├── config.json
├── pytorch_model.bin
└── ...
```

### Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| DistilBERT model | ~250MB | Cached after first download |
| BERT model | ~440MB | Cached after first download |
| RoBERTa model | ~500MB | Cached after first download |
| IMDB dataset | ~80MB | Cached after first download |
| Fine-tuned model | ~250-500MB | Saved in project directory |

**Total for one project**: ~500MB - 1GB

---

## 🔄 The Complete Workflow

```
┌─────────────────────────────────────────────┐
│  User edits config.yaml                     │
│  - model.name: "distilbert-base-uncased"   │
│  - data.dataset: "imdb"                     │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  python train.py                            │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Step 1: Load Configuration                 │
│  ✓ Read config.yaml                         │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Step 2: Download Model (if not cached)     │
│  ✓ Check ~/.cache/huggingface/hub/          │
│  ✓ Download if missing                      │
│  ✓ Load into memory                         │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Step 3: Download Dataset (if not cached)   │
│  ✓ Check ~/.cache/huggingface/datasets/     │
│  ✓ Download if missing                      │
│  ✓ Load into memory                         │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Step 4: Fine-Tune Model                    │
│  ✓ Train for N epochs                       │
│  ✓ Track metrics in MLflow                  │
│  ✓ Save best model                          │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Output: Fine-tuned model in best_model/    │
│  ✓ Ready to use for predictions             │
│  ✓ Ready to deploy via API                  │
└─────────────────────────────────────────────┘
```

---

## 💡 Key Insights

### 1. No Manual Downloads Needed
Everything happens automatically via HuggingFace libraries.

### 2. Configuration is King
The `config.yaml` file controls:
- Which model downloads
- Which data downloads
- How training happens

### 3. Caching Saves Time
First run downloads everything, subsequent runs are instant.

### 4. Fine-Tuning ≠ Training from Scratch
- **Training from scratch**: Days/weeks, massive compute
- **Fine-tuning**: Minutes/hours, moderate compute
- We do fine-tuning!

---

## ❓ Common Questions

### Q: Do I need to download models manually?
**A:** No! They download automatically from HuggingFace Hub.

### Q: Do I need to download datasets manually?
**A:** No! They download automatically from HuggingFace Datasets.

### Q: Do I need a HuggingFace account?
**A:** No! Public models and datasets are freely accessible.

### Q: Can I use my own data?
**A:** Yes! Modify `train.py` to load from CSV/JSON instead of HuggingFace.

### Q: Can I use custom models?
**A:** Yes! Upload to HuggingFace Hub or use local path in config.yaml.

### Q: What if I don't have GPU?
**A:** It works on CPU, just slower. Use smaller models (DistilBERT) and smaller datasets.

### Q: How much does it cost?
**A:** Free! Everything is open source. Only costs are:
- Your compute (electricity/cloud costs)
- Optional: paid HuggingFace Inference API

---

## 🎓 Learn More

- [HuggingFace Models](https://huggingface.co/models)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)

---
