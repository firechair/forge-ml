# Custom Data Sentiment Analysis Example

How to use ForgeML with your own CSV data instead of HuggingFace datasets.

## Quick Start

```bash
# 1. Create project
mlfactory init sentiment --name custom-sentiment
cd custom-sentiment

# 2. Prepare your data (see below)
# 3. Modify train.py to use custom data (see modifications)
# 4. Train
python train.py
```

## Data Format

Your CSV should have two columns:

```csv
text,label
"This product is amazing!",1
"Terrible experience, very disappointed.",0
"Pretty good overall.",1
"Waste of money.",0
```

- **text**: The text to classify
- **label**: 0 (negative) or 1 (positive)

## Example Dataset

See `sample_reviews.csv` in this directory for a complete example.

## Modifications to train.py

Replace the `prepare_data()` function:

```python
def prepare_data(config: dict):
    """Load custom CSV data instead of HuggingFace dataset."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load your CSV
    df = pd.read_csv("path/to/your_data.csv")

    # Validate columns
    assert 'text' in df.columns, "CSV must have 'text' column"
    assert 'label' in df.columns, "CSV must have 'label' column"

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config['training']['seed'],
        stratify=df['label']  # Balanced split
    )

    # Convert to HuggingFace dataset format
    from datasets import Dataset

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    print(f"Loaded {len(train_dataset)} train, {len(test_dataset)} test samples")

    return train_dataset, test_dataset
```

## Common Data Issues

### Imbalanced Classes

If you have more negatives than positives (or vice versa):

```python
# Add class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)

# Use in training (modify loss function)
```

### Multi-class Classification

For more than 2 classes (0, 1, 2, ...):

```python
# Update config.yaml
model:
  num_labels: 3  # Instead of 2

# Ensure labels are 0, 1, 2, ... (consecutive integers)
df['label'] = pd.Categorical(df['category']).codes
```

### Non-English Text

```python
# Update config.yaml with multilingual model
model:
  name: "bert-base-multilingual-cased"

# Or language-specific models:
# Spanish: "dccuchile/bert-base-spanish-wwm-cased"
# French: "camembert-base"
# German: "bert-base-german-cased"
```

## Data Preprocessing Tips

### Clean Text

```python
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters (optional)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

df['text'] = df['text'].apply(clean_text)
```

### Handle Missing Values

```python
# Remove rows with missing text
df = df.dropna(subset=['text', 'label'])

# Or fill with empty string
df['text'] = df['text'].fillna('')
```

### Data Augmentation

```python
# Simple augmentation: add synonyms, paraphrase, etc.
# Or use libraries like nlpaug

import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')

# Augment minority class
positive_samples = df[df['label'] == 1]
augmented = positive_samples.copy()
augmented['text'] = augmented['text'].apply(aug.augment)

df = pd.concat([df, augmented])
```

## Track Data with DVC

```bash
# Add dataset to DVC
dvc add data/your_data.csv
git add data/your_data.csv.dvc data/.gitignore
git commit -m "Add custom dataset v1"

# Push to remote
dvc push
```

## Full Example Script

See `train_custom.py` for a complete working example.

## Troubleshooting

**Q: "CSV file not found"**
A: Use absolute path or ensure you're in the correct directory

**Q: "Text column has NaN values"**
A: Clean data first: `df.dropna(subset=['text'])`

**Q: "Poor accuracy"**
A: Try:
- Larger model (bert-base instead of distilbert)
- More training data (aim for 1000+ samples per class)
- Longer training (more epochs)
- Different preprocessing
