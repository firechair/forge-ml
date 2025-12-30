# Multi-Model Comparison Example

Compare different models (BERT, DistilBERT, RoBERTa) systematically using MLflow.

## Quick Start

```bash
# 1. Create project
mlfactory init sentiment --name model-comparison
cd model-comparison

# 2. Run comparison script
python ../examples/model-comparison/compare_models.py

# 3. View results in MLflow
open http://localhost:5000
```

## What This Does

Trains 3 models automatically:
1. **DistilBERT** - Fast, lightweight (66M params)
2. **BERT** - Balanced performance (110M params)
3. **RoBERTa** - Best accuracy (125M params)

All tracked in MLflow for easy comparison.

## Comparison Script

See `compare_models.py`:

```python
import subprocess
import yaml

models = [
    ("distilbert-base-uncased", "distilbert-baseline"),
    ("bert-base-uncased", "bert-baseline"),
    ("roberta-base", "roberta-baseline"),
]

for model_name, experiment in models:
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")

    # Update config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config['model']['name'] = model_name

    with open("config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Train
    subprocess.run(
        ["python", "train.py", "--experiment", experiment],
        check=True
    )

print("\n✓ All models trained! View results in MLflow UI")
```

## Analyzing Results

### In MLflow UI

1. Go to http://localhost:5000
2. Select the experiment
3. Click "Compare" button
4. Select all runs
5. View metrics side-by-side

### Programmatic Analysis

```python
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("sentiment-analysis")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"]
)

# Create comparison DataFrame
comparison = []
for run in runs:
    comparison.append({
        'model': run.data.params['model_name'],
        'accuracy': run.data.metrics['test_accuracy'],
        'train_time': run.info.end_time - run.info.start_time,
        'num_params': run.data.params['num_parameters']
    })

df = pd.DataFrame(comparison)
print(df.to_string())
```

## Best Practices

### 1. Use Same Data

Ensure all models train on identical data:
- Same random seed
- Same train/test split
- Same preprocessing

### 2. Fair Hyperparameters

Either:
- Use same hyperparameters for all models
- Or tune each model separately (more time)

### 3. Multiple Runs

Run each model 3-5 times with different seeds:

```python
for seed in [42, 123, 456]:
    config['training']['seed'] = seed
    # Train...
```

Then compare average performance.

### 4. Resource Tracking

Log compute resources used:

```python
import mlflow
import time
import psutil

start_time = time.time()
start_mem = psutil.virtual_memory().used

# Train model...

mlflow.log_metrics({
    'training_time_seconds': time.time() - start_time,
    'memory_used_mb': (psutil.virtual_memory().used - start_mem) / 1024**2
})
```

## Visualization

Create comparison plots:

```python
import matplotlib.pyplot as plt

models = ['DistilBERT', 'BERT', 'RoBERTa']
accuracies = [0.92, 0.94, 0.95]
train_times = [30, 60, 90]  # minutes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax1.bar(models, accuracies)
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Model Accuracy Comparison')

# Time comparison
ax2.bar(models, train_times)
ax2.set_ylabel('Training Time (minutes)')
ax2.set_title('Training Time Comparison')

plt.tight_layout()
mlflow.log_figure(fig, "model_comparison.png")
```

## Decision Matrix

| Model | Accuracy | Speed | Size | Use Case |
|-------|----------|-------|------|----------|
| DistilBERT | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 66M | Prototyping, real-time API |
| BERT | ⭐⭐⭐⭐ | ⭐⭐⭐ | 110M | Production, balanced |
| RoBERTa | ⭐⭐⭐⭐⭐ | ⭐⭐ | 125M | Maximum accuracy needed |

## Advanced: Hyperparameter Tuning

Use Optuna for automatic tuning:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    # Update config and train
    # ...

    return test_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best params: {study.best_params}")
```

## Example Output

```
Model Comparison Results:
═══════════════════════════════════════
Model         Accuracy  Params    Time
───────────────────────────────────────
RoBERTa        95.2%    125M      90min
BERT           94.1%    110M      60min
DistilBERT     92.5%     66M      30min
═══════════════════════════════════════

Winner: RoBERTa (if accuracy is priority)
Alternative: DistilBERT (if speed is priority)
```
