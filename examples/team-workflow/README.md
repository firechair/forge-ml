# Team Workflow Example

Demonstrates how a team of 3 ML engineers collaborates on ForgeML.

## Scenario

Team members Alice, Bob, and Charlie work on improving a sentiment analysis model.

## Setup

```bash
# 1. Start shared infrastructure
cd forge-ml/infra
docker-compose -f docker-compose-team.yml up -d

# 2. Each team member clones repo
git clone https://github.com/company/ml-project.git
cd ml-project

# 3. Set team environment
source .env.team

# 4. Pull shared data
dvc pull
```

## Workflow Example

### Alice: Baseline Model

```bash
# Create feature branch
git checkout -b alice/baseline-bert

# Create project
mlfactory init sentiment --name sentiment-baseline
cd sentiment-baseline

# Train baseline
python train.py --experiment "alice/bert-baseline"

# View in shared MLflow
# http://mlflow-server:5000

# Results: 92% accuracy

# Commit and push
git add .
git commit -m "Add BERT baseline - 92% accuracy"
git push origin alice/baseline-bert
```

### Bob: Try Different Model

```bash
# Create his branch
git checkout -b bob/roberta-experiment

# Modify config
vim config.yaml  # Change to RoBERTa

# Train
python train.py --experiment "bob/roberta-large"

# Results: 94% accuracy (better!)

# Commit
git add config.yaml
git commit -m "Try RoBERTa - 94% accuracy"
git push origin bob/roberta-experiment
```

### Charlie: Data Augmentation

```bash
# Create branch
git checkout -b charlie/data-augmentation

# Add new augmented dataset
python scripts/augment_data.py
dvc add data/train_augmented.csv

# Push data to DVC remote
dvc push

# Commit DVC file
git add data/train_augmented.csv.dvc
git commit -m "Add augmented training data"
git push

# Train on augmented data
python train.py --experiment "charlie/augmented-data"

# Results: 95% accuracy (best!)
```

## Team Review Meeting

View all experiments in MLflow:

```python
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.test_accuracy DESC"]
)

results = []
for run in runs:
    results.append({
        'team_member': run.data.tags.get('mlflow.user', 'unknown'),
        'experiment': run.data.tags.get('mlflow.runName', run.info.run_id[:8]),
        'accuracy': run.data.metrics.get('test_accuracy', 0),
        'model': run.data.params.get('model_name', 'unknown')
    })

df = pd.DataFrame(results)
print("\nTeam Experiment Results:")
print(df.to_string(index=False))
```

Output:
```
Team Experiment Results:
team_member  experiment                 accuracy  model
alice        bert-baseline               92.0%    bert-base-uncased
bob          roberta-large               94.0%    roberta-base
charlie      augmented-data              95.0%    bert-base-uncased
```

**Decision:** Merge Charlie's approach (best results)

## Merge Process

### 1. Charlie Creates Pull Request

```markdown
## PR: Data Augmentation Improves Accuracy to 95%

**Experiment**: http://mlflow:5000/#/experiments/1/runs/abc123

**Changes:**
- Added data augmentation script
- Created augmented dataset (tracked with DVC)
- Trained BERT on augmented data

**Results:**
- Test Accuracy: 95.0% (+3% over baseline)
- No change to model architecture
- Training time: +10 minutes (acceptable)

**Reproducibility:**
- [x] Dataset tracked with DVC
- [x] Random seed fixed (42)
- [x] Config committed
```

### 2. Team Reviews

Alice and Bob review:
- Check MLflow experiment
- Verify DVC data version
- Review augmentation script
- Approve PR

### 3. Merge to Main

```bash
git checkout main
git merge charlie/data-augmentation
git push origin main
```

### 4. All Team Members Update

```bash
# Pull latest code
git pull origin main

# Pull augmented data
dvc pull

# Now everyone has best model + data
```

## Conflict Resolution Example

**Scenario:** Alice and Bob both modified config.yaml

```bash
# Alice's change
model:
  name: "bert-base-uncased"
  learning_rate: 5e-5  # Alice increased LR

# Bob's change
model:
  name: "bert-base-uncased"
  batch_size: 32  # Bob increased batch size
```

**Resolution:**

```bash
# During merge, Git shows conflict
git merge bob/experiment
# CONFLICT in config.yaml

# Manually combine both changes
model:
  name: "bert-base-uncased"
  learning_rate: 5e-5     # Alice's change
  batch_size: 32          # Bob's change

# Test combined changes
python train.py --experiment "combined-improvements"

# If good, commit merge
git add config.yaml
git commit -m "Merge: combine LR and batch size improvements"
```

## Best Practices Used

### 1. Naming Convention
- `<name>/<description>` for experiments
- Clear experiment names in MLflow

### 2. Branch Strategy
- Feature branches for experiments
- Main branch for best models
- No direct commits to main

### 3. Code Review
- Every experiment gets reviewed
- MLflow link in PR
- Reproducibility checklist

### 4. Data Versioning
- All datasets tracked with DVC
- Data changes documented
- Shared via DVC remote

### 5. Communication
- Daily standups: share experiment results
- Slack: post interesting MLflow runs
- Wiki: document what worked/didn't work

## Tools Summary

| Tool | Purpose | Command |
|------|---------|---------|
| Git | Code versioning | `git commit/push/pull` |
| DVC | Data versioning | `dvc add/push/pull` |
| MLflow | Experiment tracking | View UI, compare runs |
| Docker | Shared infrastructure | `docker-compose up` |

## Timeline Example

**Week 1:**
- Day 1: Alice creates baseline (92%)
- Day 2-3: Bob experiments with models (94%)
- Day 4-5: Charlie tries data augmentation (95%)

**Week 2:**
- Day 1: Team review, decide on Charlie's approach
- Day 2: Merge to main
- Day 3: Everyone updates, continues improving

## Metrics to Track

```python
import mlflow

# Tag all runs with team info
mlflow.set_tag("team_member", "alice")
mlflow.set_tag("team_sprint", "sprint-1")
mlflow.set_tag("experiment_type", "baseline")

# Log team-level metrics
mlflow.log_metric("team_best_accuracy", 95.0)
mlflow.log_metric("improvement_over_baseline", 3.0)
```

## Success Indicators

- ✅ All team members can see each other's experiments
- ✅ No duplicate work (MLflow shows what's been tried)
- ✅ Data is shared via DVC (no manual copying)
- ✅ Best model is in production (merged to main)
- ✅ Knowledge is documented (PRs, wiki, MLflow)

---

See [team-collaboration.md](../../docs/team-collaboration.md) for full team setup guide.
