# Team Collaboration Guide

How to use ForgeML in team environments with shared MLflow tracking and DVC data versioning.

## Quick Setup for Teams

```bash
# 1. Start shared infrastructure
cd forge-ml/infra
docker-compose -f docker-compose-team.yml up -d

# 2. Configure team members
export MLFLOW_TRACKING_URI=http://team-server:5000

# 3. Configure DVC remote
dvc remote add teamremote s3://team-bucket/dvc-storage
dvc remote default teamremote
```

## Shared MLflow Server

### Setup

The team docker-compose includes:
- MLflow with PostgreSQL backend (persistent, production-ready)
- MinIO for shared artifact storage
- Networked for team access

```yaml
# infra/docker-compose-team.yml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres/mlflow
      - MLFLOW_ARTIFACT_ROOT=s3://mlflow/artifacts
```

### Team Member Setup

Each team member adds to their environment:

```bash
# ~/.bashrc or ~/.zshrc
export MLFLOW_TRACKING_URI=http://mlflow-server.company.com:5000

# For MinIO (S3-compatible)
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://minio-server:9000
```

### Benefits

- **Shared experiments**: Everyone sees all training runs
- **Model comparison**: Compare models across team members
- **Model registry**: Centralized model versioning
- **Collaboration**: Review and iterate on experiments together

## DVC for Data Versioning

### Remote Storage Setup

#### Option 1: MinIO (S3-compatible, local)

```bash
# Configure DVC to use MinIO
dvc remote add teamremote s3://dvc-storage/data
dvc remote modify teamremote endpointurl http://minio-server:9000
dvc remote modify teamremote access_key_id minioadmin
dvc remote modify teamremote secret_access_key minioadmin
dvc remote default teamremote
```

#### Option 2: AWS S3

```bash
dvc remote add teamremote s3://company-ml-data/datasets
dvc remote default teamremote

# Team members configure AWS credentials
aws configure
```

#### Option 3: Google Cloud Storage

```bash
dvc remote add teamremote gs://company-ml-data/datasets
dvc remote default teamremote

# Authenticate
gcloud auth application-default login
```

### Team Workflow

**Data Manager** (creates/updates datasets):

```bash
# Add new dataset
dvc add data/customer_reviews_v2.csv
git add data/customer_reviews_v2.csv.dvc data/.gitignore
git commit -m "Add customer reviews v2"

# Push data to remote
dvc push

# Push metadata to git
git push
```

**Team Members** (use datasets):

```bash
# Pull latest code
git pull

# Download data
dvc pull

# Train with versioned data
python train.py
```

## Git Workflow for ML Teams

### Branch Strategy

```
main
├── develop
│   ├── feature/sentiment-bert
│   ├── feature/timeseries-lstm
│   └── experiment/hyperparameter-tuning
└── production
    └── deploy/model-v1.0
```

### Recommended Workflow

1. **Feature branches** for new capabilities
2. **Experiment branches** for ML experimentation
3. **Develop branch** for integration
4. **Production branch** for deployed models

### Example

```bash
# Create experiment branch
git checkout -b experiment/roberta-model

# Modify config
vim config.yaml  # Change model to RoBERTa

# Train and track in MLflow
python train.py --experiment "roberta-baseline"

# Review results in shared MLflow UI
open http://mlflow-server:5000

# If good, merge to develop
git checkout develop
git merge experiment/roberta-model
```

## Experiment Naming Convention

Use consistent naming for easy comparison:

```
<team-member>/<model>/<variant>
```

Examples:
- `alice/bert/baseline`
- `bob/distilbert/optimized`
- `charlie/roberta/large-lr`

In config or CLI:
```bash
python train.py --experiment "alice/bert/baseline"
```

## Code Review for ML

### What to Review

1. **Model changes**: Architecture, hyperparameters
2. **Data changes**: Dataset version, preprocessing
3. **Results**: MLflow experiment link, metrics achieved
4. **Reproducibility**: Seed set, config committed

### Pull Request Template

```markdown
## Experiment Summary
- **Model**: DistilBERT
- **Dataset**: IMDB v2
- **MLflow Run**: http://mlflow:5000/#/experiments/1/runs/abc123

## Results
- Validation Accuracy: 94.5%
- Test Accuracy: 93.2%
- Training Time: 45 minutes

## Changes
- Increased learning rate from 2e-5 to 5e-5
- Added warmup steps: 500
- Changed batch size: 16 → 32

## Reproducibility
- [x] Config committed
- [x] Seed set (42)
- [x] Data version tracked with DVC
- [x] Requirements.txt updated
```

## Conflict Resolution

### Data Conflicts

If two team members modify the same dataset:

```bash
# Situation: Both Alice and Bob modified data/train.csv

# Alice's approach (creates new version)
dvc add data/train_v3_alice.csv
dvc push
git add data/train_v3_alice.csv.dvc
git commit -m "Alice's data modifications"

# Bob's approach (creates new version)
dvc add data/train_v3_bob.csv
dvc push
git add data/train_v3_bob.csv.dvc
git commit -m "Bob's data modifications"

# Team decides which to use or merges both
```

### Model Conflicts

Use MLflow to compare:

```python
# Compare two models
import mlflow

client = mlflow.tracking.MlflowClient()

# Get Alice's run
alice_run = client.get_run("run_id_alice")
alice_accuracy = alice_run.data.metrics["test_accuracy"]

# Get Bob's run
bob_run = client.get_run("run_id_bob")
bob_accuracy = bob_run.data.metrics["test_accuracy"]

# Choose better model
if alice_accuracy > bob_accuracy:
    mlflow.register_model(f"runs:/run_id_alice/model", "production-model")
```

## Best Practices

### Communication

1. **Announce experiments**: Post in team chat before major experiments
2. **Share interesting findings**: Link to MLflow runs
3. **Document decisions**: Why certain hyperparameters were chosen
4. **Weekly sync**: Review best experiments, plan next steps

### Organization

1. **Naming conventions**: Consistent experiment/model names
2. **Tagging**: Use MLflow tags for organization
   ```python
   mlflow.set_tag("team_member", "alice")
   mlflow.set_tag("priority", "high")
   mlflow.set_tag("model_type", "transformer")
   ```
3. **Documentation**: Update README with team findings

### Resource Management

1. **GPU scheduling**: Coordinate to avoid conflicts
2. **Storage limits**: Clean up old experiments periodically
3. **Cost tracking**: Monitor cloud costs if using AWS/GCP

## Troubleshooting

**Q: Can't connect to shared MLflow server**
A: Check `MLFLOW_TRACKING_URI` environment variable and network access

**Q: DVC push fails**
A: Verify remote credentials and network connectivity

**Q: Experiments not showing up**
A: Ensure same `experiment_name` in config.yaml across team

**Q: Data version mismatch**
A: Run `dvc pull` to get latest data versions

## Example Team Setup

**Project Structure:**
```
company-ml-project/
├── .dvc/
│   └── config (team remote configured)
├── data/
│   ├── train.csv.dvc  (tracked by DVC)
│   └── test.csv.dvc
├── experiments/
│   ├── alice/
│   ├── bob/
│   └── charlie/
├── models/
│   └── production/  (registered in MLflow)
└── docs/
    └── team-experiments.md
```

**Environment File (`.env.team`):**
```bash
# Shared by all team members
MLFLOW_TRACKING_URI=http://10.0.1.50:5000
MLFLOW_S3_ENDPOINT_URL=http://10.0.1.50:9000
AWS_ACCESS_KEY_ID=teamuser
AWS_SECRET_ACCESS_KEY=teampassword
```

Load with: `source .env.team`

---

**Questions?** Open an issue or check the main [README](../README.md)
