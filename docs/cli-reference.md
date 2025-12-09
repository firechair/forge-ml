# 📘 ForgeML CLI Reference

Complete reference for all ForgeML command-line commands.

---

## Installation

```bash
pip install forge-ml
# or
pip install -e .  # for development
```

---

## Global Options

All commands support these global options:

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version information |

---

## Commands

### `mlfactory init`

Create a new ML project from a template.

#### Syntax

```bash
mlfactory init <template> --name <project-name>
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `template` | Yes | Template name (`sentiment`, `image_classification`, etc.) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--name` | string | `my-project` | Name for the new project |

#### Examples

```bash
# Basic usage
mlfactory init sentiment --name customer-reviews

# With default name
mlfactory init sentiment

# Different template (when available)
mlfactory init image_classification --name cat-dog-classifier
```

#### Output

Creates a new directory with:
- Training scripts
- Model definitions
- Configuration files
- Documentation
- Dockerfile

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Template not found or directory already exists |

---

### `mlfactory train`

Train a model using the project configuration.

#### Syntax

```bash
mlfactory train [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | `config.yaml` | Path to configuration file |
| `--experiment` | string | `default` | MLflow experiment name |

#### Examples

```bash
# Basic usage (from project directory)
mlfactory train

# Custom config
mlfactory train --config custom-config.yaml

# Named experiment
mlfactory train --experiment "baseline-v1"

# Combined
mlfactory train --config prod.yaml --experiment "production-run"
```

#### Requirements

Must be run from a ForgeML project directory containing:
- `config.yaml` (or specified config file)
- `train.py`

#### What It Does

1. Loads configuration from YAML file
2. Sets up MLflow tracking
3. Loads dataset
4. Trains model
5. Logs metrics to MLflow
6. Saves best model

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Training completed successfully |
| 1 | Config or script not found |
| 130 | Training interrupted by user (Ctrl+C) |
| Other | Training script error |

---

### `mlfactory serve`

Serve a trained model via FastAPI.

#### Syntax

```bash
mlfactory serve [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-uri` | string | `best_model/` | MLflow model URI or local path |
| `--port` | integer | `8000` | Port to serve on |

#### Examples

```bash
# Basic usage (serves best_model/)
mlfactory serve

# Custom port
mlfactory serve --port 8080

# MLflow model URI (feature in development)
mlfactory serve --model-uri runs:/abc123def/model

# Combined
mlfactory serve --port 9000 --model-uri best_model/
```

#### Requirements

Must be run from a ForgeML project directory containing:
- `serve.py`
- `best_model/` directory (or specified model)

#### What It Does

1. Loads the trained model
2. Starts FastAPI server
3. Exposes prediction endpoints
4. Provides interactive API documentation

#### Endpoints Created

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

#### Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

# Open interactive docs
open http://localhost:8000/docs
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Server stopped cleanly |
| 1 | serve.py or model not found |
| Other | Server error |

---

## Workflow Examples

### Complete Development Workflow

```bash
# 1. Create project
mlfactory init sentiment --name review-analyzer
cd review-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start MLflow
cd ../infra && docker-compose up -d && cd ../review-analyzer

# 4. Train baseline model
mlfactory train --experiment "baseline"

# 5. Try different settings
# (edit config.yaml)
mlfactory train --experiment "high-lr"

# 6. Compare in MLflow
open http://localhost:5000

# 7. Serve best model
mlfactory serve

# 8. Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

### Experimentation Workflow

```bash
# Experiment 1: Baseline
mlfactory train --experiment "exp001-baseline"

# Experiment 2: Larger model
# (change model in config.yaml to roberta-base)
mlfactory train --experiment "exp002-roberta"

# Experiment 3: More epochs
# (change num_epochs to 5)
mlfactory train --experiment "exp003-longer"

# Compare all experiments
open http://localhost:5000
```

### Production Deployment Workflow

```bash
# 1. Train production model
mlfactory train --config prod-config.yaml --experiment "prod-v1"

# 2. Test locally
mlfactory serve --port 8000

# 3. Build Docker image
docker build -t sentiment-api:v1 .

# 4. Test container
docker run -p 8000:8000 sentiment-api:v1

# 5. Deploy to cloud
# (cloud-specific commands)
```

---

## Configuration Files

### config.yaml Structure

```yaml
model:
  name: "distilbert-base-uncased"
  num_labels: 2
  max_length: 512

training:
  batch_size: 16
  learning_rate: 0.00002
  num_epochs: 3
  seed: 42

data:
  dataset: "imdb"
  train_split: "train"
  test_split: "test"

mlflow:
  experiment_name: "sentiment-analysis"
  tracking_uri: "http://localhost:5000"

logging:
  log_interval: 100
  eval_interval: 500
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name | From config.yaml |

---

## Tips & Best Practices

### Naming Experiments

Use descriptive names that include:
- Model architecture
- Key hyperparameters
- Date/version

Examples:
- `distilbert-lr2e5-bs16-v1`
- `roberta-large-prod-2024-01`
- `baseline-experiment`

### Configuration Management

Keep multiple config files for different scenarios:
```
configs/
├── dev.yaml        # Quick local testing
├── experiment.yaml # Full experimentation
└── prod.yaml      # Production settings
```

### MLflow Organization

Use consistent experiment naming:
```bash
# Group by project phase
mlfactory train --experiment "dev/baseline"
mlfactory train --experiment "dev/tuning"
mlfactory train --experiment "prod/v1"
```

---

## Troubleshooting

### Command Not Found

```bash
# Reinstall package
pip install -e .

# Or use full path
python -m cli.main init sentiment --name test
```

### Permission Errors

```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Use different port
mlfactory serve --port 8001
```

---

## Advanced Usage

### Custom Training Scripts

You can modify `train.py` while keeping the CLI:

```python
# train.py
import argparse

def main(config_path, experiment_name):
    # Your custom training logic
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment", default="default")
    args = parser.parse_args()
    main(args.config, args.experiment)
```

The CLI will still work as expected.

---

## Future Commands (Roadmap)

These commands are planned for future releases:

```bash
# MLflow UI launcher
mlfactory mlflow-ui

# Model deployment
mlfactory deploy --target aws

# Data versioning
mlfactory dvc-init

# Template management
mlfactory template list
mlfactory template install community/bert-ner

# Project validation
mlfactory validate
```

---

## Getting Help

```bash
# General help
mlfactory --help

# Command-specific help
mlfactory init --help
mlfactory train --help
mlfactory serve --help
```

---

**See Also**:
- [Quick Start Guide](quickstart.md)
- [Project README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)
