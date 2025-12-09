# DVC Configuration for Sentiment Analysis Template

## Overview

This template is configured to use DVC (Data Version Control) for tracking datasets and model artifacts.

## Setup

1. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   ```

2. **Configure Remote Storage** (optional):
   
   For S3:
   ```bash
   dvc remote add -d myremote s3://my-bucket/dvc-storage
   ```
   
   For Google Cloud Storage:
   ```bash
   dvc remote add -d myremote gs://my-bucket/dvc-storage
   ```
   
   For local storage:
   ```bash
   dvc remote add -d myremote /path/to/storage
   ```

## Usage

### Track a Dataset

```bash
# Add dataset to DVC
dvc add data/my_dataset.csv

# Commit the .dvc file to git
git add data/my_dataset.csv.dvc data/.gitignore
git commit -m "Add dataset"

# Push data to remote storage
dvc push
```

### Retrieve a Dataset

```bash
# Pull data from remote storage
dvc pull
```

### Version Control Workflow

```bash
# 1. Modify your dataset
# (edit data/my_dataset.csv)

# 2. Update DVC tracking
dvc add data/my_dataset.csv

# 3. Commit changes
git add data/my_dataset.csv.dvc
git commit -m "Update dataset with new samples"

# 4. Push to remote
dvc push
```

## Best Practices

1. **Never commit large files to git** - Use DVC instead
2. **Always push to DVC remote** - Ensures data is backed up
3. **Tag important versions** - `git tag v1.0-data`
4. **Document data changes** - Use clear commit messages

## Files

- `.dvc/` - DVC configuration directory
- `.dvcignore` - Files to ignore (like .gitignore)
- `*.dvc` - DVC tracking files (commit to git)
- `.gitignore` - Updated to ignore DVC cached data
