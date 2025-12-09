# Frequently Asked Questions

## General

**Q: What is ForgeML?**
A: A CLI tool that creates production-ready ML project templates with MLflow, DVC, Docker, and FastAPI pre-configured.

**Q: Is it free?**
A: Yes for personal, educational, and research use. See the [LICENSE](../LICENSE) for details. Commercial use requires separate licensing.

**Q: What ML frameworks does it support?**
A: Currently PyTorch with HuggingFace Transformers. TensorFlow support planned.

## Installation

**Q: Which Python version do I need?**
A: Python 3.10 or higher.

**Q: Can I use it without Docker?**
A: Yes! Docker is optional (only needed for MLflow tracking server).

**Q: Does it work on Windows?**
A: Yes, both native Windows and WSL2 are supported.

## Templates

**Q: What templates are available?**
A: Currently:
- Sentiment analysis (text classification)
- Time series forecasting (LSTM)

More templates coming soon (image classification, NER, etc.)

**Q: Can I create my own template?**
A: Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for template creation guide.

**Q: Can I use my own data?**
A: Absolutely! See [examples/custom-data-sentiment](../examples/custom-data-sentiment/) for how to use CSV data.

## Training

**Q: Training is very slow, how to speed up?**
A:
1. Use GPU (install CUDA-enabled PyTorch)
2. Reduce batch size if out of memory
3. Use smaller model (e.g., DistilBERT instead of BERT)
4. Reduce dataset size for testing

**Q: How do I use GPU?**
A:
```bash
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: Out of memory error?**
A: Reduce batch_size in config.yaml (try 8 or 16 instead of 32)

**Q: How to resume training?**
A: Training automatically saves best model. For checkpointing during training, modify train.py to save/load checkpoints.

## MLflow

**Q: Do I need MLflow server running?**
A: No, it works locally without server (saves to `mlruns/` directory).

**Q: How to access MLflow UI?**
A:
```bash
# Start server
docker-compose -f infra/docker-compose.yml up -d

# Or locally
mlflow ui

# Then open http://localhost:5000
```

**Q: Can't connect to MLflow server?**
A: Check Docker is running: `docker ps`

**Q: How to delete experiments?**
A: Through MLflow UI or:
```bash
mlflow experiments delete --experiment-id 1
```

## Model Serving

**Q: serve.py fails with ImportError?**
A: Install serving dependencies:
```bash
pip install fastapi uvicorn pydantic
```
(Should be in requirements.txt - if missing, add them)

**Q: How to serve on different port?**
A:
```bash
python serve.py  # Then edit to change port
# Or
mlfactory serve --port 8080
```

**Q: Can I deploy to production?**
A: Yes! Build Docker image:
```bash
docker build -t my-model .
docker run -p 8000:8000 my-model
```

**Q: How to add authentication?**
A: Add API key authentication to serve.py:
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

## Data Versioning (DVC)

**Q: Is DVC required?**
A: No, it's optional but recommended for team collaboration.

**Q: How to set up DVC remote?**
A:
```bash
# S3
dvc remote add myremote s3://my-bucket/dvc-storage

# Local
dvc remote add myremote /tmp/dvc-storage

# Set as default
dvc remote default myremote
```

**Q: DVC push fails?**
A: Check remote is configured and credentials are set:
```bash
dvc remote list
dvc remote modify myremote access_key_id YOUR_KEY
```

## Team Collaboration

**Q: How do teams share experiments?**
A: Use shared MLflow server (see [team-collaboration.md](team-collaboration.md))

**Q: How to avoid conflicts?**
A:
- Use feature branches
- Unique experiment names
- DVC for data
- Regular team syncs

**Q: Can multiple people train simultaneously?**
A: Yes! MLflow handles concurrent runs automatically.

## Errors & Troubleshooting

**Q: "Template not found"?**
A: Check available templates:
```bash
ls templates/
# Should show: sentiment, timeseries
```

**Q: "Config file not found"?**
A: You're not in a project directory. Run:
```bash
mlfactory init sentiment --name myproject
cd myproject
```

**Q: Dependency conflicts?**
A: Use fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

**Q: Tests fail?**
A: Ensure you've trained a model first:
```bash
python train.py
# Then run tests
pytest tests/
```

**Q: Import errors when running train.py?**
A: Make sure all project dependencies are installed:
```bash
pip install -r requirements.txt
```

**Q: CUDA out of memory error?**
A: Reduce batch size in your project's `config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16 or 32
```

**Q: Can't connect to MLflow server?**
A: Check if MLflow is running:
```bash
# Check Docker containers
docker ps

# Or verify MLflow is accessible
curl http://localhost:5000

# Alternative: Use local file tracking
# Edit config.yaml:
mlflow:
  tracking_uri: "file:./mlruns"
```

**Q: Port already in use (8000)?**
A: Use a different port:
```bash
mlfactory serve --port 8001
```

**Q: SSL certificate errors during pip install?**
A: Use trusted hosts flag:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Q: Permission denied errors on Linux/Mac?**
A: Don't use sudo with pip. Use virtual environments instead:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Performance

**Q: How to improve model accuracy?**
A:
1. More training data
2. Larger model (BERT → RoBERTa)
3. More epochs
4. Hyperparameter tuning
5. Data augmentation

**Q: How to reduce model size?**
A:
1. Use DistilBERT (66M params)
2. Model quantization
3. Pruning
4. Knowledge distillation

**Q: How fast is inference?**
A:
- DistilBERT: ~50ms/sample (CPU)
- BERT: ~100ms/sample (CPU)
- With GPU: 5-10x faster

## Advanced

**Q: Can I use models not from HuggingFace?**
A: Yes, modify model.py to load custom models.

**Q: How to add new metrics?**
A: Edit train.py and add to MLflow logging:
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
mlflow.log_metric("f1_score", f1)
```

**Q: How to use multiple GPUs?**
A: Use PyTorch DataParallel or DistributedDataParallel (requires code modifications).

**Q: Can I fine-tune GPT-3/GPT-4?**
A: No, use OpenAI's API. ForgeML is for open-source models you can run locally.

## Support

**Q: Where to get help?**
A:
- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Documentation: Check docs/ folder

**Q: How to contribute?**
A: See [CONTRIBUTING.md](../CONTRIBUTING.md)

**Q: Found a bug?**
A: Open an issue with:
- ForgeML version
- Python version
- Error message
- Steps to reproduce
