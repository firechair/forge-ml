# Time Series Forecasting Project

LSTM-based time series forecasting with MLflow tracking and FastAPI serving.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python train.py

# 3. Start API server
python serve.py

# 4. Make predictions
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[1.0], [2.0], ..., [96.0]]}'
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "LSTM"          # Model architecture
  hidden_size: 128      # LSTM hidden size
  num_layers: 2         # Number of LSTM layers

forecasting:
  sequence_length: 96   # Input window (past timesteps)
  prediction_length: 24 # Output window (future timesteps)

data:
  dataset: "ETTh1"      # ETTh1, ETTm1, weather, or CSV path
  target_column: "OT"   # Column to forecast
```

## Supported Datasets

- **ETTh1/ETTm1**: Electricity Transformer Temperature (auto-downloaded)
- **Custom CSV**: Set `dataset` to your CSV file path

## Project Structure

```
timeseries-project/
├── config.yaml         # Configuration
├── train.py           # Training script
├── model.py           # LSTM model
├── serve.py           # FastAPI server
├── requirements.txt   # Dependencies
├── Dockerfile        # Docker config
├── best_model/       # Trained model (after training)
└── tests/            # Unit tests
```

## API Endpoints

- `GET /health` - Health check
- `POST /forecast` - Single sequence forecast
- `POST /forecast/batch` - Batch forecasting
- `GET /docs` - Interactive API documentation

## Training

```bash
# Default training
python train.py

# Custom config
python train.py --config my_config.yaml

# Custom experiment name
python train.py --experiment my-experiment
```

Training automatically:
- Tracks metrics in MLflow
- Saves best model to `best_model/`
- Uses early stopping
- Normalizes data

## Model Serving

```bash
# Start server
python serve.py

# Or with mlfactory CLI
mlfactory serve

# Docker
docker build -t ts-forecaster .
docker run -p 8000:8000 ts-forecaster
```

## Testing

```bash
pytest tests/
```

## Example Usage

```python
from model import TimeSeriesForecaster
import numpy as np

# Load model
forecaster = TimeSeriesForecaster.load("best_model")

# Prepare sequence (shape: [sequence_length, input_size])
sequence = np.random.randn(96, 1)

# Predict
result = forecaster.predict(sequence)
print(f"Predictions: {result['predictions']}")
```

## Common Issues

**Q: Training is slow**
A: Install PyTorch with CUDA for GPU acceleration

**Q: Dataset not found**
A: ETT datasets auto-download. For custom data, provide full CSV path

**Q: Prediction shape mismatch**
A: Ensure input sequence shape matches (sequence_length, input_size) from config

For more help, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
