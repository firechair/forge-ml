"""
Time Series Forecasting API Server

FastAPI server for serving time series forecasting predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from pathlib import Path
import time

from model import TimeSeriesForecaster

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="LSTM-based time series forecasting service",
    version="1.0.0",
)

# Load model
MODEL_PATH = Path("best_model")
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Train a model first with: python train.py"
    )

print(f"Loading model from {MODEL_PATH}...")
forecaster = TimeSeriesForecaster.load(MODEL_PATH)
model_info = forecaster.get_model_info()
print("Model loaded successfully!")
print(f"  Input size: {model_info['input_size']}")
print(f"  Sequence length: {model_info['sequence_length']}")
print(f"  Prediction length: {model_info['prediction_length']}")


# Request/Response models
class ForecastRequest(BaseModel):
    """Request model for single forecast."""

    sequence: List[List[float]] = Field(
        ...,
        description=(
            f"Input sequence of shape "
            f"({model_info['sequence_length']}, {model_info['input_size']})"
        ),
    )
    normalize: bool = Field(
        default=True, description="Whether to normalize input (recommended: True)"
    )


class ForecastResponse(BaseModel):
    """Response model for forecast."""

    predictions: List[List[float]]
    input_sequence: List[List[float]]
    sequence_length: int
    prediction_length: int
    prediction_time_ms: float


class BatchForecastRequest(BaseModel):
    """Request model for batch forecasting."""

    sequences: List[List[List[float]]] = Field(
        ..., description="List of input sequences", max_length=100
    )
    normalize: bool = Field(default=True)


class BatchForecastResponse(BaseModel):
    """Response model for batch forecasting."""

    predictions: List[ForecastResponse]
    total_prediction_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_info: dict


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Time Series Forecasting API",
        "model": model_info,
        "endpoints": {
            "/health": "Health check",
            "/forecast": "Single sequence forecast",
            "/forecast/batch": "Batch forecasting",
            "/docs": "API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True, "model_info": model_info}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Make a forecast for a single sequence.

    The input sequence should have shape (sequence_length, input_size).
    """
    try:
        # Validate input shape
        sequence = np.array(request.sequence)

        expected_shape = (forecaster.sequence_length, forecaster.input_size)
        if sequence.shape != expected_shape:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence shape. Expected {expected_shape}, got {sequence.shape}",
            )

        # Make prediction
        start_time = time.time()
        result = forecaster.predict(sequence, normalize=request.normalize)
        prediction_time = (time.time() - start_time) * 1000

        return {
            "predictions": result["predictions"],
            "input_sequence": result["input_sequence"],
            "sequence_length": forecaster.sequence_length,
            "prediction_length": forecaster.prediction_length,
            "prediction_time_ms": prediction_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast/batch", response_model=BatchForecastResponse)
async def forecast_batch(request: BatchForecastRequest):
    """
    Make forecasts for multiple sequences.

    Maximum 100 sequences per request.
    """
    if len(request.sequences) == 0:
        raise HTTPException(status_code=400, detail="Empty sequences list")

    if len(request.sequences) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Too many sequences. Maximum 100, got {len(request.sequences)}",
        )

    try:
        # Convert to numpy arrays
        sequences = [np.array(seq) for seq in request.sequences]

        # Validate shapes
        expected_shape = (forecaster.sequence_length, forecaster.input_size)
        for i, seq in enumerate(sequences):
            if seq.shape != expected_shape:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid shape for sequence {i}. "
                        f"Expected {expected_shape}, got {seq.shape}"
                    ),
                )

        # Make predictions
        start_time = time.time()
        results = forecaster.predict_batch(sequences, normalize=request.normalize)
        total_time = (time.time() - start_time) * 1000

        # Format responses
        predictions = []
        for result in results:
            predictions.append(
                {
                    "predictions": result["predictions"],
                    "input_sequence": result["input_sequence"],
                    "sequence_length": forecaster.sequence_length,
                    "prediction_length": forecaster.prediction_length,
                    "prediction_time_ms": total_time / len(sequences),
                }
            )

        return {"predictions": predictions, "total_prediction_time_ms": total_time}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
