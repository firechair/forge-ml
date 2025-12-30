"""
FastAPI serving application for sentiment analysis model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from model import SentimentClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification using transformer models",
    version="1.0.0",
)

# Global model variable
model: Optional[SentimentClassifier] = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    texts: List[str] = Field(
        ..., description="List of texts to classify", min_items=1, max_items=100
    )

    @validator("texts")
    def texts_not_empty(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("All texts cannot be empty")
        if len(cleaned) > 100:
            raise ValueError("Maximum 100 texts allowed per batch")
        return cleaned


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    text: str
    sentiment: str  # "positive" or "negative"
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model

    # Try to load from best_model directory first
    model_path = "best_model"
    if Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        try:
            model = SentimentClassifier.from_pretrained(model_path)
            logger.info("Model loaded successfully")

            # Log model info
            info = model.get_model_info()
            logger.info(f"Model info: {info}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            model = None
    else:
        logger.warning(
            f"Model path {model_path} not found. API will not be able to make predictions."
        )
        model = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and model information.
    """
    model_loaded = model is not None
    model_info = None

    if model_loaded:
        model_info = model.get_model_info()

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for a single text.

    Args:
        request: PredictionRequest containing text

    Returns:
        PredictionResponse with sentiment and confidence
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is available.",
        )

    try:
        # Get prediction
        result = model.predict(request.text, return_probabilities=True)

        # Determine sentiment
        sentiment = "positive" if result["predicted_label"] == 1 else "negative"
        confidence = result[sentiment]

        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": result["negative"],
                "positive": result["positive"],
            },
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.

    Args:
        request: BatchPredictionRequest containing list of texts

    Returns:
        BatchPredictionResponse with predictions for all texts
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is available.",
        )

    try:
        # Get predictions
        results = model.predict(request.texts, return_probabilities=True)

        # Format responses
        predictions = []
        for text, result in zip(request.texts, results):
            sentiment = "positive" if result["predicted_label"] == 1 else "negative"
            confidence = result[sentiment]

            predictions.append(
                PredictionResponse(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    probabilities={
                        "negative": result["negative"],
                        "positive": result["positive"],
                    },
                )
            )

        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        Model information including name, parameters, etc.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model.get_model_info()


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
