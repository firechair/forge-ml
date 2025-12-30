"""Tests for TimeSeriesForecaster model."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import TimeSeriesForecaster, TimeSeriesLSTM


def test_model_initialization(config):
    """Test model initialization."""
    forecaster = TimeSeriesForecaster(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
    )

    assert forecaster.model is not None
    assert forecaster.input_size == config["model"]["input_size"]
    assert forecaster.sequence_length == config["forecasting"]["sequence_length"]


def test_prediction(config, sample_sequence):
    """Test single prediction."""
    forecaster = TimeSeriesForecaster(
        input_size=config["model"]["input_size"],
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
    )

    # Fit scaler
    forecaster.fit_scaler(sample_sequence)

    # Predict
    result = forecaster.predict(sample_sequence, normalize=False)

    assert "predictions" in result
    assert len(result["predictions"]) == config["forecasting"]["prediction_length"]


def test_save_load(config, sample_sequence, tmp_path):
    """Test model save and load."""
    forecaster1 = TimeSeriesForecaster(
        input_size=config["model"]["input_size"],
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
    )

    forecaster1.fit_scaler(sample_sequence)
    forecaster1.save(tmp_path / "test_model")

    forecaster2 = TimeSeriesForecaster.load(tmp_path / "test_model")

    assert forecaster2.input_size == forecaster1.input_size
    assert np.allclose(forecaster2.mean, forecaster1.mean)
