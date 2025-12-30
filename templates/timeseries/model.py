"""
Time Series Forecasting Model

LSTM-based model for univariate and multivariate time series forecasting.
Supports rolling window predictions and batch inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        prediction_length: int = 24,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.prediction_length = prediction_length

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, prediction_length * input_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, prediction_length, input_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Predict future values
        predictions = self.fc(last_hidden)

        # Reshape to (batch_size, prediction_length, input_size)
        batch_size = x.size(0)
        predictions = predictions.view(batch_size, self.prediction_length, self.input_size)

        return predictions


class TimeSeriesForecaster:
    """Wrapper class for time series forecasting with LSTM."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 96,
        prediction_length: int = 24,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            prediction_length=prediction_length,
        ).to(self.device)

        # Normalization parameters (fitted during training)
        self.mean = None
        self.std = None

    def get_model_info(self) -> Dict:
        """Get model information."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "prediction_length": self.prediction_length,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }

    def fit_scaler(self, data: np.ndarray):
        """Fit normalization parameters."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8  # Avoid division by zero

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted.")
        return data * self.std + self.mean

    def predict(self, sequence: np.ndarray, normalize: bool = True) -> Dict:
        """
        Make a single prediction.

        Args:
            sequence: Input sequence of shape (sequence_length, input_size)
            normalize: Whether to normalize the input

        Returns:
            Dictionary with predictions and metadata
        """
        self.model.eval()

        # Prepare input
        if normalize and self.mean is not None:
            sequence_normalized = self.normalize(sequence)
        else:
            sequence_normalized = sequence

        # Convert to tensor
        x = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(x)

        # Convert to numpy
        predictions_np = predictions.cpu().numpy()[0]

        # Denormalize
        if normalize and self.mean is not None:
            predictions_np = self.denormalize(predictions_np)

        return {
            "predictions": predictions_np.tolist(),
            "input_sequence": sequence.tolist(),
        }

    def predict_batch(self, sequences: List[np.ndarray], normalize: bool = True) -> List[Dict]:
        """Make batch predictions."""
        results = []
        for seq in sequences:
            result = self.predict(seq, normalize=normalize)
            results.append(result)
        return results

    def save(self, path: Path):
        """Save model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), path / "model.pt")

        # Save config
        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "sequence_length": self.sequence_length,
            "prediction_length": self.prediction_length,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TimeSeriesForecaster":
        """Load model from path."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create instance
        forecaster = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            sequence_length=config["sequence_length"],
            prediction_length=config["prediction_length"],
        )

        # Load weights
        forecaster.model.load_state_dict(
            torch.load(path / "model.pt", map_location=forecaster.device)
        )

        # Load normalization parameters
        if config["mean"] is not None:
            forecaster.mean = np.array(config["mean"])
            forecaster.std = np.array(config["std"])

        return forecaster
