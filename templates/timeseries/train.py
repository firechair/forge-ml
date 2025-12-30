"""
Time Series Forecasting Training Script

Trains an LSTM model for time series forecasting with MLflow tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import mlflow
import argparse
from typing import Tuple, List

from model import TimeSeriesForecaster


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_length: int,
        stride: int = 1,
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride

        # Create sequences
        self.sequences = []
        self.targets = []

        for i in range(0, len(data) - sequence_length - prediction_length + 1, stride):
            seq = data[i : i + sequence_length]
            target = data[i + sequence_length : i + sequence_length + prediction_length]
            self.sequences.append(seq)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
        )


def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split time series data.

    Returns:
        train_data, val_data, test_data as numpy arrays
    """
    dataset_name = config["data"]["dataset"]

    print(f"Loading dataset: {dataset_name}")

    # Check if it's a CSV file path
    if dataset_name.endswith(".csv"):
        df = pd.read_csv(dataset_name)
    else:
        # Load from HuggingFace or predefined datasets
        if dataset_name == "ETTh1":
            # Download ETT dataset (Electricity Transformer Temperature)
            url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
            df = pd.read_csv(url)
        elif dataset_name == "ETTm1":
            url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
            df = pd.read_csv(url)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Extract target column
    target_col = config["data"]["target_column"]
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {df.columns.tolist()}"
        )

    # Get values
    data = df[target_col].values.reshape(-1, 1).astype(np.float32)

    # Split data
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]

    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(
        f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )

    return train_data, val_data, test_data


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: dict,
    forecaster: TimeSeriesForecaster,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
        stride=config["forecasting"]["stride"],
    )

    val_dataset = TimeSeriesDataset(
        val_data,
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
        stride=config["forecasting"]["prediction_length"],  # No overlap in validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for sequences, targets in tqdm(train_loader, desc="Training"):
        sequences = sequences.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)

        # Calculate loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)

            loss = criterion(predictions, targets)
            mae = torch.mean(torch.abs(predictions - targets))

            total_loss += loss.item()
            total_mae += mae.item()

    return total_loss / len(val_loader), total_mae / len(val_loader)


def main(config_path: str = "config.yaml", experiment_name: str = None):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_data, val_data, test_data = load_data(config)

    # Initialize model
    print(f"Initializing {config['model']['name']} model...")
    forecaster = TimeSeriesForecaster(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        sequence_length=config["forecasting"]["sequence_length"],
        prediction_length=config["forecasting"]["prediction_length"],
    )

    # Fit scaler on training data
    forecaster.fit_scaler(train_data)

    # Normalize data
    train_data_norm = forecaster.normalize(train_data)
    val_data_norm = forecaster.normalize(val_data)
    test_data_norm = forecaster.normalize(test_data)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data_norm, val_data_norm, config, forecaster
    )

    # Setup training
    device = forecaster.device
    model = forecaster.model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # MLflow tracking
    mlflow_config = config["mlflow"]
    if mlflow_config.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    exp_name = experiment_name or mlflow_config["experiment_name"]
    mlflow.set_experiment(exp_name)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_config.get("run_name")):

        # Log parameters
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "hidden_size": config["model"]["hidden_size"],
                "num_layers": config["model"]["num_layers"],
                "sequence_length": config["forecasting"]["sequence_length"],
                "prediction_length": config["forecasting"]["prediction_length"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "num_epochs": config["training"]["num_epochs"],
                "seed": config["training"]["seed"],
                "dataset": config["data"]["dataset"],
            }
        )

        # Log DVC/git info (same as sentiment template)
        try:
            import subprocess

            dvc_remote = subprocess.run(
                ["dvc", "remote", "list"], capture_output=True, text=True, timeout=5
            )
            if dvc_remote.returncode == 0 and dvc_remote.stdout.strip():
                mlflow.set_tag("dvc.remote_configured", "true")

            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if git_commit.returncode == 0:
                mlflow.set_tag("git.commit", git_commit.stdout.strip()[:8])
        except:
            pass

        # Log model info
        model_info = forecaster.get_model_info()
        mlflow.log_params(
            {
                "num_parameters": model_info["num_parameters"],
                "device": model_info["device"],
            }
        )

        print(f"\nModel Info:")
        print(f"  Parameters: {model_info['num_parameters']:,}")
        print(f"  Device: {model_info['device']}")
        print(f"  Sequence Length: {model_info['sequence_length']}")
        print(f"  Prediction Length: {model_info['prediction_length']}")

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        patience = config["training"]["early_stopping_patience"]

        print(f"\nStarting training for {config['training']['num_epochs']} epochs...")

        for epoch in range(config["training"]["num_epochs"]):
            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                config["training"]["grad_clip"],
            )

            # Validate
            val_loss, val_mae = evaluate(model, val_loader, criterion, device)

            # Log metrics
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae},
                step=epoch,
            )

            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save model
                save_path = Path("best_model")
                forecaster.save(save_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_dataset = TimeSeriesDataset(
            test_data_norm,
            config["forecasting"]["sequence_length"],
            config["forecasting"]["prediction_length"],
            stride=config["forecasting"]["prediction_length"],
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["training"]["batch_size"]
        )

        test_loss, test_mae = evaluate(model, test_loader, criterion, device)

        mlflow.log_metrics({"test_loss": test_loss, "test_mae": test_mae})

        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        print(f"\n✓ Training complete! Model saved to best_model/")
        print(f"  View results: MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--experiment", default=None, help="Experiment name")
    args = parser.parse_args()

    main(args.config, args.experiment)
