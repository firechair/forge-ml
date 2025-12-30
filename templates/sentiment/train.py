"""
Training script for sentiment analysis model with MLflow tracking.
"""

import yaml
import argparse
from typing import Dict, Any
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import SentimentClassifier


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict[str, Any]) -> tuple:
    """
    Load and prepare dataset.

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    data_config = config["data"]

    # Load dataset from HuggingFace
    print(f"Loading dataset: {data_config['dataset']}")
    dataset = load_dataset(data_config["dataset"])

    # Get splits (with optional subset for quick testing)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Apply split configuration (e.g., "train[:5000]")
    if data_config.get("train_split"):
        split_str = data_config["train_split"]
        if "[" in split_str and "]" in split_str:
            # Extract slice like "train[:5000]"
            slice_part = split_str.split("[")[1].split("]")[0]
            if ":" in slice_part:
                parts = slice_part.split(":")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else len(train_data)
                train_data = train_data.select(range(start, end))

    if data_config.get("test_split"):
        split_str = data_config["test_split"]
        if "[" in split_str and "]" in split_str:
            slice_part = split_str.split("[")[1].split("]")[0]
            if ":" in slice_part:
                parts = slice_part.split(":")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else len(test_data)
                test_data = test_data.select(range(start, end))

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    return train_data, test_data


def create_dataloader(dataset, classifier, batch_size: int, shuffle: bool = True):
    """Create DataLoader from dataset."""

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch])

        # Tokenize
        encodings = classifier.tokenize(texts)

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch."""
    model.model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config["training"].get("gradient_accumulation_steps", 1) == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Get predictions
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

        # Log to MLflow
        if (batch_idx + 1) % config["logging"].get("log_interval", 100) == 0:
            step = epoch * len(dataloader) + batch_idx
            mlflow.log_metric("train_loss", loss.item(), step=step)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, dataloader, device: str) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(config_path: str = "config.yaml", experiment_name: str = None):
    """Main training function."""

    # Load configuration
    config = load_config(config_path)

    # Set seed for reproducibility
    set_seed(config["training"]["seed"])

    # Setup MLflow
    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    exp_name = experiment_name or mlflow_config["experiment_name"]
    mlflow.set_experiment(exp_name)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_config.get("run_name")):

        # Log parameters
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "num_epochs": config["training"]["num_epochs"],
                "seed": config["training"]["seed"],
                "dataset": config["data"]["dataset"],
            }
        )

        # Log DVC data version if available
        # This helps track which version of the dataset was used for training
        try:
            import subprocess

            # Get DVC remote info
            dvc_remote = subprocess.run(
                ["dvc", "remote", "list"], capture_output=True, text=True, timeout=5
            )
            if dvc_remote.returncode == 0 and dvc_remote.stdout.strip():
                mlflow.set_tag("dvc.remote_configured", "true")
                mlflow.log_param("dvc.remote", dvc_remote.stdout.strip().split()[0])

            # Get git commit for full reproducibility
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if git_commit.returncode == 0:
                mlflow.set_tag("git.commit", git_commit.stdout.strip()[:8])

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # DVC or git not available or not configured - continue without it
            pass

        # Initialize model
        print(f"Initializing model: {config['model']['name']}")
        classifier = SentimentClassifier(
            model_name=config["model"]["name"],
            num_labels=config["model"]["num_labels"],
            max_length=config["model"]["max_length"],
        )

        # Log model info
        model_info = classifier.get_model_info()
        mlflow.log_params(
            {
                "num_parameters": model_info["num_parameters"],
                "device": model_info["device"],
            }
        )

        # Prepare data
        train_dataset, test_dataset = prepare_data(config)

        train_loader = create_dataloader(
            train_dataset, classifier, config["training"]["batch_size"], shuffle=True
        )

        test_loader = create_dataloader(
            test_dataset, classifier, config["training"]["batch_size"], shuffle=False
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(
            classifier.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
        )

        total_steps = len(train_loader) * config["training"]["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["training"].get("warmup_steps", 0),
            num_training_steps=total_steps,
        )

        # Training loop
        print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
        best_f1 = 0

        for epoch in range(config["training"]["num_epochs"]):
            # Train
            train_metrics = train_epoch(
                classifier,
                train_loader,
                optimizer,
                scheduler,
                classifier.device,
                epoch,
                config,
            )

            print(f"\nEpoch {epoch + 1} Training Metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
                mlflow.log_metric(f"train_{key}", value, step=epoch)

            # Evaluate
            print("\nEvaluating on test set...")
            test_metrics = evaluate(classifier, test_loader, classifier.device)

            print(f"Epoch {epoch + 1} Test Metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")
                mlflow.log_metric(f"test_{key}", value, step=epoch)

            # Save best model
            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                print(f"New best F1 score: {best_f1:.4f} - Saving model...")

                # Save model locally
                save_path = "best_model"
                classifier.save_pretrained(save_path)

                # Log model to MLflow
                mlflow.pytorch.log_model(
                    classifier.model, "model", registered_model_name=f"{exp_name}_best"
                )

        # Log final best score
        mlflow.log_metric("best_f1_score", best_f1)

        print("\n✅ Training complete!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print("Model saved to: best_model/")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"View results at: {mlflow_config['tracking_uri']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")

    args = parser.parse_args()
    main(args.config, args.experiment)
