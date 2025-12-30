"""
Pytest fixtures for sentiment analysis tests.

Fixtures provide reusable test data and setup/teardown logic.
"""

import pytest
import torch
from pathlib import Path
import yaml
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def config():
    """Load the configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing predictions."""
    return {
        "positive": [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Great product, highly recommend!",
            "Best experience ever, will definitely come back.",
        ],
        "negative": [
            "Terrible movie, waste of time.",
            "Very disappointed with this product.",
            "Worst service I've ever experienced.",
        ],
        "neutral": [
            "The movie was okay, nothing special.",
            "It works as expected.",
        ],
    }


@pytest.fixture
def model_path():
    """Path to the best model directory."""
    return Path(__file__).parent.parent / "best_model"


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary path for testing model saving."""
    return tmp_path / "test_model"


@pytest.fixture(scope="session")
def device():
    """Get the device for testing (CPU or CUDA)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
