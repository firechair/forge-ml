"""Pytest fixtures for time series tests."""
import pytest
import numpy as np
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_sequence(config):
    """Generate sample time series sequence."""
    seq_len = config['forecasting']['sequence_length']
    input_size = config['model']['input_size']
    return np.random.randn(seq_len, input_size).astype(np.float32)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
