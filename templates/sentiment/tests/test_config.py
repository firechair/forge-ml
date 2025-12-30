"""
Tests for configuration validation.

Ensures the config.yaml file is properly formatted and contains valid values.
"""
import pytest
from pathlib import Path
import yaml


def test_config_file_exists():
    """Test that config.yaml exists."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    assert config_path.exists(), "config.yaml not found"


def test_config_is_valid_yaml(config):
    """Test that config.yaml is valid YAML."""
    assert config is not None
    assert isinstance(config, dict)


def test_config_has_required_sections(config):
    """Test that config has all required sections."""
    required_sections = ['model', 'data', 'training', 'mlflow']
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"


def test_model_config(config):
    """Test model configuration."""
    model = config['model']
    assert 'name' in model
    assert 'num_labels' in model
    assert 'max_length' in model

    assert isinstance(model['name'], str)
    assert isinstance(model['num_labels'], int)
    assert model['num_labels'] > 0
    assert isinstance(model['max_length'], int)
    assert model['max_length'] > 0


def test_data_config(config):
    """Test data configuration."""
    data = config['data']
    assert 'dataset' in data
    assert 'train_split' in data
    assert 'test_split' in data

    assert isinstance(data['dataset'], str)
    assert isinstance(data['train_split'], str)
    assert isinstance(data['test_split'], str)


def test_training_config(config):
    """Test training configuration."""
    training = config['training']

    required_params = [
        'batch_size', 'learning_rate', 'num_epochs',
        'weight_decay', 'warmup_steps', 'max_grad_norm', 'seed'
    ]

    for param in required_params:
        assert param in training, f"Missing training parameter: {param}"

    # Validate types and ranges
    assert isinstance(training['batch_size'], int)
    assert training['batch_size'] > 0

    assert isinstance(training['learning_rate'], (int, float))
    assert training['learning_rate'] > 0

    assert isinstance(training['num_epochs'], int)
    assert training['num_epochs'] > 0

    assert isinstance(training['seed'], int)


def test_mlflow_config(config):
    """Test MLflow configuration."""
    mlflow = config['mlflow']

    assert 'experiment_name' in mlflow
    assert 'run_name' in mlflow or mlflow.get('run_name') is None
    assert 'tracking_uri' in mlflow or mlflow.get('tracking_uri') is None

    assert isinstance(mlflow['experiment_name'], str)
