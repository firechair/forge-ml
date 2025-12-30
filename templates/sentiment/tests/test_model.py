"""
Tests for the SentimentClassifier model.

Tests model initialization, prediction, saving, and loading.
"""
import pytest
import torch
from pathlib import Path
import sys

# Import model class
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import SentimentClassifier


def test_model_initialization(config):
    """Test that model initializes correctly."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        max_length=config['model']['max_length']
    )

    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.num_labels == config['model']['num_labels']
    assert model.max_length == config['model']['max_length']


def test_model_device(config, device):
    """Test that model is on correct device."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # Model device should match expected device
    model_device = next(model.model.parameters()).device.type
    assert model_device == device


def test_model_info(config):
    """Test that get_model_info returns expected information."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    info = model.get_model_info()

    assert 'model_name' in info
    assert 'num_parameters' in info
    assert 'device' in info

    assert info['model_name'] == config['model']['name']
    assert isinstance(info['num_parameters'], int)
    assert info['num_parameters'] > 0


def test_single_prediction(config, sample_texts):
    """Test single text prediction."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # Test prediction on positive text
    text = sample_texts['positive'][0]
    result = model.predict(text)

    assert 'label' in result
    assert 'confidence' in result
    assert 'probabilities' in result

    assert isinstance(result['label'], int)
    assert 0 <= result['label'] < config['model']['num_labels']

    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1

    assert isinstance(result['probabilities'], list)
    assert len(result['probabilities']) == config['model']['num_labels']
    assert all(0 <= p <= 1 for p in result['probabilities'])
    assert abs(sum(result['probabilities']) - 1.0) < 0.01  # Should sum to ~1


def test_batch_prediction(config, sample_texts):
    """Test batch prediction."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # Test prediction on multiple texts
    texts = sample_texts['positive'][:2] + sample_texts['negative'][:2]
    results = model.predict_batch(texts)

    assert len(results) == len(texts)

    for result in results:
        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result


def test_model_save_load(config, temp_model_path):
    """Test saving and loading model."""
    # Create and save model
    model1 = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    model1.save(temp_model_path)

    # Verify files were created
    assert temp_model_path.exists()
    assert (temp_model_path / "config.json").exists()
    assert (temp_model_path / "pytorch_model.bin").exists() or \
           (temp_model_path / "model.safetensors").exists()

    # Load model
    model2 = SentimentClassifier.load(temp_model_path)

    assert model2 is not None
    assert model2.num_labels == model1.num_labels
    assert model2.max_length == model1.max_length

    # Test that loaded model can predict
    text = "This is a test."
    result = model2.predict(text)
    assert 'label' in result


def test_empty_text_handling(config):
    """Test model behavior with empty text."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # Empty string should still return a prediction
    result = model.predict("")
    assert 'label' in result
    assert 'confidence' in result


def test_long_text_handling(config):
    """Test model behavior with very long text."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        max_length=128  # Use smaller max_length for testing
    )

    # Create text longer than max_length
    long_text = "This is a test sentence. " * 100

    result = model.predict(long_text)
    assert 'label' in result
    assert 'confidence' in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_gpu(config):
    """Test model runs on GPU if available."""
    model = SentimentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # Model should be on CUDA
    model_device = next(model.model.parameters()).device.type
    assert model_device == "cuda"

    # Prediction should work on GPU
    result = model.predict("Test text")
    assert 'label' in result
