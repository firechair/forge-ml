"""
Sentiment Classification Model using HuggingFace Transformers.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import Dict, List, Union, Optional
import numpy as np


class SentimentClassifier:
    """
    Sentiment classification wrapper for HuggingFace transformer models.

    This class provides a simple interface for loading, training, and using
    transformer models for binary sentiment classification.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize the sentiment classifier.

        Args:
            model_name: HuggingFace model name or path
            num_labels: Number of classification labels (2 for binary sentiment)
            max_length: Maximum sequence length for tokenization
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        )
        self.model.to(self.device)

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text(s).

        Args:
            texts: Single text string or list of text strings

        Returns:
            Dictionary containing input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(
        self, texts: Union[str, List[str]], return_probabilities: bool = True
    ) -> Union[List[int], List[Dict[str, float]]]:
        """
        Predict sentiment for input text(s).

        Args:
            texts: Single text string or list of text strings
            return_probabilities: If True, return probabilities; if False, return labels

        Returns:
            List of predictions (labels or probability dictionaries)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Tokenize
        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        if return_probabilities:
            # Return probability dictionaries
            results = []
            for probs in probabilities:
                results.append(
                    {
                        "negative": float(probs[0]),
                        "positive": float(probs[1]),
                        "predicted_label": int(np.argmax(probs)),
                    }
                )
        else:
            # Return just the labels
            results = np.argmax(probabilities, axis=-1).tolist()

        # Return single result if single input
        return results[0] if single_input else results

    def save_pretrained(self, save_path: str):
        """
        Save model and tokenizer to disk.

        Args:
            save_path: Directory to save model and tokenizer
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls, model_path: str, device: Optional[str] = None
    ) -> "SentimentClassifier":
        """
        Load a pre-trained model from disk.

        Args:
            model_path: Path to saved model directory
            device: Device to load model on

        Returns:
            Loaded SentimentClassifier instance
        """
        # Load tokenizer to get config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.num_labels = model.config.num_labels
        instance.max_length = tokenizer.model_max_length

        if device is None:
            instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            instance.device = device

        instance.tokenizer = tokenizer
        instance.model = model
        instance.model.to(instance.device)

        return instance

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    classifier = SentimentClassifier()

    # Single prediction
    result = classifier.predict("This movie was absolutely amazing!")
    print(f"Single prediction: {result}")

    # Batch prediction
    texts = [
        "I loved this product!",
        "Terrible service, very disappointed.",
        "It's okay, nothing special.",
    ]
    results = classifier.predict(texts)
    for text, result in zip(texts, results):
        sentiment = "positive" if result["predicted_label"] == 1 else "negative"
        confidence = result[sentiment]
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
        print()
