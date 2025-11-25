"""
Inference Wrapper for Profanity Classifier
Fast, production-ready inference
"""
import torch
from transformers import AutoTokenizer
from typing import Dict, Any
from pathlib import Path
import logging
import time

from .model import ProfanityClassifier

logger = logging.getLogger(__name__)


class ProfanityClassifierInference:
    """
    Fast inference wrapper for profanity classifier
    """

    def __init__(
        self,
        weights_path: str = "app/ml_models/profanity_classifier/weights/profanity_classifier.pth",
        model_name: str = "TurkuNLP/bert-base-finnish-cased-v1",
        device: str = None
    ):
        """
        Initialize inference engine

        Args:
            weights_path: Path to trained model weights
            model_name: Base FinBERT model name
            device: Device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = Path(weights_path)

        logger.info(f"Initializing Profanity Classifier Inference on {self.device}")

        # Check if weights exist
        if not self.weights_path.exists():
            logger.warning(f"⚠️  Model weights not found at: {self.weights_path}")
            logger.warning("Model needs to be trained first. Run: python -m app.ml_models.profanity_classifier.train")
            self.model = None
            self.tokenizer = None
            return

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model
            self.model = ProfanityClassifier(model_name=model_name)

            # Load weights
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ Loaded profanity classifier (accuracy: {checkpoint.get('val_acc', 'N/A'):.4f})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model is not None and self.tokenizer is not None

    def predict(
        self,
        text: str,
        return_probabilities: bool = True,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict toxicity of text

        Args:
            text: Finnish text to classify
            return_probabilities: Return class probabilities
            threshold: Classification threshold

        Returns:
            Dictionary with prediction results
        """
        if not self.is_available():
            raise RuntimeError("Model not available. Train the model first.")

        start_time = time.time()

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)

        # Extract results
        toxic_probability = probabilities[0][1].item()
        is_toxic = toxic_probability >= threshold
        predicted_class = 1 if is_toxic else 0

        inference_time = (time.time() - start_time) * 1000  # ms

        result = {
            'text': text,
            'is_toxic': is_toxic,
            'toxicity_score': toxic_probability,
            'confidence': max(probabilities[0][0].item(), probabilities[0][1].item()),
            'predicted_class': predicted_class,
            'inference_time_ms': round(inference_time, 2),
            'model_version': 'v1.0.0'
        }

        if return_probabilities:
            result['probabilities'] = {
                'non_toxic': probabilities[0][0].item(),
                'toxic': probabilities[0][1].item()
            }

        return result

    def batch_predict(self, texts: list, batch_size: int = 32) -> list:
        """
        Batch prediction for multiple texts

        Args:
            texts: List of texts to classify
            batch_size: Batch size for inference

        Returns:
            List of prediction results
        """
        if not self.is_available():
            raise RuntimeError("Model not available. Train the model first.")

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # Batch inference
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)

            # Extract results for each text in batch
            for j, text in enumerate(batch_texts):
                toxic_prob = probabilities[j][1].item()
                is_toxic = toxic_prob >= 0.5

                results.append({
                    'text': text,
                    'is_toxic': is_toxic,
                    'toxicity_score': toxic_prob,
                    'confidence': max(probabilities[j][0].item(), probabilities[j][1].item())
                })

        return results
