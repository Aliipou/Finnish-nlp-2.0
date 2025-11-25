"""
Profanity Classifier Model Architecture
FinBERT + Classification Head
"""
import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProfanityClassifier(nn.Module):
    """
    Finnish profanity/toxicity classifier
    Based on FinBERT (TurkuNLP/bert-base-finnish-cased-v1)
    """

    def __init__(self, model_name: str = "TurkuNLP/bert-base-finnish-cased-v1", num_classes: int = 2):
        """
        Initialize profanity classifier

        Args:
            model_name: Pre-trained FinBERT model name
            num_classes: 2 for binary (toxic/non-toxic)
        """
        super().__init__()

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        logger.info(f"Initialized ProfanityClassifier with {model_name}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            'architecture': 'FinBERT + Classification Head',
            'base_model': 'TurkuNLP/bert-base-finnish-cased-v1',
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'classifier_head': 'Linear(768, 2)',
            'task': 'binary classification (toxic/non-toxic)'
        }
