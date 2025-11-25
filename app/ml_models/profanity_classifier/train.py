"""
Training Script for Profanity Classifier
Fine-tune FinBERT on Finnish toxicity dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json

from .model import ProfanityClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToxicityDataset(Dataset):
    """Dataset for Finnish toxicity classification"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize dataset

        Args:
            texts: List of Finnish texts
            labels: List of labels (0=non-toxic, 1=toxic)
            tokenizer: FinBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ProfanityClassifierTrainer:
    """Trainer for profanity classifier"""

    def __init__(
        self,
        model_name: str = "TurkuNLP/bert-base-finnish-cased-v1",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 5,
        device: str = None
    ):
        """
        Initialize trainer

        Args:
            model_name: Pre-trained model name
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            device: Device (cuda/cpu)
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing trainer on device: {self.device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("✅ Loaded tokenizer")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Initialize model
        self.model = ProfanityClassifier(model_name=model_name)
        self.model.to(self.device)

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

    def load_data(self, dataset_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare data

        Args:
            dataset_path: Path to CSV file with 'text' and 'is_toxic' columns

        Returns:
            Train and validation DataLoaders
        """
        logger.info(f"Loading dataset from: {dataset_path}")

        # Load CSV
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} samples")

        # Split train/val (80/20)
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

        # Create datasets
        train_dataset = ToxicityDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['is_toxic'].tolist(),
            tokenizer=self.tokenizer
        )

        val_dataset = ToxicityDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['is_toxic'].tolist(),
            tokenizer=self.tokenizer
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)

                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, dataset_path: str, save_dir: str = "app/ml_models/profanity_classifier/weights"):
        """
        Main training loop

        Args:
            dataset_path: Path to training dataset
            save_dir: Directory to save model weights
        """
        logger.info("=" * 80)
        logger.info("Starting Profanity Classifier Training")
        logger.info("=" * 80)

        # Load data
        train_loader, val_loader = self.load_data(dataset_path)

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        best_val_acc = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 80)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)

                # Save model
                model_path = save_path / "profanity_classifier.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, model_path)

                logger.info(f"✅ Saved best model to: {model_path}")

                # Save metadata
                metadata = {
                    'model_name': 'Profanity Classifier',
                    'version': '1.0.0',
                    'base_model': self.model_name,
                    'best_epoch': epoch + 1,
                    'best_val_accuracy': float(val_acc),
                    'best_val_loss': float(val_loss),
                    'num_epochs': self.num_epochs,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'trained_on': pd.Timestamp.now().isoformat()
                }

                metadata_path = save_path / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info("=" * 80)

        return history


def main():
    """Main training function"""
    # Configuration
    DATASET_PATH = "data/datasets/finnish_toxicity_corpus/finnish_toxicity_corpus.csv"
    SAVE_DIR = "app/ml_models/profanity_classifier/weights"

    # Initialize trainer
    trainer = ProfanityClassifierTrainer(
        model_name="TurkuNLP/bert-base-finnish-cased-v1",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5
    )

    # Train
    history = trainer.train(dataset_path=DATASET_PATH, save_dir=SAVE_DIR)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
