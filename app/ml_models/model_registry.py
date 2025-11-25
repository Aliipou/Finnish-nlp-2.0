"""
ML Model Registry
Centralized model loading, versioning, and management
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all custom ML models
    Handles model loading, versioning, caching, and metadata
    """

    def __init__(self, models_dir: str = "app/ml_models"):
        """
        Initialize model registry

        Args:
            models_dir: Base directory for all ML models
        """
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}

        logger.info("Initializing ML Model Registry")
        self._discover_models()

    def _discover_models(self):
        """Discover available models in the models directory"""
        model_types = [
            'profanity_classifier',
            'ambiguity_resolver',
            'sentiment_analyzer',
            'lemma_predictor',
            'code_switch_detector'
        ]

        for model_type in model_types:
            model_path = self.models_dir / model_type
            if model_path.exists():
                # Try to load metadata
                metadata_file = model_path / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.model_metadata[model_type] = json.load(f)
                        logger.info(f"✅ Found model: {model_type}")
                else:
                    logger.warning(f"⚠️  Model directory exists but no metadata: {model_type}")
            else:
                logger.info(f"ℹ️  Model not yet created: {model_type}")

    def load_model(self, model_name: str, force_reload: bool = False) -> Optional[Any]:
        """
        Load a model by name

        Args:
            model_name: Name of the model (e.g., 'profanity_classifier')
            force_reload: Force reload even if cached

        Returns:
            Loaded model or None if unavailable
        """
        # Check cache
        if not force_reload and model_name in self.loaded_models:
            logger.debug(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]

        # Try to load model
        try:
            if model_name == 'profanity_classifier':
                from .profanity_classifier.inference import ProfanityClassifierInference
                model = ProfanityClassifierInference()
                self.loaded_models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                return model

            elif model_name == 'ambiguity_resolver':
                from .ambiguity_resolver.inference import AmbiguityResolverInference
                model = AmbiguityResolverInference()
                self.loaded_models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                return model

            elif model_name == 'sentiment_analyzer':
                from .sentiment_analyzer.inference import SentimentAnalyzerInference
                model = SentimentAnalyzerInference()
                self.loaded_models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                return model

            elif model_name == 'lemma_predictor':
                from .lemma_predictor.inference import LemmaPredictorInference
                model = LemmaPredictorInference()
                self.loaded_models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                return model

            elif model_name == 'code_switch_detector':
                from .code_switch_detector.inference import CodeSwitchDetectorInference
                model = CodeSwitchDetectorInference()
                self.loaded_models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                return model

            else:
                logger.error(f"❌ Unknown model: {model_name}")
                return None

        except ImportError as e:
            logger.warning(f"⚠️  Model {model_name} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return None

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get metadata for a model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]
        else:
            return {
                'name': model_name,
                'status': 'not_available',
                'message': 'Model not yet trained or metadata missing'
            }

    def list_models(self) -> Dict[str, Any]:
        """
        List all available models

        Returns:
            Dictionary of model names and their status
        """
        models_info = {}
        model_types = [
            'profanity_classifier',
            'ambiguity_resolver',
            'sentiment_analyzer',
            'lemma_predictor',
            'code_switch_detector'
        ]

        for model_name in model_types:
            models_info[model_name] = {
                'loaded': model_name in self.loaded_models,
                'metadata_available': model_name in self.model_metadata,
                'info': self.get_model_info(model_name)
            }

        return models_info

    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")

    def reload_all_models(self):
        """Reload all models (useful for updates)"""
        logger.info("Reloading all models...")
        self.loaded_models.clear()
        self._discover_models()


# Global registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
