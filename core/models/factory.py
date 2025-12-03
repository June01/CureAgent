"""
Model factory for creating different types of models
"""
from typing import Dict, Type
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating model instances"""
    
    _model_registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]):
        """Register a model class with the factory"""
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered model type '{model_type}' with class {model_class.__name__}")
    
    @classmethod
    def create_model(cls, model_name: str, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance based on type"""
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._model_registry.keys())}")
        
        model_class = cls._model_registry[model_type]
        logger.info(f"Creating {model_type} model: {model_name}")
        
        return model_class(model_name)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available model types"""
        return list(cls._model_registry.keys())
    
    @classmethod
    def auto_detect_type(cls, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        model_name_lower = model_name.lower()
        
        # Check for different model types
        if any(name in model_name_lower for name in ["gpt", "chatgpt", "openai", "o1", "o3", "o4"]):
            return "chatgpt"
        elif "gemini" in model_name_lower:
            return "gemini"
        elif "txagent" in model_name_lower:
            return "txagent"
        elif any(name in model_name_lower for name in ["llama", "llama-3", "llama3"]):
            return "llama"
        elif any(name in model_name_lower for name in ["qwen", "qwen3"]):
            return "qwen"
        elif any(name in model_name_lower for name in ["medgemma", "med-gemma"]):
            return "medgemma"
        elif any(name in model_name_lower for name in ["gpt-oss", "gptoss"]):
            return "gpt_oss"
        elif any(name in model_name_lower for name in ["baichuan", "baichuan-m2", "baichuan-m2-32b"]):
            return "baichuan"
        else:
            return "local"  # Default to local model 