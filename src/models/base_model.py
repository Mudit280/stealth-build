# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    Defines the interface that all models must implement.
    """
    
    def __init__(self, model_name: str, **kwargs):
        # Type checking
        if not isinstance(model_name, str):
            raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
        
        # Required attributes
        self.model_name = model_name
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
    
    # Handle device (validate it's either 'cpu' or 'cuda')
        self.device = str(kwargs.get('device', 'cpu')).lower()
        if self.device not in ('cpu', 'cuda'):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device}")
    
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")