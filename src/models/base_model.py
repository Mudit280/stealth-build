# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import torch

# Set up logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    Defines the interface that all models must implement.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model with common attributes and configurations.
        
        Args:
            model_name: Name or identifier of the model
            **kwargs: Additional configuration parameters
                - device: 'cpu' or 'cuda' (default: 'cpu')
                - max_length: Maximum sequence length (default: 512)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Nucleus sampling parameter (default: 0.9)
        """
        # Type checking and validation
        if not isinstance(model_name, str):
            raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
        
        # Required attributes
        self.model_name = model_name
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        
        # Configuration
        self.device = str(kwargs.get('device', 'cpu')).lower()
        if self.device not in ('cpu', 'cuda'):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device}")
        
        # Generation parameters - these control the text generation behavior
        # Maximum number of tokens to generate in the output
        # Higher values allow longer responses but increase computation time
        # Default: 512 (typical context window for many models)
        self.max_length = int(kwargs.get('max_length', 512))
        
        # Temperature controls randomness in generation
        # - Lower (e.g., 0.2) makes output more focused and deterministic
        # - Higher (e.g., 1.0) makes output more diverse and creative
        # - Range: (0.0, 2.0), Default: 0.7 (balanced for creative but coherent text)
        self.temperature = float(kwargs.get('temperature', 0.7))
        
        # Top-p (nucleus) sampling parameter
        # - Controls diversity by limiting to top tokens that sum to this probability mass
        # - Lower values (e.g., 0.5) make output more focused
        # - Higher values (e.g., 1.0) allow more diversity
        # - Range: (0.0, 1.0), Default: 0.9 (good balance between quality and diversity)
        self.top_p = float(kwargs.get('top_p', 0.9))
        
        # Initialize empty concept detectors dictionary
        self.concept_detectors = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer.
        Should set self.model, self.tokenizer, and self.is_loaded
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: Input text prompt
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    def detect_concepts(self, text: str) -> Dict[str, float]:
        """
        Detect concepts in the given text using registered concept detectors.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping concept names to detection scores
        """
        if not self.concept_detectors:
            logger.warning("No concept detectors registered")
            return {}
            
        results = {}
        for name, detector in self.concept_detectors.items():
            try:
                results[name] = detector.detect(text)
            except Exception as e:
                logger.error(f"Error in concept detector '{name}': {str(e)}")
                results[name] = 0.0
                
        return results
    
    def register_concept_detector(self, name: str, detector: Any) -> None:
        """
        Register a concept detector.
        
        Args:
            name: Name to identify the detector
            detector: Concept detector instance with a detect() method
        """
        if not hasattr(detector, 'detect') or not callable(detector.detect):
            raise ValueError("Concept detector must implement a detect() method")
        self.concept_detectors[name] = detector
        logger.info(f"Registered concept detector: {name}")
    
    def get_activations(self, layer: int = -1) -> Optional[torch.Tensor]:
        """
        Get activations from a specific layer.
        
        Args:
            layer: Layer index to get activations from
            
        Returns:
            Tensor containing the activations, or None if not available
        """
        if not hasattr(self, 'activations') or layer not in self.activations:
            logger.warning(f"No activations available for layer {layer}")
            return None
        return self.activations[layer]
    
    def steer_output(self, concept: str, strength: float = 0.5) -> bool:
        """
        Apply steering to the model's output based on a concept.
        
        Args:
            concept: Name of the concept to steer towards/away from
            strength: Steering strength (-1.0 to 1.0)
            
        Returns:
            bool: True if steering was applied successfully
        """
        if not (-1.0 <= strength <= 1.0):
            logger.error(f"Steering strength must be between -1.0 and 1.0, got {strength}")
            return False
            
        if concept not in self.concept_detectors:
            logger.error(f"No concept detector registered for: {concept}")
            return False
            
        # Implementation will vary by model
        logger.info(f"Steering {concept} with strength {strength}")
        return True
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"