"""
GPT-2 Model Implementation

A concrete implementation of BaseModel providing access to Hugging Face's GPT-2 language model
with concept detection and steering capabilities.

Core Functionality:
1. Initialization
   - Configurable model size (e.g., "gpt2", "gpt2-medium")
   - Device management (CPU/GPU)
   - Generation parameter configuration

2. Model Management
   - Lazy loading of model weights (Why lazy loading?)
   - Resource-efficient operation
   - Model verification

3. Text Generation
   - Prompt-based text completion
   - Configurable generation parameters
   - Integrated concept detection

4. Concept Integration
   - Dynamic concept registration
   - Real-time concept detection
   - Activation analysis

5. Steering Capabilities
   - Output modification based on concepts
   - Strength-based steering
   - Multi-concept interaction

Example Usage:
    >>> model = GPT2Model("gpt2", device="cuda")
    >>> model.load_model()
    >>> output = model.generate("The future of AI is")
    >>> print(output)

Test Strategy:
- Unit tests for individual components
- Integration tests for full pipeline
- Performance benchmarks
- Edge case validation

Note: This implementation follows the interface defined in BaseModel while
adding GPT-2 specific functionality.
"""

import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, Optional, Any, List
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class GPT2Model(BaseModel):
    """
    Implementation of GPT-2 language model with concept detection and steering capabilities.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        **kwargs: Any
    ) -> None:
        """
        Initialize the GPT-2 model.

        Args:
            model_name: Name of the GPT-2 model (e.g., 'gpt2', 'gpt2-medium')
            **kwargs: Additional arguments passed to the base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.tokenizer = None
        self.model = None
        self.activations = {}  # Store layer activations

    def load_model(self) -> None:
        """
        Load the GPT-2 model and tokenizer.
        
        This method:
        1. Loads the tokenizer
        2. Loads the model
        3. Moves the model to the specified device (CPU/GPU)
        4. Sets the model to evaluation mode
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_name} is already loaded")
            return
                
        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                
            # Add padding token if not present (GPT-2 doesn't have one by default)
            # This is important for batching sequences of different lengths
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            logger.info(f"Loading model {self.model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_name,
                pad_token_id=self.tokenizer.eos_token_id  # Ensure consistent padding with tokenizer
            )
                
            # Move model to the specified device (CPU/GPU)
            # This is crucial for performance - models run much faster on GPU if available
            # The device is set during initialization (defaults to 'cpu')
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            # This disables certain layers like dropout and batch normalization
            # which behave differently during training vs inference
            self.model.eval()
                
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
                
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            self.is_loaded = False
            raise