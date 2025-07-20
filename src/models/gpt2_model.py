"""
Simple GPT-2 Model Implementation

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
from typing import Dict, Optional, Any
from .base_model import BaseModel
import numpy as np

logger = logging.getLogger(__name__)

class GPT2Model(BaseModel):
    """
    Implementation of GPT-2 language model with concept detection and steering capabilities.
    """

    def __init__(self, model_name: str = "gpt2", **kwargs: Any) -> None:
        """
        Initialize the GPT-2 model.

        Args:
            model_name: Name of the GPT-2 model (e.g., 'gpt2', 'gpt2-medium')
            **kwargs: Additional arguments passed to the base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        
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
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(inputs, **gen_kwargs)
            
            # Decode result
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def detect_concepts(self, text: str) -> Dict[str, float]:
        """Detect concepts in text using registered detectors."""
        if not self.concept_detectors:
            return {}
        
        results = {}
        for name, detector in self.concept_detectors.items():
            try:
                results[name] = detector.detect(text)
            except Exception as e:
                logger.error(f"Concept detection failed for {name}: {e}")
                results[name] = 0.0
        
        return results

    def steer_output(self, concept: str, strength: float = 0.5) -> bool:
        """Apply steering (placeholder for now)."""
        if not (-1.0 <= strength <= 1.0):
            logger.error(f"Invalid strength: {strength}")
            return False
        
        if concept not in self.concept_detectors:
            logger.error(f"No detector for concept: {concept}")
            return False
        
        logger.info(f"Steering {concept} with strength {strength}")
        return True

    def extract_features(self, texts: list, layer: int = -1, pooling: str = "mean") -> np.ndarray:
        """
        Extract features (hidden states) from input texts using GPT-2.

        Args:
            texts: List of input strings to process.
            layer: Which GPT-2 layer to extract features from (default: last).
            pooling: Pooling strategy to apply ("mean", "last").

        Returns:
            Array of extracted features for each input.
        """
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            raise RuntimeError("Model and tokenizer must be loaded before extracting features.")

        self.model.eval()
        features = []
        with torch.no_grad():
            # Tokenize and batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            # Move inputs to the model's device
            device = getattr(self, "device", "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., layerN)
            selected_layer = hidden_states[layer]  # [batch_size, seq_len, hidden_dim]

            if pooling == "mean":
                pooled = selected_layer.mean(dim=1)  # mean over sequence length
            elif pooling == "last":
                attention_mask = inputs["attention_mask"]
                lengths = attention_mask.sum(dim=1) - 1  # last token index for each input
                pooled = selected_layer[range(selected_layer.size(0)), lengths]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            features = pooled.cpu().numpy()

        return features
    
    def extract_features_multi(self, texts: list, layers: list, pooling: str = "mean") -> np.ndarray:
        """
        Extract features (hidden states) from input texts using GPT-2.

        Args:
            texts: List of input strings to process.
            layers: List of GPT-2 layers to extract features from.
            pooling: Pooling strategy to apply ("mean", "last").

        Returns:
            Array of extracted features for each input.
        """
        # Returns features for each layer in layers
        all_features = []
        for layer in layers:
            feats = self.extract_features(texts, layer=layer, pooling=pooling)
            all_features.append(feats)
        return np.stack(all_features, axis=1)  # shape: (batch, num_layers, hidden_dim)