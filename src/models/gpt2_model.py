"""
Simple GPT-2 Model Implementation

A concrete implementation of BaseModel providing access to Hugging Face's GPT-2 language model
with concept detection and steering capabilities.
"""

import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, Optional, Any, List, Tuple
from .base_model import BaseModel
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPT2Model(BaseModel):
    """
    Implementation of GPT-2 language model with concept detection and steering capabilities.
    """

    def __init__(self, model_name: str = "gpt2", **kwargs: Any) -> None:
        """
        Initialize the GPT-2 model.
        """
        super().__init__(model_name=model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        self.steering_hooks = {}

    def load_model(self) -> None:
        """
        Load the GPT-2 model and tokenizer.
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_name} is already loaded")
            return

        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, steering_vectors: Optional[List[Tuple[int, np.ndarray, float]]] = None, **kwargs) -> str:
        """
        Generate text from a prompt, with optional activation steering.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            gen_kwargs = {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with self.steering(steering_vectors if steering_vectors else []):
                with torch.no_grad():
                    outputs = self.model.generate(inputs, **gen_kwargs)

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _create_steering_hook(self, vector: np.ndarray, alpha: float):
        """
        Creates a forward hook function that adds a steering vector to the model's activations.
        """
        steering_tensor = torch.tensor(vector, dtype=self.model.dtype, device=self.device) * alpha

        def hook(module, input, output):
            # The output of a transformer block is a tuple; the first element is the hidden state.
            modified_hidden_state = output[0] + steering_tensor
            return (modified_hidden_state,) + output[1:]

        return hook

    def add_steering_vector(self, layer: int, vector: np.ndarray, alpha: float = 0.5):
        """
        Adds a steering vector to a specific layer of the model.
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before adding steering vectors.")

        if layer in self.steering_hooks:
            logger.warning(f"A steering hook already exists for layer {layer}. It will be replaced.")
            self.steering_hooks[layer].remove()

        hook_fn = self._create_steering_hook(vector, alpha)
        # The transformer blocks are in `model.transformer.h`
        hook_handle = self.model.transformer.h[layer].register_forward_hook(hook_fn)
        self.steering_hooks[layer] = hook_handle
        logger.info(f"Added steering vector to layer {layer} with strength {alpha}.")

    def remove_steering_vectors(self):
        """
        Removes all active steering hooks from the model.
        """
        for layer, handle in self.steering_hooks.items():
            handle.remove()
            logger.info(f"Removed steering hook from layer {layer}.")
        self.steering_hooks.clear()

    @contextmanager
    def steering(self, vectors: List[Tuple[int, np.ndarray, float]]):
        """
        A context manager to apply steering vectors temporarily.
        Args:
            vectors: A list of tuples, each containing (layer, vector, alpha).
        """
        self.remove_steering_vectors()  # Clear any existing hooks
        try:
            for layer, vector, alpha in vectors:
                self.add_steering_vector(layer, vector, alpha)
            yield
        finally:
            self.remove_steering_vectors()

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

    def extract_features(self, texts: list, layer: int = -1, pooling: str = "mean") -> np.ndarray:
        """
        Extract features (hidden states) from input texts using GPT-2.
        """
        if not self.is_loaded:
            raise RuntimeError("Model and tokenizer must be loaded before extracting features.")

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            selected_layer = hidden_states[layer]

            if pooling == "mean":
                pooled = selected_layer.mean(dim=1)
            elif pooling == "last":
                attention_mask = inputs["attention_mask"]
                lengths = attention_mask.sum(dim=1) - 1
                pooled = selected_layer[range(selected_layer.size(0)), lengths]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            return pooled.cpu().numpy()
    
    def extract_features_multi(self, texts: list, layers: list, pooling: str = "mean") -> np.ndarray:
        """
        Extract features (hidden states) from multiple layers.
        """
        all_features = []
        for layer in layers:
            feats = self.extract_features(texts, layer=layer, pooling=pooling)
            all_features.append(feats)
        return np.stack(all_features, axis=1)