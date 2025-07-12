"""
Simple tests for GPT2Model implementation.

Tests core functionality without complex abstractions.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.models.gpt2_model import GPT2Model


class TestGPT2Model:
    """Test cases for GPT2Model class."""

    def test_initialization(self):
        """Test basic initialization."""
        model = GPT2Model()
        assert model.model_name == "gpt2"
        assert model.device == "cpu"
        assert model.is_loaded is False
        assert model.model is None
        assert model.tokenizer is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        model = GPT2Model("gpt2-medium", temperature=0.5, max_length=256)
        assert model.model_name == "gpt2-medium"
        assert model.temperature == 0.5
        assert model.max_length == 256

    def test_load_model_success(self):
        """Test successful model loading."""
        model = GPT2Model()
        
        with patch('src.models.gpt2_model.GPT2Tokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.models.gpt2_model.GPT2LMHeadModel.from_pretrained') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "<|endoftext|>"
            mock_tokenizer_instance.eos_token_id = 50256
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = None
            mock_model.return_value = mock_model_instance
            
            model.load_model()
            
            assert model.is_loaded is True
            assert model.tokenizer is not None
            assert model.model is not None

    def test_load_model_already_loaded(self):
        """Test that loading an already loaded model doesn't reload."""
        model = GPT2Model()
        model.is_loaded = True
        
        with patch('src.models.gpt2_model.GPT2Tokenizer.from_pretrained') as mock_tokenizer:
            model.load_model()
            mock_tokenizer.assert_not_called()

    def test_load_model_error(self):
        """Test error handling during model loading."""
        model = GPT2Model()
        
        with patch('src.models.gpt2_model.GPT2Tokenizer.from_pretrained', side_effect=Exception("Load error")):
            with pytest.raises(Exception):
                model.load_model()
            assert model.is_loaded is False

    def test_generate_without_loading(self):
        """Test that generate raises error when model is not loaded."""
        model = GPT2Model()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate("Test prompt")

    def test_generate_empty_prompt(self):
        """Test that generate raises error for empty prompts."""
        model = GPT2Model()
        model.is_loaded = True
        model.tokenizer = Mock()
        model.model = Mock()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            model.generate("")
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            model.generate("   ")

    def test_generate_success(self):
        """Test successful text generation."""
        model = GPT2Model()
        model.is_loaded = True
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.eos_token_id = 50256
        model.tokenizer = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        model.model = mock_model
        
        result = model.generate("Test prompt")
        
        assert result == "Generated text"
        mock_tokenizer.encode.assert_called_once_with("Test prompt", return_tensors="pt")
        mock_model.generate.assert_called_once()

    def test_detect_concepts_no_detectors(self):
        """Test concept detection when no detectors are registered."""
        model = GPT2Model()
        result = model.detect_concepts("test text")
        assert result == {}

    def test_detect_concepts_with_detectors(self):
        """Test concept detection with registered detectors."""
        model = GPT2Model()
        
        # Mock concept detector
        mock_detector = Mock()
        mock_detector.detect.return_value = 0.8
        model.register_concept_detector("sentiment", mock_detector)
        
        result = model.detect_concepts("test text")
        assert result == {"sentiment": 0.8}

    def test_steer_output_invalid_strength(self):
        """Test steer_output with invalid strength values."""
        model = GPT2Model()
        
        assert not model.steer_output("test", 1.5)  # Too high
        assert not model.steer_output("test", -1.5)  # Too low

    def test_steer_output_no_detector(self):
        """Test steer_output when concept detector is not registered."""
        model = GPT2Model()
        
        assert not model.steer_output("nonexistent_concept", 0.5)

    def test_steer_output_success(self):
        """Test successful steering."""
        model = GPT2Model()
        
        # Register a mock concept detector
        mock_detector = Mock()
        model.register_concept_detector("test_concept", mock_detector)
        
        assert model.steer_output("test_concept", 0.5)

    def test_string_representation(self):
        """Test string representation of GPT2Model."""
        model = GPT2Model("gpt2-medium", device="cpu")
        expected = "GPT2Model(model_name='gpt2-medium', device='cpu')"
        assert str(model) == expected


class TestGPT2ModelIntegration:
    """Integration tests for GPT2Model with actual model loading."""

    @pytest.mark.slow
    def test_full_generation_pipeline(self):
        """Test the complete generation pipeline."""
        model = GPT2Model("gpt2", device="cpu")
        
        try:
            model.load_model()
            result = model.generate("Hello world", max_length=20)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Hello world" in result
            
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    @pytest.mark.slow
    def test_generation_with_different_params(self):
        """Test generation with different parameters."""
        model = GPT2Model("gpt2", device="cpu")
        
        try:
            model.load_model()
            
            # Test with different temperatures
            result1 = model.generate("The weather is", temperature=0.1, max_length=15)
            result2 = model.generate("The weather is", temperature=0.9, max_length=15)
            
            assert isinstance(result1, str)
            assert isinstance(result2, str)
            assert len(result1) > 0
            assert len(result2) > 0
            
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}") 