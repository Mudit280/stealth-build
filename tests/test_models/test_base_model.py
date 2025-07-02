import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.base_model import BaseModel

class TestBaseModel:
    """Test suite for the BaseModel class."""
    
    class ConcreteModel(BaseModel):
        """Concrete implementation of BaseModel for testing."""
        
        def load_model(self) -> None:
            """Mock model loading."""
            self.model = MagicMock()
            self.tokenizer = MagicMock()
            self.is_loaded = True
        
        def generate(self, prompt: str, **generation_kwargs) -> str:
            """Mock text generation."""
            return f"Generated response to: {prompt}"
    
    def test_initialization(self):
        """Test that the base model initializes correctly."""
        model = self.ConcreteModel("test-model")
        
        assert model.model_name == "test-model"
        assert model.is_loaded is False
        assert model.device == "cpu"
        assert model.max_length == 512
        assert model.temperature == 0.7
        assert model.top_p == 0.9
        assert model.concept_detectors == {}
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = self.ConcreteModel(
            "test-model",
            device="cuda",
            max_length=256,
            temperature=0.5,
            top_p=0.8
        )
        
        assert model.device == "cuda"
        assert model.max_length == 256
        assert model.temperature == 0.5
        assert model.top_p == 0.8
    
    def test_invalid_device(self):
        """Test that invalid device raises an error."""
        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
            self.ConcreteModel("test-model", device="invalid")
    
    def test_load_model_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel("test-model")
    
    def test_concept_detection(self):
        """Test concept detection with registered detectors."""
        model = self.ConcreteModel("test-model")
        
        # Create a mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = 0.75
        
        # Register the detector
        model.register_concept_detector("test_concept", mock_detector)
        
        # Test detection
        results = model.detect_concepts("test input")
        
        assert "test_concept" in results
        assert results["test_concept"] == 0.75
        mock_detector.detect.assert_called_once_with("test input")
    
    def test_steer_output(self):
        """Test steering output functionality."""
        model = self.ConcreteModel("test-model")
        
        # Register a mock detector
        mock_detector = MagicMock()
        model.register_concept_detector("test_concept", mock_detector)
        
        # Test valid steering
        assert model.steer_output("test_concept", 0.5) is True
        
        # Test invalid concept
        assert model.steer_output("invalid_concept", 0.5) is False
        
        # Test invalid strength
        assert model.steer_output("test_concept", 1.5) is False
        assert model.steer_output("test_concept", -1.5) is False
    
    def test_string_representation(self):
        """Test the string representation of the model."""
        model = self.ConcreteModel("test-model")
        assert "ConcreteModel(model_name='test-model', device='cpu')" in str(model)
