import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.models.gpt2_model import GPT2Model
from src.concept_detectors.probe_concept_detector import ProbeConceptDetector

class TestProbeConceptDetector(unittest.TestCase):

    @patch('joblib.load')
    def test_detect(self, mock_joblib_load):
        """
        Tests the detect method with mock objects to ensure correct logic.
        """
        # 1. Set up mock objects
        # Mock GPT2Model
        mock_model = MagicMock(spec=GPT2Model)
        # Let extract_features return a dummy activation vector
        dummy_activations = np.random.rand(1, 768) # (batch_size, hidden_dim)
        mock_model.extract_features.return_value = dummy_activations

        # Mock scikit-learn probe
        mock_probe = MagicMock()
        # Let predict_proba return a known probability
        expected_probability = 0.85
        mock_probe.predict_proba.return_value = np.array([[1 - expected_probability, expected_probability]])
        
        # Configure joblib.load to return our mock probe
        mock_joblib_load.return_value = mock_probe

        # 2. Initialize the detector with mocks
        probe_path = 'dummy/path/probe.joblib'
        layer = 8
        detector = ProbeConceptDetector(model=mock_model, probe_path=probe_path, layer=layer)

        # 3. Call the detect method
        test_text = "This is a test sentence."
        detected_score = detector.detect(test_text)

        # 4. Assertions
        # Ensure the model's feature extraction was called correctly
        mock_model.extract_features.assert_called_once_with([test_text], layer=layer)
        
        # Ensure the probe's prediction method was called with the activations
        np.testing.assert_array_equal(mock_probe.predict_proba.call_args[0][0], dummy_activations)

        # Ensure the final score is what we expect
        self.assertAlmostEqual(detected_score, expected_probability, places=5)
        
        # Ensure joblib.load was called with the correct path
        mock_joblib_load.assert_called_once_with(probe_path)

if __name__ == '__main__':
    unittest.main()
