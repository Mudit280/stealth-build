import unittest
import os

from src.models.gpt2_model import GPT2Model
from src.concept_detectors.probe_concept_detector import ProbeConceptDetector

class TestProbeDetectorIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up the model and detector for integration testing.
        """
        self.model = GPT2Model(model_name='gpt2')
        self.model.load_model()
        
        # Define the path to the probe. Assumes it's in the root directory.
        self.probe_path = 'probe.pt'
        self.layer = 8 # As determined during probe training

    def test_detector_with_real_components(self):
        """
        Tests the full integration of the model and a real probe.
        """
        # Check if the probe file exists before running the test
        if not os.path.exists(self.probe_path):
            self.skipTest(f"Probe file not found at {self.probe_path}")

        # Initialize the detector with the real model and probe
        detector = ProbeConceptDetector(
            model=self.model, 
            probe_path=self.probe_path, 
            layer=self.layer
        )

        # A sentence that should have a high score for the target concept
        # (You might want to adjust this text based on what your probe detects)
        test_text = "This movie was absolutely fantastic, a masterpiece of cinema."

        # Get the detection score
        score = detector.detect(test_text)

        # Assert that the score is a valid probability (between 0 and 1)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        print(f"\nDetected concept score for '{test_text}': {score:.4f}")

if __name__ == '__main__':
    unittest.main()
