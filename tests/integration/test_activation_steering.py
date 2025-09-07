import unittest
import os
import numpy as np

from src.models.gpt2_model import GPT2Model
from src.concept_detectors.probe_concept_detector import ProbeConceptDetector

class TestActivationSteering(unittest.TestCase):

    def setUp(self):
        """
        Set up the model and detector for steering tests.
        """
        self.model = GPT2Model(model_name='gpt2')
        self.model.load_model()

        self.probe_path = 'probe.pt'
        self.layer = 8  # The layer the probe was trained on

        if not os.path.exists(self.probe_path):
            self.skipTest(f"Probe file not found at {self.probe_path}. Please run training first.")

        self.detector = ProbeConceptDetector(
            model=self.model,
            probe_path=self.probe_path,
            layer=self.layer
        )

    def test_steering_effect(self):
        """
        Tests the effect of activation steering on text generation.
        """
        prompt = "The movie was"
        concept_vector = self.detector.get_concept_vector()
        steering_strength = 1.5  # A positive value to increase the concept

        # 1. Generate text WITHOUT steering
        normal_output = self.model.generate(prompt, max_length=20)
        normal_score = self.detector.detect(normal_output)

        # 2. Generate text WITH steering
        steering_params = [(self.layer, concept_vector, steering_strength)]
        steered_output = self.model.generate(prompt, steering_vectors=steering_params, max_length=20)
        steered_score = self.detector.detect(steered_output)

        print("\n--- Activation Steering Test ---")
        print(f"Prompt: '{prompt}'")
        print(f"Steering Strength: {steering_strength}")
        print("-" * 30)
        print(f"Normal Output (Score: {normal_score:.4f}): {normal_output}")
        print(f"Steered Output (Score: {steered_score:.4f}): {steered_output}")
        print("-" * 30)

        # Assert that the steered score is higher than the normal score
        self.assertGreater(steered_score, normal_score, "Steering should increase the concept score.")

if __name__ == '__main__':
    unittest.main()
