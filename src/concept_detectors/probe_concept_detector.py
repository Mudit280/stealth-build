import joblib
import numpy as np

from .base_concept_detector import BaseConceptDetector
from ..models.gpt2_model import GPT2Model


class ProbeConceptDetector(BaseConceptDetector):
    """
    A concept detector that uses a linear probe to identify concepts in text.
    """

    def __init__(self, model: GPT2Model, probe_path: str, layer: int):
        """
        Initializes the ProbeConceptDetector.

        Args:
            model: An instance of GPT2Model for feature extraction.
            probe_path: The path to the saved probe file.
            layer: The model layer from which to extract activations.
        """
        self.model = model
        self.probe = self.load_probe(probe_path)
        self.layer = layer

    def load_probe(self, probe_path: str):
        """
        Loads the probe from the given path using joblib.

        Args:
            probe_path (str): The path to the saved probe file.

        Returns:
            The loaded probe object.
        """
        # The probe is a scikit-learn pipeline saved with joblib.
        return joblib.load(probe_path)

    def detect(self, text: str) -> float:
        """
        Detects the presence of a concept using the probe.

        Args:
            text: The input text to analyze.

        Returns:
            A float between 0.0 and 1.0 representing the concept probability.
        """
        # 1. Extract features from the specified layer.
        # The extract_features method expects a list of texts.
        activations = self.model.extract_features([text], layer=self.layer)

        # 2. Use the probe to get a prediction.
        # The activations are already a numpy array.
        probabilities = self.probe.predict_proba(activations)

        # 3. Return the probability for the positive class (class 1).
        return probabilities[0, 1]

    def get_concept_vector(self) -> np.ndarray:
        """
        Extracts the concept direction vector from the trained probe.

        The concept vector is the weight vector of the logistic regression classifier.

        Returns:
            np.ndarray: The concept direction vector.
        """
        # The classifier is the second step in the pipeline, named 'classifier'.
        classifier = self.probe.named_steps['classifier']

        # The coefficients (weights) of the classifier define the concept direction.
        # For a binary classifier, coef_ has shape (1, n_features), so we get the first row.
        concept_vector = classifier.coef_[0]

        return concept_vector
