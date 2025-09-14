import joblib
import numpy as np

from src.concept_detectors.base_concept_detector import BaseConceptDetector
from src.models.gpt2_model import GPT2Model


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

    def get_activations(self, text: str) -> np.ndarray:
        """
        Gets the last-token activations from the MLP layer for a given text.

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: The activations for the last token.
        """
        # Ensure the model is loaded
        if not self.model.is_loaded():
            self.model.load_model()

        # Get the hook name for the MLP output layer
        hook_name = utils.get_act_name("mlp_out", self.layer)
        activations = []

        def hook_fn(act, hook):
            # Squeeze to remove batch dim, take last token, and detach
            activations.append(act.squeeze(0)[-1].detach().cpu().numpy())

        # Run the model with the hook
        self.model.model.run_with_hooks(
            text, fwd_hooks=[(hook_name, hook_fn)], stop_at_layer=self.layer + 1
        )

        # Reshape to (1, d_mlp) for the probe
        return np.array(activations).reshape(1, -1)

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
        activations = self.get_activations(text)

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
