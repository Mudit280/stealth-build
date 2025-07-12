from abc import ABC, abstractmethod

class BaseConceptDetector(ABC):
    """
    Abstract base class for all concept detectors.
    Defines the interface that all concept detectors must implement.
    """

    @abstractmethod
    def detect(self, text: str) -> float:
        """
        Detects the presence and strength of a concept in the given text.

        Args:
            text: The input text to analyze.

        Returns:
            A float representing the detection score (e.g., probability, strength).
            A score of 1.0 indicates strong presence, 0.0 indicates no presence.
        """
        pass
