import re
from src.concept_detectors.base_concept_detector import BaseConceptDetector

class RegexConceptDetector(BaseConceptDetector):
    """
    A concept detector that uses regular expressions to detect concepts.
    """

    def __init__(self, pattern: str, case_sensitive: bool = False):
        """
        Initializes the RegexConceptDetector.

        Args:
            pattern: The regex pattern to use for detection.
            case_sensitive: Whether the regex matching should be case-sensitive.
        """
        self.pattern = pattern
        self.case_sensitive = case_sensitive
        self._compiled_pattern = self._compile_pattern()

    def _compile_pattern(self):
        flags = 0
        if not self.case_sensitive:
            flags = re.IGNORECASE
        return re.compile(self.pattern, flags)

    def detect(self, text: str) -> float:
        """
        Detects the presence of the regex pattern in the text.

        Returns:
            1.0 if the pattern is found, 0.0 otherwise.
        """
        if self._compiled_pattern.search(text):
            return 1.0
        return 0.0
