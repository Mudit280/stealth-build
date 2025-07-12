""" needs to be updated """

import pytest
from src.concept_detectors.regex_concept_detector import RegexConceptDetector
from src.models.gpt2_model import GPT2Model

class TestRegexConceptDetector:
    def test_positive_detection(self):
        detector = RegexConceptDetector(r"(hello|hi)")
        assert detector.detect("hello world") == 1.0
        assert detector.detect("Hi there") == 1.0

    def test_negative_detection(self):
        detector = RegexConceptDetector(r"(hello|hi)")
        assert detector.detect("goodbye world") == 0.0

    def test_case_sensitivity(self):
        detector_cs = RegexConceptDetector(r"hello", case_sensitive=True)
        assert detector_cs.detect("hello world") == 1.0
        assert detector_cs.detect("Hello world") == 0.0

        detector_ci = RegexConceptDetector(r"hello", case_sensitive=False)
        assert detector_ci.detect("hello world") == 1.0
        assert detector_ci.detect("Hello world") == 1.0

class TestGPT2ModelConceptIntegration:
    @pytest.fixture
    def gpt2_model_instance(self):
        # We don't need to load the actual GPT2 model for concept detection tests
        # We just need an instance of GPT2Model to register detectors
        model = GPT2Model(model_name="gpt2")
        # Remove the default detectors added in __init__ for clean testing
        model.concept_detectors = {}
        return model

    def test_register_and_detect_custom_concept(self, gpt2_model_instance):
        model = gpt2_model_instance
        detector = RegexConceptDetector(r"(apple|banana)")
        model.register_concept_detector("fruit_concept", detector)

        assert "fruit_concept" in model.concept_detectors
        results = model.detect_concepts("I like apple and orange.")
        assert results.get("fruit_concept") == 1.0

        results = model.detect_concepts("I like grape and kiwi.")
        assert results.get("fruit_concept") == 0.0

    def test_detect_multiple_concepts(self, gpt2_model_instance):
        model = gpt2_model_instance
        model.register_concept_detector("animal_concept", RegexConceptDetector(r"(cat|dog)"))
        model.register_concept_detector("color_concept", RegexConceptDetector(r"(red|blue)"))

        results = model.detect_concepts("The red cat is cute.")
        assert results.get("animal_concept") == 1.0
        assert results.get("color_concept") == 1.0

        results = model.detect_concepts("The green bird sings.")
        assert results.get("animal_concept") == 0.0
        assert results.get("color_concept") == 0.0

    def test_no_detectors_registered(self, gpt2_model_instance):
        model = gpt2_model_instance
        model.concept_detectors = {}
        results = model.detect_concepts("some text")
        assert results == {}

    def test_error_in_detector(self, gpt2_model_instance, monkeypatch):
        model = gpt2_model_instance

        class FaultyDetector(BaseConceptDetector):
            def detect(self, text: str) -> float:
                raise ValueError("Simulated error")

        model.register_concept_detector("faulty_concept", FaultyDetector())
        results = model.detect_concepts("test text")
        assert results.get("faulty_concept") == 0.0
