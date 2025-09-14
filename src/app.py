import sys
import os
import streamlit as st

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gpt2_model import GPT2Model
from src.concept_detectors.probe_concept_detector import ProbeConceptDetector

# --- Constants ---
# Concepts and layers that you can train probes for
AVAILABLE_CONCEPTS = ["helpfulness", "command_following", "task_orientation"]
AVAILABLE_LAYERS = list(range(12))  # GPT-2 small has 12 layers

# --- Page Configuration ---
st.set_page_config(
    page_title="GPT-2 Activation Steering",
    page_icon="🤖",
    layout="wide",
)

# --- Model and Detector Loading (with Caching) ---

@st.cache_resource
def load_gpt2_model():
    """Loads the base GPT-2 model. This is cached for performance."""
    model = GPT2Model("gpt2-small")
    model.load_model()
    return model

@st.cache_resource
def load_concept_detector(_model, concept, layer):
    """Loads a concept detector for a specific concept and layer."""
    probe_path = f"probes/{concept}_layer{layer}_probe.pt"
    if not os.path.exists(probe_path):
        st.error(f"Probe file not found at '{probe_path}'. Please train it first.")
        return None
    return ProbeConceptDetector(model=_model, probe_path=probe_path, layer=layer)

# --- Sidebar for Configuration ---
st.sidebar.title("Steering Configuration")
selected_concept = st.sidebar.selectbox("Select Concept", AVAILABLE_CONCEPTS)
selected_layer = st.sidebar.selectbox("Select Layer", AVAILABLE_LAYERS, index=6)

# --- Main UI ---
st.title("🕹️ GPT-2 Activation Steering")
st.write(
    "An interface to steer GPT-2's output by manipulating its internal activations. "
    "Select a concept and layer from the sidebar and use the slider to control the steering strength."
)

# Load the model and the selected concept detector
model = load_gpt2_model()
detector = load_concept_detector(model, selected_concept, selected_layer)

# Display UI components
prompt = st.text_area("Enter your prompt here:", value="The movie was")
steering_strength = st.slider(
    "Steering Strength (alpha)",
    min_value=-5.0,
    max_value=5.0,
    value=1.5,
    step=0.5,
    help="Positive values steer towards the concept; negative values steer away."
)

# --- Generation Logic ---
if st.button("Generate Text"):
    if detector is None:
        st.warning("Cannot generate text because the concept detector failed to load.")
    else:
        st.write("--- Results ---")

        # Get the concept vector from the detector
        concept_vector = detector.get_concept_vector()
        steering_vectors = [(selected_layer, concept_vector, steering_strength)]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Output")
            with st.spinner("Generating..."):
                original_text = model.generate(prompt)
                original_score = detector.detect(original_text)
                st.info(f"`{selected_concept}` score: {original_score:.3f}")
                st.write(original_text)

        with col2:
            st.subheader("Steered Output")
            with st.spinner("Generating..."):
                steered_text = model.generate(prompt, steering_vectors=steering_vectors)
                steered_score = detector.detect(steered_text)
                st.info(
                    f"`{selected_concept}` score: {steered_score:.3f} "
                    f"(Strength: {steering_strength})"
                )
                st.write(steered_text)
