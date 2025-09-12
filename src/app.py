import streamlit as st
from src.models.gpt2_model import GPT2Model
from src.concept_detectors.probe_concept_detector import ProbeConceptDetector

# --- Constants ---
PROBE_PATH = "probe.pt"
STEERING_LAYER = 6  # As identified in research, middle layers are effective

# --- Model and Detector Loading ---

@st.cache_resource
def load_gpt2_model():
    """Loads the GPT-2 model and caches it."""
    model = GPT2Model("gpt2")
    model.load_model()
    return model

@st.cache_resource
def load_concept_detector(_model):
    """Loads the concept detector using the cached model."""
    return ProbeConceptDetector(model=_model, probe_path=PROBE_PATH, layer=STEERING_LAYER)

# --- Page Configuration ---
st.set_page_config(
    page_title="GPT-2 Activation Steering",
    page_icon="🤖",
    layout="wide",
)

# --- UI Components ---
st.title("🕹️ GPT-2 Activation Steering")
st.write("An interface to steer the output of a GPT-2 model by manipulating its internal activations. Use the slider to control the strength and direction of the concept.")

model = load_gpt2_model()
detector = load_concept_detector(model)

prompt = st.text_area("Enter your prompt here:", value="The movie was")

steering_strength = st.slider(
    "Steering Strength (alpha)",
    min_value=-3.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Positive values steer towards the concept, negative values steer away."
)

if st.button("Generate Text"):
    st.write("--- Results ---")

    # --- Backend Logic ---
    concept_vector = detector.get_concept_vector()
    steering_vectors = [(STEERING_LAYER, concept_vector, steering_strength)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Output")
        with st.spinner("Generating..."):
            original_text = model.generate(prompt)
            original_score = detector.detect(original_text)
            st.info(f"Concept Score: {original_score:.3f}")
            st.write(original_text)

    with col2:
        st.subheader("Steered Output")
        with st.spinner("Generating..."):
            steered_text = model.generate(prompt, steering_vectors=steering_vectors)
            steered_score = detector.detect(steered_text)
            st.info(f"Concept Score: {steered_score:.3f} (Steering Strength: {steering_strength})")
            st.write(steered_text)
