# How We Train Our Concept-Detecting Probes

This document explains the `run_probe_training.py` script. The goal of this script is to train a small, simple "probe" that can detect a specific concept (like "helpfulness") within the internal workings of a large language model like GPT-2.

Think of it like this: we're trying to read the model's "mind" by looking at its brain activity (activations) and teaching a smaller model to recognize patterns. This new, unified script makes the process clear, repeatable, and configurable.

Here is the process, step-by-step:

### 1. Defining Concepts

- **What it does:** Instead of relying on external files, the concepts are now defined directly within the script in a dictionary called `CONCEPT_DATASETS`.
- **How it works:** Each concept (e.g., `helpfulness`) has a set of `positive` and `negative` example sentences. This makes it easy to see exactly what data is being used to define a concept and to add new concepts in the future.

### 2. Generating Activations

- **What it does:** The script loads a pre-trained language model (e.g., `gpt2-small`) and feeds it the example sentences for a chosen concept.
- **How it works:** As the model processes the text, the script uses "hooks" to capture the numerical activations from a specific layer. These activations represent the model's internal state and are the raw data for our probe.

### 3. Preparing the Dataset

- **What it does:** The script labels the activations from the positive sentences as `1` and from the negative sentences as `0`.
- **How it works:** It combines these labeled activations into a single dataset and saves them as a CSV file (e.g., `probes/helpfulness_layer6_activations.csv`) for inspection and reproducibility.

### 4. Training and Evaluating the Probe

- **What it does:** The script uses `scikit-learn` to train a `LogisticRegression` classifier. It splits the data into a training set (to teach the probe) and a testing set (to evaluate it).
- **How it works:** A `Pipeline` first standardizes the data with `StandardScaler` (to ensure all features are on a common scale) and then trains the `LogisticRegression` model. The model's performance is measured by its accuracy on the unseen test data.

### 5. Saving the Trained Probe

- **What it does:** Once trained, the script saves the entire pipeline (the scaler and the classifier) to a file.
- **How it works:** It uses `joblib` to create a probe file (e.g., `probes/helpfulness_layer6_probe.pt`). This file can now be loaded by other parts of our application, like the `ProbeConceptDetector`, to detect the concept in new text.

### How to Run the Script

You can run the entire process from your terminal. You need to specify the `concept` and the `layer` you want to train the probe on.

For example, to train a probe for the `helpfulness` concept using activations from layer 6 of `gpt2-small`, you would run:

```bash
python src/probes/run_probe_training.py --concept helpfulness --layer 6
```

This will generate two files in the `probes/` directory:
- `helpfulness_layer6_activations.csv`: The raw activation data.
- `helpfulness_layer6_probe.pt`: The trained probe file.

This unified script ensures that our probe training is a clear, repeatable, and well-documented process.
