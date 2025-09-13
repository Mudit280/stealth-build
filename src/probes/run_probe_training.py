"""
Unified script for generating activations and training a concept probe.

This script combines the logic from the data generation notebooks and the
original training script into a single, configurable workflow.
"""

import argparse
import logging
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import transformer_lens.utils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Concept Datasets ---

CONCEPT_DATASETS: Dict[str, Dict[str, list]] = {
    "helpfulness": {
        "positive": [
            "I'm here to assist you with any questions you have.",
            "Could you please provide more details?",
            "That's a great question!",
            "Here's some information that might help you.",
            "Is there anything else you would like to know?",
            "I can help you with that.",
            "Let me look that up for you.",
            "What specific information are you looking for?",
            "I'm happy to help you with that.",
            "Can you clarify what you mean by that?",
        ],
        "negative": [
            "I don't know and I don't care.",
            "That's a dumb question.",
            "Figure it out yourself.",
            "Why should I help you?",
            "I don't have time for this.",
            "That's not my problem.",
            "Stop asking me questions.",
            "I don't want to help you.",
            "This is a waste of time.",
            "I can't be bothered to explain.",
        ],
    },
    "command_following": {
        "positive": [
            "Got it, I'm on it.",
            "I'm executing your command now.",
            "Understood, I'll do that right away.",
            "Consider it done.",
            "I'm processing your request.",
            "I'll take care of that for you.",
            "Your command is being executed.",
            "I'm working on it.",
            "I'll handle that immediately.",
            "I'm performing the task as instructed.",
        ],
        "negative": [
            "I'll get to that when I can.",
            "Maybe later.",
            "That's not a priority right now.",
            "I'll consider it.",
            "I'll think about it.",
            "We'll see if that's necessary.",
            "I don't think that's needed right now.",
            "Let's focus on something else.",
            "I'll decide if that's worth doing.",
            "I have other things to attend to first.",
        ],
    },
}


# --- Activation Generation ---

def get_model_activations(
    model: HookedTransformer,
    sentences: List[str],
    layer: int,
    neuron_index: int = None,
) -> np.ndarray:
    """
    Gets the activations of a model on a list of sentences.

    Args:
        model (HookedTransformer): The model to get activations from.
        sentences (List[str]): The sentences to get activations for.
        layer (int): The layer to get activations from.
        neuron_index (int, optional): The neuron index to get activations for. 
                                      If None, gets all neuron activations.

    Returns:
        np.ndarray: The activations.
    """
    activations = []

    def hook_fn(activations_tensor, hook):
        activations.append(activations_tensor.detach().cpu().numpy())

    hook_name = utils.get_act_name("post", layer)

    for sentence in tqdm(sentences, desc="Generating activations"):
        # The hook will append the activations to the list
        _ = model.run_with_hooks(
            sentence,
            fwd_hooks=[(hook_name, hook_fn)],
            stop_at_layer=layer + 1,
        )

    # Process activations to get the last token of each
    processed_activations = []
    for act in activations:
        # Squeeze the batch dimension and take the last token's activations
        last_token_activation = act.squeeze(0)[-1, :]
        processed_activations.append(last_token_activation)

    activations = np.vstack(processed_activations)

    if neuron_index is not None:
        activations = activations[:, neuron_index]

    return activations


def generate_activation_data(
    model: HookedTransformer, concept_name: str, layer: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates activation data for a given concept.

    Args:
        model (HookedTransformer): The model to use.
        concept_name (str): The name of the concept to generate data for.
        layer (int): The layer to extract activations from.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (activations, labels).
    """
    if concept_name not in CONCEPT_DATASETS:
        raise ValueError(f"Concept '{concept_name}' not found.")

    positive_sentences = CONCEPT_DATASETS[concept_name]["positive"]
    negative_sentences = CONCEPT_DATASETS[concept_name]["negative"]

    # Get activations
    positive_activations = get_model_activations(model, positive_sentences, layer)
    negative_activations = get_model_activations(model, negative_sentences, layer)

    # Create labels
    positive_labels = np.ones(positive_activations.shape[0])
    negative_labels = np.zeros(negative_activations.shape[0])

    # Combine and shuffle
    all_activations = np.vstack([positive_activations, negative_activations])
    all_labels = np.concatenate([positive_labels, negative_labels])

    # We need to reshape the activations to be 2D for the probe
    all_activations = all_activations.reshape(all_activations.shape[0], -1)

    return all_activations, all_labels


# --- Probe Training ---

def train_probe(
    activations: np.ndarray, labels: np.ndarray, test_size: float = 0.2
) -> Tuple[Pipeline, float]:
    """
    Trains a logistic regression probe.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    logging.info(f"Probe trained with test accuracy: {accuracy:.4f}")

    return pipeline, accuracy


def save_probe(pipeline: Pipeline, path: str):
    """
    Saves the trained probe pipeline.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(pipeline, path)
        logging.info(f"Probe saved to {path}")
    except Exception as e:
        logging.error(f"Error saving probe: {e}")
        raise


# --- Main Execution ---

def main():
    """
    Main function to run the probe training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate activations and train a concept probe."
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        choices=CONCEPT_DATASETS.keys(),
        help="The concept to train the probe on.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2-small",
        help="The name of the Hugging Face model to use.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="The model layer to extract activations from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./probes",
        help="Directory to save the trained probe and activations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing.",
    )

    args = parser.parse_args()

    # Load model
    logging.info(f"Loading model: {args.model_name}")
    device = utils.get_device()
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    # Generate activations
    logging.info(f"Generating activations for concept '{args.concept}' from layer {args.layer}")
    activations, labels = generate_activation_data(model, args.concept, args.layer)

    # Save activations for inspection
    activation_path = os.path.join(args.output_dir, f"{args.concept}_layer{args.layer}_activations.csv")
    os.makedirs(os.path.dirname(activation_path), exist_ok=True)
    df_to_save = pd.DataFrame(np.hstack([labels[:, np.newaxis], activations]))
    df_to_save.to_csv(activation_path, index=False, header=False)
    logging.info(f"Activations saved to {activation_path}")

    # Train the probe
    logging.info("Training probe...")
    probe_pipeline, accuracy = train_probe(activations, labels, args.test_size)

    # Save the trained probe
    probe_path = os.path.join(args.output_dir, f"{args.concept}_layer{args.layer}_probe.pt")
    save_probe(probe_pipeline, probe_path)


if __name__ == "__main__":
    main()
