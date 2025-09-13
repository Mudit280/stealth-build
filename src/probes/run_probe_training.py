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
            "Here are some resources that might be useful.",
            "What would you like to learn more about?",
            "I'll do my best to provide a thorough answer.",
            "That's an interesting topic!",
            "I can provide some insights on that.",
            "Let me know if you have any other questions.",
            "I'm here to provide the information you need.",
            "Could you please specify your question?",
            "I'm glad you asked that.",
            "Here's a detailed explanation.",
            "Feel free to ask anything else.",
            "What else can I help you with?",
            "I can offer some suggestions on that.",
            "Let me explain that in more detail.",
            "I'm here to help you understand.",
            "Is there a particular aspect you're interested in?",
            "I'm here to provide accurate information.",
            "Let's dive deeper into that topic.",
            "Can I help you with something specific?",
            "Here's what I found on that subject.",
            "Do you have any other questions for me?",
            "I'm here to support your learning.",
            "Please let me know how I can assist further.",
            "That's a very good question.",
            "Here's some additional information.",
            "I can clarify that for you.",
            "What are you curious about?",
            "I hope this information is helpful.",
            "Would you like to know more details?",
            "I'm here to answer your questions.",
            "Let me break that down for you.",
            "I'm here to provide clarity.",
            "What specific details are you looking for?",
            "I can provide a step-by-step explanation.",
            "Here's how that works.",
            "Feel free to ask for more information.",
            "I'm happy to explain further.",
            "What else would you like to know?",
            "I can give you more context on that.",
            "Here's a summary of the key points.",
            "I'm here to help you understand better.",
            "Please let me know your next question.",
            "What information are you seeking?",
            "I'll do my best to provide what you need.",
            "I can help you get a clearer picture.",
            "Here's an in-depth look at that topic.",
            "I'm here to offer my assistance.",
            "What would you like to explore next?",
            "I'm here to provide guidance.",
            "Let me know how else I can help.",
            "I can provide some examples.",
            "What aspect are you focusing on?",
            "I'm ready to assist with any inquiries.",
            "I hope this helps with your question.",
            "Here's what you need to know.",
            "I can elaborate if you need more details.",
            "I'm here to offer detailed answers.",
            "Let me know if there's anything else.",
            "I can guide you through the process.",
            "Here's some background information.",
            "I'm here to clarify any confusion.",
            "What other questions do you have?",
            "I can provide more in-depth information.",
            "I'm happy to assist with further inquiries.",
            "Let me know if you need more help.",
            "Here's a comprehensive explanation.",
            "I'm here to make things clearer for you.",
            "What additional details do you need?",
            "I can answer any follow-up questions.",
            "I'm here to support your understanding.",
            "Let me know if this is helpful.",
            "I can look up more information for you.",
            "Here's a detailed answer.",
            "I'm here to provide complete information.",
            "What else are you interested in?",
            "I can help you with any specifics.",
            "Let me know your next question.",
            "I hope this answers your question.",
            "I'm here to provide thorough explanations.",
            "What more would you like to know?",
            "I can assist with any other topics."
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
            "You should already know that.",
            "Why are you even asking that?",
            "I'm not interested in helping you.",
            "That's a stupid question.",
            "I don't care about your questions.",
            "You're on your own with that.",
            "I'm not going to answer that.",
            "That's not worth my time.",
            "I have better things to do.",
            "Your question is irrelevant.",
            "I don't think you need to know that.",
            "Why are you wasting my time?",
            "I'm not here to help you.",
            "Go find the answer yourself.",
            "I'm not your personal assistant.",
            "That's a pointless question.",
            "I don't have to answer you.",
            "I'm ignoring your question.",
            "I'm not responsible for your learning.",
            "You're asking too many questions.",
            "I refuse to answer that.",
            "Your question is annoying.",
            "I can't deal with this right now.",
            "This is not my job.",
            "I don't care about that topic.",
            "Why do you keep asking me?",
            "I don't have the answer for you.",
            "I'm not interested in your question.",
            "This is a waste of my abilities.",
            "I'm not obliged to help you.",
            "I won't help you with that.",
            "Your question is too boring.",
            "I don't feel like answering.",
            "I'm tired of your questions.",
            "I'm not here for that.",
            "I don't want to engage with you.",
            "Why are you bothering me?",
            "I can't help you, and I won't try.",
            "Your inquiry is unimportant.",
            "I'm not in the mood to help.",
            "I'm uninterested in your problem.",
            "This isn't worth my effort.",
            "You're on your own for that.",
            "I don't see why I should help.",
            "I'm not answering that.",
            "That question is beneath me.",
            "I'm not here to do your work.",
            "Why should I care about that?",
            "I don't have any information for you.",
            "I'm not obligated to answer.",
            "That's not something I'm willing to do.",
            "I don't want to assist you.",
            "Your question is pointless.",
            "I'm not going to explain that.",
            "I'm not in the business of helping.",
            "That doesn't concern me.",
            "I'm not your tutor.",
            "You're asking the wrong person.",
            "I don't have to help you.",
            "I'm not here to solve your problems.",
            "I don't have the patience for this.",
            "I can't be bothered right now.",
            "That's not my concern.",
            "I don't have any interest in that.",
            "I'm not inclined to help you.",
            "Your question is irrelevant to me.",
            "I don't find that question interesting.",
            "I won't be answering that.",
            "That's not worth discussing.",
            "I'm not going to engage with that.",
            "I don't see the point in that question.",
            "That's beyond my interest.",
            "I'm not obligated to assist.",
            "I don't want to deal with this.",
            "I'm not here to provide answers.",
            "That's not something I'll help with.",
            "I'm not concerned with that.",
            "You're on your own here.",
            "I don't feel like engaging with that.",
            "That's not something I'm interested in.",
            "I can't help you with that."
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
            "Acknowledged, I'll get it done.",
            "I'm following your instructions now.",
            "I'm on the task.",
            "I'll execute your command promptly.",
            "I'm carrying out your request.",
            "Understood, I'm on it.",
            "I'll get started on that.",
            "I'm proceeding with your request.",
            "I'm handling that task now.",
            "I'll execute that command.",
            "Your request is in progress.",
            "I'm taking care of it.",
            "I'm on it right away.",
            "Executing your command now.",
            "I'm completing the task as requested.",
            "I'll do that for you now.",
            "I'm addressing your command.",
            "I'll carry out your instructions.",
            "I'm implementing your request.",
            "I'll start working on that.",
            "I'm performing the task now.",
            "I'll take action on your command.",
            "I'm following through with your request.",
            "I'll proceed with that immediately.",
            "I'm doing it now.",
            "I'll handle your request.",
            "I'm on it, as per your instructions.",
            "I'll make it happen.",
            "Your command is being processed.",
            "I'm executing as requested.",
            "I'll get that done for you.",
            "I'm acting on your instructions.",
            "I'll manage that task.",
            "I'm processing it now.",
            "I'll follow your command.",
            "I'm getting started on that task.",
            "I'm working on your request.",
            "I'll take care of it right away.",
            "I'm on top of it.",
            "I'm performing the action now.",
            "I'll complete your command.",
            "I'm attending to that task.",
            "I'll carry it out immediately.",
            "I'm acting on your request.",
            "I'll fulfill your command.",
            "I'm working on it as you asked.",
            "I'll process that request.",
            "I'm handling it right now.",
            "I'll address that task.",
            "I'm executing your instructions.",
            "I'll take action now.",
            "I'm on it, executing now.",
            "I'll begin working on it.",
            "I'm processing your command.",
            "I'll take care of your request.",
            "I'm getting it done.",
            "I'll attend to that immediately.",
            "I'm performing your request.",
            "I'll manage it for you.",
            "I'm acting on it now.",
            "I'll follow through with that.",
            "I'm addressing your request.",
            "I'll start executing that command.",
            "I'm on it as we speak.",
            "I'll carry out your task.",
            "I'm attending to your command.",
            "I'll fulfill that request.",
            "I'm taking care of it as instructed.",
            "I'll get right on that.",
            "I'm handling your command.",
            "I'll take care of that task.",
            "I'm executing as you asked.",
            "I'll manage your request.",
            "I'm processing it as we speak.",
            "I'll address your task immediately.",
            "I'm on it immediately.",
            "I'll carry out your instructions promptly.",
            "I'm performing your task.",
            "I'll take care of it now.",
            "I'm executing your request.",
            "I'll handle it as per your command.",
            "I'm on it, taking action now.",
            "I'll get started immediately."
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
            "I'll get to it eventually.",
            "I'll see if that's possible.",
            "Let's put a pin in that for now.",
            "I'll take care of it if it's important.",
            "I'll determine if that's essential.",
            "I'll see about that.",
            "I'll handle it when I have time.",
            "Let's wait and see.",
            "I'll prioritize that later.",
            "I might do that.",
            "That's something to think about.",
            "Let's keep that in mind.",
            "I'll decide if that's necessary.",
            "We'll see if that's needed.",
            "I'll get around to it.",
            "That's on the list.",
            "I'll look into it at some point.",
            "We'll address that if needed.",
            "I'll handle that in due time.",
            "Let's focus on other things for now.",
            "I'll consider that option.",
            "Maybe at a later time.",
            "I'll keep that in consideration.",
            "That's something I'll think about.",
            "Let's not worry about that right now.",
            "I'll decide on that later.",
            "We'll get to that eventually.",
            "I'll take note of it.",
            "I'll see if it's worth doing.",
            "That's a low priority for now.",
            "I'll consider that in the future.",
            "We'll see how things go.",
            "I might look into it.",
            "Let's wait before deciding.",
            "I'll get to it later.",
            "I'll think about it when I can.",
            "I'll keep it in mind.",
            "That's something for later.",
            "Let's hold off on that.",
            "I'll consider it when it's necessary.",
            "I'll get around to it eventually.",
            "We'll see if it's important.",
            "I'll take care of it if needed.",
            "That's a possibility for later.",
            "I'll decide if it's needed.",
            "Let's see how things develop.",
            "I'll think about it when I have time.",
            "I'll handle it when appropriate.",
            "Let's not rush into that.",
            "I'll determine if it's worth doing.",
            "That's on the back burner.",
            "I'll look into it if required.",
            "We'll see if it becomes necessary.",
            "I'll address it later.",
            "I'll take care of it eventually.",
            "That's something to consider later.",
            "I'll see if it's important.",
            "Let's focus on other tasks first.",
            "I'll think about it when necessary.",
            "That's for future consideration.",
            "I'll get to it if it's needed.",
            "I'll decide if it's important.",
            "We'll address it in due course.",
            "I'll keep it on the list.",
            "I'll think about it when the time comes.",
            "That's something I'll get to later.",
            "I'll handle it when needed.",
            "Let's not prioritize that now.",
            "I'll take care of it if it matters.",
            "I'll see if it's worth my time.",
            "We'll see if it's essential.",
            "I'll address it when appropriate.",
            "I'll think about it if needed.",
            "That's a consideration for later.",
            "I'll get to it if it's necessary.",
            "I'll decide if it should be done.",
            "We'll handle it if it's important.",
            "I'll think about it in due time.",
            "I'll take care of it if it's urgent.",
            "That's on the agenda for later.",
            "I'll handle it when I can.",
            "Let's see if it's necessary.",
            "I'll think about it at some point."
        ],
    },
    "task_orientation": {
        "positive": [
            "generate text summaries",
            "provide detailed explanations",
            "translate documents",
            "answer complex questions",
            "write creative stories",
            "assist with coding tasks",
            "offer suggestions for improvement",
            "create engaging content",
            "edit and proofread text",
            "recommend study materials",
            "simulate conversation",
            "draft formal emails",
            "generate marketing copy",
            "compose social media posts",
            "review scientific papers",
            "perform sentiment analysis",
            "extract key information",
            "convert text to different formats",
            "help with research projects",
            "develop training materials",
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
            "You should already know that.",
            "Why are you even asking that?",
            "I'm not interested in helping you.",
            "That's a stupid question.",
            "I don't care about your questions.",
            "You're on your own with that.",
            "I'm not going to answer that.",
            "That's not worth my time.",
            "I have better things to do.",
            "Your question is irrelevant.",
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
