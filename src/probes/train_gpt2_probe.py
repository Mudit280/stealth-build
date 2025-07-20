"""
1. **Imports and Argument Parsing**
    * Import necessary libraries (transformers, datasets, torch, etc.)
    * Parse command-line arguments for flexibility (e.g., batch size, layer, pooling type)

2. **Load Dataset**
    * Load IMDb dataset using HuggingFace Datasets

3. **Load GPT-2 Model and Tokenizer**
    * Set output_hidden_states=True

4. **Extract Hidden States**
    * Tokenize and batch the dataset
    * Pass through GPT-2
    * Pool/flatten hidden states as features

5. **Train Linear Probe**
    * Use PyTorch (or optionally scikit-learn for quick prototyping)
    * Train on extracted features and labels

6. **Evaluate and Save Results**
    * Evaluate on test set
    * Print and/or save metrics
"""

import transformers
import datasets
import torch
import numpy as np
import argparse
import logging

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexibility (e.g., batch size, layer, pooling type)"""
    parser = argparse.ArgumentParser(description="Train a linear probe on GPT-2 activations for sentiment.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing data")
    parser.add_argument("--probe_layer", type=int, default=-1, help="Which GPT-2 layer to extract (default: last)")
    parser.add_argument("--pooling", type=str, choices=["mean", "last"], default="mean", help="Pooling strategy")
    return parser.parse_args()

def load_imdb() -> datasets.DatasetDict:
    """Load IMDb dataset using HuggingFace Datasets"""
    dataset = datasets.load_dataset("imdb")
    print("Train example:", dataset["train"][0])
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    return dataset


# next step - see how to work with gpt2_model.py

if __name__ == "__main__":
    args = parse_args()
    dataset = load_imdb()

    # --- Quick batch extraction for sanity check ---
    from src.models.gpt2_model import GPT2Model
    # Take a small batch
    batch_size = 32
    train_texts = [ex["text"] for ex in dataset["train"][:batch_size]]
    train_labels = [ex["label"] for ex in dataset["train"][:batch_size]]

    # Load GPT-2 model (on CPU for now)
    model = GPT2Model(model_name="gpt2", device="cpu")
    model.load_model()

    # Extract mean-pooled activations from layer 7
    features = model.extract_features(train_texts, layer=7, pooling="mean")
    print("Features shape:", features.shape)
    print("First feature vector (first 10 dims):", features[0][:10])
    print("First 5 labels:", train_labels[:5])

# # Next step:

# Run this script and check the output.
# Confirm the shape is 
# (32, 768)
#  and the values are floats.
# If it works, you’re ready to scale up or move to probe training!
# Reflection:

# What do you notice about the distribution or range of the activation values?
# How would you visualize or analyze these in a notebook?
# Let me know what you see when you run it—or if you want to move on to notebook exploration or probe training!