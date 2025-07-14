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


