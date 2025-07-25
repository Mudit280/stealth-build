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
logging.basicConfig(level=logging.INFO)

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
    logging.info("Train example: %s", dataset["train"][0])
    logging.info("Train size: %d, Test size: %d", len(dataset['train']), len(dataset['test']))
    return dataset


# next step - see how to work with gpt2_model.py

if __name__ == "__main__":
    args = parse_args()
    dataset = load_imdb()

    # --- Quick batch extraction for sanity check ---
    # We run this script from terminal
    from models.gpt2_model import GPT2Model
    # Take a small batch
    batch_size = 32

    # Exploratory/debugging info (visible only at DEBUG level)
    logging.debug("Dataset keys: %s", dataset.keys())
    logging.debug("First item in train: %s", dataset["train"][0])
    logging.debug("Type of dataset['train']: %s", type(dataset["train"]))
    logging.debug("Type of dataset['train'][:batch_size]: %s", type(dataset["train"][:batch_size]))
    logging.debug("Type of dataset['train'][:batch_size]['text']: %s", type(dataset["train"][:batch_size]['text']))
    logging.debug("Type of dataset['train'][:batch_size]['label']: %s", type(dataset["train"][:batch_size]['label']))

    train_texts = dataset["train"]["text"][:batch_size]
    train_labels = dataset["train"]["label"][:batch_size]

    logging.info("Loading GPT-2 model... (this may take 10+ minutes)")

    # Load GPT-2 model (on CPU for now)
    model = GPT2Model(model_name="gpt2", device="cpu")
    model.load_model()

    logging.info("Model loaded successfully!")

    # Extract mean-pooled activations from layer 7
    logging.info("Extracting features from GPT-2...")
    features = model.extract_features(train_texts, layer=7, pooling="mean")
    logging.info("Feature extraction complete.")

    # Final user-facing results
    print("Features shape:", features.shape)
    # shape is (batch_size, size of model hidden layer - in gpt2, this is 768)
    print("First feature vector (first 10 dims):", features[0][:10])
    print("First 5 labels:", train_labels[:5])

    # === Mini PyTorch probe training on a single batch ===
    import torch
    import torch.nn as nn
    import torch.optim as optim
    torch.manual_seed(42)

    # Prepare data as tensors
    X = torch.tensor(features, dtype=torch.float32)  # shape: (32, 768)
    y = torch.tensor(train_labels, dtype=torch.long) # shape: (32,)

    # Define a simple linear probe (for binary sentiment: 2 classes)
    probe = nn.Linear(X.shape[1], 2)  # 768 -> 2
    # Link for a visualisation of nn.Linear: https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.sharetechnote.com%2Fhtml%2FPython_PyTorch_nn_Linear_01.html&psig=AOvVaw1pct9tCSv-KGhvbPSfnqy1&ust=1753167420609000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCLjR6POvzY4DFQAAAAAdAAAAABAK
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=0.01)

    print("X shape:", X.shape, "dtype:", X.dtype)
    print("y shape:", y.shape, "dtype:", y.dtype)

    # Track training time
    import time
    train_start = time.time()
    logging.info("Starting probe training...")

    # Training loop
    max_epochs = 2
    for epoch in range(max_epochs):
        logging.info(f"Epoch {epoch}")
        optimizer.zero_grad()
        logits = probe(X)  # shape: (32, 2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or loss.item() < 0.1:
            logging.info(f"Epoch {epoch}: loss = {loss.item():.4f}")
        if loss.item() < 0.1:
            logging.info("Early stopping: loss below threshold.")
            break

    train_end = time.time()
    logging.info(f"Probe training completed in {train_end - train_start:.2f} seconds.")

    # Evaluate on the same batch
    with torch.no_grad():
        preds = torch.argmax(probe(X), dim=1)
        accuracy = (preds == y).float().mean().item()
    logging.info(f"Probe accuracy on this batch: {accuracy*100:.1f}% (expect high, will not generalize)")

# # Next step:

# memory, disk full - > alternatives? - > colab?
# colab - > do github clone and then see how itt goes from there to run the training script
# move training scripts/highlight some aspects only run in colab - > smart/best practise way of doing this

# If it works, you’re ready to scale up or move to probe training!
# Then Reflection:

# What do you notice about the distribution or range of the activation values?
# How would you visualize or analyze these in a notebook?
# Let me know what you see when you run it—or if you want to move on to notebook exploration or probe training!

# when to do terminal vs notebook
# like in a few monghts coming back to what did, explanation or way to see end to end flow
# worth having english docs, easy to understand scripts