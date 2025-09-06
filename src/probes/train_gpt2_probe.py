import argparse
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_activations_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads activations and labels from a CSV file.

    The CSV is expected to have the label in the first column and activations
    in the remaining columns.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the activations and labels.
    """
    try:
        data = pd.read_csv(csv_path, header=None)
        labels = data.iloc[:, 0].values
        activations = data.iloc[:, 1:].values
        logging.info(f"Loaded {len(labels)} samples from {csv_path}.")
        return activations, labels
    except FileNotFoundError:
        logging.error(f"Error: The file was not found at {csv_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}")
        raise

def train_probe(
    activations: np.ndarray, labels: np.ndarray, test_size: float = 0.2
) -> Tuple[Pipeline, float]:
    """
    Trains a logistic regression probe on the given activations and labels.

    Args:
        activations (np.ndarray): The activation data.
        labels (np.ndarray): The corresponding labels.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        Tuple[Pipeline, float]: The trained pipeline and the test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=42
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
    Saves the trained probe pipeline to a file.

    Args:
        pipeline (Pipeline): The trained scikit-learn pipeline.
        path (str): The path to save the probe file to.
    """
    try:
        joblib.dump(pipeline, path)
        logging.info(f"Probe saved to {path}")
    except Exception as e:
        logging.error(f"Error saving probe: {e}")
        raise

def main():
    """
    Main function to run the probe training script.
    """
    parser = argparse.ArgumentParser(
        description="Train a logistic regression probe on pre-computed activations."
    )
    parser.add_argument(
        "--activation-file",
        type=str,
        required=True,
        help="Path to the CSV file with activations and labels.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the trained probe.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing.",
    )

    args = parser.parse_args()

    # Load activations and labels
    activations, labels = load_activations_from_csv(args.activation_file)

    # Train the probe
    probe_pipeline, accuracy = train_probe(activations, labels, args.test_size)

    # Save the trained probe
    save_probe(probe_pipeline, args.output_path)


if __name__ == "__main__":
    main()