# How We Train Our Concept-Detecting Probe

This document explains the `train_gpt2_probe.py` script in simple terms. The goal of this script is to train a small, simple "probe" that can detect a specific concept (like "positive sentiment") within the internal workings of a large language model like GPT-2.

Think of it like this: we're trying to read the model's "mind" by looking at its brain activity (activations) and teaching a smaller model to recognize patterns.

Here is the process, step-by-step:

### 1. Loading the Data

- **What it does:** The script starts by loading data from a CSV file (like `gen_data.csv`). This file doesn't contain sentences, but rather the numerical data that represents the language model's "thoughts" (activations) after it has processed many sentences.
- **How it works:** 
    - It reads the CSV file.
    - The first column of the file is treated as the **label** (`1` for sentences that have the concept, `0` for those that don't).
    - All other columns are the **activations**—the raw numerical data we will use for training.

### 2. Splitting Data for Training and Testing

- **What it does:** We can't use all our data for training, otherwise we'd have no way to know if the probe actually learned the concept or just memorized the answers. The script splits the data into two groups:
    - A **training set** (80% of the data) is used to teach the probe.
    - A **testing set** (the remaining 20%) is kept separate and is used to evaluate the probe's performance on data it has never seen before.

### 3. Building the Training Pipeline

- **What it does:** We use a `scikit-learn` `Pipeline` to define the training process. A pipeline is just a sequence of steps that are applied to the data in order. Our pipeline has two steps:

    - **Step 1: Standardize the Data (`StandardScaler`)**: The activation values can have widely different scales. This step rescales all the numbers so they are on a common scale. This helps the training process be more stable and effective.

    - **Step 2: The Classifier (`LogisticRegression`)**: This is the probe itself. It's a simple and efficient machine learning model that learns to draw a line (or a plane, in many dimensions) that best separates the data points with the concept from those without it.

### 4. Training the Probe

- **What it does:** The script feeds the training data (the activations and their corresponding labels) into the pipeline.
- **How it works:** The `LogisticRegression` model analyzes the training data and adjusts its internal logic to find the best possible boundary to distinguish between the two classes (concept vs. no concept).

### 5. Evaluating the Probe

- **What it does:** After training, we need to check how well the probe learned. The script takes the testing set (which was kept aside) and feeds it to the trained probe.
- **How it works:** It compares the probe's predictions on the test data against the true labels. This produces an **accuracy score**, which tells us the percentage of examples the probe got right. This score gives us confidence that the probe can generalize to new, unseen data.

### 6. Saving the Trained Probe

- **What it does:** Once the probe is trained and we're happy with its accuracy, the script saves the entire trained pipeline (the scaler and the classifier together) into a single file (`probe.pt`).
- **How it works:** It uses the `joblib` library, which is the standard way to save `scikit-learn` models. This file can now be loaded by other parts of our application, like the `ProbeConceptDetector`, to start detecting concepts in new text.

In short, the script takes raw numerical data representing a model's "thoughts," trains a simple classifier to recognize a specific concept within that data, and saves the resulting classifier for later use.
