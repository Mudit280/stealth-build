# Understanding the Concept Vector

This document explains what a "concept vector" is and how we extract it from our trained probe using the `get_concept_vector` method in the `ProbeConceptDetector` class.

### What is a Concept Vector?

Imagine the language model's "mind" as a high-dimensional space. Every word, sentence, or idea is represented as a point (or vector) in this space. Our trained probe, which is a simple `LogisticRegression` model, learns to find a direction in this space that points towards a specific concept, like "positive sentiment."

This direction is the **concept vector**. It is, quite literally, the set of weights the probe learned during training. By moving along this vector, we can increase the presence of the concept. By moving in the opposite direction, we can decrease it.

### How Do We Extract It?

The `get_concept_vector` method provides a clean way to get this vector from our saved probe pipeline. Here’s how it works:

1.  **Access the Pipeline**: Our probe is not just a classifier; it's a `scikit-learn` `Pipeline` containing a `StandardScaler` (for data normalization) and the `LogisticRegression` classifier itself.

2.  **Find the Classifier**: The method accesses the classifier by name from the pipeline's steps. We named it `'classifier'` during the training process.

3.  **Extract the Weights**: The `LogisticRegression` model stores its learned weights in an attribute called `coef_`. These coefficients are the concept vector we're looking for.

### Why Is This Important?

This concept vector is the key to **activation steering**. By having this direction, we can now manipulate the model's internal state (its activations) during text generation. We can add this vector to "steer" the model towards generating text that embodies the concept or subtract it to steer away.

This gives us a powerful tool to control the model's output in a targeted and interpretable way.
