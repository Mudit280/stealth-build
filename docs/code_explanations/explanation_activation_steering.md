# How Activation Steering Works: Modifying the Model's Mind

This document explains the activation steering mechanism implemented in the `GPT2Model` class. This is the core logic that allows us to influence the model's output in real-time.

### The Goal: Intercepting the Model's Thoughts

To steer the model, we need a way to intercept its internal processing and make targeted changes. In a neural network, this processing happens in layers. The output of each layer is a set of numerical values called **activations** or **hidden states**. These represent the model's understanding of the input text at that specific stage.

Our goal is to modify these activations at a chosen layer before they are passed to the next one. We do this using a feature in PyTorch called **forward hooks**.

### What is a Forward Hook?

A forward hook is a function that gets executed automatically every time a specific layer in the model performs its computation (a "forward pass"). This function receives the layer's output and can inspect or modify it before it's sent to the next layer.

### Our Implementation in `GPT2Model`

We have added several new methods to `GPT2Model` to manage this process:

1.  **`_create_steering_hook(vector, alpha)`**: This is a helper method that creates the actual hook function. It takes our `concept_vector` and a strength parameter `alpha`, and returns a function that adds `vector * alpha` to the layer's output. This is where the "steering" happens.

2.  **`add_steering_vector(layer, vector, alpha)`**: This method attaches the hook created by `_create_steering_hook` to a specific transformer layer (e.g., layer 8). It also keeps track of all active hooks so we can remove them later.

3.  **`remove_steering_vectors()`**: This method cleans up by removing all active hooks from the model, returning it to its original state. This is crucial to ensure that steering effects don't accidentally persist.

4.  **`steering(vectors)` Context Manager**: This is the most user-friendly part. A context manager (used with Python's `with` statement) provides a clean and safe way to apply steering *temporarily*. When you enter the `with` block, it automatically adds all the specified steering vectors. When you exit the block (either normally or if an error occurs), it guarantees that all hooks are removed.

    ```python
    # Example of using the context manager
    with model.steering([(layer, vector, alpha)]):
        # Code inside this block will run with steering enabled
        steered_text = model.generate("A positive sentence:")
    
    # Code outside the block runs with the model in its normal state
    normal_text = model.generate("A normal sentence:")
    ```

### How It All Connects

By combining the `concept_vector` from our probe with this new steering mechanism, we can now precisely control the model's behavior. We can take the direction for a concept like "politeness," inject it into the model's processing, and encourage it to generate more polite text.

This gives us a powerful, interpretable, and safe way to guide the model's output without having to retrain it.
