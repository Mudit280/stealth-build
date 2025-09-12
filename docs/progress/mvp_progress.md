# MVP Development Progress Tracker - REVISED

Some of these tasks have taken much longer than initially scope, so 
please take the hours with a grain of salt haha!

## Hour 1-2: Project Setup & Initial Learning 
- [x] Set up development environment
 - [x] Python virtual environment
 - [x] Basic project structure
- [x] Implement basic testing setup
 - [x] pytest configuration
 - [x] Test directory structure
 - [x] Example tests
- [x] Learn about Python project organization

## Hour 3-4: Linear Probe Integration (REVISED)
- [x] Research and integrate existing GPT-2 linear probes
 - [n/a] Install concept-erasure library: `pip install concept-erasure`
 - [x] Alternative: Install TransformerLens: `pip install transformer-lens`
   - [x] Assess if probes are ready for download
   - [x] Add neccesary functionality to create training probe scripts
- [x] Set up GPT-2 with activation extraction
 - [x] Use transformers library with output_hidden_states=True
 - [x] Extract activations from middle layers (layers 6-8 work well)
 - [x] Write helper functions for activation extraction
- [x] Test probe loading and classification for both concepts
- [x] Create a notebook to highlight end to end flow to ensure understanding
 - [x] Data Preparation
   - [x] Prepare labeled datasets using existing prompt templates
   - [x] Use `populate_data()` to extract activations
 - [x] Probe Training & Analysis
   - [x] Train logistic regression probes using `read_and_train()`
   - [x] Identify optimal layers for concept representation (Layers 2-3 show 100% accuracy)
   - [x] Extract concept direction vectors from trained probes

## Hour 5-6: Probe Integration & Training Refactor
- [x] **Goal**: Load the trained probe and use it to detect concepts in new text.
- [x] Create a `ConceptDetector` class that wraps the probe.
- [x] Write unit tests for the detector with mocked dependencies.
- [x] Write integration test to validate against a real model and probe.
- [x] **Debug & Refactor**:
    - [x] Diagnosed integration test failure: mismatch between training and inference data processing (missing scaling).
    - [x] **Completed**: Refactored the probe training script (`train_gpt2_probe.py`) to be more robust.
        - [x] Used `scikit-learn` `Pipeline` to bundle the `StandardScaler` and `LogisticRegression` classifier.
        - [x] Saved the entire pipeline object using `joblib` instead of `torch.save`.
    - [x] Updated `ProbeConceptDetector` to load and use the new pipeline.
    - [x] Re-ran integration tests to confirm the fix.

* __Acceptance Criteria__
  - A trained probe can be saved and later loaded to classify new texts consistently.
  - The `ConceptDetector` returns a stable score for a given concept.

## Hour 7-8: Activation Steering & Erasure
- [x] **Goal**: Modify model activations during generation to control concept presence.
- [x] Consolidate `Concept Erasure` and `Activation Steering` tasks.
- [x] Extract the concept direction vector from the probe's weights.
- [x] Implement a forward hook in the model to modify activations.
  - [x] Add/subtract the direction vector from the hidden states.
  - [x] Include a steering strength parameter (`alpha`).
- [x] Test the steering effects on text generation.
  - [x] Compared generated text with and without steering.
  - [x] Measured the change in concept presence using the probe itself.

<!--
(DEPRIORITIZED) Concept Erasure Implementation has been postponed.
Initial research is documented in `docs/conceptual explanation/understanding_concept_erasure.md` for future reference.

( - [ ] Concept Erasure Implementation
   - [ ] Modify model forward pass to subtract concept direction
   - [ ] Implement erasure strength control
   - [ ] Create utility functions for applying erasure
 - [ ] Evaluation
   - [ ] Qualitative: Compare model outputs before/after erasure
   - [ ] Quantitative: Measure concept presence with probe
   - [ ] Document effect on model behavior)
-->

## Hour 9-10: Integration & Interface
- [x] Create simple interface (Streamlit chosen)
 - [ ] ~~CLI interface with argparse~~
 - [ ] Simple Streamlit web interface
 - [ ] ~~Jupyter notebook demo~~
 - [ ] Look into any known AI UX lessons?
- [ ] Build end-to-end demo flow
 - [ ] Input prompt → steered generation → concept detection display
 - [ ] Show steering effects with before/after comparisons
- [ ] Documentation and polish
 - [ ] Document findings and limitations
 - [ ] Create usage examples
 - [ ] Note which concepts work best
 - [x] At the end, have a place where each script is explained in plain english
    - [x] Created `explanation_train_gpt2_probe.md`
  - See what to do with csv files and how play into code


## Key Technical Resources
- **Libraries**: concept-erasure, transformer-lens, transformers
- **Models**: GPT-2 (small/medium for speed)
- **Concepts**: sentiment, formality, toxicity (pre-trained probes available)
- **Layers**: Focus on layers 6-8 for steering (middle layers work best)

## Success Metrics
- [ ] Can load and run pre-trained linear probes
- [ ] Can detect concepts in generated text with reasonable accuracy
- [ ] Can demonstrate steering effects (even if subtle)
- [ ] Can show end-to-end pipeline working
- [ ] Interface allows experimentation with different prompts/steering

## Notes
- Steering effects will be subtle, not dramatic - this is expected
- Some concepts (sentiment) work better than others (complex reasoning)
- Focus on getting basic pipeline working rather than perfect results
- Pre-trained probes save significant development time

## Next Steps After MVP
- Add hallucination toolkit
- Address the `torch.load` issue with scikit-learn probes. The current workaround is using `weights_only=False`, but a more robust solution is needed to ensure model portability and security. This may involve saving and loading the probe's state dictionary instead of the entire object.
- Train gpt-2 myself, working with Karpathy or other things. 
Deep understanding of transformers will help
- https://ameliorology.substack.com/p/probing-activations-in-gpt2-small
- https://github.com/milosal/activation-probes/blob/main/main.ipynb
- Local agents/different things or Cloud options (either way, see if can build)
- Train custom probes on domain-specific concepts
- Experiment with different steering techniques (activation patching, etc.)
- Let's  do simple train/test sets, get something working with decent performance and then as post MVP next steps, we can note down different lienar probe options
- Build more sophisticated interface
- Explore commercial applications

Last Updated: 2025-09-07

## Key Metrics
- **Test Coverage**: 78% (base_model.py)
- **Open Issues**: 0
- **Completed Tasks**: 12/25 (48%)

## Notes
- Base model implementation is complete and well-tested
- Ready to proceed with GPT-2 model implementation
- Good test coverage established for core functionality
- Next focus: Implement GPT2Model class and basic concept detectors