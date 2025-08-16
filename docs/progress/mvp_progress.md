# MVP Development Progress Tracker - REVISED

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
- [ ] Test probe loading and classification for both concepts
- [ ] Create a notebook to highlight end to end flow to ensure understanding
- [ ] Implement concept erasure using existing probe script
 - [ ] Data Preparation
   - [ ] Prepare labeled datasets using existing prompt templates
   - [ ] Use `populate_data()` to extract activations
 - [ ] Probe Training & Analysis
   - [ ] Train logistic regression probes using `read_and_train()`
   - [ ] Identify optimal layers for concept representation
   - [ ] Extract concept direction vectors from trained probes
 - [ ] Concept Erasure Implementation
   - [ ] Modify model forward pass to subtract concept direction
   - [ ] Implement erasure strength control
   - [ ] Create utility functions for applying erasure
 - [ ] Evaluation
   - [ ] Qualitative: Compare model outputs before/after erasure
   - [ ] Quantitative: Measure concept presence with probe
   - [ ] Document effect on model behavior

## Hour 5-6: Probe Integration & Testing
- [ ] Set up probe-based concept detection
 - [ ] Use existing `get_layer_actives()` for activation extraction
 - [ ] Implement probe inference on new text
 - [ ] Validate detection with example texts
- [ ] Create concept erasure demo
 - [ ] Jupyter notebook showing end-to-end workflow
 - [ ] Examples of successful concept erasure
 - [ ] Documentation of limitations and edge cases

* __Tests__
  - Unit: probe IO round-trip (save→load→identical outputs on fixed inputs).
  - Unit: shape/device tests for feature extraction and probe forward pass.
  - Integration: tiny batch through `GPT2Model.extract_features()` + loaded probe; verify probabilities sum to 1 and shapes match.

* __Acceptance criteria__
  - A trained probe can be saved and later loaded to classify new texts consistently.
  - Given metadata (`layer`, `pooling`), inference uses the correct hidden states.
  - Optional detector wrapper returns a stable score via `detect_concepts`.

Status (initial):
- [x] Feature extractor available: `GPT2Model.extract_features()`
- [ ] IO helpers implemented
- [ ] Model helper/wrapper implemented
- [ ] Concept-detector wrapper implemented (optional)
- [ ] Tests added and passing

## Hour 7-8: Basic Activation Steering
- [ ] Implement activation steering mechanism
 - [ ] Extract probe direction vectors from linear weights
 - [ ] Add/subtract vectors from activations during generation
 - [ ] Use generate() with custom forward hooks in PyTorch
 - [ ] Implement steering strength parameter (alpha)
- [ ] Test steering effects
 - [ ] Simple prompts with different steering values
 - [ ] Measure before/after concept scores
 - [ ] Document which concepts steer effectively

## Hour 9-10: Integration & Interface
- [ ] Create simple interface (choose one)
 - [ ] CLI interface with argparse
 - [ ] Simple Gradio web interface
 - [ ] Jupyter notebook demo
 - [ ] Look into any known AI UX lessons?
- [ ] Build end-to-end demo flow
 - [ ] Input prompt → steered generation → concept detection display
 - [ ] Show steering effects with before/after comparisons
- [ ] Documentation and polish
 - [ ] Document findings and limitations
 - [ ] Create usage examples
 - [ ] Note which concepts work best


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

Last Updated: 2025-08-15

## Key Metrics
- **Test Coverage**: 78% (base_model.py)
- **Open Issues**: 0
- **Completed Tasks**: 12/25 (48%)

## Notes
- Base model implementation is complete and well-tested
- Ready to proceed with GPT-2 model implementation
- Good test coverage established for core functionality
- Next focus: Implement GPT2Model class and basic concept detectors