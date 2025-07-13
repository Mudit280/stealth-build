
# MVP Development Progress Tracker - REVISED

## Hour 1-2: Project Setup & Initial Learning ✅
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
   - [ ] Download or train probes for: sentiment and toxicity
   - [ ] Test probe loading and classification for both concepts
- [ ] Set up GPT-2 with activation extraction
 - [ ] Use transformers library with output_hidden_states=True
 - [ ] Extract activations from middle layers (layers 6-8 work well)
 - [ ] Write helper functions for activation extraction
- [ ] Implement concept erasure for sentiment and/or toxicity
 - [ ] Apply probe direction to erase concept
 - [ ] Demonstrate and document effect of erasure on model outputs (qualitative/quantitative)

## Hour 5-6: Probe Loading & Concept Detection
- [ ] Implement probe loading system
 - [ ] Load pre-trained linear probe weights
 - [ ] Create probe interface for consistent API
 - [ ] Handle different probe formats/architectures
- [ ] Build concept detection pipeline
 - [ ] Input text → GPT-2 → extract activations → probe predictions
 - [ ] Test with example texts for each concept
 - [ ] Validate probe outputs make sense (sanity checks)
- [ ] Write tests for concept detection accuracy

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
- Train custom probes on domain-specific concepts
- Experiment with different steering techniques (activation patching, etc.)
- Build more sophisticated interface
- Explore commercial applications

Last Updated: [Current Date]

## Key Metrics
- **Test Coverage**: 78% (base_model.py)
- **Open Issues**: 0
- **Completed Tasks**: 12/25 (48%)

## Notes
- Base model implementation is complete and well-tested
- Ready to proceed with GPT-2 model implementation
- Good test coverage established for core functionality
- Next focus: Implement GPT2Model class and basic concept detectors