# AI Safety Linear Probe Project

## 📋 Project Overview
Build an explainable, steerable chatbot using small language models with linear probes to detect and manipulate concepts in real-time. This project aims to bridge the gap between AI safety research and commercial applications.

## 🚀 Quick Start
```bash
# 1. Clone the repository
git clone <repository-url>
cd stealth-build

# 2. Set up the environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the development server
python src/main.py
```

## 📚 Documentation

### Project Planning
- [MVP Vision & Scope](./docs/planning/mvp_vision.md)
- [10-Hour Implementation Plan](./docs/planning/10_hour_plan.md)
- [Learning Plan & Progress](./docs/planning/learning_plan.md)

### Architecture
- [System Architecture](./docs/architecture/overview.md)
- [API Documentation](./docs/architecture/api.md)
- [Testing Strategy](./docs/architecture/testing.md)

## 🎯 Project Goals

### Phase 1: Core Functionality (10-Hour MVP)
- [x] Project setup and planning
- [ ] GPT-2 model integration
- [ ] Basic concept detection
- [ ] Simple steering interface
- [ ] Test coverage

### Future Goals
- Real-time chatbot with explainable concept manipulation
- Advanced steering capabilities
- Support for multiple models
- Commercial integration patterns

## 🏗️ Architecture

### Core Components
- **Model**: GPT-2 Small (for initial development)
- **Probe Type**: Linear probes on hidden states
- **Interface**: Command-line with basic web UI
- **Testing**: Pytest with 80%+ coverage target

## 📂 Project Structure
```
.
├── src/                    # Source code
├── tests/                  # Test files
├── docs/                   # Documentation
│   ├── planning/           # Planning documents
│   └── architecture/       # Architecture docs
├── data/                   # Training and test data
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
- **Interface**: Web app focused on steering demonstration
- **Target Concepts**: Toxicity detection & company values alignment

## Core Features
- Real-time concept detection in text
- Concept steering/clamping interface
- Explainable activations visualization
- Multi-concept support (toxicity, company values)

## First Tasks
[Defined below]
