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

# 4. Train a Concept Probe
# Choose a concept (e.g., 'helpfulness') and a layer (e.g., 6)
python src/probes/run_probe_training.py --concept helpfulness --layer 6

# 5. Run the Streamlit Application
streamlit run src/app.py
```

## 📚 Documentation

- **Code Explanations**: Detailed walkthroughs of key scripts can be found in `docs/code explanations`.
- **Project Progress**: The high-level project plan and progress tracker is in `docs/progress/mvp_progress.md`.

## 📂 Project Structure
```
.
├── src/                    # Source code (models, detectors, app)
├── probes/                 # Trained probe files and activation data
├── tests/                  # Test files
├── docs/                   # Documentation
├── archive/                # Old notebooks and data files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License
This project is licensed under the MIT License.
