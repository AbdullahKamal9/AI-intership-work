# AI Course Projects (Weeks 3–6)

A four-week hands-on journey through core machine learning: from a first classifier to a neural network built from scratch, and finally a complete wine-analysis AI system with a recommender.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch%20NN-lightgrey)

> All results come from real runs (see [`logs/`](logs/)). A fixed random seed (42) is used everywhere, so results are reproducible.

## Table of Contents
- [Results at a Glance](#results-at-a-glance)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Project Details](#project-details)
- [Output Charts](#output-charts)
- [Limitations](#limitations)
- [Author](#author)

## Results at a Glance

| Week | Project | Dataset | Model | Test Accuracy |
|------|---------|---------|-------|---------------|
| 3 | Basic ML model | Iris (150 rows) | Logistic Regression | **96.67%** |
| 4 | Neural network from scratch (NumPy) | Digits (1,797 images) | 64 → 64 (ReLU) → 10 (Softmax) | **96.67%** (train 99.86%) |
| 5 | Breast cancer intelligent system | Breast Cancer (569 rows) | Logistic Regression (best by CV) | **98.25%** (CV 98.02%) |
| 6 | Wine Intelligence System | Wine (178 rows) | Random Forest (best by CV) | **100%** (CV 98.62%) |

Week 6 also compared: MLP neural network (94.44% test / 98.62% CV) and Logistic Regression (97.22% test / 98.60% CV).

## Project Structure

```
ai-course-projects/
├── README.md
├── requirements.txt
├── logs/                          # Console output of each run
├── week3_basic_ml/                # basic_ml.py
├── week4_neural_network/          # neural_network.py  + outputs/
├── week5_intelligent_system/      # intelligent_system.py + outputs/
└── week6_final_project/           # wine_ai_system.py + outputs/
```

## Setup

```bash
git clone https://github.com/AbdullahKamal9/AI-intership-work.git
cd AI-intership-work
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## How to Run

```bash
python week3_basic_ml/basic_ml.py
python week4_neural_network/neural_network.py
python week5_intelligent_system/intelligent_system.py          # interactive
python week5_intelligent_system/intelligent_system.py --demo   # no typing needed
python week6_final_project/wine_ai_system.py                   # interactive menu
python week6_final_project/wine_ai_system.py --demo            # no typing needed
```

## Project Details

### Week 3 – Basic ML
Loads Iris, splits 80/20, trains Logistic Regression, predicts, and prints accuracy, confusion matrix and classification report.

### Week 4 – Neural Network from Scratch
Built with NumPy only (no deep-learning framework):
1. Scale pixels to 0–1 and one-hot encode labels.
2. Forward pass: input → hidden layer (ReLU) → softmax output.
3. Compute cross-entropy loss.
4. Backpropagation: gradients for every weight.
5. Mini-batch gradient descent for 100 epochs.
6. Evaluate on unseen images; charts saved to `outputs/`.

### Week 5 – Intelligent System
- Preprocessing: missing values, duplicates, scaling.
- Compares 3 models using 5-fold cross-validation.
- Evaluation with accuracy, confusion matrix, feature importance.
- Interactive prediction loop with input validation.

### Week 6 – Final Project: Wine Intelligence System
- Data preprocessing and 3 models, including an MLP neural network.
- Accuracy evaluation and 5 charts (comparison, confusion matrix, loss curve, PCA clusters, feature importance).
- Content-based wine **recommender** using nearest neighbours.
- Interactive menu (or `--demo` mode).

## Output Charts
Generated charts are stored in each week's `outputs/` folder, e.g. `week4_neural_network/outputs/training_curves.png` and `week6_final_project/outputs/4_pca_clusters.png`.

## Limitations
- Test sets are small (30 / 114 / 36 samples); one wrong sample moves accuracy by 1–3%. A 100% score on Week 6 does not mean a perfect model, so cross-validation scores are more reliable.
- Datasets are small, clean, built-in scikit-learn sets: good for coursework, not real-world proof.
- Week 5 is an educational demo, not a medical tool.

## Author
**Abdullah Kamal** – [GitHub](https://github.com/AbdullahKamal9)
