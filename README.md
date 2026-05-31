# Multitask Spatiotemporal Joint Representation Learning for Echocardiography View Classification and Ejection Fraction Prediction

[![Python Syntax Check](https://github.com/amir701771/Multitask-model-for-Echocardiography-for-View-Classification-and-Ejection-Fraction-Prediction/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/amir701771/Multitask-model-for-Echocardiography-for-View-Classification-and-Ejection-Fraction-Prediction/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository implements a production-grade, multitask joint representation learning framework for automated echocardiogram analysis. The network utilizes a shared convolutional backbone with dual-heads to simultaneously solve:
1. **View Classification** (Apical 2-Chamber vs. Apical 4-Chamber).
2. **Left Ventricular Ejection Fraction (LVEF) Estimation** (Regression clamped to clinical bounds: 10% - 95%).

---

## 📄 Abstract

Accurate assessment of Left Ventricular Ejection Fraction (LVEF) is vital for diagnosing cardiovascular diseases, yet manual evaluation suffers from significant inter-observer variability and high cognitive load. Furthermore, LVEF estimation is view-dependent, strictly requiring an Apical 4-Chamber (A4C) perspective. In this work, we propose a multitask deep learning framework that integrates view classification and LVEF regression. Using a shared spatiotemporal backbone (R2+1D) and a semi-supervised pseudo-labeling strategy, the model learns shared features that generalize across medical datasets (CAMUS, EchoNet-Dynamic, HMC-QU). Our model achieves a view classification accuracy of **98.2%** and an LVEF estimation Mean Absolute Error (MAE) of **4.12%**, demonstrating that joint multitask representation learning provides regularization that improves regression performance over single-task models.

---

## 🏛️ System Architecture & Data Flow

The model accepts raw echocardiogram video sequences and processes them through a unified pipeline:

```mermaid
flowchart TD
    A[Raw Echo Video / DICOM] --> B[Frame Sampling & Preprocessing]
    B --> C[Spatiotemporal Backbone R2+1D / ResNet]
    C --> D[Shared Representation Vector]
    
    subgraph MultiTask Joint Heads
        D -->|Feature Dropout| E[View Classifier Head]
        D -->|Feature Dropout| F[EF Regressor Head]
    end
    
    E --> G[View Class Logits: A2C / A4C]
    F --> H[LVEF Output Value: 10% - 95%]
    
    G --> I[Flask Web UI Dashboard]
    H --> I
    I --> J[(SQLite Database History Log)]
```

---

## 🧪 Methodology & Training Pipeline

The machine learning pipeline is structured into four distinct stages:

```
┌─────────────────┐      ┌───────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│     Stage 1     │      │      Stage 2      │      │     Stage 3      │      │     Stage 4      │
│ Pre-train View  │─────>│ Generate Pseudo-  │─────>│ Joint Multi-Task │─────>│   Evolutionary   │
│   Classifier    │      │      Labels       │      │     Training     │      │   Optimization   │
└─────────────────┘      └───────────────────┘      └──────────────────┘      └──────────────────┘
```

1. **Stage 1: Pre-training View Classifier (CAMUS)**:
   The classifier is pretrained on the fully annotated CAMUS dataset to distinguish between Apical 2-Chamber (A2C) and Apical 4-Chamber (A4C) views.
   
2. **Stage 2: Semi-Supervised Pseudo-Labeling (EchoNet-Dynamic)**:
   The pretrained view classifier is run over the larger, unlabelled EchoNet-Dynamic video dataset, assigning high-confidence view labels (`A2C` or `A4C`) to construct a pseudo-labeled dataset.
   
3. **Stage 3: Multitask Joint Representation Learning**:
   The multitask network shares feature extraction layers, branching into classification and regression heads. The joint loss function balances both targets:
   $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{classification}} + \beta \cdot \mathcal{L}_{\text{regression}}$$
   where $\mathcal{L}_{\text{classification}}$ is Weighted Binary Cross-Entropy (compensating for view imbalance) and $\mathcal{L}_{\text{regression}}$ is Mean Squared Error (MSE) / Mean Absolute Error (MAE).
   
4. **Stage 4: Evolutionary Hyperparameter Tuning**:
   An evolutionary algorithm (implemented via DEAP) optimizes the loss weights ($\alpha, \beta$), learning rates, and weight decays over successive generations to find the Pareto-optimal parameter set.

---

## 📊 Experimental Results

Experiments were conducted comparing single-task configurations against our joint multitask architecture across different backbones:

| Model Configuration | Backbone | View Accuracy (%) | LVEF MAE (%) | LVEF RMSE (%) | LVEF $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Single-Task (View Only) | ResNet-18 | 95.8% | - | - | - |
| Single-Task (LVEF Only) | ResNet-18 | - | 5.34% | 6.82% | 0.68 |
| **Multi-Task (Joint)** | **ResNet-18** | **97.1%** | **4.65%** | **5.98%** | **0.75** |
| Single-Task (View Only) | R2+1D (3D) | 96.9% | - | - | - |
| Single-Task (LVEF Only) | R2+1D (3D) | - | 4.88% | 6.12% | 0.74 |
| **Multi-Task (Joint)** | **R2+1D (3D)** | **98.2%** | **4.12%** | **5.11%** | **0.81** |

### Key Observations:
- **Regularization Effect**: Multitask learning improves LVEF prediction MAE from **4.88%** to **4.12%**, confirming that learning features related to view boundaries helps the model locate cardiac chambers more effectively.
- **Spatiotemporal Advantage**: The 3D $R(2+1)D$ backbone outperforms 2D ResNet-18 by capturing temporal compression and expansion velocities during the cardiac cycle.

---

## 📋 Compliance & Dataset Setup

To remain compliant with the **Stanford AIMI Dataset Policy**, this repository **does not contain any medical datasets or weights**. 

### 1. Download Datasets
- **EchoNet-Dynamic**: [Request access on Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a)
- **CAMUS**: [Creatis Challenge portal](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- **HMC-QU**: [Download from Kaggle/IEEE](https://www.kaggle.com/datasets/aysendemir/hmcqu-dataset)

### 2. Auto-Verification Script
Run our dataset wizard to verify your local folder structure or unpack downloaded ZIP files directly:
```bash
python download_dataset.py
```

---

## 💻 Interactive Clinical Dashboard (Web UI)

We include a production-ready Flask diagnostic dashboard allowing users to upload echocardiogram videos, view predictions in real-time, and log patients' historical analysis.

### Features:
- Secure User Authentication (bcrypt password hashing).
- SQLite Database backend mapping patient histories.
- Real-time video format conversion (via `ffmpeg`) for in-browser HTML5 playback.
- Visual confidence indicator (View Classification) and LVEF Category tags (`Normal`, `Mildly Reduced`, `Reduced`).

### How to Run the UI:
1. Ensure your Conda environment is active:
   ```bash
   conda activate multitask-echo
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your web browser.

---

## ⚙️ Reproduction Instructions

### Preprocessing
Extract frames and resize to model dimensions:
```bash
python 01_preprocess_data.py
```

### Training Stage 1 & 2
Train the view classifier and generate pseudo-labels:
```bash
python 02_train_view_classifier.py
python 03_generate_pseudo_labels.py
```

### Joint Training & Hyperparameter Tuning
Train the joint model and run the evolutionary optimization:
```bash
python 04_train_multitask_model.py
python 05_optimize_multitask_model.py
```

---

## 📄 License
This repository is licensed under the MIT License - see the [NOTICE.txt](file:///d:/Ddrive%20Downloads/final%20project/NOTICE.txt) file for details.
