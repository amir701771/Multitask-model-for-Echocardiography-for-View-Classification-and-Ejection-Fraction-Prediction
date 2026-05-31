# Multitask Model for Echocardiography: View Classification and Ejection Fraction Prediction

This repository contains a multitask deep learning model for Echocardiography that simultaneously performs **View Classification** (Apical 2-Chamber vs. Apical 4-Chamber) and **Ejection Fraction (EF) Prediction**. The pipeline includes preprocessing scripts, a view classifier training stage, pseudo-label generation, multitask model training, evolutionary hyperparameter optimization, and a production-ready Flask web interface.

---

## 📊 Dataset Requirement & Stanford AIMI Compliance

In compliance with the **Stanford AIMI Dataset Usage Policy**, this repository **does not store or distribute any medical datasets, video files, or model weights**. You must download the datasets directly from their official sources.

### EchoNet-Dynamic Dataset
- **Official Dataset Link**: [Request Access on Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a)
- **Description**: A dataset of 10,036 echocardiography videos with ejection fraction labels and volume tracings.
- **Local Location**: The training and preprocessing pipelines expect the dataset to be placed under `data/ECHONET-Dynamic/`.

### Other Supported Datasets
- **CAMUS Dataset**: [Download on Creatis](https://www.creatis.insa-lyon.fr/Challenge/camus/) (Expected local path: `data/camus/`)
- **HMC-QU Dataset**: [Download on Kaggle/IEEE](https://www.kaggle.com/datasets/aysendemir/hmcqu-dataset) (Expected local path: `data/hmcqu/`)

---

## ⚙️ Project Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/amir701771/Multitask-model-for-Echocardiography-for-View-Classification-and-Ejection-Fraction-Prediction.git
   cd Multitask-model-for-Echocardiography-for-View-Classification-and-Ejection-Fraction-Prediction
   ```

2. **Set up the Environment**:
   We recommend using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate multitask-echo
   ```

3. **Verify and Extract Dataset**:
   We provide a helper script to verify your local dataset path:
   ```bash
   python download_dataset.py
   ```
   Follow the interactive prompts to extract a downloaded EchoNet zip file or verify that the directory contains the necessary files.

---

## 🚀 Execution Pipeline

Execute the following stages in sequence to reproduce training:

1. **Data Preprocessing**:
   Reads raw datasets (MHD/AVI formats), extracts relevant frames, resizes, and structures the images.
   ```bash
   python 01_preprocess_data.py
   ```

2. **Train View Classifier**:
   Trains the network to distinguish between Apical 2-Chamber (A2C) and Apical 4-Chamber (A4C) views.
   ```bash
   python 02_train_view_classifier.py
   ```

3. **Generate Pseudo-Labels**:
   Applies the trained view classifier to unlabelled files to generate a pseudo-labelled dataset.
   ```bash
   python 03_generate_pseudo_labels.py
   ```

4. **Train Multitask Model**:
   Trains the joint classification and regression network.
   ```bash
   python 04_train_multitask_model.py
   ```

5. **Optimize Hyperparameters**:
   Runs the evolutionary algorithm (using DEAP) to find optimal loss-balancing parameters ($\alpha$, $\beta$) and learning rates.
   ```bash
   python 05_optimize_multitask_model.py
   ```

---

## 🖥️ Production Web Interface

To launch the web interface for uploading echocardiography videos and getting real-time view classification and EF predictions:
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser. Note that you will need to place a model checkpoint (e.g. `best_model.pth.tar`) under the `results/` folder as configured in the config file.

---

## 📝 License
Please check the license and usage policies of individual datasets (Stanford AIMI, Creatis, HMC-QU) before using this codebase for research or clinical purposes.
