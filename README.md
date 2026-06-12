# Quantum MRI Brain Tumor Classification

This repository implements a **High-Resolution Spatial-Ensemble Quantum Variational Framework** for classifying multi-class brain tumors (Meningioma, Glioma, and Pituitary tumors) using T1-weighted contrast-enhanced MRI scans. 

It provides an end-to-end medical image classification pipeline that combines deep transfer learning (ResNet-50 and VGG-19) with PennyLane-based variational quantum circuits (QVC) on a GPU-enabled environment. 

Additionally, this repository contains **7 models** implemented in PyTorch and PennyLane to replicate and benchmark against the state-of-the-art:
1. **Classical CNN**: A custom, lightweight convolutional neural network baseline.
2. **VGG CNN**: Pre-trained classical VGG-19 backbone with a classical classification head.
3. **ResNet CNN**: Pre-trained classical ResNet-50 backbone with a classical classification head.
4. **DenseNet CNN**: Replicates the classical benchmark in the base paper (DenseNet-121).
5. **Ensemble CNN**: Fuses features from ResNet-50 and VGG-19 classically using a 1x1 convolution.
6. **Normal Quantum CNN (HQC-CNN)**: Replicates the base paper's quantum model using a custom CNN frontend and a variational quantum circuit.
7. **Ensemble QCNN (Proposed Model)**: Spatial-ensemble framework that fuses ResNet-50 and VGG-19 features using a 1x1 convolution and maps them to a 4-qubit variational quantum circuit.

**Base Paper Reference**: *Investigating Hybrid Quantum-Assisted Classical and Deep Learning Model for MRI Brain Tumor Classification* (Anandhavalli Muniasamy et al., JOIG 2025).

---

## Proposed Model Architecture (Ensemble QCNN)

The proposed Ensemble QCNN is a hybrid quantum-classical framework designed for multi-class brain tumor classification from MRI images. The architecture combines the complementary feature extraction capabilities of VGG-19 and ResNet-50 to capture both local and high-level tumor characteristics. The extracted representations are integrated and processed through a quantum-enhanced learning module, enabling the model to learn complex feature relationships beyond conventional deep learning approaches.

The quantum component performs feature transformation, correlation learning, and dimensionality reduction before the final classification stage. By leveraging both classical transfer learning and quantum feature processing, the proposed framework aims to improve discriminative capability while maintaining efficient model complexity.

---

## Experimental Results

The proposed Ensemble QCNN was evaluated against six benchmark architectures, including Classical CNN, DenseNet121, VGG19, ResNet50, Ensemble CNN, and a Hybrid Quantum CNN (HQC-CNN).

| Model | Test Accuracy (%) | Precision | Recall | F1-Score |
|---------|---------|---------|---------|---------|
| Classical CNN | 87.46 | 0.8621 | 0.8654 | 0.8637 |
| DenseNet121 | 89.46 | 0.8837 | 0.8815 | 0.8813 |
| VGG19 CNN | 88.46 | 0.8740 | 0.8780 | 0.8747 |
| ResNet50 CNN | 93.14 | 0.9263 | 0.9250 | 0.9256 |
| Ensemble CNN | 93.31 | 0.9269 | 0.9315 | 0.9287 |
| HQC-CNN | 79.26 | 0.7885 | 0.7654 | 0.7686 |
| **Proposed Ensemble QCNN** | **93.48** | **0.9315** | **0.9282** | **0.9294** |

### Performance Summary

The proposed **Ensemble QCNN achieved the highest overall classification accuracy of 93.48%**, outperforming all benchmark models evaluated in this study. Compared to the strongest classical baseline (**Ensemble CNN**, 93.31%), the proposed model achieved a modest but consistent improvement while also obtaining the highest macro-precision and F1-score. These results demonstrate the effectiveness of combining transfer learning-based feature extraction with quantum-enhanced representation learning for brain tumor MRI classification.

---

## File Structure

```
Quantum-Brain-Tumor-Classification/
├── data_preprocessing.py               # Downloads, extracts, parses MAT files, outputs PNG dataset
├── models.py                           # PyTorch + PennyLane module definitions for the 7 networks
├── train.py                            # Standardized training loop with learning rate schedules
├── evaluate.py                         # Test set validation, confusion matrix & ROC curve plotting
├── run_experiments.py                  # Automation script to train & evaluate all 7 models sequentially
├── Brain_Tumor_Ensemble_QCNN.ipynb     # Self-contained Kaggle-ready notebook
└── README.md                           # Setup and execution instructions (this file)
```

---

## Installation & Prerequisites

To run these scripts locally, install the following dependencies:

```bash
pip install torch torchvision
pip install pennylane
pip install scipy pandas scikit-learn matplotlib seaborn tabulate
```

---

## How to Run (Local Environment)

### 1. Data Preprocessing
Run the preprocessing script to download the Figshare dataset, extract, and convert MAT files to split PNG datasets:
```bash
python data_preprocessing.py
```
This downloads the raw data, groups images by Patient ID to prevent data leakage, and exports them to:
- `dataset_processed/train/`
- `dataset_processed/val/`
- `dataset_processed/test/`

### 2. Run All Experiments (Automated Benchmark)
Train and evaluate all 7 models sequentially, print the final comparison table, and generate benchmark charts:
```bash
python run_experiments.py --epochs 20 --batch_size 32
```
Outputs (comparative charts and reports) will be saved under the `results/` folder.

### 3. Run Single Model Training
To train a single specific model (e.g. the proposed Ensemble QCNN):
```bash
python train.py --model ensemble_qcnn --epochs 20 --batch_size 32 --lr 0.001
```
Available model choices: `classical_cnn`, `vgg_cnn`, `resnet_cnn`, `densenet_cnn`, `ensemble_cnn`, `normal_qcnn`, `ensemble_qcnn`.

### 4. Evaluate a Trained Model
To evaluate a trained model checkpoint on the test split and generate its confusion matrix and ROC curve:
```bash
python evaluate.py --model ensemble_qcnn
```
This saves:
- `results/confusion_matrix_ensemble_qcnn.png`
- `results/roc_curves_ensemble_qcnn.png`
- `results/metrics_ensemble_qcnn.json`

---

## Kaggle Integration

To run on Kaggle:
1. Upload the `Brain_Tumor_Ensemble_QCNN.ipynb` file directly as a Kaggle Notebook.
2. In the right panel under **Settings**, set the **Accelerator** to **GPU T4 x2** or **GPU P100**.
3. Under **Internet**, ensure it is turned **ON** (required to download the dataset from Figshare and install `pennylane` via pip).
4. Run all cells. The notebook will automatically download the dataset, execute the preprocessing, train all 7 models, and generate the comparative plots and tables.
