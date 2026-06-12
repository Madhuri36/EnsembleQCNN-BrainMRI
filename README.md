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

The Ensemble QCNN operates by extracting multi-scale feature maps from pre-trained ResNet-50 and VGG-19 backbones. These feature maps are concatenated and passed through a **1x1 convolution** layer which fuses channels and performs spatial-channel reduction to project the representation to $4$ features (1 feature per qubit). Global Average Pooling (GAP) reduces spatial dimensions to a $4$-dimensional vector.

The vector is mapped to a 4-qubit quantum system using RX and RZ rotations. The quantum convolution layer applies parameterized RX and RZ rotations on all qubits followed by cyclic CNOT entanglement. A SWAP-test based quantum pooling layer compares qubit pairs (0 vs 1 using ancilla 4, and 2 vs 3 using ancilla 5) to compute overlap similarity. Expectation values of PauliZ on the 2 ancilla qubits yield a 2-dimensional pooled representation, which is mapped to class logits via a final linear layer.

---

## File Structure

```
C:\Users\saima\.gemini\antigravity\scratch\quantum_mri_classification/
├── data_preprocessing.py      # Downloads, extracts, parses MAT files, outputs PNG dataset
├── models.py                  # PyTorch + PennyLane module definitions for the 7 networks
├── train.py                  # Standardized training loop with learning rate schedules
├── evaluate.py               # Test set validation, confusion matrix & ROC curve plotting
├── run_experiments.py        # Automation script to train & evaluate all 7 models sequentially
├── Kaggle_Notebook.ipynb     # Self-contained Kaggle-ready notebook
└── README.md                 # Setup and execution instructions (this file)
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
1. Upload the `Kaggle_Notebook.ipynb` file directly as a Kaggle Notebook.
2. In the right panel under **Settings**, set the **Accelerator** to **GPU T4 x2** or **GPU P100**.
3. Under **Internet**, ensure it is turned **ON** (required to download the dataset from Figshare and install `pennylane` via pip).
4. Run all cells. The notebook will automatically download the dataset, execute the preprocessing, train all 7 models, and generate the comparative plots and tables.
