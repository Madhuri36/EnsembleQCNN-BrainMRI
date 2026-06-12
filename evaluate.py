import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
import seaborn as sns

from models import (
    ClassicalCNN,
    VGGCNN,
    ResNetCNN,
    DenseNetCNN,
    EnsembleCNN,
    NormalQuantumCNN,
    EnsembleQCNN
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model mapping
MODEL_MAP = {
    'classical_cnn': ClassicalCNN,
    'vgg_cnn': VGGCNN,
    'resnet_cnn': ResNetCNN,
    'densenet_cnn': DenseNetCNN,
    'ensemble_cnn': EnsembleCNN,
    'normal_qcnn': NormalQuantumCNN,
    'ensemble_qcnn': EnsembleQCNN
}

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]

def get_test_loader(dataset_dir, batch_size=32):
    """Sets up transformations and DataLoader for the test split."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader

def evaluate_model(model, dataloader, checkpoint_path):
    """Loads weights and evaluates the model on test loader."""
    print(f"[*] Loading model weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total += targets.size(0)
            
    test_loss = running_loss / total
    return np.array(all_targets), np.array(all_preds), np.array(all_probs), test_loss

def save_confusion_matrix(targets, preds, save_path):
    """Generates and saves a seaborn heatmap confusion matrix."""
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    
    # Custom aesthetic palette
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_roc_curves(targets, probs, save_path):
    """Generates and saves multi-class One-vs-Rest ROC curves."""
    plt.figure(figsize=(8, 6))
    
    # Convert targets to one-hot encoding
    n_classes = len(CLASS_NAMES)
    one_hot_targets = np.eye(n_classes)[targets]
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(one_hot_targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained MRI Models")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help="Model architecture name to evaluate")
    parser.add_argument('--dataset_dir', type=str, default="dataset_processed",
                        help="Path to preprocessed dataset directory")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to specific model checkpoint")
    parser.add_argument('--save_dir', type=str, default="results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Deduce default checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join("checkpoints", f"best_{args.model}.pth")
        
    if not os.path.exists(checkpoint_path):
        print(f"[-] Error: Checkpoint file '{checkpoint_path}' does not exist.")
        return
        
    # Load loaders
    print(f"[*] Loading test set from: {args.dataset_dir}...")
    test_loader = get_test_loader(args.dataset_dir)
    
    # Instantiate Model
    model = MODEL_MAP[args.model]()
    
    # Run evaluation
    targets, preds, probs, test_loss = evaluate_model(model, test_loader, checkpoint_path)
    
    # Classification report
    print(f"\n[+] Classification Report for {args.model}:")
    report = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4)
    print(report)
    
    # Calculate precision, recall, f1, accuracy
    precision, recall, fscore, _ = precision_recall_fscore_support(targets, preds, average='macro')
    test_acc = np.mean(targets == preds)
    
    metrics = {
        'model': args.model,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(fscore)
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(args.save_dir, f"metrics_{args.model}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Metrics exported to: {metrics_path}")
        
    # Generate and save plots
    cm_path = os.path.join(args.save_dir, f"confusion_matrix_{args.model}.png")
    save_confusion_matrix(targets, preds, cm_path)
    print(f"[+] Confusion matrix plot saved to: {cm_path}")
    
    roc_path = os.path.join(args.save_dir, f"roc_curves_{args.model}.png")
    save_roc_curves(targets, probs, roc_path)
    print(f"[+] ROC curves plot saved to: {roc_path}")

if __name__ == "__main__":
    main()
