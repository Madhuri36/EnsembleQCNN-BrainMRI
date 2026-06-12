import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

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
print(f"[*] Training will run on device: {device}")

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

def get_dataloaders(dataset_dir, batch_size=32):
    """Sets up transformations and DataLoaders for train, val, and test splits."""
    # ImageNet standard normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_val(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def plot_and_save_curves(history, output_path):
    """Plots and saves loss and accuracy history curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#4A90E2', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#E06666', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#4A90E2', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc', color='#E06666', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train MRI Brain Tumor Classification Models")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help="Model architecture name to train")
    parser.add_argument('--dataset_dir', type=str, default="dataset_processed",
                        help="Path to preprocessed dataset directory")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('--save_dir', type=str, default="checkpoints",
                        help="Directory to save trained model check points")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load loaders
    print(f"[*] Loading datasets from: {args.dataset_dir}...")
    train_loader, val_loader, _ = get_dataloaders(args.dataset_dir, args.batch_size)
    
    # Instantiate Model
    print(f"[*] Instantiating model: {args.model}...")
    model = MODEL_MAP[args.model]().to(device)
    
    # Set optimizer
    # For hybrid quantum models, we can optimize quantum weights and classical weights together.
    # PennyLane parameters are natively optimized by PyTorch optimizers.
    # In PyTorch, nn.CrossEntropyLoss() is the exact mathematical equivalent of Keras's
    # 'sparse_categorical_crossentropy' loss combined with a 'softmax' activation layer.
    # We keep the model outputs as raw logits (no softmax at the end) and let CrossEntropyLoss
    # compute log_softmax internally for maximum numerical stability and to prevent model collapse.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    early_stopping_patience = 15
    epochs_no_improve = 0
    
    print(f"[*] Starting training loop for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_val(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
              
        # Save best checkpoint (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    [+] Saved new best model checkpoint based on val_acc: {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break
                
    # Save training history as JSON
    history_path = os.path.join(args.save_dir, f"history_{args.model}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Plot curves
    curve_path = os.path.join(args.save_dir, f"curves_{args.model}.png")
    plot_and_save_curves(history, curve_path)
    print(f"[+] Training completed successfully. Results saved to '{args.save_dir}'.")

if __name__ == "__main__":
    main()
