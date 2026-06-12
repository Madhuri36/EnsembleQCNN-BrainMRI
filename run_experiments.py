import os
import subprocess
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

MODELS = [
    'classical_cnn',
    'densenet_cnn',
    'normal_qcnn',
    'vgg_cnn',
    'resnet_cnn',
    'ensemble_cnn',
    'ensemble_qcnn'
]

def check_dataset():
    """Forces preprocessing script to run to ensure stratified random splits are applied."""
    print("[*] Running data_preprocessing.py to ensure stratified random splits...")
    subprocess.run(["python", "data_preprocessing.py"], check=True)

def run_command(cmd_args):
    """Utility function to print and run subprocess commands."""
    print(f"[*] Running command: {' '.join(cmd_args)}")
    subprocess.run(cmd_args, check=True)

def compile_results():
    """Reads all metrics files and outputs a comparative markdown report."""
    results_dir = "results"
    all_metrics = []
    
    if not os.path.exists(results_dir):
        print("[-] Results directory not found. No metrics to compile.")
        return
        
    for model in MODELS:
        metrics_file = os.path.join(results_dir, f"metrics_{model}.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                all_metrics.append(data)
        else:
            print(f"[-] Warning: Metrics file for '{model}' was not found at {metrics_file}.")
            
    if len(all_metrics) == 0:
        print("[-] No valid metrics found to compile.")
        return
        
    df = pd.DataFrame(all_metrics)
    
    # Capitalize model names for display
    df['model_display'] = df['model'].str.upper().str.replace('_', ' ')
    df = df[['model_display', 'test_loss', 'test_accuracy', 'macro_precision', 'macro_recall', 'macro_f1']]
    
    # Rename columns for table
    df.columns = ['Model Architecture', 'Test Loss', 'Test Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    
    # Save as Markdown
    md_table = df.to_markdown(index=False)
    
    report = f"""# Benchmark Experiment Results Comparison

This report summarizes the comparative analysis of 7 distinct models trained on the Cheng Figshare brain tumor MRI dataset. The comparison includes classical CNNs, transfer learning backbones, classically-fused ensembling, a basic hybrid quantum-classical network (HQC-CNN), and the proposed **Ensemble QCNN** spatial-ensemble quantum framework.

## Performance Metrics Table

{md_table}

---

## Key Analysis
- **DenseNet CNN**: Represents the base paper's classical benchmark (target accuracy ~94%).
- **Normal Quantum CNN (HQC-CNN)**: Replicates the base paper's quantum model (target accuracy ~88%).
- **Ensemble QCNN**: The proposed spatial-ensemble model combining ResNet-50 and VGG-19 features into a 4-qubit quantum classifier.
"""
    
    report_path = os.path.join(results_dir, "results_comparison.md")
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\n[+] Compiled comparison report saved to: {report_path}")
    print(report)
    
    # Generate Comparison Plot
    plt.figure(figsize=(10, 6))
    colors = ['#B6D7A8' if 'QUANTUM' in m or 'QCNN' in m else '#9FC5E8' for m in df['Model Architecture']]
    bars = plt.bar(df['Model Architecture'], df['Test Accuracy'] * 100, color=colors, edgecolor='grey', width=0.6)
    
    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.title('Test Accuracy Comparison across Models', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.ylim([0, 110])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    print("[+] Accuracy comparison bar chart saved to results/accuracy_comparison.png")


# Model-specific configurations for epochs and learning rates
MODEL_CONFIGS = {
    'classical_cnn': {'epochs': 40, 'lr': 0.001},
    'densenet_cnn':  {'epochs': 40, 'lr': 0.001},
    'normal_qcnn':   {'epochs': 40, 'lr': 0.001},
    'vgg_cnn':       {'epochs': 40, 'lr': 0.001},
    'resnet_cnn':    {'epochs': 40, 'lr': 0.001},
    'ensemble_cnn':  {'epochs': 40, 'lr': 0.001},
    'ensemble_qcnn': {'epochs': 40, 'lr': 0.001}
}

def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark experiments")
    parser.add_argument('--classical_epochs', type=int, default=None, help="Number of training epochs for classical models")
    parser.add_argument('--quantum_epochs', type=int, default=None, help="Number of training epochs for quantum models")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate")
    args = parser.parse_args()
    
    # 1. Check dataset
    check_dataset()
    
    # 2. Train and Evaluate each model
    for model in MODELS:
        print("\n" + "="*60)
        print(f"[*] STARTING EXPERIMENT FOR MODEL: {model.upper()}")
        print("="*60)
        
        # Determine parameters (use model-specific defaults, override with CLI args if specified)
        config = MODEL_CONFIGS[model]
        
        is_quantum = 'qcnn' in model or 'quantum' in model
        
        # Override defaults if specified on the command line
        if is_quantum:
            epochs = args.quantum_epochs if args.quantum_epochs is not None else config['epochs']
        else:
            epochs = args.classical_epochs if args.classical_epochs is not None else config['epochs']
            
        lr = args.lr if args.lr is not None else config['lr']
        batch_size = args.batch_size
        
        print(f"[*] Configured training parameters: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        
        # Training Command
        train_cmd = [
            "python", "train.py",
            "--model", model,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr)
        ]
        try:
            run_command(train_cmd)
        except subprocess.CalledProcessError as e:
            print(f"[-] Training failed for model '{model}': {e}. Continuing to next model...")
            continue
            
        # Evaluation Command
        eval_cmd = [
            "python", "evaluate.py",
            "--model", model
        ]
        try:
            run_command(eval_cmd)
        except subprocess.CalledProcessError as e:
            print(f"[-] Evaluation failed for model '{model}': {e}.")
            
    # 3. Compile results
    print("\n" + "="*60)
    print("[*] COMPILING EXPERIMENT RESULTS")
    print("="*60)
    compile_results()

if __name__ == "__main__":
    main()
