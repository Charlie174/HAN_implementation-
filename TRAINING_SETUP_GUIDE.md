# Complete Training Setup Guide

## Overview
This guide provides a complete, proper training setup for the HAN medical prediction model with accuracy tracking, enhanced plotting, and proper data path configuration.

## ✅ Data Files Setup

### Required Files (in `data/` folder):
```
data/
├── filtered_patient_reports.csv          # Patient features & test values
├── test-disease-organ.csv                # Medical knowledge graph
└── patient-one-hot-labeled-disease-new.csv   # Pre-computed labels (optional)
```

### Understanding Labels
**Labels ARE REQUIRED for supervised learning!** 

- **Automatic Label Generation**: The `MedicalGraphData` class automatically computes labels by comparing patient test values against normal ranges from `test-disease-organ.csv`
- **Pre-computed Labels**: If you have `patient-one-hot-labeled-disease-new.csv`, it contains pre-computed labels, but the system will generate them automatically if not provided
- **Label Format**: Binary multi-label (one-hot encoded) indicating abnormal conditions per organ system

---

## 📋 Complete Training Setup Code

### 1. Install Requirements
```bash
cd /Users/charlie/Documents/Coding/VS Code/Language_python/FYP/New_2026/HAN-implementation
pip install -r requirements.txt
```

### 2. Import Packages
```python
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Import HAN components with enhanced validation metrics
from HAN import (
    MedicalGraphData,
    AttentionLayer,
    HANModel,
    SubgraphSampler,
    evaluate_model,
    compute_accuracy,              # NEW: For accuracy computation
    plot_training_metrics_enhanced  # NEW: For enhanced 6-subplot visualization
)

print("✓ All packages loaded successfully!")
print("✓ Enhanced validation metrics loaded!")
```

### 3. Configuration
```python
# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
PATIENCE = 10

# Data paths - Using relative paths for local data folder
PATH_RECORDS = "../data/filtered_patient_reports.csv"
PATH_SYMPTOM = "../data/test-disease-organ.csv"

# Data preprocessing parameters
SYMPTOM_FREQ_THRESHOLD = 5
PRUNE_PER_PATIENT = True
NNZ_THRESHOLD = 3

# Output directory
OUT_DIR = "../output"
os.makedirs(OUT_DIR, exist_ok=True)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"✓ Device: {DEVICE}")
print(f"✓ Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print(f"✓ Data: {PATH_RECORDS}")
print(f"✓ Symptoms: {PATH_SYMPTOM}")
```

### 4. Verify Data Files
```python
print("\n" + "="*80)
print("VERIFYING DATA FILES")
print("="*80)

files_to_check = {
    "Patient Records (Features)": PATH_RECORDS,
    "Symptom/Disease Map": PATH_SYMPTOM,
    "Label File (pre-computed)": "../data/patient-one-hot-labeled-disease-new.csv"
}

all_required_exist = True
for name, path in files_to_check.items():
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    required = "(optional)" if "Label File" in name else "(required)"
    status = "✅" if exists else "❌"
    
    print(f"{status} {name:30} {required:12}")
    print(f"   Path: {abs_path}")
    
    if "required" in required and not exists:
        all_required_exist = False

print()
if all_required_exist:
    print("✅ All required data files found!")
    print("📝 Labels will be computed automatically from patient test records")
    print("   by comparing test values against normal ranges.")
else:
    print("❌ Some required files are missing!")
    raise FileNotFoundError("Required data files missing!")
```

### 5. Load and Process Data
```python
print("\n" + "="*80)
print("LOADING AND PROCESSING DATA")
print("="*80)

# Initialize data loader
data_loader = MedicalGraphData(
    path_records=PATH_RECORDS,
    path_symptom=PATH_SYMPTOM,
    symptom_freq_threshold=SYMPTOM_FREQ_THRESHOLD,
    prune_per_patient=PRUNE_PER_PATIENT,
    nnz_threshold=NNZ_THRESHOLD,
    seed=SEED
)

# Load and process data (this automatically generates labels!)
data_loader.load_data()
data_loader.build_labels_and_features()
data_loader.build_adjacency_matrices()

print(f"✅ Data loaded successfully!")
print(f"   Patients: {len(data_loader.patient_list)}")
print(f"   Symptoms: {len(data_loader.symptom_list)}")
print(f"   Labels shape: {data_loader.labels.shape}")
print(f"   Features shape: {data_loader.features.shape}")
```

### 6. Training Loop with Accuracy Tracking
```python
def train_model_with_accuracy(
    model, optimizer, criterion, 
    features, labels, adj_dict,
    train_idx, val_idx,
    epochs=40, patience=10, device='cpu'
):
    """
    Training loop with accuracy tracking for enhanced visualization.
    
    Returns:
        dict: Contains all metrics including train/val losses, F1 scores, and accuracies
    """
    print("\n" + "="*80)
    print("STARTING TRAINING WITH ACCURACY TRACKING")
    print("="*80)
    
    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)
    
    # Initialize metric tracking
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    train_accuracies = []  # NEW: Track training accuracy
    val_accuracies = []    # NEW: Track validation accuracy
    
    val_micro_f1 = []
    val_macro_f1 = []
    
    best_val_f1 = 0.0
    best_val_acc = 0.0  # NEW: Track best validation accuracy
    patience_counter = 0
    
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features, adj_dict)
        train_outputs = outputs[train_idx]
        train_labels = labels[train_idx]
        
        # Compute loss
        loss = criterion(train_outputs, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.zero_grad()
        
        # Record training metrics
        train_losses.append(loss.item())
        
        # Compute training F1 score
        with torch.no_grad():
            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
            train_f1 = f1_score(
                train_labels.cpu().numpy(),
                train_preds.cpu().numpy(),
                average='samples',
                zero_division=0
            )
            train_f1_scores.append(train_f1)
            
            # NEW: Compute training accuracy
            train_acc_dict = compute_accuracy(
                train_labels.cpu().numpy(),
                train_preds.cpu().numpy()
            )
            train_accuracies.append(train_acc_dict['overall_accuracy'])
        
        # === VALIDATION PHASE ===
        model.eval()
        with torch.no_grad():
            val_outputs = outputs[val_idx]
            val_labels_batch = labels[val_idx]
            
            # Compute validation loss
            val_loss = criterion(val_outputs, val_labels_batch)
            val_losses.append(val_loss.item())
            
            # Compute validation predictions
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            
            # Compute validation F1 scores
            val_f1 = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='samples',
                zero_division=0
            )
            val_f1_scores.append(val_f1)
            
            val_micro = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='micro',
                zero_division=0
            )
            val_micro_f1.append(val_micro)
            
            val_macro = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='macro',
                zero_division=0
            )
            val_macro_f1.append(val_macro)
            
            # NEW: Compute validation accuracy
            val_acc_dict = compute_accuracy(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy()
            )
            val_accuracies.append(val_acc_dict['overall_accuracy'])
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.4f} → {val_loss.item():.4f} | "
                  f"F1: {train_f1:.4f} → {val_f1:.4f} | "
                  f"Acc: {train_acc_dict['overall_accuracy']:.4f} → {val_acc_dict['overall_accuracy']:.4f}")
        
        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc_dict['overall_accuracy']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation F1: {best_val_f1:.4f}")
    print(f"   Best validation Accuracy: {best_val_acc:.4f}")
    
    # Return all metrics
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_accuracies': train_accuracies,  # NEW
        'val_accuracies': val_accuracies,      # NEW
        'val_micro_f1': val_micro_f1,
        'val_macro_f1': val_macro_f1,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,          # NEW
        'total_epochs': len(train_losses)
    }
```

### 7. Initialize Model and Run Training
```python
# Model architecture parameters
INPUT_DIM = data_loader.features.shape[1]
HIDDEN_DIM = 64
OUTPUT_DIM = data_loader.labels.shape[1]
NUM_HEADS = 8
DROPOUT = 0.3

# Initialize model
model = HANModel(
    in_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUTPUT_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

print(f"✓ Model initialized: {INPUT_DIM} → {HIDDEN_DIM} → {OUTPUT_DIM}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Get train/val split
train_idx = data_loader.train_idx
val_idx = data_loader.val_idx

print(f"✓ Training set: {len(train_idx)} patients")
print(f"✓ Validation set: {len(val_idx)} patients")

# Run training
results = train_model_with_accuracy(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    features=data_loader.features,
    labels=data_loader.labels,
    adj_dict=data_loader.adj_dict,
    train_idx=train_idx,
    val_idx=val_idx,
    epochs=EPOCHS,
    patience=PATIENCE,
    device=DEVICE
)
```

### 8. Enhanced Visualization with Accuracy
```python
print("\n" + "="*80)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*80)

# Create enhanced 6-subplot visualization
plot_path = os.path.join(OUT_DIR, "training_metrics_enhanced.png")

plot_training_metrics_enhanced(
    train_losses=results['train_losses'],
    val_losses=results['val_losses'],
    train_f1=results['train_f1_scores'],
    val_f1=results['val_f1_scores'],
    val_micro_f1=results['val_micro_f1'],
    val_macro_f1=results['val_macro_f1'],
    train_accuracies=results['train_accuracies'],  # NEW
    val_accuracies=results['val_accuracies'],      # NEW
    model_name="HAN Medical Predictor",
    figsize=(18, 12),
    save_path=plot_path
)

print(f"✅ Enhanced visualization saved to: {plot_path}")
print("   Plot contains 6 subplots:")
print("   1. Training & Validation Loss")
print("   2. F1 Score (Samples Average)")
print("   3. Micro & Macro F1 Scores")
print("   4. Overall Accuracy")  # NEW
print("   5. F1 vs Accuracy Comparison")  # NEW
print("   6. Final Metrics Summary Table")  # NEW
```

### 9. Results Summary
```python
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"{'Metric':<25} {'Training':<15} {'Validation':<15}")
print("-" * 55)
print(f"{'Loss':<25} {results['train_losses'][-1]:<15.4f} {results['val_losses'][-1]:<15.4f}")
print(f"{'F1 Score (Samples)':<25} {results['train_f1_scores'][-1]:<15.4f} {results['val_f1_scores'][-1]:<15.4f}")
print(f"{'Accuracy':<25} {results['train_accuracies'][-1]:<15.4f} {results['val_accuracies'][-1]:<15.4f}")
print(f"{'Micro F1':<25} {'-':<15} {results['val_micro_f1'][-1]:<15.4f}")
print(f"{'Macro F1':<25} {'-':<15} {results['val_macro_f1'][-1]:<15.4f}")
print()
print(f"{'Best Validation F1':<25} {results['best_val_f1']:<15.4f}")
print(f"{'Best Validation Accuracy':<25} {results['best_val_acc']:<15.4f}")
print(f"{'Total Epochs':<25} {results['total_epochs']:<15}")

# Save results to JSON
import json
results_json = {
    'best_val_f1': float(results['best_val_f1']),
    'best_val_acc': float(results['best_val_acc']),
    'final_train_loss': float(results['train_losses'][-1]),
    'final_val_loss': float(results['val_losses'][-1]),
    'final_val_micro_f1': float(results['val_micro_f1'][-1]),
    'final_val_macro_f1': float(results['val_macro_f1'][-1]),
    'total_epochs': results['total_epochs']
}

json_path = os.path.join(OUT_DIR, 'training_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n✅ Results saved to: {json_path}")
```

---

## 📊 Understanding the Enhanced Plots

### Plot 1: Training & Validation Loss
- Shows how well the model is learning over time
- Both curves should decrease and converge
- Large gap indicates overfitting

### Plot 2: F1 Score (Samples Average)
- Primary metric for multi-label classification
- Measures precision-recall trade-off for each sample
- Higher is better (max = 1.0)

### Plot 3: Micro & Macro F1
- **Micro F1**: Aggregates all predictions (sensitive to class imbalance)
- **Macro F1**: Averages per-class F1 (treats all classes equally)

### Plot 4: Overall Accuracy ⭐ **NEW**
- Shows percentage of correctly predicted labels
- Tracks both training and validation accuracy
- Complements F1 score with interpretable metric

### Plot 5: F1 vs Accuracy Comparison ⭐ **NEW**
- Directly compares F1 and Accuracy trends
- Helps identify if metrics agree or diverge
- Useful for understanding model behavior

### Plot 6: Final Metrics Summary Table ⭐ **NEW**
- Displays all final metrics in one place
- Includes best scores achieved during training
- Easy reference for model performance

---

## 🎯 Key Features of This Setup

### ✅ Proper Data Paths
- Uses relative paths (`../data/`) for portability
- Verifies files exist before training
- Clear error messages if files missing

### ✅ Automatic Label Generation
- No need to provide pre-computed labels
- `MedicalGraphData` computes labels from test values
- Compares against normal ranges from knowledge graph

### ✅ Enhanced Accuracy Tracking
- `compute_accuracy()` provides overall and per-organ accuracy
- Tracks both training and validation accuracy
- Complements F1 scores for comprehensive evaluation

### ✅ Enhanced Visualization
- `plot_training_metrics_enhanced()` creates 6-subplot figure
- Shows loss, F1, accuracy, and comparisons
- Includes summary table for quick reference

### ✅ Best Model Saving
- Automatically saves model with best validation F1
- Also tracks best accuracy
- Includes early stopping to prevent overfitting

### ✅ Comprehensive Logging
- Progress printed every 5 epochs
- Final summary with all metrics
- Results saved to JSON for record-keeping

---

## 🚀 Quick Start

1. **Ensure data files are in place**:
   ```bash
   ls -la data/
   # Should see: filtered_patient_reports.csv, test-disease-organ.csv
   ```

2. **Open notebook**: `notebooks/train.ipynb`

3. **Run cells sequentially**:
   - Configuration → Data Loading → Training → Visualization

4. **Check outputs**:
   ```bash
   ls -la output/
   # Should see: best_model.pth, training_metrics_enhanced.png, training_results.json
   ```

---

## 📝 Notes

- **Labels**: Automatically generated from test values vs normal ranges
- **Device**: Automatically uses GPU if available, falls back to CPU
- **Early Stopping**: Training stops if no improvement for 10 epochs
- **Reproducibility**: Set `SEED=42` for consistent results
- **Customization**: Adjust `BATCH_SIZE`, `EPOCHS`, `LR` in configuration

---

## 🔧 Troubleshooting

### "FileNotFoundError: No such file"
- Check that data files are in `data/` folder
- Use absolute paths if needed: `/Users/charlie/.../data/...`

### "CUDA out of memory"
- Reduce `BATCH_SIZE` (try 64 or 32)
- Use CPU instead: `DEVICE = torch.device('cpu')`

### "Low F1/Accuracy scores"
- Increase `EPOCHS` (try 60-80)
- Adjust learning rate `LR` (try 5e-4 or 5e-3)
- Increase `HIDDEN_DIM` (try 128 or 256)
- Add more symptoms (`SYMPTOM_FREQ_THRESHOLD=3`)

### "Model not improving"
- Check for data leakage (train/val split)
- Ensure labels are balanced
- Try different `SEED` values
- Adjust `DROPOUT` (try 0.2-0.5)

---

## ✨ Summary

This setup provides:
- ✅ Proper data path configuration
- ✅ Automatic label generation
- ✅ Accuracy tracking alongside F1 scores
- ✅ Enhanced 6-subplot visualization
- ✅ Best model saving with early stopping
- ✅ Comprehensive results logging

**You now have a complete, production-ready training pipeline!** 🎉
