"""
Training Script for Model 4 - Consensus RetNet
==============================================

Task: Multi-class classification (5 consensus mechanisms)
Target: >96.9% classification accuracy
Device: MPS (Apple Silicon GPU)

Real-time output: ✅ Terminal displays training progress

Author: NOK KO
Date: 2026-01-28
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
import time
from pathlib import Path
import sys

# Add models path
SRC_DIR = Path(__file__).parent               # src/
BASE_DIR = SRC_DIR.parent                     # GIT_MODEL4/
sys.path.insert(0, str(SRC_DIR / 'models'))
from retnet_consensus import create_model


class ConsensusDataset(Dataset):
    """Dataset for consensus mechanism selection"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Feature names (12 features)
        self.feature_names = [
            'num_nodes', 'connectivity', 'latency_requirement_sec',
            'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
            'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
            'decentralization_requirement', 'network_load', 'attack_risk'
        ]
        
        # Label mapping
        self.label_map = {'PoW': 0, 'PoS': 1, 'PBFT': 2, 'DPoS': 3, 'Hybrid': 4}
        self.label_names = ['PoW', 'PoS', 'PBFT', 'DPoS', 'Hybrid']
        
        # Extract features and labels
        self.features = []
        self.labels = []
        
        for sample in self.data:
            # Features
            feat = [sample[name] for name in self.feature_names]
            self.features.append(feat)
            
            # Label
            label = self.label_map[sample['optimal_mechanism']]
            self.labels.append(label)
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Normalize features
        self.feat_mean = self.features.mean(dim=0)
        self.feat_std = self.features.std(dim=0) + 1e-8
        self.features = (self.features - self.feat_mean) / self.feat_std
        
        print(f"✅ Loaded {len(self)} samples")
        print(f"   Features: {self.features.shape}")
        print(f"   Labels: {self.labels.shape}")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_class_weights(self):
        """Compute class weights for imbalanced data"""
        class_counts = torch.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Real-time progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\r  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"({progress:5.1f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:5.2f}% | "
                  f"Time: {elapsed:5.1f}s", end='', flush=True)
    
    print()  # Newline after progress bar
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    epoch_time = time.time() - start_time
    
    return avg_loss, accuracy, epoch_time


def evaluate(model, val_loader, criterion, device, dataset):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class statistics
    class_correct = [0] * 5
    class_total = [0] * 5
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Per-class accuracy
    class_acc = {}
    for i, name in enumerate(dataset.label_names):
        if class_total[i] > 0:
            class_acc[name] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0.0
    
    return avg_loss, accuracy, class_acc


def main():
    print("\n" + "=" * 80)
    print("Model 4: Consensus RetNet Training")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"  Task: Multi-class classification (5 classes)")
    print(f"  Architecture: RetNet (Multi-scale Retention)")
    print(f"  Device: MPS (Apple Silicon GPU)")
    print(f"  Target Accuracy: >96.9%")
    print("\n" + "=" * 80 + "\n")
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✅ Using device: {device}\n")
    
    # Hyperparameters
    BATCH_SIZE = 128  # Larger batch for stability
    EPOCHS = 100
    BASE_LR = 3e-4  # Higher learning rate
    WEIGHT_DECAY = 0.01
    
    # Load dataset (V2 - balanced and discriminative)
    print("📊 Loading dataset V2 (balanced)...")
    data_path = BASE_DIR / 'data' / 'training_data' / 'consensus_training_data.json'
    dataset = ConsensusDataset(str(data_path))
    
    # Split dataset (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model(
        input_dim=12,
        output_dim=5,
        d_model=384,
        num_layers=3,
        num_heads=3,
        dropout=0.1
    )
    model = model.to(device)
    
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}\n")
    
    # No class weights needed - dataset is balanced!
    print(f"📊 Dataset is balanced - using equal weights\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # No class weights
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR * 2, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler (3-stage)
    def lr_lambda(epoch):
        if epoch < 10:  # Warmup
            return (epoch + 1) / 10
        elif epoch < 40:  # Stable
            return 1.0
        else:  # Cosine decay
            progress = (epoch - 40) / (EPOCHS - 40)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("=" * 80)
    print("🚀 Starting training...")
    print("=" * 80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'per_class_acc': []
    }
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, class_acc = evaluate(
            model, val_loader, criterion, device, dataset
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:6.2f}%")
        print(f"  Time: {train_time:.1f}s")
        
        # Print per-class accuracy
        print(f"\n  Per-Class Accuracy:")
        for name in dataset.label_names:
            acc = class_acc[name]
            status = "✅" if acc > 90 else "⚠️" if acc > 80 else "❌"
            print(f"    {name:8s}: {acc:5.2f}% {status}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['lr'].append(current_lr)
        training_history['per_class_acc'].append(class_acc)
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_path = BASE_DIR / 'training' / 'checkpoints' / 'best_consensus.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_acc': class_acc,
                'feature_stats': {
                    'mean': dataset.feat_mean.tolist(),
                    'std': dataset.feat_std.tolist()
                }
            }, save_path)
            
            print(f"\n  🎯 NEW BEST! Val Acc: {val_acc:.2f}% (saved to {save_path.name})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹️  Early stopping triggered (patience={patience})")
                break
        
        print("\n" + "=" * 80 + "\n")
    
    # Final evaluation on test set
    print("=" * 80)
    print("🎯 Final Evaluation on Test Set")
    print("=" * 80 + "\n")
    
    # Load best model
    checkpoint = torch.load(str(save_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_class_acc = evaluate(
        model, test_loader, criterion, device, dataset
    )
    
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    print(f"\n  Per-Class Test Accuracy:")
    for name in dataset.label_names:
        acc = test_class_acc[name]
        status = "✅" if acc > 90 else "⚠️" if acc > 80 else "❌"
        print(f"    {name:8s}: {acc:5.2f}% {status}")
    
    # Save training history
    history_path = BASE_DIR / 'training' / 'history' / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n  ✅ Training history saved to {history_path.name}")
    
    print("\n" + "=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)
    print(f"\n  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Final Test Accuracy: {test_acc:.2f}%")
    print(f"  Target Achieved: {'✅ YES' if test_acc > 96.9 else '⚠️  NO (adjust and retrain)'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
