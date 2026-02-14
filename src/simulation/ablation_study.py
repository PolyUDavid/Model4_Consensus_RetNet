#!/usr/bin/env python3
"""
Model 4: ConsensusRetNet - Comprehensive Ablation Study
========================================================

Ablation experiments to validate each architectural design choice:

1. Number of RetNet Layers (1, 2, 3, 4, 5)
2. Model Dimension d_model (64, 128, 192, 256, 384, 512)
3. Number of Attention Heads (1, 2, 3)
4. Decay Rate γ variations
5. Feed-Forward Expansion ratio (2x, 4x, 6x)
6. Dropout rate (0.0, 0.05, 0.1, 0.2, 0.3)
7. Label Smoothing (0.0, 0.05, 0.1, 0.15, 0.2)
8. Feature Ablation (remove each feature group)

Author: NOK KO
Date: 2026-02-05
"""

import sys, os, json, time, copy
import numpy as np
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent       # reaches src/
BASE_DIR = SRC_DIR.parent                    # reaches GIT_MODEL4/
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / 'models'))

import torch
import torch.nn as nn

from models.retnet_consensus import ConsensusRetNet, create_model

# Output directory
OUT_DIR = BASE_DIR / 'data' / 'ablation_study'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['PoW', 'PoS', 'PBFT', 'DPoS', 'Hybrid']
FEATURE_KEYS = [
    'num_nodes', 'connectivity', 'latency_requirement_sec',
    'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
    'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
    'decentralization_requirement', 'network_load', 'attack_risk'
]

FEATURE_GROUPS = {
    'Topology': ['num_nodes', 'connectivity'],
    'Performance': ['latency_requirement_sec', 'throughput_requirement_tps'],
    'Security': ['byzantine_tolerance', 'security_priority', 'attack_risk'],
    'Resource': ['energy_budget', 'bandwidth_mbps'],
    'Consensus': ['consistency_requirement', 'decentralization_requirement'],
    'Load': ['network_load'],
}


def load_data():
    """Load and split training data"""
    data_path = BASE_DIR / 'data' / 'training_data' / 'consensus_training_data.json'
    
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    features_list = []
    labels_list = []
    for sample in dataset:
        feats = [sample[k] for k in FEATURE_KEYS]
        features_list.append(feats)
        labels_list.append(CLASSES.index(sample['optimal_mechanism']))
    
    X = torch.tensor(features_list, dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.long)
    
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    torch.manual_seed(42)
    indices = torch.randperm(n)
    
    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val = X[indices[n_train:n_train+n_val]]
    y_val = y[indices[n_train:n_train+n_val]]
    X_test = X[indices[n_train+n_val:]]
    y_test = y[indices[n_train+n_val:]]
    
    # Compute normalization stats from training set
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    
    feature_stats = {'mean': mean, 'std': std}
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_stats


def normalize(X, feature_stats):
    mean = feature_stats['mean']
    std = feature_stats['std']
    return (X - mean.unsqueeze(0)) / (std.unsqueeze(0) + 1e-8)


def train_model(model, X_train, y_train, X_val, y_val, feature_stats,
                epochs=80, lr=6e-4, weight_decay=0.01, label_smoothing=0.1,
                batch_size=128, patience=15, verbose=False):
    """Train a model and return best val accuracy + metrics"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Warmup + cosine schedule
    warmup_epochs = min(10, epochs // 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    X_tr = normalize(X_train, feature_stats)
    X_v = normalize(X_val, feature_stats)
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    best_epoch = 0
    train_losses = []
    val_accs = []
    
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_tr), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_tr[batch_idx]
            yb = y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item() * 100
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = epoch + 1
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    training_time = time.time() - t0
    
    if best_state:
        model.load_state_dict(best_state)
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'training_time': training_time,
        'final_train_loss': train_losses[-1] if train_losses else 0,
    }


def evaluate_test(model, X_test, y_test, feature_stats):
    """Evaluate on test set"""
    model.eval()
    X_t = normalize(X_test, feature_stats)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1)
        test_acc = (preds == y_test).float().mean().item() * 100
        
        # Per-class
        per_class = {}
        for i, c in enumerate(CLASSES):
            mask = y_test == i
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == y_test[mask]).float().mean().item() * 100
    
    return test_acc, per_class


def measure_inference_time(model, input_dim=12, n_runs=1000, batch_size=1):
    """Measure inference latency"""
    model.eval()
    x = torch.randn(batch_size, input_dim)
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }


# ==============================================================================
# Ablation Studies
# ==============================================================================

def ablation_num_layers(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 1: Number of RetNet layers"""
    print("\n" + "=" * 70)
    print("  ABLATION 1: Number of RetNet Layers")
    print("=" * 70)
    
    results = {}
    for n_layers in [1, 2, 3, 4, 5]:
        print(f"  Training with {n_layers} layers...")
        model = ConsensusRetNet(num_layers=n_layers, d_model=384, num_heads=3)
        n_params = sum(p.numel() for p in model.parameters())
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        latency = measure_inference_time(model)
        
        results[n_layers] = {
            'num_layers': n_layers,
            'parameters': n_params,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'total_epochs': train_res['total_epochs'],
            'training_time_s': round(train_res['training_time'], 1),
            'inference_ms': round(latency['mean_ms'], 3),
        }
        print(f"    Layers={n_layers}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%, "
              f"Params={n_params:,}, Time={train_res['training_time']:.1f}s, "
              f"Latency={latency['mean_ms']:.3f}ms")
    
    return results


def ablation_d_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 2: Model dimension d_model"""
    print("\n" + "=" * 70)
    print("  ABLATION 2: Model Dimension (d_model)")
    print("=" * 70)
    
    results = {}
    for d in [48, 96, 192, 384, 576]:
        # num_heads must divide d_model
        n_heads = 3 if d % 3 == 0 else (2 if d % 2 == 0 else 1)
        print(f"  Training with d_model={d}, heads={n_heads}...")
        model = ConsensusRetNet(d_model=d, num_layers=3, num_heads=n_heads)
        n_params = sum(p.numel() for p in model.parameters())
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        latency = measure_inference_time(model)
        
        results[d] = {
            'd_model': d,
            'num_heads': n_heads,
            'parameters': n_params,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'training_time_s': round(train_res['training_time'], 1),
            'inference_ms': round(latency['mean_ms'], 3),
        }
        print(f"    d={d}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%, "
              f"Params={n_params:,}, Latency={latency['mean_ms']:.3f}ms")
    
    return results


def ablation_num_heads(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 3: Number of attention heads"""
    print("\n" + "=" * 70)
    print("  ABLATION 3: Number of Attention Heads")
    print("=" * 70)
    
    # d_model=384 is divisible by 1,2,3,4,6,8
    results = {}
    for n_heads in [1, 2, 3, 4, 6, 8]:
        print(f"  Training with {n_heads} heads...")
        model = ConsensusRetNet(d_model=384, num_layers=3, num_heads=n_heads)
        n_params = sum(p.numel() for p in model.parameters())
        
        # Need to adjust gammas for different head counts
        for layer in model.layers:
            retention = layer.retention
            if n_heads == 1:
                gammas = torch.tensor([0.95])
            elif n_heads == 2:
                gammas = torch.tensor([0.9, 0.99])
            elif n_heads == 3:
                gammas = torch.tensor([0.9, 0.95, 0.99])
            elif n_heads == 4:
                gammas = torch.tensor([0.85, 0.9, 0.95, 0.99])
            elif n_heads == 6:
                gammas = torch.tensor([0.8, 0.85, 0.9, 0.95, 0.97, 0.99])
            elif n_heads == 8:
                gammas = torch.tensor([0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99])
            retention.gammas = nn.Parameter(gammas, requires_grad=False)
            retention.group_norm = nn.GroupNorm(n_heads, 384)
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        
        results[n_heads] = {
            'num_heads': n_heads,
            'parameters': n_params,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'training_time_s': round(train_res['training_time'], 1),
        }
        print(f"    Heads={n_heads}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%")
    
    return results


def ablation_dropout(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 4: Dropout rate"""
    print("\n" + "=" * 70)
    print("  ABLATION 4: Dropout Rate")
    print("=" * 70)
    
    results = {}
    for drop in [0.0, 0.05, 0.1, 0.2, 0.3]:
        print(f"  Training with dropout={drop}...")
        model = ConsensusRetNet(d_model=384, num_layers=3, num_heads=3, dropout=drop)
        n_params = sum(p.numel() for p in model.parameters())
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        
        results[str(drop)] = {
            'dropout': drop,
            'parameters': n_params,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'training_time_s': round(train_res['training_time'], 1),
        }
        print(f"    Drop={drop}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%")
    
    return results


def ablation_label_smoothing(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 5: Label smoothing"""
    print("\n" + "=" * 70)
    print("  ABLATION 5: Label Smoothing (ε)")
    print("=" * 70)
    
    results = {}
    for eps in [0.0, 0.05, 0.1, 0.15, 0.2]:
        print(f"  Training with ε={eps}...")
        model = ConsensusRetNet(d_model=384, num_layers=3, num_heads=3)
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats,
                               label_smoothing=eps)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        
        results[str(eps)] = {
            'label_smoothing': eps,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'training_time_s': round(train_res['training_time'], 1),
        }
        print(f"    ε={eps}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%")
    
    return results


def ablation_feature_groups(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 6: Feature group importance"""
    print("\n" + "=" * 70)
    print("  ABLATION 6: Feature Group Importance (Leave-One-Group-Out)")
    print("=" * 70)
    
    results = {}
    
    # Baseline: all features
    print(f"  Training with ALL features (baseline)...")
    model = ConsensusRetNet(input_dim=12, d_model=384, num_layers=3, num_heads=3)
    train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
    test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
    results['All_Features'] = {
        'removed_group': 'None',
        'remaining_features': 12,
        'val_acc': train_res['best_val_acc'],
        'test_acc': test_acc,
        'accuracy_drop': 0.0,
    }
    baseline_acc = test_acc
    print(f"    All Features: Test={test_acc:.2f}%")
    
    # Remove each feature group
    for group_name, group_features in FEATURE_GROUPS.items():
        feature_indices = [FEATURE_KEYS.index(f) for f in group_features]
        keep_indices = [i for i in range(12) if i not in feature_indices]
        n_remaining = len(keep_indices)
        
        print(f"  Training WITHOUT {group_name} ({group_features})...")
        
        # Select features
        X_tr_sub = X_train[:, keep_indices]
        X_v_sub = X_val[:, keep_indices]
        X_te_sub = X_test[:, keep_indices]
        
        # Recompute feature stats for subset
        sub_stats = {
            'mean': feature_stats['mean'][keep_indices],
            'std': feature_stats['std'][keep_indices],
        }
        
        # Adjust model for fewer features, use smaller d_model proportionally
        d_model = 384
        n_heads = 3
        model = ConsensusRetNet(input_dim=n_remaining, d_model=d_model, num_layers=3, num_heads=n_heads)
        
        train_res = train_model(model, X_tr_sub, y_train, X_v_sub, y_val, sub_stats)
        
        model.eval()
        X_t = normalize(X_te_sub, sub_stats)
        with torch.no_grad():
            preds = model(X_t).argmax(dim=1)
            test_acc = (preds == y_test).float().mean().item() * 100
        
        acc_drop = baseline_acc - test_acc
        
        results[group_name] = {
            'removed_group': group_name,
            'removed_features': group_features,
            'remaining_features': n_remaining,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'accuracy_drop': round(acc_drop, 4),
        }
        print(f"    Without {group_name}: Test={test_acc:.2f}% (Δ={acc_drop:+.2f}%)")
    
    return results


def ablation_decay_rates(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats):
    """Ablation 7: Decay rate configurations"""
    print("\n" + "=" * 70)
    print("  ABLATION 7: Decay Rate (γ) Configurations")
    print("=" * 70)
    
    configs = {
        'Short-only (0.9, 0.9, 0.9)': [0.9, 0.9, 0.9],
        'Long-only (0.99, 0.99, 0.99)': [0.99, 0.99, 0.99],
        'Uniform (0.95, 0.95, 0.95)': [0.95, 0.95, 0.95],
        'Default (0.9, 0.95, 0.99)': [0.9, 0.95, 0.99],
        'Wide (0.8, 0.95, 0.999)': [0.8, 0.95, 0.999],
        'Narrow (0.92, 0.95, 0.98)': [0.92, 0.95, 0.98],
    }
    
    results = {}
    for name, gammas in configs.items():
        print(f"  Training with γ = {gammas}...")
        model = ConsensusRetNet(d_model=384, num_layers=3, num_heads=3)
        
        # Set custom gammas
        for layer in model.layers:
            layer.retention.gammas = nn.Parameter(torch.tensor(gammas), requires_grad=False)
        
        train_res = train_model(model, X_train, y_train, X_val, y_val, feature_stats)
        test_acc, per_class = evaluate_test(model, X_test, y_test, feature_stats)
        
        results[name] = {
            'gammas': gammas,
            'val_acc': train_res['best_val_acc'],
            'test_acc': test_acc,
            'per_class': per_class,
            'best_epoch': train_res['best_epoch'],
            'training_time_s': round(train_res['training_time'], 1),
        }
        print(f"    {name}: Val={train_res['best_val_acc']:.2f}%, Test={test_acc:.2f}%")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("  Model 4: ConsensusRetNet — Comprehensive Ablation Study")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_stats = load_data()
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    all_results = {}
    total_t0 = time.time()
    
    # Run all ablations
    all_results['1_num_layers'] = ablation_num_layers(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['2_d_model'] = ablation_d_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['3_num_heads'] = ablation_num_heads(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['4_dropout'] = ablation_dropout(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['5_label_smoothing'] = ablation_label_smoothing(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['6_feature_groups'] = ablation_feature_groups(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    all_results['7_decay_rates'] = ablation_decay_rates(X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    
    total_time = time.time() - total_t0
    
    # Save results
    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    serializable = make_serializable(all_results)
    serializable['_meta'] = {
        'total_time_seconds': round(total_time, 1),
        'total_time_minutes': round(total_time / 60, 1),
        'dataset_size': len(X_train) + len(X_val) + len(X_test),
        'device': 'CPU',
        'author': 'NOK KO',
        'date': '2026-02-05',
    }
    
    output_path = OUT_DIR / 'ablation_study_results.json'
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results saved: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
