#!/usr/bin/env python3
"""
Model 4: ConsensusRetNet - Comprehensive Validation Experiments
===============================================================

Experiment 1: Accuracy vs Architecture Comparison
    - RetNet vs MLP vs LSTM vs CNN baselines on the same data
    
Experiment 2: Dynamic Scenario Switching
    - Test model's consensus selection across 3 vehicular scenarios
    - Validate physical reasonableness of predictions

Experiment 3: Byzantine Attack Resilience
    - Sweep attack_risk from 0 to 1.0
    - Show how consensus selection adapts

Experiment 4: Confusion Matrix & Per-Class Analysis
    - Full test set evaluation
    - Confusion matrix, precision, recall, F1

Experiment 5: Latency-Throughput Trade-off
    - 2D sweep: latency requirement vs throughput requirement
    - Heatmap of selected consensus mechanisms

Author: NOK KO
Date: 2026-02-05
"""

import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
SRC_DIR = Path(__file__).parent.parent       # reaches src/
BASE_DIR = SRC_DIR.parent                    # reaches GIT_MODEL4/
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / 'models'))

import torch
import torch.nn as nn
from models.retnet_consensus import create_model

# Output directory
OUT_DIR = BASE_DIR / 'figures' / 'experiment_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['PoW', 'PoS', 'PBFT', 'DPoS', 'Hybrid']
CLASS_COLORS = ['#F1C40F', '#2ECC71', '#9B59B6', '#3498DB', '#E67E22']

FEATURE_KEYS = [
    'num_nodes', 'connectivity', 'latency_requirement_sec',
    'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
    'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
    'decentralization_requirement', 'network_load', 'attack_risk'
]


# ==============================================================================
# Utilities
# ==============================================================================

def load_model_and_data():
    """Load trained model and test data"""
    # Model
    ckpt_path = BASE_DIR / 'training' / 'checkpoints' / 'best_consensus.pth'
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
    
    model = create_model()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    feature_stats = ckpt.get('feature_stats', None)
    
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Val accuracy: {ckpt.get('val_acc', 'N/A'):.2f}%")
    
    # Data
    data_path = BASE_DIR / 'data' / 'training_data' / 'consensus_training_data.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    # Convert to tensors
    features_list = []
    labels_list = []
    for sample in dataset:
        feats = [sample[k] for k in FEATURE_KEYS]
        features_list.append(feats)
        labels_list.append(CLASSES.index(sample['optimal_mechanism']))
    
    X = torch.tensor(features_list, dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.long)
    
    # Split: 70/15/15
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
    
    print(f"  Data: {n} total -> {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    
    return model, feature_stats, X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X, feature_stats):
    """Apply feature normalization"""
    if feature_stats is None:
        return X
    mean = feature_stats['mean']
    std = feature_stats['std']
    if isinstance(mean, torch.Tensor):
        return (X - mean.unsqueeze(0)) / (std.unsqueeze(0) + 1e-8)
    else:
        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)
        return (X - mean_t.unsqueeze(0)) / (std_t.unsqueeze(0) + 1e-8)


def predict_batch(model, X, feature_stats, batch_size=512):
    """Batch prediction"""
    model.eval()
    all_preds = []
    all_probs = []
    X_norm = normalize(X, feature_stats)
    
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = X_norm[i:i+batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds)
            all_probs.append(probs)
    
    return torch.cat(all_preds), torch.cat(all_probs)


# ==============================================================================
# Baseline Models
# ==============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=12, hidden=256, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden=128, output_dim=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 12)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 12)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


def train_baseline(model_class, X_train, y_train, X_val, y_val, 
                   feature_stats, epochs=150, name="Model"):
    """Train a baseline model"""
    model = model_class()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    X_tr = normalize(X_train, feature_stats)
    X_v = normalize(X_val, feature_stats)
    
    best_acc = 0
    best_state = None
    patience = 30
    no_improve = 0
    batch_size = 256
    
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item() * 100
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"    {name}: {best_acc:.2f}% (params: {n_params:,}, epochs: {epoch+1})")
    return model, best_acc, n_params


# ==============================================================================
# Experiment 1: Architecture Comparison
# ==============================================================================

def experiment_1_architecture_comparison(model, feature_stats, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Architecture Comparison")
    print("=" * 70)
    
    results = {}
    
    # RetNet (already trained)
    preds, probs = predict_batch(model, X_test, feature_stats)
    retnet_acc = (preds == y_test).float().mean().item() * 100
    retnet_params = sum(p.numel() for p in model.parameters())
    results['RetNet'] = {'acc': retnet_acc, 'params': retnet_params}
    print(f"    RetNet:     {retnet_acc:.2f}% (params: {retnet_params:,})")
    
    # Train baselines
    for name, cls in [('MLP', SimpleMLP), ('LSTM', SimpleLSTM), ('CNN', SimpleCNN)]:
        m, acc, params = train_baseline(cls, X_train, y_train, X_val, y_val, feature_stats, epochs=150, name=name)
        # Test
        m.eval()
        X_t_norm = normalize(X_test, feature_stats)
        with torch.no_grad():
            test_preds = m(X_t_norm).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item() * 100
        results[name] = {'acc': test_acc, 'params': params}
        print(f"    {name} Test:  {test_acc:.2f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    params = [results[n]['params'] for n in names]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F']
    
    # Accuracy bar chart
    ax = axes[0]
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(accs) - 2, 101])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Parameters bar chart
    ax = axes[1]
    bars = ax.bar(names, [p/1000 for p in params], color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Parameters (K)', fontsize=12)
    ax.set_title('Model Complexity', fontsize=14, fontweight='bold')
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)/1000*0.02,
                f'{p/1000:.1f}K', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '1_architecture_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_architecture_comparison.png")
    
    return results


# ==============================================================================
# Experiment 2: Dynamic Scenario Switching
# ==============================================================================

def experiment_2_scenario_switching(model, feature_stats):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Dynamic Scenario Switching")
    print("=" * 70)
    
    # Simulate a 100-second vehicular journey through 5 distinct phases
    # Each phase targets a different consensus mechanism based on actual data distributions
    time_steps = 1000  # 0.1s resolution
    timestamps = np.linspace(0, 100, time_steps)
    
    # Phases designed to match V2 data generator's decision boundaries:
    # 0-20s:  Normal (Hybrid) - Balanced requirements
    # 20-40s: Energy Crisis (PoS) - Low energy, medium security
    # 40-60s: Small Network + Low Latency (PBFT) - Few nodes, instant finality
    # 60-80s: Byzantine Attack (PoW) - High security, high energy, high decentralization
    # 80-100s: Mass Scale-up (DPoS) - Large network, high throughput
    
    predictions = []
    confidences = []
    features_over_time = {k: [] for k in FEATURE_KEYS}
    
    for i, t in enumerate(timestamps):
        if t < 20:
            # Phase 1: Hybrid - balanced scenario
            phase = t / 20
            state = {
                'num_nodes': 150,
                'connectivity': 0.72,
                'latency_requirement_sec': 12.0,
                'throughput_requirement_tps': 1200,
                'byzantine_tolerance': 0.17,
                'security_priority': 0.80,
                'energy_budget': 0.60,
                'bandwidth_mbps': 1500,
                'consistency_requirement': 0.75,
                'decentralization_requirement': 0.70,
                'network_load': 0.45 + 0.05 * phase,
                'attack_risk': 0.55,
            }
        elif t < 40:
            # Phase 2: PoS - energy constrained, medium security
            phase = (t - 20) / 20
            state = {
                'num_nodes': int(150 + 50 * phase),
                'connectivity': 0.72 + 0.03 * phase,
                'latency_requirement_sec': 12.0 + 8 * phase,
                'throughput_requirement_tps': int(1200 - 1140 * phase),
                'byzantine_tolerance': 0.17 - 0.02 * phase,
                'security_priority': 0.80 - 0.02 * phase,
                'energy_budget': 0.60 - 0.35 * phase,  # Drop to 0.25 (LOW)
                'bandwidth_mbps': 1500 - 700 * phase,
                'consistency_requirement': 0.75 - 0.0 * phase,
                'decentralization_requirement': 0.70 + 0.05 * phase,
                'network_load': 0.50 - 0.05 * phase,
                'attack_risk': 0.55 + 0.08 * phase,
            }
        elif t < 60:
            # Phase 3: PBFT - small network, ultra-low latency, strong consistency
            phase = (t - 40) / 20
            state = {
                'num_nodes': int(200 - 160 * phase),  # Shrink to ~40
                'connectivity': 0.75 + 0.15 * phase,  # High connectivity
                'latency_requirement_sec': 20.0 - 17.5 * phase,  # Down to 2.5s
                'throughput_requirement_tps': int(60 + 2000 * phase),  # Ramp up
                'byzantine_tolerance': 0.15 + 0.07 * phase,
                'security_priority': 0.78 - 0.03 * phase,
                'energy_budget': 0.25 + 0.25 * phase,  # Back to medium
                'bandwidth_mbps': 800 + 2000 * phase,
                'consistency_requirement': 0.75 + 0.17 * phase,  # Strong >0.90
                'decentralization_requirement': 0.75 - 0.30 * phase,  # Low ~0.45
                'network_load': 0.45 + 0.10 * phase,
                'attack_risk': 0.63 - 0.0 * phase,
            }
        elif t < 80:
            # Phase 4: PoW - high security, high energy, high decentralization, high attack
            phase = (t - 60) / 20
            state = {
                'num_nodes': int(40 + 700 * phase),
                'connectivity': 0.90 - 0.05 * phase,
                'latency_requirement_sec': 2.5 + 42 * phase,  # High tolerance
                'throughput_requirement_tps': int(2060 - 2030 * phase),  # Low
                'byzantine_tolerance': 0.22 + 0.06 * phase,  # High BFT
                'security_priority': 0.75 + 0.19 * phase,  # Very high >0.90
                'energy_budget': 0.50 + 0.38 * phase,  # High >0.80
                'bandwidth_mbps': 2800 - 2300 * phase,
                'consistency_requirement': 0.92 - 0.02 * phase,
                'decentralization_requirement': 0.45 + 0.47 * phase,  # Very high >0.90
                'network_load': 0.55 - 0.25 * phase,
                'attack_risk': 0.63 + 0.22 * phase,  # High >0.80
            }
        else:
            # Phase 5: DPoS - large network, massive throughput, moderate decentralization
            phase = (t - 80) / 20
            state = {
                'num_nodes': int(740 - 340 * phase),  # Large >300
                'connectivity': 0.85 - 0.13 * phase,
                'latency_requirement_sec': 44.5 - 37 * phase,  # Low latency ~7
                'throughput_requirement_tps': int(30 + 7700 * phase),  # VERY HIGH
                'byzantine_tolerance': 0.28 - 0.16 * phase,
                'security_priority': 0.94 - 0.22 * phase,
                'energy_budget': 0.88 - 0.58 * phase,  # Low energy
                'bandwidth_mbps': 500 + 5000 * phase,  # High BW
                'consistency_requirement': 0.90 - 0.25 * phase,
                'decentralization_requirement': 0.92 - 0.30 * phase,  # Moderate
                'network_load': 0.30 + 0.40 * phase,
                'attack_risk': 0.85 - 0.45 * phase,
            }
        
        # Record features
        for k in FEATURE_KEYS:
            features_over_time[k].append(state[k])
        
        # Predict
        feats = [state[k] for k in FEATURE_KEYS]
        x = torch.tensor([feats], dtype=torch.float32)
        x_norm = normalize(x, feature_stats)
        
        with torch.no_grad():
            logits = model(x_norm)
            probs = torch.softmax(logits, dim=1)[0]
        
        pred_idx = probs.argmax().item()
        predictions.append(pred_idx)
        confidences.append(probs[pred_idx].item())
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # Panel 1: Consensus selection over time
    ax = axes[0]
    for i, c in enumerate(CLASSES):
        mask = np.array(predictions) == i
        if mask.any():
            ax.scatter(timestamps[mask], np.array(confidences)[mask], 
                      c=CLASS_COLORS[i], label=c, s=8, alpha=0.7)
    
    # Phase dividers
    for boundary in [20, 40, 60, 80]:
        ax.axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.3)
    
    # Phase labels
    phase_labels = [
        (10, 'HYBRID\n(Normal)', '#E67E22'),
        (30, 'PoS\n(Low Energy)', '#2ECC71'),
        (50, 'PBFT\n(Small Net)', '#9B59B6'),
        (70, 'PoW\n(Attack)', '#F1C40F'),
        (90, 'DPoS\n(Scale-up)', '#3498DB'),
    ]
    for x, label, color in phase_labels:
        ax.text(x, 1.03, label, ha='center', fontsize=10, fontweight='bold', color=color,
                transform=ax.get_xaxis_transform())
    
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('ConsensusRetNet: Dynamic Consensus Selection Under Changing Network Conditions', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0.3, 1.08])
    ax.grid(alpha=0.2)
    
    # Panel 2: Key network features
    ax = axes[1]
    ax.plot(timestamps, features_over_time['attack_risk'], 'r-', label='Attack Risk', linewidth=2)
    ax.plot(timestamps, features_over_time['network_load'], 'b-', label='Network Load', linewidth=2)
    ax.plot(timestamps, features_over_time['security_priority'], 'g-', label='Security Priority', linewidth=1.5)
    ax.plot(timestamps, features_over_time['energy_budget'], 'y-', label='Energy Budget', linewidth=1.5)
    for boundary in [20, 40, 60, 80]:
        ax.axvline(boundary, color='white', linestyle='--', alpha=0.3)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Network Conditions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_ylim([-0.05, 1.1])
    ax.grid(alpha=0.2)
    
    # Panel 3: Throughput requirement
    ax = axes[2]
    tps = np.array(features_over_time['throughput_requirement_tps'])
    ax.fill_between(timestamps, 0, tps, alpha=0.3, color='#3498DB')
    ax.plot(timestamps, tps, color='#3498DB', linewidth=2, label='Throughput Req (TPS)')
    for boundary in [20, 40, 60, 80]:
        ax.axvline(boundary, color='white', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TPS', fontsize=12)
    ax.set_title('Throughput Requirements', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '2_dynamic_scenario_switching.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 2_dynamic_scenario_switching.png")
    
    # Print summary
    phase_defs = [
        ('Hybrid (Normal)', 0, 200),
        ('PoS (Low Energy)', 200, 400),
        ('PBFT (Small Net)', 400, 600),
        ('PoW (Attack)', 600, 800),
        ('DPoS (Scale-up)', 800, 1000),
    ]
    for phase_name, start, end in phase_defs:
        phase_preds = predictions[start:end]
        counts = {c: phase_preds.count(i) for i, c in enumerate(CLASSES)}
        dominant = max(counts, key=counts.get)
        print(f"    {phase_name:20s}: {dominant} ({counts[dominant]/len(phase_preds):.0%})")


# ==============================================================================
# Experiment 3: Byzantine Attack Resilience
# ==============================================================================

def experiment_3_byzantine_resilience(model, feature_stats):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Byzantine Attack Resilience")
    print("=" * 70)
    
    attack_risks = np.linspace(0, 1.0, 50)
    results_per_risk = {c: [] for c in CLASSES}
    selected = []
    
    # Start from a Hybrid-like baseline, sweep features toward PoW as attack_risk increases
    for risk in attack_risks:
        # As attack risk rises, the network responds:
        # - Security priority increases (operators demand higher security)
        # - Energy budget increases (willing to spend more for safety)
        # - Decentralization requirement increases (avoid single points of failure)
        # - Throughput requirement drops (security over speed)
        # - Latency tolerance increases (security more important than speed)
        state = {
            'num_nodes': int(150 + 550 * risk),    # More nodes join for resilience
            'connectivity': max(0.60, 0.85 - 0.10 * risk),
            'latency_requirement_sec': 12.0 + 38 * risk,    # Tolerate more latency
            'throughput_requirement_tps': int(1200 - 1170 * risk),  # Less throughput needed
            'byzantine_tolerance': min(0.33, 0.12 + 0.21 * risk),  # Need more BFT
            'security_priority': min(1.0, 0.72 + 0.26 * risk),  # Demand more security
            'energy_budget': min(1.0, 0.40 + 0.55 * risk),     # Willing to spend energy
            'bandwidth_mbps': 1500 - 1000 * risk,
            'consistency_requirement': 0.75 + 0.15 * risk,      # Need strong consistency
            'decentralization_requirement': min(1.0, 0.65 + 0.30 * risk),  # Go decentralized
            'network_load': 0.40 + 0.10 * risk,
            'attack_risk': risk,
        }
        
        feats = [state[k] for k in FEATURE_KEYS]
        x = torch.tensor([feats], dtype=torch.float32)
        x_norm = normalize(x, feature_stats)
        
        with torch.no_grad():
            logits = model(x_norm)
            probs = torch.softmax(logits, dim=1)[0]
        
        for i, c in enumerate(CLASSES):
            results_per_risk[c].append(probs[i].item())
        
        selected.append(CLASSES[probs.argmax().item()])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Probability curves
    ax = axes[0]
    for i, c in enumerate(CLASSES):
        ax.plot(attack_risks, results_per_risk[c], color=CLASS_COLORS[i], 
                label=c, linewidth=2.5)
    ax.set_xlabel('Attack Risk', fontsize=12)
    ax.set_ylabel('Prediction Probability', fontsize=12)
    ax.set_title('Consensus Probability vs Attack Risk', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    # Transition visualization
    ax = axes[1]
    # Color-code the attack risk axis by selected consensus
    for i, (risk, cons) in enumerate(zip(attack_risks, selected)):
        c_idx = CLASSES.index(cons)
        ax.barh(0, 1/len(attack_risks), left=risk, color=CLASS_COLORS[c_idx], height=1)
    
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[i], label=CLASSES[i]) for i in range(5)]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.set_xlabel('Attack Risk', fontsize=12)
    ax.set_title('Selected Consensus Mechanism', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim([0, 1])
    
    # Annotate transitions
    prev = selected[0]
    for i, cons in enumerate(selected):
        if cons != prev:
            ax.axvline(attack_risks[i], color='white', linestyle='--', linewidth=2)
            ax.text(attack_risks[i], 0.5, f'{prev}→{cons}', fontsize=9, 
                    ha='center', va='center', rotation=90, color='white', fontweight='bold')
            prev = cons
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '3_byzantine_resilience.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 3_byzantine_resilience.png")
    
    # Print transitions
    prev = selected[0]
    print(f"    Risk=0.00: {prev}")
    for i, cons in enumerate(selected):
        if cons != prev:
            print(f"    Risk={attack_risks[i]:.2f}: {prev} -> {cons}")
            prev = cons


# ==============================================================================
# Experiment 4: Confusion Matrix & Per-Class Analysis
# ==============================================================================

def experiment_4_confusion_matrix(model, feature_stats, X_test, y_test):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Confusion Matrix & Per-Class Analysis")
    print("=" * 70)
    
    preds, probs = predict_batch(model, X_test, feature_stats)
    
    # Build confusion matrix
    n_classes = len(CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test.numpy(), preds.numpy()):
        cm[true][pred] += 1
    
    # Per-class metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    overall_acc = (preds == y_test).float().mean().item() * 100
    
    print(f"    Overall Accuracy: {overall_acc:.2f}%")
    print(f"    {'Class':8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for i, c in enumerate(CLASSES):
        print(f"    {c:8s} {precision[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Confusion Matrix (Acc: {overall_acc:.2f}%)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    # Add counts to cells
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm[i][j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=12, color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Per-class F1
    ax = axes[1]
    bars = ax.bar(CLASSES, f1, color=CLASS_COLORS, edgecolor='white', linewidth=2)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score', fontsize=14, fontweight='bold')
    ax.set_ylim([min(f1) - 0.02, 1.02])
    for bar, score in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '4_confusion_matrix_f1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 4_confusion_matrix_f1.png")
    
    return {'accuracy': overall_acc, 'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist()}


# ==============================================================================
# Experiment 5: Latency-Throughput Trade-off Heatmap
# ==============================================================================

def experiment_5_latency_throughput_heatmap(model, feature_stats):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Latency-Throughput Trade-off Heatmap")
    print("=" * 70)
    
    latencies = np.linspace(0.5, 60, 40)
    throughputs = np.linspace(5, 10000, 40)
    
    consensus_map = np.zeros((len(throughputs), len(latencies)), dtype=int)
    confidence_map = np.zeros((len(throughputs), len(latencies)))
    
    base_state = {
        'num_nodes': 200,
        'connectivity': 0.80,
        'latency_requirement_sec': 0,
        'throughput_requirement_tps': 0,
        'byzantine_tolerance': 0.15,
        'security_priority': 0.75,
        'energy_budget': 0.5,
        'bandwidth_mbps': 1000,
        'consistency_requirement': 0.75,
        'decentralization_requirement': 0.7,
        'network_load': 0.5,
        'attack_risk': 0.3,
    }
    
    # Batch all predictions
    all_features = []
    for j, tps in enumerate(throughputs):
        for i, lat in enumerate(latencies):
            state = base_state.copy()
            state['latency_requirement_sec'] = lat
            state['throughput_requirement_tps'] = tps
            feats = [state[k] for k in FEATURE_KEYS]
            all_features.append(feats)
    
    X = torch.tensor(all_features, dtype=torch.float32)
    preds, probs = predict_batch(model, X, feature_stats)
    
    idx = 0
    for j in range(len(throughputs)):
        for i in range(len(latencies)):
            consensus_map[j, i] = preds[idx].item()
            confidence_map[j, i] = probs[idx].max().item()
            idx += 1
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Custom colormap for 5 classes
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(CLASS_COLORS)
    
    # Consensus selection heatmap
    ax = axes[0]
    im = ax.imshow(consensus_map, origin='lower', aspect='auto', cmap=cmap, 
                   vmin=-0.5, vmax=4.5,
                   extent=[latencies[0], latencies[-1], throughputs[0], throughputs[-1]])
    ax.set_xlabel('Latency Requirement (s)', fontsize=12)
    ax.set_ylabel('Throughput Requirement (TPS)', fontsize=12)
    ax.set_title('Consensus Selection Map', fontsize=14, fontweight='bold')
    
    # Custom colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[i], label=CLASSES[i]) for i in range(5)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Confidence heatmap
    ax = axes[1]
    im2 = ax.imshow(confidence_map, origin='lower', aspect='auto', cmap='viridis',
                    extent=[latencies[0], latencies[-1], throughputs[0], throughputs[-1]])
    ax.set_xlabel('Latency Requirement (s)', fontsize=12)
    ax.set_ylabel('Throughput Requirement (TPS)', fontsize=12)
    ax.set_title('Prediction Confidence Map', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '5_latency_throughput_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 5_latency_throughput_heatmap.png")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("  Model 4: ConsensusRetNet - Comprehensive Validation")
    print("=" * 70)
    
    # Load
    model, feature_stats, X_train, y_train, X_val, y_val, X_test, y_test = load_model_and_data()
    
    # Experiments
    t0 = time.time()
    
    arch_results = experiment_1_architecture_comparison(model, feature_stats, X_train, y_train, X_val, y_val, X_test, y_test)
    experiment_2_scenario_switching(model, feature_stats)
    experiment_3_byzantine_resilience(model, feature_stats)
    cm_results = experiment_4_confusion_matrix(model, feature_stats, X_test, y_test)
    experiment_5_latency_throughput_heatmap(model, feature_stats)
    
    elapsed = time.time() - t0
    
    # Summary
    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(OUT_DIR.glob('*.png')):
        print(f"    - {f.name}")
    
    # Save summary JSON
    summary = {
        'model': 'ConsensusRetNet',
        'architecture': 'RetNet (ICML 2023)',
        'parameters': sum(p.numel() for p in model.parameters()),
        'test_accuracy': cm_results['accuracy'],
        'per_class_f1': dict(zip(CLASSES, cm_results['f1'])),
        'architecture_comparison': arch_results,
        'total_time_seconds': elapsed,
    }
    results_dir = BASE_DIR / 'data' / 'experiment_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: validation_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
