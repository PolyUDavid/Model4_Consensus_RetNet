# ConsensusRetNet: Adaptive Consensus Mechanism Selection via Retentive Networks

> **A Multi-Scale Retentive Network (RetNet) for Intelligent Dynamic Consensus Protocol Selection in Vehicular Blockchain Networks**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)

---

## Overview

This repository contains the complete implementation of **ConsensusRetNet** — my research framework for adaptive blockchain consensus mechanism selection in Internet of Vehicles (IoV) environments. The core innovation leverages the **Retentive Network (RetNet)** architecture (ICML 2023) with multi-scale exponential decay retention to dynamically select optimal consensus protocols from {PoW, PoS, PBFT, DPoS, Hybrid} based on real-time network conditions.

### Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.98% (4,499/4,500 correct) |
| **Macro F1 Score** | 0.9997 |
| **Total Parameters** | 5,402,126 |
| **Inference Latency** | 0.93 ms (model) / ~2 ms (API) |
| **Physical Consistency** | 100% (5/5 scenarios) |
| **Ablation Variants Tested** | 39 (all > 99.5%) |
| **Training Dataset** | 30,000 samples (balanced, 6K/class) |
| **Best Epoch** | 6/26 (early stopping) |

---

## Architecture

```
Input: x ∈ ℝ¹² (12-dim network state vector)
  ↓
Linear Projection: ℝ¹² → ℝ³⁸⁴
  ↓
+ Learnable Positional Embedding
  ↓
3 × RetNet Blocks:
  ├─ LayerNorm → Multi-Scale Retention (γ = {0.9, 0.95, 0.99}) → Dropout → Residual
  └─ LayerNorm → FFN (384 → 1536 → 384, GELU) → Dropout → Residual
  ↓
Final LayerNorm → Squeeze
  ↓
Classification Head: 384 → 192 (GELU) → Dropout → 5
  ↓
Output: P(y = k | x) for k ∈ {PoW, PoS, PBFT, DPoS, Hybrid}
```

The **multi-scale retention mechanism** is the core innovation:
- **Head 1** (γ = 0.9, half-life ≈ 6.6 steps): Captures sudden network anomalies (DDoS bursts, node failures)
- **Head 2** (γ = 0.95, half-life ≈ 13.5 steps): Tracks medium-term trends (traffic density evolution)
- **Head 3** (γ = 0.99, half-life ≈ 69.0 steps): Monitors persistent threats (slow-rate Sybil infiltration)

---

## Repository Structure

```
GIT_MODEL4/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── models/
│   │   └── retnet_consensus.py        # Core RetNet architecture
│   ├── api/
│   │   └── main.py                    # FastAPI inference server
│   ├── simulation/
│   │   ├── consensus_simulation.py    # Pygame real-time simulation
│   │   ├── validate_consensus.py      # 5 validation experiments
│   │   ├── ablation_study.py          # 7-dimension ablation study
│   │   └── run_cases_via_api.py       # API-verified case experiments
│   ├── dashboard/
│   │   └── dashboard_app.py           # Streamlit interactive dashboard
│   ├── data_generator/
│   │   └── generate_consensus_data.py # Physics-informed data generator
│   ├── visualization/
│   │   ├── generate_all_plots.py      # Publication-quality plot generator
│   │   ├── generate_backbone_diagram.py
│   │   └── generate_training_plots.py
│   └── train_consensus.py             # Training script
│
├── training/                          # Training artifacts
│   ├── checkpoints/
│   │   └── best_consensus.pth         # Best model checkpoint (~62MB)
│   ├── logs/
│   │   └── training_output.log        # Training console log
│   └── history/
│       └── training_history.json      # Epoch-wise metrics
│
├── data/                              # Experiment data (JSON)
│   ├── training_data/
│   │   └── consensus_training_data.json   # 30,000 balanced samples
│   ├── experiment_results/
│   │   ├── test_results.json              # Test set performance
│   │   ├── validation_summary.json        # Validation summary
│   │   ├── first_principles_formulas.json # Physics validation rules
│   │   ├── COMPLETE_EXPERIMENT_RESULTS.md # Full experiment report
│   │   └── MODEL_EXCELLENCE_PROOF.md      # Model excellence analysis
│   ├── verified_api_data/
│   │   └── verified_experiment_data.json  # API-verified case results
│   ├── case_studies/
│   │   ├── scenario_predictions.json      # Case study predictions
│   │   └── CASE_STUDIES_DETAILED_REPORT.md
│   ├── ablation_study/
│   │   └── ablation_study_results.json    # 39-variant ablation data
│   ├── dynamic_scenario/
│   │   └── dynamic_scenario_results.json  # 100-step dynamic test
│   └── byzantine_resilience/              # (data in verified_api_data)
│
├── figures/                           # Visualizations
│   ├── training_curves/               # Loss, accuracy, LR plots
│   ├── experiment_plots/              # Experiment result plots
│   └── architecture_diagrams/         # Model backbone diagrams
│
└── docs/                              # Documentation
    ├── PAPER_STORYLINE_Full_Narrative.md   # Complete paper narrative
    ├── formulas/                      # 188 mathematical formulas (5 parts)
    │   ├── PART1_RetNet_Architecture_Formulas.md
    │   ├── PART2_Consensus_Physics_Formulas.md
    │   ├── PART3_Simulation_Algorithm_Formulas.md
    │   ├── PART4_Experiment_Cases_Formulas.md
    │   └── PART5_Model_Advantages_Performance.md
    ├── architecture/
    │   └── retnet_architecture.json   # Architecture specification
    └── paper_analysis/                # Paper-ready analyses
        ├── 00_PAPER_EXPERIMENT_INDEX.md
        ├── TRAINING_ANALYSIS.md
        ├── CROSS_CASE_COMPARISON.md
        ├── ARCHITECTURE_COMPARISON.md
        ├── DYNAMIC_SCENARIO_ANALYSIS.md
        ├── BYZANTINE_RESILIENCE.md
        └── ABLATION_STUDY_ANALYSIS.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python src/data_generator/generate_consensus_data.py
```

This creates 30,000 balanced samples (6,000 per consensus class) with physics-informed decision boundaries.

### 3. Train the Model

```bash
python src/train_consensus.py
```

Training configuration:
- **Optimizer**: AdamW (lr=6e-4, weight_decay=0.01)
- **LR Schedule**: 3-phase (warmup → plateau → cosine annealing)
- **Label Smoothing**: ε = 0.1
- **Early Stopping**: patience = 20 epochs
- **Batch Size**: 128
- **Hardware**: Intel i9-14900K + NVIDIA RTX 4090 24GB

### 4. Start the API Server

```bash
cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API documentation available at `http://localhost:8000/docs`.

### 5. Launch the Dashboard

```bash
cd src && streamlit run dashboard/dashboard_app.py --server.port 8501
```

### 6. Run the Simulation

```bash
python src/simulation/consensus_simulation.py
```

Controls: `SPACE` (switch scenario), `1/2/3` (jump to scenario), `P` (pause), `ESC` (quit).

### 7. Run Validation Experiments

```bash
python src/simulation/validate_consensus.py
```

### 8. Run Ablation Study

```bash
python src/simulation/ablation_study.py
```

---

## Experiments

### Experiment 1: Architecture Comparison

| Model | Test Acc | F1 Macro | Parameters | Convergence |
|-------|----------|----------|------------|-------------|
| **ConsensusRetNet** | **99.98%** | **0.9997** | 5.40M | **26 epochs** |
| MLP Baseline | 99.98% | 0.9997 | 0.07M | 48 epochs |
| LSTM Baseline | 99.98% | 0.9997 | 0.21M | 41 epochs |
| CNN Baseline | 99.09% | 0.9902 | 0.03M | 107 epochs |

### Experiment 2: Dynamic Scenario Switching (100-second journey)

| Phase | Time | Scenario | Selected Consensus | Confidence |
|-------|------|----------|--------------------|------------|
| 1 | 0-20s | Normal V2X | Hybrid | 92.76% |
| 2 | 20-40s | Energy Crisis | PoS | 92.55% |
| 3 | 40-60s | Small Cluster | PBFT | 93.32% |
| 4 | 60-80s | Byzantine Attack | PoW | 92.01% |
| 5 | 80-100s | Mass Scale-up | DPoS | 92.95% |

### Experiment 3: Byzantine Resilience

Critical transition point detected at **α* = 0.41** (Hybrid → PoW), consistent with BFT theoretical danger zone (0.33–0.50).

### Ablation Study (39 variants across 7 dimensions)

Key findings:
- **Optimal depth**: 3 layers (unique 100.00% accuracy)
- **Optimal d_model**: 192 (100.00% with 75% fewer parameters)
- **Optimal dropout**: 0.05 (unique 100.00% with all 5 classes perfect)
- **Multi-scale γ superiority**: Wide [0.8, 0.95, 0.999] achieves 100.00% vs homogeneous 99.96%
- **Critical features**: Performance group (latency + throughput) most impactful

---

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "network_state": {
      "num_nodes": 150,
      "connectivity": 0.72,
      "latency_requirement_sec": 12.0,
      "throughput_requirement_tps": 1200,
      "byzantine_tolerance": 0.17,
      "security_priority": 0.80,
      "energy_budget": 0.60,
      "bandwidth_mbps": 1500,
      "consistency_requirement": 0.75,
      "decentralization_requirement": 0.70,
      "network_load": 0.45,
      "attack_risk": 0.55
    }
  }'
```

### Response

```json
{
  "success": true,
  "prediction": {
    "predicted_consensus": "Hybrid",
    "full_name": "Hybrid Consensus",
    "confidence": 0.927648,
    "probabilities": {
      "PoW": 0.021,
      "PoS": 0.018,
      "PBFT": 0.015,
      "DPoS": 0.018,
      "Hybrid": 0.928
    }
  },
  "metadata": {
    "inference_time_ms": 1.85,
    "model": "ConsensusRetNet"
  }
}
```

---

## 12-Dimensional Input Feature Space

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `num_nodes` | [10, 1000] | Active vehicular node count |
| 2 | `connectivity` | [0.6, 0.95] | Network graph connectivity ratio |
| 3 | `latency_requirement_sec` | [0.5, 60] | Maximum tolerable finality latency |
| 4 | `throughput_requirement_tps` | [5, 10000] | Required transactions per second |
| 5 | `byzantine_tolerance` | [0, 0.33] | Detected Byzantine fault ratio |
| 6 | `security_priority` | [0.6, 1.0] | Security importance threshold |
| 7 | `energy_budget` | [0.1, 1.0] | Normalized energy availability |
| 8 | `bandwidth_mbps` | [100, 10000] | Available wireless bandwidth |
| 9 | `consistency_requirement` | [0.5, 1.0] | Ledger consistency strength |
| 10 | `decentralization_requirement` | [0.3, 1.0] | Decentralization mandate |
| 11 | `network_load` | [0.1, 0.9] | Channel utilization percentage |
| 12 | `attack_risk` | [0.0, 1.0] | Aggregated cyber-attack probability |

---

## Mathematical Foundation

The multi-scale retention mechanism employs exponential decay:

$$D_\gamma[i,j] = \begin{cases} \gamma^{i-j} & \text{if } i \geq j \\ 0 & \text{otherwise} \end{cases}$$

$$\text{Retention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot D_\gamma\right) V$$

The effective half-life of each retention head:

$$\tau_{1/2} = \frac{\ln(0.5)}{\ln(\gamma)}$$

The complete mathematical framework (188 formulas) is documented in `docs/formulas/`.

---

## Hardware Environment

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Core i9-14900K (24C/32T, 6.0 GHz boost) |
| **GPU** | NVIDIA GeForce RTX 4090 24GB GDDR6X |
| **RAM** | 64 GB DDR5 |
| **Framework** | PyTorch 2.0+ with CUDA acceleration |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ko2026consensusretnet,
  title={ConsensusRetNet: A Retentive Network-Based Dynamic Heterogeneous Redundancy 
         Framework for Adaptive Consensus Selection in Vehicular Blockchains},
  author={NOK KO},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Author**: NOK KO  
**Contact**: [GitHub](https://github.com/nokko)  
**Last Updated**: February 2026
