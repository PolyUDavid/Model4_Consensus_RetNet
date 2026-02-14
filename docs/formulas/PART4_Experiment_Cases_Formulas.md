# Part 4: Experiment Cases 1–5 — Scenario Mathematics & Validation

> **Model 4: ConsensusRetNet**  
> Experimental Validation with First-Principles Alignment  
> Author: NOK KO  
> Total formulas in this document: **45**

---

## Experiment 1: Architecture Comparison

### Formula 1.1 — Test Accuracy
$$\text{Acc}_{test} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{1}[\hat{y}_i = y_i]$$

### Formula 1.2 — Accuracy Advantage (RetNet vs Baseline)
$$\Delta\text{Acc} = \text{Acc}_{RetNet} - \text{Acc}_{baseline}$$

### Formula 1.3 — Parameter Efficiency Ratio
$$\eta_{param} = \frac{\text{Acc}_{model}}{N_{params} / 10^6}$$

### Formula 1.4 — Results Summary
| Model | Test Accuracy | Parameters | Epochs | $\eta_{param}$ |
|-------|--------------|------------|--------|-----------------|
| **RetNet** | **99.98%** | 5,402,126 | 26 | 18.51 |
| MLP | 99.98% | 70,405 | 48 | 1420.11 |
| LSTM | 99.98% | 205,445 | 41 | 486.75 |
| CNN | 99.09% | 25,605 | 107 | 3870.72 |

### Formula 1.5 — Convergence Speed
$$\text{Speed}_{conv} = \frac{\text{Acc}_{target}}{\text{Epochs}_{required}}$$

$$\text{Speed}_{RetNet} = \frac{99.98}{26} = 3.84\%/\text{epoch}$$

### Formula 1.6 — Statistical Significance (McNemar's Test)
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where $b$ = samples correct by RetNet but wrong by baseline, $c$ = vice versa.

---

## Experiment 2: Dynamic Scenario Switching (5 Phases)

### Phase 1: Normal → Hybrid (t = 0–20s)

### Formula 2.1 — Hybrid Utility (Balanced Conditions)
$$U_{Hybrid} = 0.20 \cdot S + 0.20 \cdot E + 0.20 \cdot \Theta + 0.20 \cdot D + 0.20 \cdot C$$

When all weights are approximately equal and features are moderate, Hybrid dominates.

### Formula 2.2 — Balanced Feature Vector
$$\mathbf{x}_{normal} = [150, 0.72, 12.0, 1200, 0.17, 0.80, 0.60, 1500, 0.75, 0.70, 0.45, 0.55]$$

### Phase 2: Energy Crisis → PoS (t = 20–40s)

### Formula 2.3 — Energy Budget Decline
$$\mathcal{E}(t) = 0.60 - 0.35 \cdot \frac{t - 20}{20}, \quad t \in [20, 40]$$

### Formula 2.4 — PoS Selection Criterion
$$\text{Select PoS} \iff \mathcal{E}_{budget} < 0.4 \text{ AND } \sigma_{sec} \in [0.7, 0.9]$$

### Formula 2.5 — PoS Energy Advantage
$$\frac{E_{PoW}}{E_{PoS}} = \frac{H \cdot P_{hash} \cdot T}{N_v \cdot P_{idle} \cdot T} \approx 10^4$$

### Phase 3: Small Network → PBFT (t = 40–60s)

### Formula 2.6 — Node Count Decline
$$N_{nodes}(t) = 200 - 160 \cdot \frac{t - 40}{20}, \quad \text{reaching } N \approx 40$$

### Formula 2.7 — PBFT Selection Criterion
$$\text{Select PBFT} \iff N < 100 \text{ AND } L_{req} < 5s \text{ AND } \mathcal{C}_{req} > 0.85$$

### Formula 2.8 — PBFT Feasibility Check
$$N^2 \cdot 3 \cdot S_{msg} \leq BW_{available} \cdot T_{target}$$

$$40^2 \cdot 3 \cdot 256 = 1.23 \text{ MB} \leq 2500 \text{ Mbps} \cdot 2.5s \quad \checkmark$$

### Phase 4: Byzantine Attack → PoW (t = 60–80s)

### Formula 2.9 — Attack Risk Escalation
$$\alpha_{risk}(t) = 0.63 + 0.22 \cdot \frac{t - 60}{20}$$

### Formula 2.10 — Security Priority Escalation
$$\sigma_{sec}(t) = 0.75 + 0.19 \cdot \frac{t - 60}{20} \to 0.94$$

### Formula 2.11 — PoW Selection Criterion
$$\text{Select PoW} \iff \sigma_{sec} > 0.85 \text{ AND } \mathcal{E}_{budget} > 0.7 \text{ AND } \mathcal{D}_{req} > 0.85$$

### Formula 2.12 — PoW Security Guarantee
$$C_{attack}^{PoW} = 0.51 \cdot H_{network} \cdot P_{elec} \cdot T \gg C_{attack}^{other}$$

### Phase 5: Mass Scale-up → DPoS (t = 80–100s)

### Formula 2.13 — Throughput Requirement Spike
$$\Theta_{req}(t) = 30 + 7700 \cdot \frac{t - 80}{20} \to 7730 \text{ TPS}$$

### Formula 2.14 — DPoS Selection Criterion
$$\text{Select DPoS} \iff N > 200 \text{ AND } \Theta_{req} > 2000 \text{ AND } \mathcal{D}_{req} \in [0.5, 0.75]$$

### Formula 2.15 — DPoS Throughput Advantage
$$\Theta_{DPoS} = \frac{n_{tx}}{T_{slot}} = \frac{10000}{0.5} = 20000 \text{ TPS} \gg \Theta_{PBFT}$$

### Formula 2.16 — Phase Transition Detection
$$\text{Transition at } t^* \iff m(t^* - \delta) \neq m(t^* + \delta)$$

---

## Experiment 3: Byzantine Attack Resilience

### Formula 3.1 — Attack Risk Sweep
$$\alpha_{risk} \in \{0.00, 0.02, 0.04, \ldots, 1.00\}, \quad |\text{sweep}| = 50$$

### Formula 3.2 — Correlated Feature Evolution
$$\sigma_{sec}(\alpha) = \min(1.0, \; 0.72 + 0.26\alpha)$$
$$\mathcal{E}_{budget}(\alpha) = \min(1.0, \; 0.40 + 0.55\alpha)$$
$$\mathcal{D}_{req}(\alpha) = \min(1.0, \; 0.65 + 0.30\alpha)$$
$$L_{req}(\alpha) = 12.0 + 38\alpha$$
$$\Theta_{req}(\alpha) = 1200 - 1170\alpha$$

### Formula 3.3 — Transition Point
$$\alpha^* = \inf\{\alpha : m(\alpha) = \text{PoW}\} \approx 0.41$$

### Formula 3.4 — Physical Justification of Transition
At $\alpha^* = 0.41$:
$$\sigma_{sec}(0.41) = 0.72 + 0.26 \times 0.41 = 0.827$$
$$\mathcal{E}_{budget}(0.41) = 0.40 + 0.55 \times 0.41 = 0.626$$
$$\mathcal{D}_{req}(0.41) = 0.65 + 0.30 \times 0.41 = 0.773$$

These values cross the PoW decision boundary learned from training data.

### Formula 3.5 — Security-Energy Trade-off Curve
$$\text{Pareto front}: \; \mathcal{E} = f(\sigma_{sec}) \text{ s.t. } U_{PoW} = U_{Hybrid}$$

### Formula 3.6 — Byzantine Tolerance Impact
$$f_{byz}(\alpha) = \min(0.33, \; 0.12 + 0.21\alpha)$$

At $\alpha = 1.0$: $f_{byz} = 0.33$, requiring $N_{min} = 3 \times \lfloor N \times 0.33 \rfloor + 1$.

---

## Experiment 4: Confusion Matrix Analysis

### Formula 4.1 — True Positive Rate (Sensitivity)
$$\text{TPR}_k = \frac{TP_k}{TP_k + FN_k}$$

### Formula 4.2 — False Positive Rate
$$\text{FPR}_k = \frac{FP_k}{FP_k + TN_k}$$

### Formula 4.3 — Confusion Matrix Element
$$CM[i][j] = \left|\{n : y_n = i \text{ AND } \hat{y}_n = j\}\right|$$

### Formula 4.4 — Cohen's Kappa
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o = \text{Accuracy}$, $p_e = \sum_k \frac{n_{k,true} \cdot n_{k,pred}}{N^2}$.

### Formula 4.5 — Per-Class Results

| Class | TP | FP | FN | Precision | Recall | F1 |
|-------|-----|-----|-----|-----------|--------|------|
| PoW | 872 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| PoS | 863 | 1 | 0 | 0.9988 | 1.0000 | 0.9994 |
| PBFT | 898 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| DPoS | 920 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid | 946 | 0 | 1 | 1.0000 | 0.9989 | 0.9995 |

### Formula 4.6 — Weighted F1
$$F_1^{weighted} = \sum_{k=1}^{5} \frac{N_k}{N_{total}} \cdot F_1^k = 0.99976$$

---

## Experiment 5: Latency-Throughput Trade-off Heatmap

### Formula 5.1 — 2D Parameter Sweep
$$L_{req} \in \{0.5, 2.0, 3.5, \ldots, 60.0\}, \quad |\text{grid}_L| = 40$$
$$\Theta_{req} \in \{5, 260, 515, \ldots, 10000\}, \quad |\text{grid}_\Theta| = 40$$
$$\text{Total predictions} = 40 \times 40 = 1600$$

### Formula 5.2 — Decision Boundary (PBFT ↔ DPoS)
$$\text{PBFT region}: L_{req} < 5s \text{ AND } \Theta_{req} > 1000$$
$$\text{DPoS region}: \Theta_{req} > 2000 \text{ AND } N > 200$$

### Formula 5.3 — Decision Boundary (PoW ↔ Hybrid)
$$\text{PoW region}: L_{req} > 30s \text{ AND } \Theta_{req} < 50$$
$$\text{Hybrid region}: 5 < L_{req} < 20 \text{ AND } 500 < \Theta_{req} < 2000$$

### Formula 5.4 — Confidence Map
$$\text{Conf}(L, \Theta) = \max_k P(y = k | L_{req} = L, \Theta_{req} = \Theta, \mathbf{x}_{base})$$

### Formula 5.5 — Decision Region Area
$$A_k = \frac{|\{(i,j) : m_{ij} = k\}|}{|\text{grid}|} \times 100\%$$

### Formula 5.6 — Boundary Sharpness
$$\nabla_{boundary} = \frac{\Delta P_{dominant}}{\Delta L + \Delta \Theta}\bigg|_{\text{boundary}}$$

Higher gradient = sharper, more confident boundary.

---

## Cross-Experiment Validation

### Formula C.1 — Physical Consistency Score
$$\text{PCS} = \frac{1}{N_{scenarios}} \sum_{i=1}^{N_{scenarios}} \mathbb{1}[m_{predicted}^i = m_{physics}^i]$$

$$\text{PCS} = \frac{5}{5} = 100\%$$

(All 5 scenario predictions match first-principles expectations)

### Formula C.2 — Transition Consistency
$$\text{TC} = \frac{|\text{correct transitions}|}{|\text{expected transitions}|}$$

### Formula C.3 — Confidence-Correctness Correlation
$$r_{conf} = \text{corr}(\text{confidence}, \; \mathbb{1}[\text{correct}])$$

### Formula C.4 — Cross-Scenario Generalization
$$\text{Gen} = 1 - \frac{\sigma_{Acc}^{scenarios}}{\mu_{Acc}^{scenarios}}$$

---

**Total formulas in Part 4: 45**
