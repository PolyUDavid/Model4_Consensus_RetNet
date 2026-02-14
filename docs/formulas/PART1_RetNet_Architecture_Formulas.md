# Part 1: RetNet Architecture — Core Mathematical Formulas

> **Model 4: ConsensusRetNet**  
> Architecture: Retentive Network (ICML 2023)  
> Author: NOK KO  
> Total formulas in this document: **38**

---

## 1. Input Representation

### Formula 1.1 — Input Feature Vector
$$\mathbf{x} = [x_1, x_2, \ldots, x_{12}] \in \mathbb{R}^{12}$$

where each $x_i$ represents a network condition feature:

| Index | Feature | Symbol | Range |
|-------|---------|--------|-------|
| 1 | Number of Nodes | $N_{nodes}$ | [10, 1000] |
| 2 | Connectivity | $\kappa$ | [0.6, 0.95] |
| 3 | Latency Requirement | $L_{req}$ (sec) | [0.5, 60] |
| 4 | Throughput Requirement | $\Theta_{req}$ (TPS) | [5, 10000] |
| 5 | Byzantine Tolerance | $f_{byz}$ | [0, 0.33] |
| 6 | Security Priority | $\sigma_{sec}$ | [0.6, 1.0] |
| 7 | Energy Budget | $\mathcal{E}_{budget}$ | [0.1, 1.0] |
| 8 | Bandwidth | $B_{net}$ (Mbps) | [100, 10000] |
| 9 | Consistency Requirement | $\mathcal{C}_{req}$ | [0.5, 1.0] |
| 10 | Decentralization Requirement | $\mathcal{D}_{req}$ | [0.3, 1.0] |
| 11 | Network Load | $\rho_{load}$ | [0.1, 0.9] |
| 12 | Attack Risk | $\alpha_{risk}$ | [0, 1.0] |

### Formula 1.2 — Feature Normalization (Z-score)
$$\hat{x}_i = \frac{x_i - \mu_i}{\sigma_i + \epsilon}, \quad \epsilon = 10^{-8}$$

where $\mu_i$ and $\sigma_i$ are the training set mean and standard deviation for feature $i$.

### Formula 1.3 — Normalized Input Vector
$$\hat{\mathbf{x}} = [\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_{12}] \in \mathbb{R}^{12}$$

---

## 2. Input Projection Layer

### Formula 2.1 — Linear Projection
$$\mathbf{h}_0 = \mathbf{W}_{proj} \hat{\mathbf{x}} + \mathbf{b}_{proj}$$

where $\mathbf{W}_{proj} \in \mathbb{R}^{384 \times 12}$, $\mathbf{b}_{proj} \in \mathbb{R}^{384}$.

### Formula 2.2 — Sequence Dimension Expansion
$$\mathbf{H}_0 = \text{unsqueeze}(\mathbf{h}_0, \text{dim}=1) \in \mathbb{R}^{B \times 1 \times 384}$$

### Formula 2.3 — Positional Embedding Addition
$$\mathbf{H}_0' = \mathbf{H}_0 + \mathbf{P}_{emb}$$

where $\mathbf{P}_{emb} \in \mathbb{R}^{1 \times 1 \times 384}$ is a learnable positional embedding.

---

## 3. Multi-Scale Retention Mechanism (Core Innovation)

### Formula 3.1 — Query, Key, Value Projections
$$\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_{model}}$, all bias-free.

### Formula 3.2 — Multi-Head Reshaping
$$\mathbf{Q}_h = \mathbf{Q}[:, :, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{B \times L \times d_k}$$

where $d_k = d_{model} / n_{heads} = 384 / 3 = 128$.

### Formula 3.3 — Retention Score Computation
$$\mathbf{S}_h = \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}$$

### Formula 3.4 — Exponential Decay Matrix (Key Innovation)
$$D_\gamma[i, j] = \begin{cases} \gamma^{i-j} & \text{if } i \geq j \\ 0 & \text{otherwise} \end{cases}$$

This is the **causal decay mask** that distinguishes RetNet from Transformer attention.

### Formula 3.5 — Multi-Scale Decay Rates
$$\gamma_1 = 0.9 \quad \text{(short-term memory)}$$
$$\gamma_2 = 0.95 \quad \text{(medium-term memory)}$$
$$\gamma_3 = 0.99 \quad \text{(long-term memory)}$$

### Formula 3.6 — Decay-Weighted Attention Scores
$$\mathbf{A}_h = \mathbf{S}_h \odot D_{\gamma_h}$$

where $\odot$ denotes element-wise multiplication.

### Formula 3.7 — Retention Weights (Softmax Normalization)
$$\mathbf{R}_h = \text{softmax}(\mathbf{A}_h)$$

### Formula 3.8 — Retention Output per Head
$$\mathbf{O}_h = \mathbf{R}_h \mathbf{V}_h \in \mathbb{R}^{B \times L \times d_k}$$

### Formula 3.9 — Multi-Head Concatenation
$$\mathbf{O}_{concat} = [\mathbf{O}_1 ; \mathbf{O}_2 ; \mathbf{O}_3] \in \mathbb{R}^{B \times L \times d_{model}}$$

### Formula 3.10 — Group Normalization
$$\mathbf{O}_{norm} = \text{GroupNorm}(\mathbf{O}_{concat}, \text{groups}=n_{heads})$$

### Formula 3.11 — Output Projection
$$\mathbf{O}_{ret} = \mathbf{O}_{norm} \mathbf{W}_{out}$$

where $\mathbf{W}_{out} \in \mathbb{R}^{d_{model} \times d_{model}}$.

### Formula 3.12 — Complete Multi-Scale Retention
$$\text{MSRetention}(\mathbf{H}) = \text{Proj}_{out}\left(\text{GroupNorm}\left(\bigoplus_{h=1}^{3} \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} \odot D_{\gamma_h}\right) \mathbf{V}_h\right)\right)$$

---

## 4. Feed-Forward Network (FFN)

### Formula 4.1 — FFN First Layer
$$\mathbf{F}_1 = \text{GELU}(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)$$

where $\mathbf{W}_1 \in \mathbb{R}^{384 \times 1536}$.

### Formula 4.2 — GELU Activation Function
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

### Formula 4.3 — FFN Second Layer (Projection Back)
$$\mathbf{F}_{out} = \mathbf{F}_1 \mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_2 \in \mathbb{R}^{1536 \times 384}$.

### Formula 4.4 — Complete FFN
$$\text{FFN}(\mathbf{X}) = \text{Linear}_{1536 \to 384}(\text{Dropout}(\text{GELU}(\text{Linear}_{384 \to 1536}(\mathbf{X}))))$$

---

## 5. RetNet Block (Pre-Norm Architecture)

### Formula 5.1 — Retention Sub-Block with Residual
$$\mathbf{H}_{ret} = \mathbf{X} + \text{Dropout}(\text{MSRetention}(\text{LayerNorm}(\mathbf{X})))$$

### Formula 5.2 — FFN Sub-Block with Residual
$$\mathbf{H}_{out} = \mathbf{H}_{ret} + \text{Dropout}(\text{FFN}(\text{LayerNorm}(\mathbf{H}_{ret})))$$

### Formula 5.3 — Layer Normalization
$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mathbb{E}[\mathbf{x}]}{\sqrt{\text{Var}[\mathbf{x}] + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

### Formula 5.4 — Complete RetNet Block
$$\text{RetNetBlock}(\mathbf{X}) = \text{ResFFN}(\text{ResRetention}(\mathbf{X}))$$

---

## 6. Stacked RetNet Layers

### Formula 6.1 — Layer 1 Output
$$\mathbf{H}_1 = \text{RetNetBlock}_1(\mathbf{H}_0')$$

### Formula 6.2 — Layer 2 Output
$$\mathbf{H}_2 = \text{RetNetBlock}_2(\mathbf{H}_1)$$

### Formula 6.3 — Layer 3 Output
$$\mathbf{H}_3 = \text{RetNetBlock}_3(\mathbf{H}_2)$$

### Formula 6.4 — Final Normalization
$$\mathbf{H}_{final} = \text{LayerNorm}_{final}(\mathbf{H}_3)$$

---

## 7. Classification Head

### Formula 7.1 — Sequence Squeeze (Global Average Pooling)
$$\mathbf{h}_{cls} = \text{squeeze}(\mathbf{H}_{final}) \in \mathbb{R}^{B \times 384}$$

### Formula 7.2 — First Classification Layer
$$\mathbf{z}_1 = \text{GELU}(\mathbf{W}_{cls1} \mathbf{h}_{cls} + \mathbf{b}_{cls1})$$

where $\mathbf{W}_{cls1} \in \mathbb{R}^{192 \times 384}$.

### Formula 7.3 — Dropout Regularization
$$\mathbf{z}_1' = \text{Dropout}(\mathbf{z}_1, p=0.1)$$

### Formula 7.4 — Output Logits
$$\mathbf{z}_{out} = \mathbf{W}_{cls2} \mathbf{z}_1' + \mathbf{b}_{cls2}$$

where $\mathbf{W}_{cls2} \in \mathbb{R}^{5 \times 192}$, $\mathbf{z}_{out} \in \mathbb{R}^{B \times 5}$.

### Formula 7.5 — Softmax Probability Distribution
$$P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{5} e^{z_j}}, \quad k \in \{\text{PoW, PoS, PBFT, DPoS, Hybrid}\}$$

### Formula 7.6 — Predicted Class
$$\hat{y} = \arg\max_{k} P(y = k | \mathbf{x})$$

---

## 8. Training Formulas

### Formula 8.1 — Cross-Entropy Loss with Label Smoothing
$$\mathcal{L}_{CE} = -\sum_{k=1}^{5} q_k \log P(y = k | \mathbf{x})$$

where $q_k = (1 - \epsilon) \cdot \mathbb{1}[k = y_{true}] + \frac{\epsilon}{K}$, $\epsilon = 0.1$, $K = 5$.

### Formula 8.2 — AdamW Optimizer Update
$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

where $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$, $\lambda = 0.01$.

### Formula 8.3 — Learning Rate Schedule (3-Stage)
$$\eta(t) = \begin{cases} \eta_{base} \cdot \frac{t}{T_{warmup}} & t \leq T_{warmup} = 10 \\ \eta_{base} & T_{warmup} < t \leq T_{stable} \\ \eta_{min} + \frac{\eta_{base} - \eta_{min}}{2}\left(1 + \cos\frac{\pi(t - T_{stable})}{T_{max} - T_{stable}}\right) & t > T_{stable} \end{cases}$$

### Formula 8.4 — Gradient Clipping
$$\hat{\mathbf{g}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq 1.0 \\ \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{otherwise} \end{cases}$$

### Formula 8.5 — Xavier Initialization
$$W_{ij} \sim \mathcal{U}\left[-\frac{\text{gain} \cdot \sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\text{gain} \cdot \sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right], \quad \text{gain} = 0.5$$

---

## 9. Inference Complexity

### Formula 9.1 — Parallel Mode Complexity (Training)
$$\mathcal{O}_{parallel} = O(L^2 \cdot d_{model}) = O(N^2 \cdot 384)$$

### Formula 9.2 — Recurrent Mode Complexity (Inference)
$$\mathcal{O}_{recurrent} = O(d_{model}^2) = O(384^2) \text{ per step}$$

### Formula 9.3 — Memory Footprint
$$\text{Memory} = \sum_{l=1}^{3} (4 \cdot d_{model}^2 + 2 \cdot d_{model} \cdot d_{ff} + 4 \cdot d_{model}) \approx 5.4M \text{ params}$$

---

**Total formulas in Part 1: 38**
