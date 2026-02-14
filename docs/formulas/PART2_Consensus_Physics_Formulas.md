# Part 2: Consensus Mechanism Physics — First-Principles Formulas

> **Model 4: ConsensusRetNet**  
> Domain: Blockchain Consensus for Vehicular Networks (V2X)  
> Author: NOK KO  
> Total formulas in this document: **42**

---

## 1. Byzantine Fault Tolerance (BFT) Theory

### Formula 1.1 — BFT Minimum Node Requirement
$$N_{min} = 3f + 1$$

where $f$ is the maximum number of tolerable Byzantine (faulty/malicious) nodes.

### Formula 1.2 — Byzantine Tolerance Ratio
$$f = \lfloor N \cdot f_{byz} \rfloor$$

where $f_{byz} \in [0, 0.33]$ is the Byzantine tolerance parameter.

### Formula 1.3 — Safety Condition (Consensus Validity)
$$N_{honest} = N - f > \frac{2N}{3}$$

A consensus protocol is safe iff more than 2/3 of nodes are honest.

### Formula 1.4 — Liveness Condition
$$\text{Liveness} \iff N_{responsive} \geq 2f + 1$$

The system can make progress if at least $2f+1$ nodes respond.

### Formula 1.5 — Byzantine Failure Probability
$$P_{fail} = \sum_{k=f+1}^{N} \binom{N}{k} p_{byz}^k (1 - p_{byz})^{N-k}$$

where $p_{byz}$ is the probability each node is Byzantine.

---

## 2. Proof of Work (PoW) Physics

### Formula 2.1 — Hash Rate and Difficulty
$$T_{block} = \frac{D}{H_{total}}$$

where $D$ is mining difficulty, $H_{total}$ is total network hash rate (H/s).

### Formula 2.2 — PoW Security (51% Attack Cost)
$$C_{51\%} = 0.51 \cdot H_{network} \cdot P_{elec} \cdot T_{attack}$$

### Formula 2.3 — Energy Consumption per Block
$$E_{block}^{PoW} = H_{total} \cdot P_{per\_hash} \cdot T_{block}$$

### Formula 2.4 — PoW Throughput
$$\Theta_{PoW} = \frac{S_{block}}{T_{block} \cdot S_{tx}} = \frac{S_{block}}{(D / H_{total}) \cdot S_{tx}}$$

### Formula 2.5 — Finality Time (Probabilistic)
$$T_{finality}^{PoW} = k \cdot T_{block}, \quad k = 6 \text{ (Bitcoin standard)}$$

### Formula 2.6 — Double-Spend Probability (Nakamoto)
$$P_{ds}(z) = 1 - \sum_{k=0}^{z} \frac{(\lambda z)^k e^{-\lambda z}}{k!} \left(1 - \left(\frac{q}{p}\right)^{z-k}\right)$$

where $q$ is attacker's hash fraction, $p = 1-q$, $z$ is confirmation blocks.

### Formula 2.7 — Mining Reward Economics
$$R_{miner} = R_{block} + \sum_{i=1}^{n_{tx}} \text{fee}_i$$

### Formula 2.8 — Difficulty Adjustment
$$D_{new} = D_{old} \cdot \frac{T_{target}}{T_{actual}}$$

---

## 3. Proof of Stake (PoS) Physics

### Formula 3.1 — Validator Selection Probability
$$P_{select}(v) = \frac{S_v}{\sum_{i=1}^{N_v} S_i}$$

where $S_v$ is validator $v$'s staked amount.

### Formula 3.2 — PoS Energy Consumption
$$E_{block}^{PoS} = N_{validators} \cdot P_{idle} \cdot T_{slot}$$

### Formula 3.3 — Energy Ratio (PoS vs PoW)
$$\frac{E^{PoW}}{E^{PoS}} \approx 10^4 \text{ to } 10^6$$

### Formula 3.4 — Slashing Condition
$$\text{Slash}(v) = \begin{cases} \alpha \cdot S_v & \text{if double-sign detected} \\ \beta \cdot S_v & \text{if offline too long} \end{cases}$$

### Formula 3.5 — PoS Finality (Casper FFG)
$$T_{finality}^{PoS} = 2 \cdot T_{epoch} = 2 \times 32 \times T_{slot}$$

### Formula 3.6 — Effective Stake Yield
$$Y_{annual} = \frac{R_{attestation} \cdot N_{epochs/year}}{S_{staked}}$$

### Formula 3.7 — Security Deposit Requirement
$$S_{min} = \frac{C_{attack}^{target}}{P_{slash}}$$

---

## 4. Practical Byzantine Fault Tolerance (PBFT)

### Formula 4.1 — PBFT Message Complexity
$$M_{total} = N \cdot (N - 1) \cdot 3 = O(N^2)$$

(pre-prepare + prepare + commit phases)

### Formula 4.2 — PBFT Bandwidth Requirement
$$BW_{PBFT} = \frac{M_{total} \cdot S_{msg}}{T_{round}}$$

### Formula 4.3 — PBFT Latency
$$T_{PBFT} = T_{pre-prepare} + T_{prepare} + T_{commit} + T_{reply}$$

$$T_{PBFT} \approx 4 \cdot T_{network\_RTT} + 3 \cdot T_{verify}$$

### Formula 4.4 — PBFT Throughput
$$\Theta_{PBFT} = \frac{n_{tx/block}}{T_{PBFT}} = \frac{n_{tx/block}}{4 \cdot RTT + 3 \cdot T_{verify}}$$

### Formula 4.5 — View Change Timeout
$$T_{timeout} = T_{base} \cdot 2^{v - v_0}$$

where $v$ is the current view number (exponential backoff).

### Formula 4.6 — PBFT Maximum Network Size (Practical)
$$N_{max}^{PBFT} \leq \sqrt{\frac{BW_{available} \cdot T_{target}}{3 \cdot S_{msg}}}$$

---

## 5. Delegated Proof of Stake (DPoS)

### Formula 5.1 — Delegate Election (Weighted Voting)
$$\text{Score}(d) = \sum_{v \in \text{voters}(d)} S_v \cdot w_v$$

### Formula 5.2 — DPoS Communication Complexity
$$M_{DPoS} = O(D), \quad D \ll N$$

where $D$ is the number of delegates (typically 21–101).

### Formula 5.3 — DPoS Throughput
$$\Theta_{DPoS} = \frac{D \cdot n_{tx/block}}{D \cdot T_{slot}} = \frac{n_{tx/block}}{T_{slot}}$$

### Formula 5.4 — Block Production Schedule
$$\text{Producer}(t) = \text{Delegates}\left[\lfloor t / T_{slot} \rfloor \mod D\right]$$

### Formula 5.5 — DPoS Decentralization (Nakamoto Coefficient)
$$NC_{DPoS} = \min\left\{k : \sum_{i=1}^{k} \frac{S_{d_i}}{S_{total}} > 0.51\right\}$$

### Formula 5.6 — Delegate Rotation Period
$$T_{rotation} = D \cdot T_{slot}$$

---

## 6. Hybrid Consensus

### Formula 6.1 — Hybrid Utility Function
$$U_{hybrid}(\mathbf{x}) = w_s \cdot S_{sec}(\mathbf{x}) + w_e \cdot E_{eff}(\mathbf{x}) + w_t \cdot \Theta(\mathbf{x}) + w_d \cdot D_{dec}(\mathbf{x}) + w_c \cdot C_{fin}(\mathbf{x})$$

### Formula 6.2 — Weight Adaptation
$$w_i(t) = \frac{\exp(\lambda \cdot r_i(t))}{\sum_j \exp(\lambda \cdot r_j(t))}$$

where $r_i(t)$ is the priority signal for objective $i$ at time $t$.

### Formula 6.3 — Mode Switching Criterion
$$m^*(t) = \arg\max_{m \in \mathcal{M}} U_{hybrid}^m(\mathbf{x}(t))$$

where $\mathcal{M} = \{\text{PoW, PoS, PBFT, DPoS}\}$.

### Formula 6.4 — Transition Cost
$$C_{switch}(m_1 \to m_2) = T_{reconfig} \cdot \Theta_{lost} + E_{overhead}$$

---

## 7. Network Communication Physics

### Formula 7.1 — Network Propagation Delay
$$T_{prop} = \frac{d_{physical}}{c_{medium}} + T_{processing}$$

### Formula 7.2 — Effective Bandwidth Under Load
$$BW_{eff} = BW_{max} \cdot (1 - \rho_{load})$$

### Formula 7.3 — Message Delivery Probability
$$P_{deliver} = (1 - p_{loss})^{n_{hops}} \cdot \kappa$$

where $\kappa$ is the connectivity ratio.

### Formula 7.4 — Consensus Convergence Time
$$T_{converge} = \frac{N}{BW_{eff}} \cdot S_{state} + T_{prop} \cdot \text{diameter}(G)$$

### Formula 7.5 — Network Partition Probability
$$P_{partition} = 1 - \kappa^{N_{edges}}$$

---

## 8. Consensus Performance Metrics

### Formula 8.1 — Classification Accuracy
$$\text{Accuracy} = \frac{\sum_{k=1}^{5} TP_k}{\sum_{k=1}^{5} (TP_k + FP_k)} = \frac{N_{correct}}{N_{total}}$$

### Formula 8.2 — Per-Class Precision
$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

### Formula 8.3 — Per-Class Recall
$$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

### Formula 8.4 — F1 Score
$$F_1^k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### Formula 8.5 — Macro-Averaged F1
$$F_1^{macro} = \frac{1}{K} \sum_{k=1}^{K} F_1^k$$

### Formula 8.6 — Confidence Calibration (Expected Calibration Error)
$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

---

**Total formulas in Part 2: 42**
