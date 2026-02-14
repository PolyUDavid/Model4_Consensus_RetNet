# Part 3: Simulation Platform Algorithm Formulas

> **Model 4: ConsensusRetNet — Pygame V2X Blockchain Simulation**  
> Author: NOK KO  
> Total formulas in this document: **35**

---

## 1. Vehicle (Node) Dynamics

### Formula 1.1 — Position Update
$$x_{node}(t + \Delta t) = x_{node}(t) + v(t) \cdot \Delta t$$

### Formula 1.2 — Speed Acceleration
$$v(t + \Delta t) = \min\left(v_{target}, \; v(t) + a_{accel} \cdot \Delta t\right), \quad a_{accel} = 70 \text{ px/s}^2$$

### Formula 1.3 — Speed Deceleration
$$v(t + \Delta t) = \max\left(v_{target}, \; v(t) - a_{decel} \cdot \Delta t\right), \quad a_{decel} = 50 \text{ px/s}^2$$

### Formula 1.4 — Traffic Light Braking
$$v(t + \Delta t) = \max\left(0, \; v(t) - a_{brake} \cdot \Delta t\right), \quad a_{brake} = 120 \text{ px/s}^2$$

Condition: $x_{light} - 120 < x_{node} < x_{light} + 50$ AND light is red.

### Formula 1.5 — Collision Avoidance
$$v_{follower}(t + \Delta t) = \max\left(v_{leader}(t), \; v_{follower}(t) - a_{avoid} \cdot \Delta t\right)$$

Condition: Same lane, gap $< 90$ px, leader is slower.

### Formula 1.6 — Camera Following
$$x_{cam}(t + \Delta t) = x_{cam}(t) + \alpha_{smooth} \cdot (x_{target} - x_{cam}(t))$$

where $\alpha_{smooth} = 0.06$ and $x_{target} = \bar{x}_{nodes} - W_{screen}/2$.

---

## 2. Traffic Light State Machine

### Formula 2.1 — State Transition
$$\text{State}(t) = \begin{cases} \text{green} \to \text{yellow} & \text{if } \tau \geq T_{green} \\ \text{yellow} \to \text{red} & \text{if } \tau \geq T_{yellow} = 3s \\ \text{red} \to \text{green} & \text{if } \tau \geq T_{red} \end{cases}$$

### Formula 2.2 — Randomized Duration
$$T_{green} = 50 + U(-8, 8), \quad T_{red} = 38 + U(-8, 8)$$

where $U(a,b)$ is a uniform random integer.

### Formula 2.3 — Phase Cycle Period
$$T_{cycle} = T_{green} + T_{yellow} + T_{red} \approx 91 \text{s}$$

---

## 3. RSU Signal Animation

### Formula 3.1 — Signal Phase Update
$$\phi_{RSU}(t + \Delta t) = \phi_{RSU}(t) + 2.5 \cdot \Delta t$$

### Formula 3.2 — Wave Radius
$$r_i(t) = 35 + i \cdot 25 + \lfloor(\phi_{RSU} \mod 1) \cdot 25\rfloor, \quad i \in \{0, 1, 2\}$$

### Formula 3.3 — Wave Alpha (Transparency)
$$\alpha_i(t) = \max\left(0, \; 120 - i \cdot 35 - (\phi_{RSU} \mod 1) \cdot 120\right)$$

---

## 4. Dynamic Network State Computation

### Formula 4.1 — Visible Node Count
$$N_{visible}(t) = \left|\{n : x_{cam} - 200 < x_n < x_{cam} + W_{screen} + 200\}\right|$$

### Formula 4.2 — Scaled Node Count
$$N_{nodes}(t) = \max(10, \; 5 \cdot N_{visible})$$

### Formula 4.3 — Dynamic Connectivity
$$\kappa(t) = \text{clamp}\left(0.9 - 0.003 \cdot N_{visible}, \; 0.6, \; 0.95\right)$$

### Formula 4.4 — Dynamic Network Load
$$\rho_{load}(t) = \min\left(0.95, \; 0.2 + 0.015 \cdot N_{visible}\right)$$

### Formula 4.5 — Byzantine Ratio Detection
$$r_{byz}(t) = \frac{N_{byzantine}}{N_{visible}}$$

### Formula 4.6 — Dynamic Attack Risk (Scenario 2)
$$\alpha_{risk}(t) = \min\left(1.0, \; 0.6 + 0.35 \cdot \sin(0.3t)\right)$$

### Formula 4.7 — Emergency Throughput Spikes (Scenario 3)
$$\Theta_{req}(t) = 3000 \cdot \left(1 + 0.5 \cdot \sin(0.5t)\right)$$

### Formula 4.8 — Emergency Latency Tightening
$$L_{req}(t) = \max\left(0.5, \; 2.0 - 0.01t\right)$$

---

## 5. Transaction Visualization

### Formula 5.1 — Transaction Progress
$$p_{tx}(t + \Delta t) = p_{tx}(t) + v_{tx} \cdot \Delta t, \quad v_{tx} \in [1.5, 3.0]$$

### Formula 5.2 — Transaction Position (Linear Interpolation)
$$\mathbf{pos}_{tx}(t) = (1 - p_{tx}) \cdot \mathbf{pos}_{src} + p_{tx} \cdot \mathbf{pos}_{dst}$$

### Formula 5.3 — Transaction Completion
$$\text{complete} = (p_{tx} \geq 1.0)$$

---

## 6. Block Generation

### Formula 6.1 — Block Timer
$$\tau_{block}(t + \Delta t) = \tau_{block}(t) + \Delta t$$

### Formula 6.2 — Block Confirmation Condition
$$\text{confirm\_block} \iff \tau_{block} \geq T_{interval}$$

where $T_{interval} = \{2.0, 3.5, 0.5\}$ seconds for scenarios $\{1, 2, 3\}$.

### Formula 6.3 — Blocks Per Second (Simulation TPS)
$$\text{BPS}_{sim} = \frac{1}{T_{interval}}$$

---

## 7. Node Glow Effects

### Formula 7.1 — Pulse Phase
$$\phi_{pulse}(t + \Delta t) = \phi_{pulse}(t) + 3.0 \cdot \Delta t$$

### Formula 7.2 — Glow Alpha Oscillation
$$\alpha_{glow}(t) = 60 + 40 \cdot \sin(\phi_{pulse})$$

### Formula 7.3 — Byzantine Danger Alpha
$$\alpha_{danger}(t) = 80 + 60 \cdot \sin(2 \cdot \phi_{pulse})$$

---

## 8. Attack Effect Particles

### Formula 8.1 — Particle Velocity
$$\mathbf{v}_{particle} = (v_x, v_y), \quad v_x, v_y \sim U(-80, 80)$$

### Formula 8.2 — Particle Position Update
$$x_p(t + \Delta t) = x_p(t) + v_x \cdot \Delta t$$
$$y_p(t + \Delta t) = y_p(t) + v_y \cdot \Delta t$$

### Formula 8.3 — Particle Lifetime
$$\text{alive} = (\text{age} < \text{life}), \quad \text{life} \sim U(0.5, 1.5) \text{s}$$

### Formula 8.4 — Particle Alpha Decay
$$\alpha_p(t) = 255 \cdot \left(1 - \frac{\text{age}}{\text{life}}\right)$$

### Formula 8.5 — Particle Size Decay
$$s_p(t) = 6 \cdot \left(1 - \frac{\text{age}}{\text{life}}\right)$$

---

## 9. API Integration

### Formula 9.1 — Prediction Source Priority
$$\text{source} = \begin{cases} \text{API} & \text{if API responds within 500ms} \\ \text{Local Model} & \text{if model loaded} \\ \text{Rule-based} & \text{fallback} \end{cases}$$

### Formula 9.2 — Prediction Refresh Rate
$$f_{predict} = 1 \text{ Hz} \quad (\text{every } 1.0 \text{s})$$

---

**Total formulas in Part 3: 35**
