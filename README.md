# Error-360°
**Locomotive Control & Latent Proprioception for Generative Stability**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Status](https://img.shields.io/badge/Status-Research_Preview-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

> "Precision is the ability to correct a stumble, not just the ability to measure a fall."

## 1. The Core Thesis

Conventional ML relies on **exteroception**: measuring error against external ground truth ($\mathcal{L}$). However, in generative tasks (diffusion, planning), this is a lagging indicator. By the time the "visual" output is corrupted, the internal reasoning process has already collapsed.

**Error-360 introduces Latent Proprioception: the capacity for a model to sense the stability of its own internal state trajectory.**

Just as a biological motor system uses proprioception to detect a slip before a fall, Error-360 monitors the "gait" of the generative process. We track three kinematic signals to detect **stumble precursors** before they manifest as hallucinations:

1. **Velocity ($\dot{z}$)**: The stride length or rate of representational change.
2. **Angular Velocity ($\omega$)**: The turning rate. High values indicate the trajectory is "cornering" hard.
3. **Angular Acceleration ($\dot{\omega}$)**: The **Instability Pulse**. A sudden spike here represents a "jerk" or structural slip.

## 2. Why "Locomotive" Control?

We reframe generative inference not as a static calculation, but as **high-dimensional locomotion** subject to dynamic constraints. The model must maintain "balance" (manifold adherence) while moving through the latent space.

Error-360 acts as the **Digital Cerebellum**, using proprioceptive feedback to apply micro-corrections (dampening, re-balancing) ensuring the model maintains a stable gait throughout the generation.

## 3. Methodology

The system acts as a "Feed-360" loop wrapping standard inference. It forecasts instability and intervenes before failure manifests.

```ascii
    ┌─────────────────────────────────────┐
    │     Generative Model (Diffusion)    │
    └──────────────┬──────────────────────┘
                   │ z_t (latent state)
                   ▼
    ┌─────────────────────────────────────┐
    │       Error-360 Monitor             │
    │      (Latent Proprioception)        │
    │  ─────────────────────────────      │
    │  • Measure Gait (Velocity/Turn)     │
    │  • Detect Slip (Acceleration > θ)   │
    │  • Signal Instability               │
    └──────────────┬──────────────────────┘
                   │ metrics
                   ▼
    ┌─────────────────────────────────────┐
    │       Controller (Reflex Layer)     │
    │  ─────────────────────────────      │
    │  • Level 1: Micro-Adjust Temp       │
    │  • Level 2: Dampen Momentum         │
    │  • Level 3: Backtrack & Perturb     │
    └─────────────────────────────────────┘
```

### The Pulse Signal

We utilize the derivative of cosine similarity to detect "panic" in the model's reasoning process:

$$\omega_t = 1 - \cos(\vec{v}_{t-1}, \vec{v}_t) \quad \text{where} \quad \vec{v}_t = z_t - z_{t-1}$$

$$\alpha_t = \omega_t - \omega_{t-1} \quad \text{(The Pulse)}$$

If $\text{Pulse} > \text{Threshold}$, the controller triggers a Reflex Cascade within the Pulse Reactive Time (PRT) window.

### Intervention Strategy: The Reflex Cascade

Interventions are applied based on the severity of the slip, minimizing impact on creativity:

1. **Micro-Correction (Low Severity)**: Dynamic Temperature Adjustment. Slightly reduce randomness to stabilize footing.
2. **Dampening (Medium Severity)**: Momentum Decay. Blend current update with previous stable vector.
3. **Branch & Bound (High Severity)**: Backtrack to the last "Safe Harbor" checkpoint and perturb the trajectory (noise injection) to avoid repeating the same error.

## 4. Relationship to Existing Paradigms

Error-360 complements probabilistic learning by adding a geometric control layer.

| Paradigm | Objective | Mechanism | Limitation |
|----------|-----------|-----------|------------|
| Standard ML | Minimize Loss | Exteroception (Output Check) | Reactive (Too late) |
| RL | Maximize Reward | Sparse Feedback | Slow convergence |
| Error-360 | Maintain Stability | Proprioception (Internal Sense) | Anticipatory |

## 5. Installation

```bash
git clone https://github.com/yourusername/error-360.git
cd error-360
pip install -r requirements.txt
```

## 6. Usage

Error-360 is designed to wrap around any PyTorch inference loop (LLM or Diffusion).

### Basic Example: Monitoring Only

```python
from error360 import Error360Monitor

# Initialize with robust statistics (Median + MAD) for outlier resistance
monitor = Error360Monitor(calibration_mode="robust")

# Inside your generation loop
for t, latent in enumerate(diffusion_steps):
    
    # 1. Proprioceptive Check
    metrics = monitor.update(latent)
    
    # 2. Reflex Action
    if metrics['alpha'] > monitor.threshold:
        print(f"[Reflex] Slip detected at step {t}. Correcting...")
        # Intervention logic here...
        
    # 3. Standard Step
    latent = model(latent)
```

## 7. Theoretical Foundations

Error-360 draws from control theory and differential geometry:

- **Latent Proprioception**: Provides the "inner sense" required for autonomous stability regulation.
- **Geodesic Deviation**: Monitors how trajectories diverge from shortest paths in latent space.
- **Lyapunov Stability**: Angular acceleration serves as a proxy for exponential divergence.
- **Computational Efficiency**: Operates with $O(d)$ complexity (vector dot products), introducing negligible overhead (<0.01%) compared to the $O(d^2)$ cost of the underlying generative inference.

## 8. Roadmap & Validation

- [x] Core Proprioception Monitor (Velocity, Omega, Alpha)
- [x] Reflex Cascade (Temperature, Dampen, Backtrack)
- [ ] **Robust Calibration**: Implement Median/MAD thresholding to handle heavy-tailed jerk distributions.
- [ ] **Phase Space Visualization**: Plot $\omega$ vs $\dot{\omega}$ to visually distinguish "Creative Turns" (Green Zone) from "Instability Slips" (Red Zone).
- [ ] **Benchmarking**:
  - **Detection**: "Canary Plots" showing Jerk spiking before visual collapse.
  - **Diversity**: Track LPIPS scores to ensure stability does not compromise output diversity.

## 9. Citation

If you use Error-360 in your research, please cite:

```bibtex
@software{error360_2025,
  author = {Vishal J.},
  title = {Error-360: Locomotive Control and Latent Proprioception for Generative Stability},
  year = {2025},
  url = {https://github.com/yourusername/error-360}
}
```

## 10. License

MIT License - see LICENSE for details.
