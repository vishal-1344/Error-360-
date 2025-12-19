# Error-360°
**Anticipatory Error Handling via Latent Proprioception & Locomotion Control**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Status](https://img.shields.io/badge/Status-Research_Preview-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

> "Precision is the ability to react within the pulse window of instability, not the ability to minimize error after it occurs"

## 1. The Problem

Generative models, despite their success, suffer from critical reliability failures including **hallucination, mode collapse, and temporal decoherence** which render them unsafe for high-stakes deployment. Current error-handling paradigms are fundamentally **reactive**, relying on exteroceptive signals (e.g., loss functions or classifiers) that detect failure only after the generation has degraded.

**Error-360 is a framework for anticipatory error handling that monitors the geometric stability of internal state trajectories during inference.**

We introduce **Latent Proprioception**: the capacity for a model to sense instability in its own trajectory through the latent/activation space. By monitoring the kinematics of this path—specifically the angular acceleration ($\dot{\omega}$) or "jerk"—we detect geometric precursors to output degradation *before* quality collapse occurs.

### The Value Proposition

Think of Error-360 as **ABS brakes for Generative Models**. It operates as a closed-loop inference controller with negligible overhead (one dot product + two norms per step, <0.01% of total compute) to prevent trajectory instability without stopping generation, ensuring stable outputs even during high-speed inference.

## 2. Methodology: Locomotion Control

To achieve robust error handling, we treat inference as **locomotion** through a high-dimensional manifold. The model must maintain "balance" (adherence to valid regions) while traversing the latent space.

Error-360 implements a **closed-loop inference controller** (monitor → trigger → intervention) that applies geometric micro-corrections to maintain trajectory stability throughout generation.

### State Space Definition

The monitored state `z_t` varies by model architecture:

**Diffusion Models:**
- **Primary**: `x_t` (the noisy sample at denoising timestep t)
- **Alternative**: VAE latent `z_t` for Stable Diffusion variants
- **Space**: ℝ^d where d = C × H × W (flattened image tensor)

**LLMs (Future Work):**
- **Primary**: Last-layer hidden states h_t ∈ ℝ^d before output projection
- **Alternative**: Logits or attention pattern entropy
- **Space**: Model hidden dimension (e.g., 4096 for Llama-2-7B)

### The Pulse Signal

We compute trajectory kinematics using finite differences and angular measurements:

$$v_t = z_t - z_{t-1} \quad \text{(velocity vector)}$$

$$\omega_t = 1 - \cos(\vec{v}_{t-1}, \vec{v}_t) = 1 - \frac{\vec{v}_{t-1} \cdot \vec{v}_t}{||\vec{v}_{t-1}|| \, ||\vec{v}_t||} \quad \text{(angular velocity)}$$

$$\alpha_t = \omega_t - \omega_{t-1} \quad \text{(angular acceleration / "pulse")}$$

**Stability Condition:** If $\alpha_t > \theta$ (calibrated threshold) **and** $||v_t|| > \epsilon_{\text{min}}$ (velocity guard), trigger intervention.

#### Guardrails

To prevent numerical instability:
- **Minimum velocity threshold**: Skip pulse calculation when $||v_t|| < 10^{-6}$ (near-stationary)
- **Per-scheduler calibration**: Threshold $\theta$ is set to 95th percentile of baseline pulse distribution for the specific solver (e.g., DPM++, DDIM, Euler)
- **Whitening option**: Normalize pulse in per-dimension scaled coordinates during calibration

### Intervention Strategy: The Reflex Cascade

Interventions are model-native and severity-adaptive:

**For Diffusion Models:**
1. **Level 1 (Low Severity)**: Guidance rescaling. Reduce CFG scale by 10% to dampen trajectory curvature.
2. **Level 2 (Medium Severity)**: Step size reduction. Decrease solver step size (for adaptive solvers) or blend with momentum-averaged update.
3. **Level 3 (High Severity)**: Backtrack & perturb. Revert to last stable checkpoint (Safe Harbor: last state where $\alpha_t \approx 0$) and inject small noise ($\sigma = 0.01$) to avoid repeating the unstable path.

**For LLMs (Future):**
1. **Level 1**: Temperature reduction (×0.95)
2. **Level 2**: Top-p constraint (×0.9)
3. **Level 3**: Switch to greedy/beam search for next N tokens

## 3. System Architecture

```ascii
    ┌─────────────────────────────────────┐
    │    Generative Model (e.g., SDXL)    │
    │          Scheduler: DPM++           │
    └──────────────┬──────────────────────┘
                   │ x_t (noisy sample)
                   ▼
    ┌─────────────────────────────────────┐
    │         Error-360 Monitor           │
    │       (Latent Proprioception)       │
    │  ─────────────────────────────      │
    │  • Compute v_t = x_t - x_{t-1}      │
    │  • Compute ω_t (angular velocity)   │
    │  • Compute α_t (pulse / jerk)       │
    │  • Check: α_t > θ AND ||v_t|| > ε   │
    └──────────────┬──────────────────────┘
                   │ metrics + trigger
                   ▼
    ┌─────────────────────────────────────┐
    │        Controller (Reflex Layer)    │
    │  ─────────────────────────────      │
    │  • Level 1: CFG rescale             │
    │  • Level 2: Step damping            │
    │  • Level 3: Backtrack + noise       │
    └─────────────────────────────────────┘
```

## 4. Validation Protocol (In Progress)

Our immediate focus is empirical validation to prove geometric instability predicts output degradation.

### Minimal Falsifiable Claim

> "On SDXL with DPM++ solver, pulse spikes (α > θ) predict CLIP score drops (Δ > 0.2) within 3 steps with AUC ≥ 0.75 across 500 DrawBench prompts. Interventions reduce collapse rate by 30% with ≤5% FID increase."

### Validation Steps

- [ ] **Dataset**: 500 prompts from DrawBench (known-hard cases)
- [ ] **Collapse definition**: CLIP score drop >0.2 OR aesthetic score drop >0.15 within 5 steps
- [ ] **Lead time measurement**: Distribution of (collapse_step - pulse_spike_step)
- [ ] **Baseline comparisons**:
  - Latent norm spike: $||x_t|| > \text{threshold}$
  - Velocity spike: $||v_t|| > \text{threshold}$
  - CFG heuristics
  - No intervention
- [ ] **Diversity check**: LPIPS and FID scores to ensure stability ≠ homogenization

### Visualizations

- [ ] **Phase Portrait**: Real-time $\omega$ vs $\alpha$ plot with stability zones:
  - **Green**: Low ω, low α → Stable trajectory
  - **Yellow**: High ω, low α → Creative turn (smooth)
  - **Red**: High α → Instability precursor (kinetic fracture)
- [ ] **Canary Plot**: Timeline showing pulse spike at step t, quality collapse at step t+3
- [ ] **Rescue Rate**: Bar chart comparing failure rates with/without Error-360

## 5. Installation

```bash
git clone https://github.com/yourusername/error-360.git
cd error-360
pip install -r requirements.txt
```

## 6. Usage

Error-360 is designed to wrap around PyTorch diffusion inference loops.

### Basic Example: Monitoring Only

```python
from error360 import Error360Monitor

# Initialize with per-scheduler calibration
monitor = Error360Monitor(
    calibration_mode="robust",  # Use Median + MAD
    scheduler="dpm++",           # Specify your solver
    min_velocity=1e-6            # Velocity guard
)

# Calibration phase (optional but recommended)
monitor.calibrate(model, calibration_prompts, num_steps=20)

# Inside your generation loop
for t, x_t in enumerate(diffusion_steps):
    
    # 1. Proprioceptive Check
    metrics = monitor.update(x_t)
    
    # 2. Reflex Action
    if metrics['trigger']:
        severity = metrics['severity']  # 'low', 'medium', 'high'
        print(f"[Reflex] Instability detected at step {t} (severity: {severity})")
        
        if severity == 'low':
            cfg_scale *= 0.9
        elif severity == 'medium':
            # Implement momentum blending or step reduction
            pass
        elif severity == 'high':
            x_t = monitor.get_safe_harbor()  # Backtrack
            x_t += torch.randn_like(x_t) * 0.01  # Perturb
        
    # 3. Standard Diffusion Step
    x_t = scheduler.step(model, x_t, t)
```

## 7. Theoretical Foundations

Error-360 draws from control theory and differential geometry:

- **Latent Proprioception**: Internal stability sensing for autonomous trajectory regulation.
- **Geodesic Deviation**: Monitors how paths diverge from shortest routes in latent manifolds.
- **Lyapunov-Inspired**: Angular acceleration correlates with trajectory divergence (not a formal stability proof).
- **Computational Efficiency**: O(d) per step via dot products and norms, <0.01% overhead vs O(d²) model forward pass.

## 8. Relationship to Existing Paradigms

| Paradigm | Objective | Mechanism | Timing | Limitation |
|----------|-----------|-----------|--------|------------|
| Standard ML | Minimize Loss | Exteroception (Loss) | Post-generation | Reactive (too late) |
| Guidance Rescaling | Prevent saturation | CFG adjustment | Fixed schedule | Not adaptive to instability |
| Error-360 | Prevent Collapse | Proprioception (Trajectory) | Real-time (anticipatory) | Requires calibration |

## 9. Roadmap

- [x] Core proprioception monitor (velocity, ω, α)
- [x] Reflex cascade framework
- [x] Per-scheduler calibration protocol
- [ ] SDXL + DPM++ validation experiment
- [ ] Phase portrait visualization tool
- [ ] Diversity-preserving threshold tuning
- [ ] LLM extension (hidden state monitoring)
- [ ] Paper: "Anticipatory Error Handling via Geometric Trajectory Monitoring"

## 10. Citation

If you use Error-360 in your research, please cite:

```bibtex
@software{error360_2025,
  author = {Vishal J.},
  title = {Error-360: Anticipatory Error Handling via Latent Proprioception and Locomotion Control},
  year = {2025},
  url = {https://github.com/yourusername/error-360}
}
```

## 11. License

MIT License - see LICENSE for details.

## 12. Positioning

**What Error-360 is:**
- An inference-time control loop using internal trajectory geometry as an early-warning signal for output degradation
- A lightweight monitor (one dot product + two norms per step)
- Model-agnostic framework adaptable to diffusion, autoregressive, and planning models

**What Error-360 is not yet:**
- A formal proof of Lyapunov stability
- A universal fix for semantic hallucination
- A replacement for training-time improvements

---

*"The red line doesn't snap. It rotates."*
