"""
Error-360 Demo: Diffusion Model Integration

This example demonstrates how to integrate Error-360 into a standard
diffusion model inference loop for anticipatory geometric control.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from error360 import Error360Monitor, Error360Controller


class SimpleDiffusionModel(nn.Module):
    """
    A minimal diffusion model for demonstration purposes.
    In practice, replace this with your actual diffusion model.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simple denoising network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x, t):
        """Predict noise at timestep t"""
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_embed], dim=-1)
        return self.net(inp)


def run_baseline_diffusion(model, num_steps=50, seed=42):
    """
    Standard diffusion sampling without Error-360.
    """
    torch.manual_seed(seed)
    
    # Initialize with noise
    latent = torch.randn(1, model.latent_dim)
    trajectory = [latent.clone()]
    
    print("=== Baseline Diffusion (No Control) ===")
    
    for t in range(num_steps):
        # Timestep (normalized)
        timestep = torch.tensor([t / num_steps])
        
        # Predict and subtract noise
        noise_pred = model(latent, timestep)
        latent = latent - 0.1 * noise_pred
        
        trajectory.append(latent.clone())
    
    print(f"Completed {num_steps} steps\n")
    return torch.stack(trajectory)


def run_error360_diffusion(model, num_steps=50, seed=42, strategy="dampen"):
    """
    Diffusion sampling WITH Error-360 anticipatory control.
    """
    torch.manual_seed(seed)
    
    # Initialize Error-360
    monitor = Error360Monitor(
        history_len=3,
        instability_threshold=0.15,
        smooth_factor=0.9
    )
    controller = Error360Controller(monitor, strategy=strategy)
    
    # Initialize with noise
    latent = torch.randn(1, model.latent_dim)
    trajectory = [latent.clone()]
    
    # Tracking
    interventions = []
    metrics_history = []
    
    print(f"=== Error-360 Diffusion (Strategy: {strategy}) ===")
    
    for t in range(num_steps):
        # Timestep (normalized)
        timestep = torch.tensor([t / num_steps])
        
        # Predict noise
        noise_pred = model(latent, timestep)
        latent = latent - 0.1 * noise_pred
        
        # Error-360 Control Step
        result = controller.step(latent, temperature=1.0)
        
        # Use potentially modified state
        latent = result['latent']
        metrics = result['metrics']
        
        # Log metrics
        metrics_history.append(metrics)
        trajectory.append(latent.clone())
        
        # Report interventions
        if result['intervened']:
            interventions.append(t)
            print(f"  Step {t:3d}: INTERVENTION via {result['strategy']}")
            print(f"    └─ Alpha: {metrics['alpha']:.4f} | Risk: {metrics['risk']:.2f}x")
    
    print(f"\nCompleted {num_steps} steps")
    print(f"Total interventions: {len(interventions)}")
    if interventions:
        print(f"Intervention steps: {interventions}\n")
    
    return torch.stack(trajectory), metrics_history


def visualize_trajectories(baseline_traj, controlled_traj, metrics_history):
    """
    Visualize the latent trajectories and geometric metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trajectory Norms (Overall magnitude)
    ax = axes[0, 0]
    baseline_norms = torch.norm(baseline_traj, dim=-1).squeeze().numpy()
    controlled_norms = torch.norm(controlled_traj, dim=-1).squeeze().numpy()
    
    ax.plot(baseline_norms, label='Baseline', color='gray', alpha=0.7)
    ax.plot(controlled_norms, label='Error-360', color='blue', linewidth=2)
    ax.set_title('Latent Trajectory Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Angular Velocity (Omega)
    ax = axes[0, 1]
    omegas = [m['omega'] for m in metrics_history]
    ax.plot(omegas, color='orange', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Turn')
    ax.set_title('Angular Velocity (ω)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('ω (Turn Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Angular Acceleration (Alpha - The Pulse Signal)
    ax = axes[1, 0]
    alphas = [m['alpha'] for m in metrics_history]
    threshold = 0.15
    
    ax.plot(alphas, color='red', linewidth=2, label='α (Pulse)')
    ax.axhline(y=threshold, color='darkred', linestyle='--', 
               linewidth=2, label=f'Threshold ({threshold})')
    ax.fill_between(range(len(alphas)), threshold, [max(a, threshold) for a in alphas],
                     where=[a > threshold for a in alphas], 
                     alpha=0.3, color='red', label='Unstable Region')
    ax.set_title('Angular Acceleration (α) - The Pulse Signal', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('α (Instability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Risk Score
    ax = axes[1, 1]
    risks = [m['risk'] for m in metrics_history]
    ax.plot(risks, color='purple', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Risk = 1.0x')
    ax.fill_between(range(len(risks)), 1.0, risks,
                     where=[r > 1.0 for r in risks],
                     alpha=0.3, color='purple', label='High Risk')
    ax.set_title('Risk Score (α / threshold)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Risk Multiplier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error360_demo_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: error360_demo_results.png")
    plt.show()


def main():
    """
    Main demonstration comparing baseline vs Error-360 control.
    """
    print("=" * 60)
    print("Error-360 Diffusion Demo")
    print("=" * 60)
    print()
    
    # Initialize model
    model = SimpleDiffusionModel(latent_dim=128)
    model.eval()
    
    num_steps = 50
    seed = 42
    
    # Run baseline
    print("Running baseline diffusion...")
    baseline_traj = run_baseline_diffusion(model, num_steps, seed)
    
    # Run with Error-360
    print("Running Error-360 controlled diffusion...")
    controlled_traj, metrics = run_error360_diffusion(
        model, num_steps, seed, strategy="dampen"
    )
    
    # Visualize results
    print("Generating visualizations...")
    visualize_trajectories(baseline_traj, controlled_traj, metrics)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
