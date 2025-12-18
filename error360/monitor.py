import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict, Tuple

class Error360Monitor:
    """
    Error-360: Anticipatory Geometric Control.
    
    Tracks the latent trajectory of a generative model to detect 
    instability signals (angular acceleration) BEFORE error realization.
    """
    def __init__(self, 
                 history_len: int = 3, 
                 instability_threshold: float = 0.15,
                 smooth_factor: float = 0.9):
        
        self.history = deque(maxlen=history_len)
        self.threshold = instability_threshold
        self.smooth_factor = smooth_factor
        
        # State tracking
        self.prev_omega = 0.0
        self.angular_acceleration = 0.0
        self.is_unstable = False

    def update(self, z_t: torch.Tensor) -> Dict[str, float]:
        """
        Ingest the current latent state z_t and compute geometric diagnostics.
        """
        # Detach and flatten for geometric calculation
        curr_vec = z_t.detach().view(-1)
        self.history.append(curr_vec)

        metrics = {
            "velocity": 0.0,
            "omega": 0.0, # Angular velocity
            "alpha": 0.0, # Angular acceleration
            "risk": 0.0
        }

        # We need at least 3 points to compute acceleration
        if len(self.history) < 3:
            return metrics

        # 1. Get vectors
        z0, z1, z2 = self.history[-3], self.history[-2], self.history[-1]

        # 2. Compute Displacement Vectors
        v1 = z1 - z0
        v2 = z2 - z1
        
        # 3. Velocity (Magnitude of change)
        velocity = torch.norm(v2).item()

        # 4. Angular Velocity (Cosine Distance)
        # Cosine similarity ranges [-1, 1]. 
        # 1.0 = Straight line. < 1.0 = Turning.
        cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        
        # Clamp for numerical stability
        cos_sim = max(-1.0, min(1.0, cos_sim))
        
        # Omega: How much did we turn? (0 = straight, 2 = u-turn)
        omega = 1.0 - cos_sim

        # 5. Angular Acceleration (The Instability Pulse)
        # Did the turn get sharper suddenly?
        alpha = omega - self.prev_omega
        
        # Smooth the signal (EMA) to reduce noise
        self.angular_acceleration = (self.smooth_factor * self.angular_acceleration) + \
                                    ((1 - self.smooth_factor) * alpha)

        # 6. Store state
        self.prev_omega = omega
        
        # 7. Evaluate Risk
        if self.angular_acceleration > self.threshold:
            self.is_unstable = True
            
        metrics = {
            "velocity": velocity,
            "omega": omega,
            "alpha": self.angular_acceleration,
            "risk": max(0.0, self.angular_acceleration / self.threshold)
        }
        
        return metrics

    def reset(self):
        self.history.clear()
        self.prev_omega = 0.0
        self.angular_acceleration = 0.0
        self.is_unstable = False
