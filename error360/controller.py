import torch
from typing import Dict, Callable, Optional
from .monitor import Error360Monitor


class Error360Controller:
    """
    Feed-360 Controller: Intervention Mechanisms.
    
    Provides anticipatory control strategies that activate when 
    geometric instability is detected by the Error360Monitor.
    """
    
    def __init__(self, 
                 monitor: Error360Monitor,
                 strategy: str = "dampen"):
        """
        Args:
            monitor: An Error360Monitor instance for tracking geometry
            strategy: Control strategy ("dampen", "backtrack", "temperature")
        """
        self.monitor = monitor
        self.strategy = strategy
        
        # Control parameters
        self.damping_factor = 0.8
        self.temperature_reduction = 0.7
        self.backtrack_steps = 1
        
        # State history for backtracking
        self.state_history = []
        self.max_history = 5
        
    def step(self, 
             latent: torch.Tensor, 
             temperature: float = 1.0,
             **kwargs) -> Dict:
        """
        Process a single step with anticipatory control.
        
        Args:
            latent: Current latent state tensor
            temperature: Current temperature parameter
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict containing:
                - 'latent': Potentially modified latent state
                - 'temperature': Adjusted temperature
                - 'metrics': Geometric diagnostics
                - 'intervened': Whether intervention occurred
        """
        # 1. Update geometric monitor
        metrics = self.monitor.update(latent)
        
        # 2. Store state for potential backtracking
        self._store_state(latent, temperature)
        
        # 3. Check for instability
        intervened = False
        modified_latent = latent
        modified_temp = temperature
        
        if metrics['alpha'] > self.monitor.threshold:
            intervened = True
            
            if self.strategy == "dampen":
                # Reduce the magnitude of latent changes
                modified_latent = self._apply_damping(latent)
                
            elif self.strategy == "backtrack":
                # Revert to a previous stable state
                modified_latent, modified_temp = self._backtrack()
                
            elif self.strategy == "temperature":
                # Reduce generation temperature
                modified_temp = temperature * self.temperature_reduction
                
        return {
            'latent': modified_latent,
            'temperature': modified_temp,
            'metrics': metrics,
            'intervened': intervened,
            'strategy': self.strategy if intervened else None
        }
    
    def _apply_damping(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Dampen the current latent state by interpolating with history.
        """
        if len(self.state_history) < 2:
            return latent
            
        prev_state = self.state_history[-2]['latent']
        
        # Interpolate between previous and current state
        damped = (self.damping_factor * prev_state + 
                  (1 - self.damping_factor) * latent)
        
        return damped
    
    def _backtrack(self) -> tuple:
        """
        Revert to a previous stable state.
        """
        # Find the most recent stable state
        for i in range(len(self.state_history) - 1, -1, -1):
            if not self.state_history[i].get('unstable', False):
                return (
                    self.state_history[i]['latent'],
                    self.state_history[i]['temperature']
                )
        
        # If no stable state found, return last state
        if self.state_history:
            return (
                self.state_history[-1]['latent'],
                self.state_history[-1]['temperature']
            )
        
        # Fallback to zeros
        return torch.zeros_like(latent), 1.0
    
    def _store_state(self, latent: torch.Tensor, temperature: float):
        """
        Store current state for potential backtracking.
        """
        self.state_history.append({
            'latent': latent.clone().detach(),
            'temperature': temperature,
            'unstable': self.monitor.is_unstable
        })
        
        # Limit history size
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
    
    def reset(self):
        """
        Reset controller and monitor state.
        """
        self.monitor.reset()
        self.state_history.clear()
    
    def set_strategy(self, strategy: str):
        """
        Change the intervention strategy.
        
        Args:
            strategy: One of "dampen", "backtrack", "temperature"
        """
        if strategy not in ["dampen", "backtrack", "temperature"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy


class AdaptiveController(Error360Controller):
    """
    Advanced controller with adaptive thresholding based on 
    local manifold curvature.
    """
    
    def __init__(self, monitor: Error360Monitor, **kwargs):
        super().__init__(monitor, **kwargs)
        self.base_threshold = monitor.threshold
        self.curvature_history = []
        
    def adapt_threshold(self):
        """
        Adjust instability threshold based on recent curvature.
        Higher curvature regions tolerate more angular acceleration.
        """
        if len(self.curvature_history) < 5:
            return
        
        avg_curvature = sum(self.curvature_history[-5:]) / 5
        
        # Gain scheduling: higher curvature -> higher threshold
        self.monitor.threshold = self.base_threshold * (1 + 0.5 * avg_curvature)
