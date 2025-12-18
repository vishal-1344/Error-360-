"""
Error-360: Anticipatory Geometric Control for Generative Models

A "System 2" supervisor that detects instability signals in latent 
trajectories BEFORE error realization.
"""

__version__ = "0.1.0"
__author__ = "Ahab"

from .monitor import Error360Monitor
from .controller import Error360Controller, AdaptiveController

__all__ = [
    "Error360Monitor",
    "Error360Controller", 
    "AdaptiveController"
]
