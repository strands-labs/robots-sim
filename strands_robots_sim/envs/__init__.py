"""
Environment implementations for strands-robots-sim.

This module provides various simulation environment implementations.
"""

from .base import SimulationEnvironment
from .env_libero import LiberoEnvironment, MockLiberoEnvironment


def create_simulation_environment(env_type: str, **kwargs) -> SimulationEnvironment:
    """Create a simulation environment instance.

    Args:
        env_type: Environment type ("libero", "mock_libero", etc.)
        **kwargs: Environment-specific parameters

    Returns:
        SimulationEnvironment instance

    Raises:
        ValueError: If environment type is not supported
    """
    if env_type == "libero":
        return LiberoEnvironment(**kwargs)
    elif env_type == "mock_libero":
        return MockLiberoEnvironment(**kwargs)
    else:
        available = ["libero", "mock_libero"]
        raise ValueError(f"Unknown environment type: {env_type}. Available: {available}")


__all__ = [
    "SimulationEnvironment",
    "MockLiberoEnvironment",
    "LiberoEnvironment",
    "create_simulation_environment",
]
