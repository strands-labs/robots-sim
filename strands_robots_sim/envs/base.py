#!/usr/bin/env python3
"""
Base Environment Classes

This module provides the abstract base class for all simulation environments.
"""

from typing import Any, Dict, List, Optional


class SimulationEnvironment:
    """Abstract base class for simulation environments."""

    def __init__(self, env_name: str, **kwargs):
        self.env_name = env_name
        self.env = None
        self.is_initialized = False  # nosemgrep: python.lang.maintainability.is-function-without-parentheses

    async def initialize(self) -> bool:
        """Initialize the simulation environment."""
        raise NotImplementedError("Subclasses must implement initialize()")

    async def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment to initial state."""
        raise NotImplementedError("Subclasses must implement reset()")

    async def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        raise NotImplementedError("Subclasses must implement step()")

    async def get_observation(self) -> Dict[str, Any]:
        """Get current observation from environment."""
        raise NotImplementedError("Subclasses must implement get_observation()")

    def get_robot_state_keys(self) -> List[str]:
        """Get robot state keys for this environment."""
        raise NotImplementedError("Subclasses must implement get_robot_state_keys()")

    async def cleanup(self):
        """Cleanup environment resources."""
        pass
