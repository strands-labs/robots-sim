#!/usr/bin/env python3
"""
Policy Abstraction for Universal VLA Support

This module provides a clean abstraction for Vision-Language-Action (VLA) models,
allowing the Robot class to work with any VLA provider without hardcoding.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Policy(ABC):
    """Abstract base class for VLA policies."""

    @abstractmethod
    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state)
            instruction: Natural language instruction
            **kwargs: Provider-specific parameters

        Returns:
            List of action dictionaries for robot execution
        """
        pass

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass


class MockPolicy(Policy):
    """Mock policy for testing and development."""

    def __init__(self, **kwargs):
        """Initialize mock policy."""
        self.robot_state_keys = []
        logger.info("🎭 Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys."""
        self.robot_state_keys = robot_state_keys
        logger.info(f"🔧 Mock robot state keys: {self.robot_state_keys}")

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Return mock actions."""
        import numpy as np

        # Generate mock actions
        mock_actions = []
        for _ in range(8):  # Mock action horizon
            action_dict = {}
            for key in self.robot_state_keys:
                # Generate small random movements
                action_dict[key] = float(np.random.uniform(-0.1, 0.1))
            mock_actions.append(action_dict)

        logger.info(f"🎭 Mock policy generated {len(mock_actions)} actions for: '{instruction}'")
        return mock_actions


# Factory function for creating policies
def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance based on provider name.

    Args:
        provider: Provider name ("groot", "mock", etc.)
        **kwargs: Provider-specific parameters
            For groot: data_config (str or object), host, port, etc.
            For mock: any parameters (ignored)

    Returns:
        Policy instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "mock":
        return MockPolicy(**kwargs)
    elif provider == "groot":
        from .groot import Gr00tPolicy

        # Gr00tPolicy requires data_config as first positional argument
        data_config = kwargs.pop("data_config", None)
        if data_config is None:
            raise ValueError("data_config is required for groot policy")

        return Gr00tPolicy(data_config, **kwargs)
    else:
        # Try dynamic import for extensibility
        try:
            module = __import__(f"strands_robots.policies.{provider}", fromlist=[f"{provider.capitalize()}Policy"])
            PolicyClass = getattr(module, f"{provider.capitalize()}Policy")
            return PolicyClass(**kwargs)
        except (ImportError, AttributeError):
            available = ["groot", "mock"]
            raise ValueError(f"Unknown policy provider: {provider}. Available: {available}")


__all__ = ["Policy", "MockPolicy", "create_policy"]
