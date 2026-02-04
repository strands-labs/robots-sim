#!/usr/bin/env python3
"""
Strands Robotics Simulation - Robot Control in Simulated Environments

A unified Python interface for controlling robots in simulation environments
with policy abstraction architecture.

Key features:
- Policy abstraction for any VLA provider (GR00T, ACT, SmolVLA, etc.)
- Simulation support (Libero, etc.)
- Clean separation between environment control and policy inference
- Multi-camera support with rich configuration options
"""

import warnings

try:
    from strands_robots_sim.policies import MockPolicy, Policy, create_policy
    from strands_robots_sim.sim_env import SimEnv
    from strands_robots_sim.stepped_sim_env import SteppedSimEnv
    from strands_robots_sim.tools.gr00t_inference import gr00t_inference

    try:
        from strands_robots_sim.policies.groot import Gr00tPolicy

        __all__ = [
            "SimEnv",
            "SteppedSimEnv",
            "Policy",
            "Gr00tPolicy",
            "MockPolicy",
            "create_policy",
            "gr00t_inference",
        ]
    except ImportError as e:
        warnings.warn(f"GR00T policy not available (missing dependencies): {e}")
        __all__ = [
            "SimEnv",
            "SteppedSimEnv",
            "Policy",
            "MockPolicy",
            "create_policy",
            "gr00t_inference",
        ]

except ImportError as e:
    warnings.warn(f"Could not import core components: {e}")
    __all__ = []

__version__ = "0.1.0"
