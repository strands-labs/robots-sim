#!/usr/bin/env python3
"""GR00T data configurations — robot embodiment key mappings.

Each config maps modality names (video, state, action, language) to the
keys expected by the GR00T inference server.

Configs may optionally include a ``protocol`` field that selects the
transport format.  Two protocols exist:

* ``"sim_wrapper"`` — For ``Gr00tSimPolicyWrapper`` servers (Isaac-GR00T
  N1.6+).  5-D video, float32 state, wrapped request envelope.
* ``"direct"`` — For bare-policy servers (Isaac-GR00T N1.5 and earlier).
  4-D video, float64 state, flat request.

When ``protocol`` is absent the client auto-detects at runtime.

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

# ---------------------------------------------------------------------------
# Protocol descriptors — HOW observations are formatted and sent.
# ---------------------------------------------------------------------------

PROTOCOLS = {
    "sim_wrapper": {
        "video_ndim": 5,  # (B, T, H, W, C)
        "state_dtype": np.float32,
        "request_wrap": "observation",
        "response_batch_dim": True,
        "language_type": "tuple",
    },
    "direct": {
        "video_ndim": 4,  # (B, H, W, C)
        "state_dtype": np.float64,
        "request_wrap": None,
        "response_batch_dim": False,
        "language_type": "list",
    },
}

# ---------------------------------------------------------------------------
# Embodiment data configs — WHAT keys to use (unchanged from original).
# ---------------------------------------------------------------------------

DATA_CONFIGS = {
    "fourier_gr1_arms_only": {
        "video": ["video.ego_view"],
        "state": ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
        "action": ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
        "language": ["annotation.human.action.task_description"],
    },
    "bimanual_panda_gripper": {
        "video": ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"],
        "state": [
            "state.right_arm_eef_pos",
            "state.right_arm_eef_quat",
            "state.right_gripper_qpos",
            "state.left_arm_eef_pos",
            "state.left_arm_eef_quat",
            "state.left_gripper_qpos",
        ],
        "action": [
            "action.right_arm_eef_pos",
            "action.right_arm_eef_rot",
            "action.right_gripper_close",
            "action.left_arm_eef_pos",
            "action.left_arm_eef_rot",
            "action.left_gripper_close",
        ],
        "language": ["annotation.human.action.task_description"],
    },
    "unitree_g1": {
        "video": ["video.rs_view"],
        "state": ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
        "action": ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
        "language": ["annotation.human.task_description"],
    },
    "libero": {
        "video": ["video.image", "video.wrist_image"],
        "state": ["state"],
        "action": [
            "action.robot0_joint_pos",
            "action.robot0_joint_vel",
            "action.robot0_eef_pos",
            "action.robot0_eef_quat",
            "action.robot0_gripper_qpos",
        ],
        "language": ["annotation.human.action.task_description"],
        "protocol": "sim_wrapper",
    },
    "libero_spatial": {
        "video": ["video.image", "video.wrist_image"],
        "state": ["state"],
        "action": [
            "action.robot0_joint_pos",
            "action.robot0_joint_vel",
            "action.robot0_eef_pos",
            "action.robot0_eef_quat",
            "action.robot0_gripper_qpos",
        ],
        "language": ["annotation.human.action.task_description"],
        "protocol": "sim_wrapper",
    },
    "libero_goal": {
        "video": ["video.image", "video.wrist_image"],
        "state": ["state"],
        "action": [
            "action.robot0_joint_pos",
            "action.robot0_joint_vel",
            "action.robot0_eef_pos",
            "action.robot0_eef_quat",
            "action.robot0_gripper_qpos",
        ],
        "language": ["annotation.human.action.task_description"],
        "protocol": "sim_wrapper",
    },
    "libero_meanstd": {
        "video": ["video.image", "video.wrist_image"],
        "state": ["state"],
        "action": [
            "action.robot0_joint_pos",
            "action.robot0_joint_vel",
            "action.robot0_eef_pos",
            "action.robot0_eef_quat",
            "action.robot0_gripper_qpos",
        ],
        "language": ["annotation.human.action.task_description"],
    },
}


def get_protocol(name):
    """Return a copy of the named protocol descriptor, or empty dict."""
    return dict(PROTOCOLS.get(name, {}))


def load_data_config(name, protocol=None):
    """Load a data config by name.

    Args:
        name: Config name (e.g. "libero") or dict.
            Supports "name:protocol" syntax (e.g. "libero:sim_wrapper").
        protocol: Override the protocol field in the returned config.

    Returns:
        Dict with video / state / action / language keys and optional
        ``protocol`` field.
    """
    if isinstance(name, dict):
        return name

    # Support "name:protocol" shorthand
    if isinstance(name, str) and ":" in name:
        name, protocol = name.split(":", 1)

    config = _lookup(name)
    if config is None:
        raise ValueError(f"Unknown data_config '{name}'. Available: {list(DATA_CONFIGS.keys())}")

    if protocol:
        config["protocol"] = protocol

    return config


def _lookup(name):
    """Exact or fuzzy config lookup.  Returns a shallow copy.

    Resolution order:
    1. Exact match against DATA_CONFIGS keys.
    2. Fuzzy: any name containing "libero" falls back to the base
       ``libero`` config (or ``libero_meanstd`` if "goal"/"meanstd"
       appears in the name).  This lets callers pass suite names like
       ``"libero_object"`` without registering every variant.
    3. None if nothing matches — caller should raise ValueError.

    Note: the ``protocol`` field in the returned config can still be
    overridden by ``load_data_config(..., protocol=...)`` after lookup.
    """
    if name in DATA_CONFIGS:
        return dict(DATA_CONFIGS[name])
    if isinstance(name, str) and "libero" in name.lower():
        if "goal" in name.lower() or "meanstd" in name.lower():
            return dict(DATA_CONFIGS["libero_meanstd"])
        return dict(DATA_CONFIGS["libero"])
    return None
