#!/usr/bin/env python3
"""GR00T data configurations — robot embodiment key mappings.

SPDX-License-Identifier: Apache-2.0
"""

# Each config: (video_keys, state_keys, action_keys, language_keys)
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


def load_data_config(name):
    """Load a data config by name. Returns dict with video/state/action/language keys."""
    if isinstance(name, dict):
        return name
    if name in DATA_CONFIGS:
        return DATA_CONFIGS[name]
    # Fuzzy match: any name containing "libero" falls back to the base libero config
    if isinstance(name, str) and "libero" in name.lower():
        if "goal" in name.lower() or "meanstd" in name.lower():
            return DATA_CONFIGS["libero_meanstd"]
        return DATA_CONFIGS["libero"]
    raise ValueError(f"Unknown data_config '{name}'. Available: {list(DATA_CONFIGS.keys())}")
