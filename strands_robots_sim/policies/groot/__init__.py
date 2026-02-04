#!/usr/bin/env python3
"""GR00T Policy — natural language robot control via GR00T inference servers.

SPDX-License-Identifier: Apache-2.0
"""

import logging
import math
from typing import Any, Dict, List, Union

import numpy as np

from .. import Policy
from .client import GR00TClient
from .data_config import load_data_config

logger = logging.getLogger(__name__)


class Gr00tPolicy(Policy):
    """GR00T policy: connects to a GR00T inference server via ZMQ."""

    def __init__(self, data_config: Union[str, dict], host: str = "localhost", port: int = 5555, **kwargs):
        """Initialize GR00T policy.

        Args:
            data_config: Config name (e.g. "libero") or dict with video/state/action/language keys
            host: Inference service host
            port: Inference service port
        """
        self.config = load_data_config(data_config)
        self.data_config_name = data_config if isinstance(data_config, str) else "custom"
        self.client = GR00TClient(host=host, port=port)

        self.camera_keys = self.config["video"]
        self.state_keys = self.config["state"]
        self.action_keys = self.config["action"]
        self.language_keys = self.config["language"]
        self.robot_state_keys = []

        logger.info(f"🧠 GR00T Policy: {self.data_config_name} @ {host}:{port}")

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from GR00T policy server.

        Args:
            observation_dict: Robot observations (cameras + state)
            instruction: Natural language instruction

        Returns:
            List of action dicts for robot execution
        """
        obs = {}

        # Camera observations with resizing
        for vkey in self.camera_keys:
            cam = self._find_camera(vkey, observation_dict)
            if cam and cam in observation_dict:
                image = observation_dict[cam]
                if "so100" in self.data_config_name.lower():
                    obs[vkey] = self._resize_image(image, target_size=(720, 1280))
                else:
                    obs[vkey] = self._resize_image(image, target_size=(256, 256))
            else:
                if "so100" in self.data_config_name.lower():
                    obs[vkey] = np.zeros((720, 1280, 3), dtype=np.uint8)
                else:
                    obs[vkey] = np.zeros((256, 256, 3), dtype=np.uint8)

        # State observations
        robot_state_parts = []
        for k in self.robot_state_keys:
            value = observation_dict.get(k, 0.0)
            if isinstance(value, (list, np.ndarray)):
                robot_state_parts.extend(np.atleast_1d(value).flatten())
            else:
                robot_state_parts.append(float(value))
        robot_state = np.array(robot_state_parts, dtype=np.float64)

        if "libero" in self.data_config_name.lower():
            self._map_libero_state(obs, observation_dict)
        else:
            self._map_state(obs, robot_state)

        # Language instruction
        if self.language_keys:
            obs[self.language_keys[0]] = instruction

        # Batch dimension
        for k in obs:
            if isinstance(obs[k], np.ndarray) and k.startswith("video."):
                obs[k] = np.expand_dims(obs[k], axis=0)
            elif isinstance(obs[k], str):
                obs[k] = [obs[k]]

        try:
            action_chunk = self.client.get_action(obs)
        except Exception as e:
            logger.error(f"GR00T inference failed: {e}")
            action_chunk = self._create_fallback_actions()

        return self._to_robot_actions(action_chunk)

    def _find_camera(self, video_key: str, obs: dict) -> str:
        """Map GR00T video key to available camera key."""
        if video_key in obs:
            return video_key

        name = video_key.replace("video.", "")
        if name in obs:
            return name

        # Libero-specific aliases
        libero_aliases = {
            "image": ["front_camera", "agentview_image", "front", "webcam", "main"],
            "wrist_image": ["wrist_camera", "robot0_eye_in_hand_image", "wrist", "hand", "end_effector"],
        }
        if name in libero_aliases:
            for candidate in libero_aliases[name]:
                if candidate in obs:
                    return candidate

        aliases = {
            "webcam": ["webcam", "front", "wrist", "main"],
            "front": ["front", "webcam", "top", "ego_view", "main"],
            "wrist": ["wrist", "hand", "end_effector", "gripper"],
            "ego_view": ["front", "ego_view", "webcam", "main"],
            "top": ["top", "overhead", "front"],
            "side": ["side", "lateral", "left", "right"],
            "rs_view": ["rs_view", "front", "ego_view", "webcam"],
        }
        for candidate in aliases.get(name, [name]):
            if candidate in obs:
                return candidate

        # Fallback: first camera-like key
        cams = [
            k
            for k in obs
            if any(n in k.lower() for n in ["camera", "image", "webcam", "front", "wrist", "video", "rgb", "depth"])
            and not k.startswith("state.")
            and not k.startswith("robot0_joint")
            and not k.startswith("robot0_eef")
            and not k.startswith("robot0_gripper")
        ]
        return cams[0] if cams else None

    def _resize_image(self, image: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
        """Resize image to match GR00T server expectations."""
        try:
            if len(image.shape) == 4:
                image = image[0]
            elif len(image.shape) == 2:
                image = image[..., np.newaxis]

            h, w = image.shape[:2]
            th, tw = target_size
            if h == th and w == tw:
                return image

            try:
                import cv2  # nosec B404

                return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                pass

            try:
                from scipy.ndimage import zoom

                factors = (th / h, tw / w, 1) if len(image.shape) == 3 else (th / h, tw / w)
                return zoom(image, factors, order=1).astype(image.dtype)
            except ImportError:
                pass

            # Numpy nearest-neighbor fallback
            h_idx = np.linspace(0, h - 1, th).astype(int)
            w_idx = np.linspace(0, w - 1, tw).astype(int)
            if len(image.shape) == 3:
                return image[np.ix_(h_idx, w_idx, range(image.shape[2]))]
            return image[np.ix_(h_idx, w_idx)]
        except Exception:
            return image

    def _map_libero_state(self, obs: dict, observation_dict: dict):
        """Map Libero end-effector pose to GR00T state format."""
        if "robot0_eef_pos" in observation_dict and "robot0_eef_quat" in observation_dict:
            xyz = observation_dict["robot0_eef_pos"]
            quat = observation_dict["robot0_eef_quat"]
            gripper = observation_dict.get("robot0_gripper_qpos", np.array([0.0, 0.0]))
            rpy = self._quat2axisangle(quat)
            obs["state.x"] = np.array([[xyz[0]]])
            obs["state.y"] = np.array([[xyz[1]]])
            obs["state.z"] = np.array([[xyz[2]]])
            obs["state.roll"] = np.array([[rpy[0]]])
            obs["state.pitch"] = np.array([[rpy[1]]])
            obs["state.yaw"] = np.array([[rpy[2]]])
            obs["state.gripper"] = np.expand_dims(gripper, axis=0)
        else:
            for key in ("x", "y", "z", "roll", "pitch", "yaw"):
                obs[f"state.{key}"] = np.array([[0.0]], dtype=np.float64)
            obs["state.gripper"] = np.array([[0.0]], dtype=np.float64)

    def _map_state(self, obs: dict, state: np.ndarray):
        """Map robot state array to GR00T state keys."""
        name = self.data_config_name.lower()
        if "so100" in name and len(state) >= 6:
            obs["state.single_arm"] = state[:5].astype(np.float64)
            obs["state.gripper"] = state[5:6].astype(np.float64)
        elif "fourier_gr1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7].astype(np.float64)
            obs["state.right_arm"] = state[7:14].astype(np.float64)
        elif "unitree_g1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7].astype(np.float64)
            obs["state.right_arm"] = state[7:14].astype(np.float64)
        elif "bimanual_panda" in name and len(state) >= 12:
            obs["state.right_arm_eef_pos"] = state[:3].astype(np.float64)
            obs["state.right_arm_eef_quat"] = state[3:7].astype(np.float64)
            obs["state.left_arm_eef_pos"] = state[7:10].astype(np.float64)
            obs["state.left_arm_eef_quat"] = state[10:14].astype(np.float64)
        elif self.state_keys and len(state) > 0:
            obs[self.state_keys[0]] = state.astype(np.float64)

    def _to_robot_actions(self, chunk: dict) -> List[Dict[str, Any]]:
        """Convert GR00T action chunk to list of robot action dicts."""
        act_key = None
        for k in self.action_keys:
            base = k.replace("action.", "") if k.startswith("action.") else k
            full = f"action.{base}"
            if full in chunk:
                act_key = full
                break
        if not act_key:
            act_keys = [k for k in chunk if k.startswith("action.")]
            act_key = act_keys[0] if act_keys else None
        if not act_key:
            return []

        horizon = chunk[act_key].shape[0]
        actions = []

        if "libero" in self.data_config_name.lower():
            for i in range(horizon):
                action_array = self._to_libero_action(chunk, idx=i)
                actions.append({"action": action_array.tolist()})
        else:
            for i in range(horizon):
                parts = []
                for k in self.action_keys:
                    mod = k.split(".")[-1]
                    if f"action.{mod}" in chunk:
                        parts.append(np.atleast_1d(chunk[f"action.{mod}"][i]))
                if not parts:
                    for k, v in chunk.items():
                        if k.startswith("action."):
                            parts.append(np.atleast_1d(v[i]))

                concat = np.concatenate(parts) if parts else np.zeros(len(self.robot_state_keys) or 6)
                actions.append(
                    {k: float(concat[j]) if j < len(concat) else 0.0 for j, k in enumerate(self.robot_state_keys)}
                )

        return actions

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (x,y,z,w) to axis-angle (roll,pitch,yaw)."""
        quat = np.array(quat)
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _to_libero_action(self, action_chunk: dict, idx: int = 0) -> np.ndarray:
        """Convert GR00T action chunk to Libero 7-dim: [dx,dy,dz,droll,dpitch,dyaw,gripper]."""
        components = []
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            full_key = f"action.{key}"
            if full_key in action_chunk:
                components.append(np.atleast_1d(action_chunk[full_key][idx])[0])
            else:
                components.append(0.0)
        action = np.array(components, dtype=np.float32)
        action = self._normalize_gripper(action)
        assert len(action) == 7  # nosec B101
        return action

    @staticmethod
    def _normalize_gripper(action: np.ndarray, binarize: bool = True) -> np.ndarray:
        """Normalize gripper action from [0,1] to [+1,-1]."""
        action[..., -1] = 1 - 2 * action[..., -1]
        if binarize:
            action[..., -1] = np.sign(action[..., -1])
        return action

    def _create_fallback_actions(self) -> dict:
        """Create zero-action fallback when inference fails."""
        chunk = {}
        horizon = 8
        for key in self.action_keys:
            mod = key.split(".")[-1]
            if "joint_pos" in mod.lower():
                dim = 7
            elif "eef_pos" in mod.lower():
                dim = 3
            elif "eef_quat" in mod.lower():
                dim = 4
            elif "gripper" in mod.lower():
                dim = 1
            else:
                dim = len(self.robot_state_keys) // 5 if self.robot_state_keys else 7
            chunk[f"action.{mod}"] = np.zeros((horizon, dim), dtype=np.float64)
        if not chunk:
            chunk["action.robot0_joint_pos"] = np.zeros((horizon, 7), dtype=np.float64)
        return chunk


__all__ = ["Gr00tPolicy"]
