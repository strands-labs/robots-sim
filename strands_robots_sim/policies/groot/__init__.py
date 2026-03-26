#!/usr/bin/env python3
"""GR00T Policy — natural language robot control via GR00T inference servers.

Adapts observation formatting and transport to the active protocol
(``sim_wrapper`` or ``direct``).  Data configs are unchanged; the
protocol only controls HOW data is shaped and sent.

SPDX-License-Identifier: Apache-2.0
"""

import logging
import math
from typing import Any, Dict, List, Union

import numpy as np

from .. import Policy
from .client import GR00TClient
from .data_config import get_protocol, load_data_config

logger = logging.getLogger(__name__)


class Gr00tPolicy(Policy):
    """GR00T policy: connects to a GR00T inference server via ZMQ."""

    def __init__(self, data_config: Union[str, dict], host: str = "localhost", port: int = 5555, **kwargs):
        protocol_override = kwargs.pop("protocol", kwargs.pop("groot_version", None))
        # Map legacy version aliases to protocol names
        _aliases = {"n1d6": "sim_wrapper", "n1.6": "sim_wrapper", "n1d5": "direct", "n1.5": "direct"}
        if protocol_override in _aliases:
            protocol_override = _aliases[protocol_override]

        self.config = load_data_config(data_config, protocol=protocol_override)
        self.data_config_name = data_config if isinstance(data_config, str) else "custom"

        self.protocol_name = self.config.get("protocol", "auto")
        self.protocol = get_protocol(self.protocol_name)

        self.client = GR00TClient(host=host, port=port, protocol=self.protocol_name)

        self.camera_keys = self.config["video"]
        self.state_keys = self.config["state"]
        self.action_keys = self.config["action"]
        self.language_keys = self.config["language"]
        self.robot_state_keys: List[str] = []

        logger.info(f"🧠 GR00T Policy: {self.data_config_name} @ {host}:{port} (protocol: {self.protocol_name})")

    # Backward-compat alias
    @property
    def groot_version(self) -> str:
        return self.protocol_name

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        obs = self._build_observation(observation_dict, instruction)
        try:
            action_chunk = self.client.get_action(obs)
        except Exception as e:
            logger.error(f"GR00T inference failed: {e}")
            action_chunk = self._create_fallback_actions()
        return self._to_robot_actions(action_chunk)

    # ------------------------------------------------------------------
    # Observation building — driven by protocol descriptor
    # ------------------------------------------------------------------

    def _build_observation(self, observation_dict: Dict[str, Any], instruction: str) -> dict:
        """Build observation dict.  Protocol controls shape/dtype/wrapping."""
        obs: dict = {}
        video_ndim = self.protocol.get("video_ndim", 4)
        state_dtype = self.protocol.get("state_dtype", np.float64)

        # Video
        for vkey in self.camera_keys:
            cam = self._find_camera(vkey, observation_dict)
            img = observation_dict[cam] if cam and cam in observation_dict else None
            img = (
                self._resize_image(img, self._image_size())
                if img is not None
                else np.zeros((*self._image_size(), 3), dtype=np.uint8)
            )
            obs[vkey] = self._add_video_dims(img, video_ndim)

        # State
        if "libero" in self.data_config_name.lower():
            self._map_libero_state(obs, observation_dict, state_dtype, video_ndim)
        else:
            self._map_state(obs, observation_dict, state_dtype)

        # Language
        if self.language_keys:
            lang_type = self.protocol.get("language_type", "list")
            obs[self.language_keys[0]] = (instruction,) if lang_type == "tuple" else [instruction]

        return obs

    @staticmethod
    def _add_video_dims(image: np.ndarray, ndim: int) -> np.ndarray:
        image = image.astype(np.uint8)
        assert image.ndim == 3, f"Expected (H, W, C) image, got ndim={image.ndim} shape={image.shape}"
        if ndim == 5:
            return image.reshape(1, 1, *image.shape)  # (B, T, H, W, C)
        return np.expand_dims(image, 0)  # (B, H, W, C)

    def _image_size(self) -> tuple:
        return (720, 1280) if "so100" in self.data_config_name.lower() else (256, 256)

    # ------------------------------------------------------------------
    # State mapping
    # ------------------------------------------------------------------

    def _map_libero_state(self, obs: dict, env_obs: dict, dtype, video_ndim: int):
        """Decompose Libero eef_pos / eef_quat into state.x, state.y … state.gripper.

        If eef_pos/eef_quat are missing, zero-valued state entries are still
        added so the server always receives a complete observation.
        """
        eef_pos = env_obs.get("robot0_eef_pos")
        eef_quat = env_obs.get("robot0_eef_quat")
        gripper = env_obs.get("robot0_gripper_qpos", np.array([0.0, 0.0]))

        if eef_pos is None or eef_quat is None:
            logger.warning("robot0_eef_pos/eef_quat missing from observation — using zeros")
            eef_pos = eef_pos if eef_pos is not None else np.zeros(3)
            eef_quat = eef_quat if eef_quat is not None else np.array([0, 0, 0, 1.0])
        rpy = self._quat2axisangle(eef_quat)

        scalars = {"x": eef_pos[0], "y": eef_pos[1], "z": eef_pos[2], "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}

        for name, val in scalars.items():
            key = f"state.{name}"
            if video_ndim == 5:
                obs[key] = np.array([[[val]]], dtype=dtype)
            else:
                obs[key] = np.array([[val]], dtype=dtype)

        gripper_arr = np.asarray(gripper, dtype=dtype)
        if video_ndim == 5:
            obs["state.gripper"] = gripper_arr.reshape(1, 1, -1)
        else:
            obs["state.gripper"] = np.expand_dims(gripper_arr, 0)

    def _map_state(self, obs: dict, env_obs: dict, dtype):
        parts = []
        for k in self.robot_state_keys:
            v = env_obs.get(k, 0.0)
            parts.extend(np.atleast_1d(v).flatten() if isinstance(v, (list, np.ndarray)) else [float(v)])
        state = np.array(parts, dtype=dtype)

        name = self.data_config_name.lower()
        if "so100" in name and len(state) >= 6:
            obs["state.single_arm"] = state[:5]
            obs["state.gripper"] = state[5:6]
        elif "fourier_gr1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7]
            obs["state.right_arm"] = state[7:14]
        elif "unitree_g1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7]
            obs["state.right_arm"] = state[7:14]
        elif "bimanual_panda" in name and len(state) >= 12:
            obs["state.right_arm_eef_pos"] = state[:3]
            obs["state.right_arm_eef_quat"] = state[3:7]
            obs["state.left_arm_eef_pos"] = state[7:10]
            obs["state.left_arm_eef_quat"] = state[10:14]
        elif self.state_keys and len(state) > 0:
            obs[self.state_keys[0]] = state

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _find_camera(self, video_key: str, obs: dict) -> str:
        name = video_key.replace("video.", "")
        for candidate in (video_key, name):
            if candidate in obs:
                return candidate
        aliases = {
            "image": ["front_camera", "agentview_image", "front", "webcam", "main"],
            "wrist_image": ["wrist_camera", "robot0_eye_in_hand_image", "wrist", "hand", "end_effector"],
            "webcam": ["webcam", "front", "wrist", "main"],
            "front": ["front", "webcam", "top", "ego_view", "main"],
            "wrist": ["wrist", "hand", "end_effector", "gripper"],
            "ego_view": ["front", "ego_view", "webcam", "main"],
            "rs_view": ["rs_view", "front", "ego_view", "webcam"],
        }
        for c in aliases.get(name, [name]):
            if c in obs:
                return c
        cams = [
            k
            for k in obs
            if any(n in k.lower() for n in ("camera", "image", "webcam", "front", "wrist", "video", "rgb"))
            and not k.startswith(("state.", "robot0_joint", "robot0_eef", "robot0_gripper"))
        ]
        return cams[0] if cams else None

    def _resize_image(self, image: np.ndarray, target: tuple = (256, 256)) -> np.ndarray:
        """Resize image to target (H, W).  Always returns a 3-D (H, W, C) array."""
        try:
            if image.ndim == 4:
                image = image[0]
            elif image.ndim == 2:
                image = image[..., np.newaxis]
            h, w = image.shape[:2]
            th, tw = target
            if (h, w) == (th, tw):
                return image
            try:
                import cv2

                return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                pass
            try:
                from scipy.ndimage import zoom

                return zoom(image, (th / h, tw / w, 1) if image.ndim == 3 else (th / h, tw / w), order=1).astype(
                    image.dtype
                )
            except ImportError:
                pass
            hi, wi = np.linspace(0, h - 1, th).astype(int), np.linspace(0, w - 1, tw).astype(int)
            return image[np.ix_(hi, wi, range(image.shape[2]))] if image.ndim == 3 else image[np.ix_(hi, wi)]
        except Exception:
            # Ensure we always return 3-D so _add_video_dims doesn't fail
            if image.ndim == 2:
                return image[..., np.newaxis]
            if image.ndim == 4:
                return image[0]
            return image

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _to_robot_actions(self, chunk: dict) -> List[Dict[str, Any]]:
        # Strip batch dim if protocol says response has one
        if self.protocol.get("response_batch_dim"):
            chunk = {
                k: v[0] if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] == 1 else v
                for k, v in chunk.items()
            }

        act_key = self._find_action_key(chunk)
        if not act_key:
            return []
        horizon = chunk[act_key].shape[0]
        actions: list = []

        if "libero" in self.data_config_name.lower():
            for i in range(horizon):
                actions.append({"action": self._to_libero_action(chunk, i).tolist()})
        else:
            for i in range(horizon):
                parts = []
                for k in self.action_keys:
                    mod = k.split(".")[-1] if "." in k else k
                    for c in (k, f"action.{mod}", mod):
                        if c in chunk:
                            parts.append(np.atleast_1d(chunk[c][i]).flatten())
                            break
                cat = np.concatenate(parts) if parts else np.zeros(len(self.robot_state_keys) or 6)
                actions.append({k: float(cat[j]) if j < len(cat) else 0.0 for j, k in enumerate(self.robot_state_keys)})
        return actions

    def _find_action_key(self, chunk: dict) -> str:
        for k in self.action_keys:
            base = k.split(".")[-1] if "." in k else k
            for c in (k, f"action.{base}", base):
                if c in chunk:
                    return c
        for k in chunk:
            if k.startswith("action."):
                return k
        return None

    def _to_libero_action(self, chunk: dict, idx: int = 0) -> np.ndarray:
        parts = []
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            for c in (f"action.{key}", key):
                if c in chunk:
                    parts.append(float(np.asarray(chunk[c][idx]).flatten()[0]))
                    break
            else:
                parts.append(0.0)
        action = np.array(parts, dtype=np.float32)
        action[-1] = np.sign(1 - 2 * action[-1])  # gripper [0,1] → {+1,−1}
        return action

    # Typical dimensionality for known action key fragments.
    _ACTION_DIM_PATTERNS = {
        "joint_pos": 7,
        "joint_vel": 7,
        "eef_pos": 3,
        "eef_quat": 4,
        "eef_rot": 3,
        "gripper_qpos": 1,
        "gripper_close": 1,
        "gripper": 1,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 1,
        "right_hand": 1,
        "single_arm": 5,
    }

    @classmethod
    def _infer_action_dim(cls, key: str) -> int:
        """Infer action dimensionality from a key name like 'action.robot0_joint_pos'."""
        name = key.split(".")[-1] if "." in key else key
        # Try longest suffix match first (e.g. "gripper_qpos" before "gripper")
        for pattern in sorted(cls._ACTION_DIM_PATTERNS, key=len, reverse=True):
            if name.endswith(pattern):
                return cls._ACTION_DIM_PATTERNS[pattern]
        return 1

    def _create_fallback_actions(self) -> dict:
        h = 16 if self.protocol_name == "sim_wrapper" else 8
        chunk = {}
        for key in self.action_keys:
            dim = self._infer_action_dim(key)
            chunk[key] = np.zeros((h, dim), dtype=np.float32)
        return chunk

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        q = np.array(quat)
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(1.0 - q[3] * q[3])
        return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


__all__ = ["Gr00tPolicy"]
