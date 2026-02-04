#!/usr/bin/env python3
"""
Libero Environment Implementations

This module provides Libero-specific simulation environment implementations,
including mock environments for testing and real Libero integration.
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .base import SimulationEnvironment

# Suppress GLFW and OpenGL warnings early
warnings.filterwarnings("ignore", message=".*GLFW.*")
warnings.filterwarnings("ignore", message=".*OpenGL.*")
warnings.filterwarnings("ignore", message=".*X11.*")
warnings.filterwarnings("ignore", message=".*EGL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*egl.*")

# Set headless environment variables early
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logger = logging.getLogger(__name__)


class LiberoEnvironment(SimulationEnvironment):
    """Libero simulation environment implementation."""

    def __init__(self, task_suite: str = "libero_spatial", **kwargs):
        super().__init__(f"libero_{task_suite}", **kwargs)
        self.task_suite = task_suite
        self.available_tasks = []
        self.current_task_idx = 0
        self.task_name = None  # Can be set to specify a particular task

    async def initialize(self) -> bool:
        """Initialize Libero environment."""
        try:
            # Set up environment to avoid interactive prompts

            # Mock the input to automatically answer "N" to dataset path question
            original_input = __builtins__.get("input", input)

            def mock_input(prompt=""):
                logger.info(f"🤖 Auto-answering prompt: {prompt.strip()}")
                return "N"  # Answer "N" to dataset path question

            # Temporarily replace input function
            if hasattr(__builtins__, "__setitem__"):
                __builtins__["input"] = mock_input
            else:
                __builtins__.input = mock_input

            try:
                # Import libero components (HuggingFace LeRobot version)
                from libero.libero import benchmark

                # Get the benchmark task suite
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite_name = self.task_suite.replace("libero_", "")  # Remove prefix for compatibility

                if task_suite_name not in benchmark_dict:
                    # Try with libero_ prefix
                    task_suite_name = self.task_suite
                    if task_suite_name not in benchmark_dict:
                        # Fall back to first available suite
                        available_suites = list(benchmark_dict.keys())
                        logger.warning(f"⚠️ Task suite '{self.task_suite}' not found. Available: {available_suites}")
                        task_suite_name = available_suites[0] if available_suites else "libero_spatial"

                # Create task suite instance
                task_suite = benchmark_dict[task_suite_name]()

                # Get available tasks
                self.available_tasks = []
                for i in range(task_suite.n_tasks):
                    try:
                        task = task_suite.get_task(i)
                        # self.available_tasks.append(f"{task_suite_name.upper()}_{task.name.replace(' ', '_')}")
                        self.available_tasks.append(f"{task.language}")
                    except (AttributeError, KeyError, IndexError, TypeError):
                        # Skip tasks that can't be loaded
                        continue

                self.task_suite_instance = task_suite
                self.task_suite_name = task_suite_name

                print(f"🎮 Libero {task_suite_name} initialized")
                print(f"📋 Available tasks: {len(self.available_tasks)}")
                print(
                    f"🎯 Tasks: {self.available_tasks[:15]}..."
                    if len(self.available_tasks) > 15
                    else f"🎯 Tasks: {self.available_tasks}"
                )

                self.is_initialized = True

                return True

            finally:
                # Restore original input function
                if hasattr(__builtins__, "__setitem__"):
                    __builtins__["input"] = original_input
                else:
                    __builtins__.input = original_input

        except ImportError as e:
            logger.error(f"❌ Libero not installed: {e}")
            logger.error("💡 Install with: pip install lerobot-libero")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Libero: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def set_task_name(self, task_name: str) -> bool:
        """Set a specific task name to use for resets.

        Args:
            task_name: Task name from self.available_tasks

        Returns:
            True if task_name is valid, False otherwise
        """
        if not self.is_initialized:

            logger.error("❌ Environment must be initialized before setting task_name")
            return False

        if task_name not in self.available_tasks:
            logger.error(f"❌ Task '{task_name}' not found in available tasks")
            logger.info(f"Available tasks: {self.available_tasks}")
            return False

        self.task_name = task_name
        logger.info(f"✅ Task name set to: {task_name}")
        return True

    async def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset Libero environment."""
        try:
            if not self.is_initialized:

                raise RuntimeError("Environment not initialized")

            # Use self.task_name if set, otherwise use provided task_name or cycle through tasks
            if task_name is None and self.task_name is not None:
                task_name = self.task_name

            # Select task
            if task_name:
                # Extract task id from name if it contains task suite prefix
                task_names_only = [t.split("_", 2)[-1] if t.count("_") >= 2 else t for t in self.available_tasks]
                if task_name in self.available_tasks:
                    task_id = self.available_tasks.index(task_name)
                elif task_name in task_names_only:
                    task_id = task_names_only.index(task_name)
                else:
                    raise ValueError(f"Task {task_name} not found in {self.task_suite}")
            else:
                task_id = self.current_task_idx % len(self.available_tasks)
                task_name = self.available_tasks[task_id]

            # Get task from suite
            task = self.task_suite_instance.get_task(task_id)

            # Create environment using libero's OffScreenRenderEnv
            import os

            from libero.libero.envs import OffScreenRenderEnv
            from libero.libero.utils import get_libero_path

            # Get BDDL file path
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 256,  # Match Isaac-GR00T reference (increased from 128)
                "camera_widths": 256,  # Match Isaac-GR00T reference (increased from 128)
            }

            self.env = OffScreenRenderEnv(**env_args)
            self.env.seed(42)

            # Reset environment
            obs = self.env.reset()

            # Set initial states if available (handle PyTorch 2.6 compatibility)
            try:
                init_states = self.task_suite_instance.get_task_init_states(task_id)
                if init_states:
                    self.env.set_init_state(init_states[0])
                    obs = self.env.reset()  # Reset again with initial state
            except Exception as init_error:
                # Handle PyTorch loading issues with initial states
                logger.warning(f"⚠️ Could not load initial states: {init_error}")
                logger.info("🔄 Continuing without initial states (environment will still work)")
                # Continue without initial states - the environment will still function

            # self.current_task_name = task.name
            self.current_task_name = task.language

            print(f"🔄 Libero environment reset to task: {task.name}")
            return self._process_observation(obs)

        except Exception as e:
            logger.warning(f"❌ Failed to reset Libero environment: {e}")
            import traceback

            logger.warning(f"❌ Traceback: {traceback.format_exc()}")
            raise

    async def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in Libero environment."""
        try:
            if not self.env:
                raise RuntimeError("Environment not reset")

            # Convert action dict to libero format
            libero_action = self._convert_action_to_libero(action)

            # Execute step
            obs, reward, done, info = self.env.step(libero_action)

            return (self._process_observation(obs), float(reward), bool(done), dict(info))

        except Exception as e:
            logger.warning(f"❌ Failed to step in Libero environment: {e}")
            raise

    async def get_observation(self) -> Dict[str, Any]:
        """Get current observation from Libero environment."""
        try:
            if not self.env:
                raise RuntimeError("Environment not reset")

            # Get current observation
            obs = self.env._get_observations()
            return self._process_observation(obs)

        except Exception as e:
            logger.warning(f"❌ Failed to get observation: {e}")
            raise

    def get_robot_state_keys(self) -> List[str]:
        """Get robot state keys for Libero environment."""
        # Common Libero robot state keys
        return ["robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw Libero observation into standard format."""
        processed_obs = {}

        # Process robot state
        for key in self.get_robot_state_keys():
            if key in obs:
                processed_obs[key] = obs[key]

        # Process camera observations with 180-degree rotation
        # IMPORTANT: Rotate images 180 degrees to match Isaac-GR00T training preprocessing
        # Reference: Isaac-GR00T/examples/Libero/eval/utils.py get_libero_image()
        if "agentview_image" in obs:
            # Rotate 180 degrees: img[::-1, ::-1]
            rotated_img = obs["agentview_image"][::-1, ::-1]
            processed_obs["agentview_image"] = rotated_img
            processed_obs["front_camera"] = rotated_img

        if "robot0_eye_in_hand_image" in obs:
            # Rotate 180 degrees: wrist_img[::-1, ::-1]
            rotated_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
            processed_obs["robot0_eye_in_hand_image"] = rotated_wrist
            processed_obs["wrist_camera"] = rotated_wrist

        # Add other observations as needed
        for key, value in obs.items():
            if key not in processed_obs:
                processed_obs[key] = value

        return processed_obs

    def _convert_action_to_libero(self, action: Dict[str, Any]) -> np.ndarray:
        """Convert action dict to Libero action format.

        IMPORTANT: GR00T outputs 7-dim delta pose actions [dx,dy,dz,dr,dp,dy,gripper]
        Reference: Isaac-GR00T/examples/Libero/eval/run_libero_eval.py
        """
        try:
            # Check if action is already in the correct format from GR00T
            if "action" in action and isinstance(action["action"], (list, np.ndarray)):
                # GR00T format: {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}
                libero_action = np.array(action["action"], dtype=np.float32)

                # Ensure exactly 7 dimensions
                if len(libero_action) == 7:
                    return libero_action
                elif len(libero_action) > 7:
                    return libero_action[:7]
                else:
                    # Pad with zeros if needed
                    padded = np.zeros(7, dtype=np.float32)
                    padded[: len(libero_action)] = libero_action
                    return padded

            # Fallback: Try legacy format with joint positions
            elif "robot0_joint_pos" in action:
                joint_pos = action["robot0_joint_pos"]
                if len(joint_pos) >= 7:
                    libero_action = np.array(joint_pos[:7])
                else:
                    libero_action = np.zeros(7)
                    libero_action[: len(joint_pos)] = joint_pos
                return libero_action.astype(np.float32)

            # Final fallback: zero action
            else:
                logger.warning("⚠️ No recognized action format, using zero action")
                return np.zeros(7, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Failed to convert action: {e}")
            logger.error(f"   Action: {action}")
            # Return zero action as safe fallback
            return np.zeros(7, dtype=np.float32)

    async def cleanup(self):
        """Cleanup Libero environment."""
        try:
            if self.env:
                # Suppress EGL/OpenGL cleanup warnings that are harmless but noisy
                import os
                import warnings

                # Temporarily suppress OpenGL warnings during cleanup
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Set environment variable to suppress EGL warnings
                    old_egl_debug = os.environ.get("EGL_LOG_LEVEL", None)
                    os.environ["EGL_LOG_LEVEL"] = "fatal"

                    try:
                        self.env.close()
                    finally:
                        # Restore EGL debug level
                        if old_egl_debug is not None:
                            os.environ["EGL_LOG_LEVEL"] = old_egl_debug
                        elif "EGL_LOG_LEVEL" in os.environ:
                            del os.environ["EGL_LOG_LEVEL"]

                self.env = None
            logger.info("🧹 Libero environment cleaned up")
        except Exception as e:
            # Suppress EGL cleanup errors as they're harmless during shutdown
            if "EGL" in str(e) or "eglDestroy" in str(e):
                logger.debug(f"🔇 Suppressed harmless EGL cleanup warning: {e}")
            else:
                logger.error(f"❌ Cleanup error: {e}")


class MockLiberoEnvironment(LiberoEnvironment):
    """Mock Libero environment for testing without Libero dependencies.

    This class inherits from LiberoEnvironment but overrides key methods
    to provide mock data, allowing testing without installing Libero.
    """

    def __init__(self, task_suite: str = "libero_spatial", **kwargs):
        # Call parent's __init__ but override env_name
        SimulationEnvironment.__init__(self, f"mock_{task_suite}", **kwargs)
        self.task_suite = task_suite
        self.available_tasks = [
            "pick up the red block and put it in the drawer",
            "put the book on the bookshelf",
            "stack the blocks",
            "open the drawer",
            "close the microwave",
        ]
        self.current_task_idx = 0
        self.step_counter = 0

    async def initialize(self) -> bool:
        """Initialize mock environment without Libero dependencies."""
        logger.info(f"🎮 Mock {self.task_suite} initialized (no dependencies)")
        logger.info(f"📋 Available mock tasks: {len(self.available_tasks)}")
        logger.info(f"🎯 Mock tasks: {self.available_tasks[:3]}...")

        self.is_initialized = True

        return True

    async def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset mock environment with mock observations."""
        if task_name and task_name in self.available_tasks:
            selected_task = task_name
        else:
            selected_task = self.available_tasks[self.current_task_idx % len(self.available_tasks)]
            self.current_task_idx += 1

        self.step_counter = 0
        self.current_task_name = selected_task

        # Generate mock observation using parent's observation format
        mock_obs = {
            "robot0_joint_pos": np.random.uniform(-1, 1, 7),
            "robot0_joint_vel": np.zeros(7),
            "robot0_eef_pos": np.random.uniform(-0.5, 0.5, 3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0]),
            "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        }

        logger.info(f"🔄 Mock environment reset to task: {selected_task}")

        # Use parent's _process_observation to ensure consistent formatting
        return self._process_observation(mock_obs)

    async def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in mock environment."""
        self.step_counter += 1

        # Generate mock observation
        mock_obs = {
            "robot0_joint_pos": np.random.uniform(-1, 1, 7),
            "robot0_joint_vel": np.random.uniform(-0.1, 0.1, 7),
            "robot0_eef_pos": np.random.uniform(-0.5, 0.5, 3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([np.random.uniform(-1, 1)]),
            "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        }

        # Mock reward and done
        reward = np.random.uniform(0, 1)
        done = self.step_counter >= 50 or np.random.random() < 0.1  # Random episode termination

        # Mock success (random success for demonstration)
        success = done and np.random.random() < 0.3

        info = {"success": success, "step": self.step_counter, "task": "mock_task"}

        # Use parent's _process_observation for consistent formatting
        return self._process_observation(mock_obs), reward, done, info

    async def get_observation(self) -> Dict[str, Any]:
        """Get current mock observation."""
        mock_obs = {
            "robot0_joint_pos": np.random.uniform(-1, 1, 7),
            "robot0_joint_vel": np.zeros(7),
            "robot0_eef_pos": np.random.uniform(-0.5, 0.5, 3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0]),
            "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        }
        return self._process_observation(mock_obs)

    async def cleanup(self):
        """Cleanup mock environment."""
        logger.info("🧹 Mock Libero environment cleaned up")
