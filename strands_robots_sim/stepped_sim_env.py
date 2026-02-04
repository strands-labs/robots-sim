#!/usr/bin/env python3
"""
Stepped Simulated Environment Control for Agent-Driven Planning

This module provides a stepped execution interface where an agent acts as a "brain"
to plan and decompose complex instructions into simpler ones. The tool executes
a limited number of steps and returns observations (camera images + state) to the
agent for decision-making.

Key Features:
- Execute limited steps per call (e.g., 10 steps)
- Return camera images and state information to agent
- Agent decides whether to continue, change instruction, or stop
- Stateful execution across multiple tool calls
- Support for hierarchical planning and instruction decomposition
"""

import base64
import io
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from .envs import create_simulation_environment
from .policies import Policy, create_policy

logger = logging.getLogger(__name__)


class StepExecutionStatus(Enum):
    """Step execution status"""

    IDLE = "idle"
    RUNNING = "running"
    EPISODE_DONE = "episode_done"
    ERROR = "error"


@dataclass
class StepExecutionState:
    """State maintained across stepped executions"""

    status: StepExecutionStatus = StepExecutionStatus.IDLE
    current_instruction: str = ""
    current_episode: int = 0
    total_steps: int = 0
    episode_steps: int = 0
    cumulative_reward: float = 0.0
    episode_reward: float = 0.0
    success_count: int = 0
    last_observation: Optional[Dict[str, Any]] = None
    error_message: str = ""
    task_name: Optional[str] = None
    # Video recording state
    record_video: bool = False
    video_path: Optional[str] = None
    top_view_frames: List[np.ndarray] = None
    wrist_view_frames: List[np.ndarray] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.top_view_frames is None:
            self.top_view_frames = []
        if self.wrist_view_frames is None:
            self.wrist_view_frames = []


class SteppedSimEnv(AgentTool):
    """
    Stepped simulation environment for agent-driven planning.

    The agent acts as a "brain" that:
    1. Provides an instruction
    2. Tool executes N steps
    3. Returns camera images + state
    4. Agent analyzes and decides next action (continue, change instruction, reset, etc.)
    """

    def __init__(
        self,
        tool_name: str,
        env_type: str = "libero",
        task_suite: str = "libero_spatial",
        action_horizon: int = 8,
        steps_per_call: int = 10,
        max_steps_per_episode: int = 500,
        data_config: Any = None,
        **kwargs,
    ):
        """
        Initialize SteppedSimEnv.

        Args:
            tool_name: Name for this simulation tool
            env_type: Environment type ("libero", "robocasa", etc.)
            task_suite: Task suite name
            action_horizon: Actions per inference step
            steps_per_call: Number of steps to execute per tool call
            max_steps_per_episode: Maximum steps per episode
            data_config: Data configuration (for GR00T compatibility)
            **kwargs: Environment-specific parameters
        """
        super().__init__()

        self.tool_name_str = tool_name
        self.env_type = env_type
        self.task_suite = task_suite
        self.action_horizon = action_horizon
        self.steps_per_call = steps_per_call
        self.max_steps_per_episode = max_steps_per_episode
        self.data_config = data_config

        # Execution state
        self._state = StepExecutionState()

        # Policy instance (created on first use)
        self._policy: Optional[Policy] = None
        self._policy_config: Optional[Dict] = None

        # Initialize simulation environment
        self.sim_env = create_simulation_environment(env_type, task_suite=task_suite, **kwargs)

        logger.info(f"🎮 {tool_name} stepped simulation environment initialized")
        logger.info(f"🌍 Environment: {env_type} ({task_suite})")
        logger.info(f"👣 Steps per call: {steps_per_call}")

    async def _ensure_environment_initialized(self) -> bool:
        """Ensure simulation environment is initialized."""
        try:
            if not self.sim_env.is_initialized:

                success = await self.sim_env.initialize()
                if not success:
                    return False
                logger.info(f"✅ {self.sim_env.env_name} environment ready")
            return True
        except Exception as e:
            logger.error(f"❌ Environment initialization failed: {e}")
            return False

    async def _ensure_policy_initialized(
        self, policy_port: int, policy_host: str = "localhost", policy_provider: str = "groot"
    ) -> bool:
        """Ensure policy is initialized with given configuration."""
        try:
            # Check if we need to create/recreate policy
            new_config = {"port": policy_port, "host": policy_host, "provider": policy_provider}

            if self._policy is None or self._policy_config != new_config:
                # Create new policy
                policy_config = {"port": policy_port, "host": policy_host}
                if self.data_config:
                    policy_config["data_config"] = self.data_config

                self._policy = create_policy(policy_provider, **policy_config)
                self._policy_config = new_config

                # Set robot state keys
                robot_state_keys = self.sim_env.get_robot_state_keys()
                self._policy.set_robot_state_keys(robot_state_keys)

                logger.info(f"🧠 Policy initialized: {policy_provider} on {policy_host}:{policy_port}")

            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize policy: {e}")
            return False

    async def _reset_episode(
        self, task_name: Optional[str] = None, record_video: bool = False, video_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reset environment for new episode."""
        try:
            observation = await self.sim_env.reset(task_name)

            # Update state
            self._state.status = StepExecutionStatus.IDLE
            self._state.current_episode += 1
            self._state.episode_steps = 0
            self._state.episode_reward = 0.0
            self._state.last_observation = observation
            self._state.task_name = task_name

            # Initialize video recording state
            self._state.record_video = record_video
            self._state.video_path = video_path
            self._state.top_view_frames = []
            self._state.wrist_view_frames = []

            if record_video:
                logger.info(f"🎥 Video recording enabled for episode {self._state.current_episode}")

            logger.info(f"🔄 Episode {self._state.current_episode} reset (task: {task_name or 'random'})")

            return observation
        except Exception as e:
            logger.warning(f"❌ Episode reset failed: {e}")
            self._state.status = StepExecutionStatus.ERROR
            self._state.error_message = str(e)
            raise

    async def _execute_steps(self, instruction: str, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a limited number of steps with current instruction.

        Returns:
            Dict containing execution results and observations
        """
        if num_steps is None:
            num_steps = self.steps_per_call

        # Validate state
        if self._state.last_observation is None:
            raise ValueError("No active episode. Call reset_episode first.")

        if self._policy is None:
            raise ValueError("Policy not initialized. Provide policy_port.")

        # Update instruction if changed
        if instruction != self._state.current_instruction:
            logger.info(f"📝 Instruction updated: '{instruction}'")
            self._state.current_instruction = instruction

        # Execute steps
        self._state.status = StepExecutionStatus.RUNNING
        observation = self._state.last_observation
        step_results = []

        for step_idx in range(num_steps):
            # Check episode termination
            if self._state.episode_steps >= self.max_steps_per_episode:
                logger.info(f"⏱️ Max episode steps reached ({self.max_steps_per_episode})")
                self._state.status = StepExecutionStatus.EPISODE_DONE

                # Save video if recording was enabled
                if self._state.record_video and (self._state.top_view_frames or self._state.wrist_view_frames):
                    try:
                        video_path = self._save_rollout_video(
                            self._state.top_view_frames,
                            self._state.wrist_view_frames,
                            self._state.current_episode,
                            False,  # Not successful - timed out
                            self._state.current_instruction or self._state.task_name or "task",
                            self._state.video_path,
                        )
                        logger.info(f"🎥 Episode video saved to: {video_path}")
                    except Exception as video_error:
                        logger.error(f"❌ Failed to save episode video: {video_error}")

                break

            try:
                # Get actions from policy
                robot_actions = await self._policy.get_actions(observation, instruction)

                # Execute action chunk
                for action_dict in robot_actions[: self.action_horizon]:
                    # Execute step
                    observation, reward, done, info = await self.sim_env.step(action_dict)

                    # Update state
                    self._state.episode_steps += 1
                    self._state.total_steps += 1
                    self._state.episode_reward += reward
                    self._state.cumulative_reward += reward
                    self._state.last_observation = observation

                    # Record video frames if enabled
                    if self._state.record_video:
                        top_frame, wrist_frame = self._capture_video_frames(observation)
                        if top_frame is not None:
                            self._state.top_view_frames.append(top_frame)
                        if wrist_frame is not None:
                            self._state.wrist_view_frames.append(wrist_frame)

                    # Record step result
                    step_results.append(
                        {"step": self._state.episode_steps, "reward": reward, "done": done, "info": info}
                    )

                    # Check if episode done
                    if done:
                        if info.get("success", False):
                            self._state.success_count += 1
                        self._state.status = StepExecutionStatus.EPISODE_DONE
                        logger.info(
                            f"🏁 Episode {self._state.current_episode} completed: "
                            f"{'✅ Success' if info.get('success') else '❌ Failed'} "
                            f"(steps: {self._state.episode_steps}, reward: {self._state.episode_reward:.2f})"
                        )

                        # Save video if recording was enabled
                        if self._state.record_video and (self._state.top_view_frames or self._state.wrist_view_frames):
                            try:
                                video_path = self._save_rollout_video(
                                    self._state.top_view_frames,
                                    self._state.wrist_view_frames,
                                    self._state.current_episode,
                                    done,
                                    self._state.current_instruction or self._state.task_name or "task",
                                    self._state.video_path,
                                )
                                logger.info(f"🎥 Episode video saved to: {video_path}")
                            except Exception as video_error:
                                logger.error(f"❌ Failed to save episode video: {video_error}")

                        break

                if done:
                    break

            except Exception as e:
                logger.warning(f"❌ Step execution error: {e}")
                self._state.status = StepExecutionStatus.ERROR
                self._state.error_message = str(e)
                raise

        # Prepare result
        result = {
            "steps_executed": len(step_results),
            "step_results": step_results,
            "final_observation": observation,
            "episode_done": self._state.status == StepExecutionStatus.EPISODE_DONE,
        }

        return result

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        try:
            from PIL import Image

            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Convert to PIL Image
            pil_img = Image.fromarray(image)

            # Encode to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"❌ Image encoding failed: {e}")
            return ""

    def _extract_camera_images(self, observation: Dict[str, Any]) -> Dict[str, str]:
        """Extract and encode camera images from observation."""
        images = {}

        # Camera key mappings
        camera_mappings = {
            "front_camera": ["front_camera", "agentview_image", "video.webcam", "webcam"],
            "wrist_camera": ["wrist_camera", "robot0_eye_in_hand_image"],
        }

        for camera_name, keys in camera_mappings.items():
            for key in keys:
                if key in observation:
                    frame = observation[key]
                    if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                        # Process frame
                        if len(frame.shape) == 4:
                            frame = frame[0]  # Remove batch dim
                        elif len(frame.shape) == 2:
                            frame = np.stack([frame] * 3, axis=-1)  # Grayscale to RGB

                        # Encode
                        encoded = self._encode_image(frame)
                        if encoded:
                            images[camera_name] = encoded
                            break

        return images

    def _extract_camera_images_as_bytes(self, observation: Dict[str, Any]) -> Dict[str, bytes]:
        """Extract camera images from observation and return as bytes for ToolResultEvent.

        Returns:
            Dict mapping camera names to PNG image bytes
        """
        images = {}

        try:
            from PIL import Image

            # Camera key mappings
            camera_mappings = {
                "front_camera": ["front_camera", "agentview_image", "video.webcam", "webcam"],
                "wrist_camera": ["wrist_camera", "robot0_eye_in_hand_image"],
            }

            for camera_name, keys in camera_mappings.items():
                for key in keys:
                    if key in observation:
                        frame = observation[key]
                        if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                            # Process frame
                            processed_frame = self._process_frame(frame)

                            # Convert to PIL Image
                            pil_img = Image.fromarray(processed_frame)

                            # Encode to PNG bytes
                            buffer = io.BytesIO()
                            pil_img.save(buffer, format="PNG")
                            img_bytes = buffer.getvalue()

                            images[camera_name] = img_bytes
                            break

            return images

        except Exception as e:
            logger.error(f"❌ Failed to extract camera images as bytes: {e}")
            return {}

    def _capture_video_frames(self, observation: Dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture both top view and wrist view frames from observation.

        Returns:
            Tuple of (top_view_frame, wrist_view_frame) as numpy arrays
        """
        try:
            # Top view camera keys (front/agent view)
            top_keys = ["front_camera", "agentview_image", "video.webcam", "webcam"]
            # Wrist view camera keys
            wrist_keys = ["wrist_camera", "robot0_eye_in_hand_image"]

            top_frame = None
            wrist_frame = None

            # Find top view
            for key in top_keys:
                if key in observation:
                    frame = observation[key]
                    if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                        top_frame = self._process_frame(frame)
                        break

            # Find wrist view
            for key in wrist_keys:
                if key in observation:
                    frame = observation[key]
                    if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                        wrist_frame = self._process_frame(frame)
                        break

            return top_frame, wrist_frame

        except Exception as e:
            logger.debug(f"⚠️ Error capturing video frames: {e}")
            return None, None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame to ensure correct format."""
        # Remove batch dimension if present
        if len(frame.shape) == 4:
            frame = frame[0]
        # Convert grayscale to RGB
        elif len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        return frame

    def _save_rollout_video(
        self,
        top_view: List[np.ndarray],
        wrist_view: List[np.ndarray],
        episode_idx: int,
        success: bool,
        task_description: str,
        base_path: Optional[str] = None,
    ) -> str:
        """Save rollout video with side-by-side views.

        Similar to reference: save_rollout_video in Isaac-GR00T examples.
        """
        try:
            import imageio

            # Create rollout directory
            DATE = time.strftime("%Y_%m_%d")
            DATE_TIME = time.strftime("%Y_%m_%d_%H_%M_%S")
            rollout_dir = f"./rollouts/{DATE}"
            os.makedirs(rollout_dir, exist_ok=True)

            # Process task description for filename
            processed_task = self._state.task_name.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]

            # Generate filename (always in rollouts folder)
            mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task}.mp4"

            # Create video writer
            video_writer = imageio.get_writer(mp4_path, fps=30)

            # Combine views side-by-side
            for i in range(max(len(top_view), len(wrist_view))):
                # Get frames (use last frame if one list is shorter)
                img1 = top_view[min(i, len(top_view) - 1)] if top_view else None
                img2 = wrist_view[min(i, len(wrist_view) - 1)] if wrist_view else None

                if img1 is not None and img2 is not None:
                    # Side-by-side concatenation
                    combined = np.hstack((img1, img2))
                elif img1 is not None:
                    # Only top view available
                    combined = img1
                elif img2 is not None:
                    # Only wrist view available
                    combined = img2
                else:
                    continue

                video_writer.append_data(combined)

            video_writer.close()
            logger.info(f"🎥 Saved rollout MP4 at path {mp4_path}")
            return mp4_path

        except Exception as e:
            logger.warning(f"❌ Failed to save rollout video: {e}")
            raise

    def _format_state_text(self) -> str:
        """Format current state as text for agent."""
        ep = self._state.current_episode
        success_pct = (self._state.success_count / ep * 100) if ep > 0 else 0
        text = f"""## Simulation State

**Environment**: {self.env_type} ({self.task_suite})
**Task**: {self._state.task_name or 'random'}

**Current Instruction**: {self._state.current_instruction}

**Episode Progress**:
- Episode: {self._state.current_episode}
- Episode Steps: {self._state.episode_steps} / {self.max_steps_per_episode}
- Episode Reward: {self._state.episode_reward:.3f}
- Status: {self._state.status.value}

**Overall Progress**:
- Total Steps: {self._state.total_steps}
- Total Reward: {self._state.cumulative_reward:.3f}
- Success Count: {self._state.success_count}
- Success Rate: {success_pct:.1f}%
"""

        if self._state.error_message:
            text += f"\n**Error**: {self._state.error_message}\n"

        return text

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "stepped_sim_env"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification."""
        return {
            "name": self.tool_name_str,
            "description": (
                f"Stepped simulation environment control for agent-driven planning ({self.env_type}). "
                f"Execute {self.steps_per_call} steps and return camera images + state for "
                f"agent decision-making. "
                f"Actions: execute_steps (run N steps), reset_episode (start new episode), "
                f"get_state (get current state). "
                f"Agent acts as 'brain' to decompose complex tasks into simpler instructions."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                            "enum": ["execute_steps", "reset_episode", "get_state"],
                            "default": "execute_steps",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Current instruction for the robot (required for execute_steps)",
                        },
                        "num_steps": {
                            "type": "integer",
                            "description": f"Number of steps to execute (default: {self.steps_per_call})",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute_steps)",
                            "default": 8000,
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider (groot, openai, etc.)",
                            "default": "groot",
                        },
                        "task_name": {
                            "type": "string",
                            "description": "Specific task name for reset_episode (optional, random if not specified)",
                        },
                        "record_video": {
                            "type": "boolean",
                            "description": "Whether to record video of the episode (default: false)",
                            "default": False,
                        },
                        "video_path": {
                            "type": "string",
                            "description": "Path to save recorded videos (optional, defaults to ./rollouts/)",
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream stepped simulation execution."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            action = input_data.get("action", "execute_steps")

            # Ensure environment is initialized
            if not await self._ensure_environment_initialized():
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": "❌ Failed to initialize environment"}],
                    }
                )
                return

            if action == "execute_steps":
                # Execute steps with instruction
                instruction = input_data.get("instruction", "")
                num_steps = input_data.get("num_steps")
                policy_port = input_data.get("policy_port", 8000)
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")

                if not instruction:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": "❌ instruction is required for execute_steps action"}],
                        }
                    )
                    return

                # Ensure policy is initialized
                if not await self._ensure_policy_initialized(policy_port, policy_host, policy_provider):
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": "❌ Failed to initialize policy"}],
                        }
                    )
                    return

                # Check if we need to reset (first call or episode done)
                if self._state.last_observation is None or self._state.status == StepExecutionStatus.EPISODE_DONE:
                    record_video = input_data.get("record_video", True)
                    video_path = input_data.get("video_path")
                    await self._reset_episode(input_data.get("task_name"), record_video, video_path)

                # Execute steps
                result = await self._execute_steps(instruction, num_steps)

                # Format state text
                state_text = self._format_state_text()

                # Build content with text and images
                content = [{"text": state_text}]

                # Extract and add camera images
                images = self._extract_camera_images_as_bytes(result["final_observation"])
                for camera_name, img_bytes in images.items():
                    content.append(
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": img_bytes},
                            }
                        }
                    )

                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": content,
                    }
                )

            elif action == "reset_episode":
                # Reset episode
                task_name = input_data.get("task_name")
                record_video = input_data.get("record_video", True)
                video_path = input_data.get("video_path")
                await self._reset_episode(task_name, record_video, video_path)

                # Extract images from initial observation
                images = self._extract_camera_images(self._state.last_observation)
                state_text = self._format_state_text()

                # Add image availability info
                if images:
                    state_text += f"\n**Camera Images Available**: {', '.join(images.keys())}\n"
                    state_text += f"(Initial observation captured with {len(images)} camera view(s))\n"

                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": state_text}],
                    }
                )

            elif action == "get_state":
                # Get current state
                state_text = self._format_state_text()

                # Add images if we have observation
                if self._state.last_observation:
                    images = self._extract_camera_images(self._state.last_observation)
                    if images:
                        state_text += f"\n**Camera Images Available**: {', '.join(images.keys())}\n"

                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": state_text}],
                    }
                )

            else:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"❌ Unknown action: {action}"}],
                    }
                )

        except Exception as e:
            logger.error(f"❌ {self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"❌ {self.tool_name_str} error: {str(e)}"}],
                }
            )

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.sim_env:
                await self.sim_env.cleanup()
            logger.info(f"🧹 {self.tool_name_str} cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

    def __del__(self):
        """Destructor."""
        try:
            if hasattr(self, "sim_env") and self.sim_env:
                import asyncio

                try:
                    asyncio.get_event_loop().create_task(self.cleanup())
                except RuntimeError:  # nosec B110
                    # Event loop not available during shutdown
                    pass
        except Exception:  # nosec B110
            # Ignore all errors in destructor to prevent exceptions during shutdown
            pass
