#!/usr/bin/env python3
"""
Universal Simulated Environment Control with Policy Abstraction

This module provides a clean interface for controlling simulated robots in
environments like Libero, RoboCasa, etc. using the same policy abstraction
as the Robot class.

Features:
- Async simulation task execution with real-time status reporting
- Non-blocking operations - simulation runs while tool returns status
- Stop functionality to interrupt running tasks
- Environment state management with proper error handling
- Policy abstraction for any VLA provider (same as Robot class)
"""

import asyncio
import logging
import os
import socket
import sys
import threading
import time

# Suppress GLFW and OpenGL warnings early
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Suppress all EGL/OpenGL related warnings
warnings.filterwarnings("ignore", message=".*GLFW.*")
warnings.filterwarnings("ignore", message=".*OpenGL.*")
warnings.filterwarnings("ignore", message=".*X11.*")
warnings.filterwarnings("ignore", message=".*EGL.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*egl.*")

# Set headless environment variables early
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# Suppress EGL errors in stderr (these are harmless during cleanup)
class SuppressEGLErrors:
    """Context manager to suppress EGL error messages during cleanup."""

    def __enter__(self):
        self.original_stderr = sys.stderr
        self.devnull_file = open(os.devnull, "w", encoding="utf-8")  # Closed in __exit__
        sys.stderr = self.devnull_file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        self.devnull_file.close()
        return False


# Install a custom exception hook to suppress EGL errors during interpreter shutdown
_original_excepthook = sys.excepthook


def _suppress_egl_excepthook(exc_type, exc_value, exc_traceback):
    """Suppress EGL-related errors during cleanup."""
    if exc_type.__name__ == "EGLError" or "EGL_NOT_INITIALIZED" in str(exc_value):
        # Silently ignore EGL cleanup errors - they're harmless
        return
    # Call original handler for other exceptions
    _original_excepthook(exc_type, exc_value, exc_traceback)


sys.excepthook = _suppress_egl_excepthook

import numpy as np  # noqa: E402
from strands.tools.tools import AgentTool  # noqa: E402
from strands.types._events import ToolResultEvent  # noqa: E402
from strands.types.tools import ToolSpec, ToolUse  # noqa: E402

from .envs import create_simulation_environment  # noqa: E402
from .policies import Policy, create_policy  # noqa: E402

logger = logging.getLogger(__name__)


# Monkey-patch sys.stderr to suppress EGL errors during cleanup
class EGLErrorFilter:
    """Filter that suppresses EGL_NOT_INITIALIZED errors from stderr."""

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self._suppress = False
        self._in_egl_traceback = False

    def write(self, text):
        # Comprehensive list of EGL/OpenGL error patterns to suppress
        error_patterns = [
            "EGL_NOT_INITIALIZED",
            "EGLError",
            "eglDestroy",
            "eglMakeCurrent",
            "Exception ignored in:",
            "MjRenderContext.__del__",
            "EGLGLContext.__del__",
            "OpenGL.raw.EGL._errors",
            "binding_utils.py",
            "egl_context.py",
            "OpenGL_accelerate",
            "glCheckError",
        ]

        # Check if this line contains any error patterns
        if any(pattern in text for pattern in error_patterns):
            self._in_egl_traceback = True
            return  # Suppress this line

        # Check if we're in an EGL traceback
        if self._in_egl_traceback:
            # If line is part of traceback (starts with spaces or File), suppress it
            if text.strip().startswith(("File", "at 0x", "Traceback", "result =")) or not text.strip():
                return  # Suppress traceback continuation
            else:
                # End of traceback, reset flag
                self._in_egl_traceback = False

        # Write non-EGL content
        try:
            self.original_stderr.write(text)
        except (OSError, ValueError, AttributeError):
            pass  # Ignore errors during shutdown

    def flush(self):
        try:
            self.original_stderr.flush()
        except (OSError, ValueError, AttributeError):
            pass  # Ignore errors during shutdown

    def fileno(self):
        try:
            return self.original_stderr.fileno()
        except:  # noqa: E722
            return -1


# Install the stderr filter globally
sys.stderr = EGLErrorFilter(sys.stderr)

# Add atexit handler to completely suppress stderr during final cleanup
import atexit  # noqa: E402

# Keep reference to devnull file to ensure proper cleanup
_devnull_file = None


def _suppress_stderr_on_exit():
    """Suppress all stderr output during final Python shutdown."""
    global _devnull_file
    try:
        # Replace stderr with devnull during final cleanup
        _devnull_file = open(os.devnull, "w", encoding="utf-8")  # Intentionally left open until Python shutdown
        sys.stderr = _devnull_file
    except (OSError, AttributeError, RuntimeError):
        pass  # Ignore if this fails during Python shutdown


# Register the atexit handler
atexit.register(_suppress_stderr_on_exit)


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


class SimTaskStatus(Enum):
    """Simulation task execution status"""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SimTaskState:
    """Simulation task execution state"""

    status: SimTaskStatus = SimTaskStatus.IDLE
    instruction: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    episode_count: int = 0
    success_count: int = 0
    error_message: str = ""
    task_future: Optional[Future] = None
    current_task: str = ""


class SimEnv(AgentTool):
    """Universal simulated environment control with async task execution."""

    def __init__(
        self,
        tool_name: str,
        env_type: str = "libero",
        task_suite: str = "libero_spatial",
        action_horizon: int = 8,
        data_config: Union[str, Any, None] = None,
        **kwargs,
    ):
        """Initialize SimEnv with async capabilities.

        Args:
            tool_name: Name for this simulation tool
            env_type: Environment type ("libero", "robocasa", etc.)
            task_suite: Task suite name (e.g., "libero_spatial", "libero_goal")
            action_horizon: Actions per inference step
            data_config: Data configuration (for GR00T compatibility)
            **kwargs: Environment-specific parameters
        """
        super().__init__()

        self.tool_name_str = tool_name
        self.env_type = env_type
        self.task_suite = task_suite
        self.action_horizon = action_horizon
        self.data_config = data_config

        # Task execution state
        self._task_state = SimTaskState()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{tool_name}_executor")
        self._shutdown_event = threading.Event()

        # Initialize simulation environment
        self.sim_env = create_simulation_environment(env_type, task_suite=task_suite, **kwargs)

        logger.info(f"🎮 {tool_name} simulation environment initialized")
        logger.info(f"🌍 Environment: {env_type} ({task_suite})")

        if data_config:
            logger.info(f"⚙️ Data config: {data_config}")

    async def _get_policy(
        self, policy_port: Optional[int] = None, policy_host: str = "localhost", policy_provider: str = "groot"
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters."""

        if not policy_port:
            raise ValueError("policy_port is required for simulation operation")

        policy_config = {"port": policy_port, "host": policy_host}

        if self.data_config:
            policy_config["data_config"] = self.data_config

        return create_policy(policy_provider, **policy_config)

    async def _initialize_environment(self) -> bool:
        """Initialize simulation environment."""
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

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with environment state keys."""
        try:
            # Get robot state keys from environment
            robot_state_keys = self.sim_env.get_robot_state_keys()

            # Set robot state keys in policy
            policy.set_robot_state_keys(robot_state_keys)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize policy: {e}")
            return False

    async def _execute_task_async(
        self,
        instruction: str,
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        max_episodes: int = 10,
        max_steps_per_episode: int = 500,
        task_name: Optional[str] = None,
        record_video: bool = False,
        video_path: Optional[str] = None,
    ) -> None:
        """Execute simulation task in background (internal method)."""
        try:
            # Update task state
            self._task_state.status = SimTaskStatus.INITIALIZING
            self._task_state.instruction = instruction
            self._task_state.start_time = time.time()
            self._task_state.step_count = 0
            self._task_state.episode_count = 0
            self._task_state.success_count = 0
            self._task_state.error_message = ""
            self._task_state.current_task = task_name or "random"

            # Initialize environment
            initialized = await self._initialize_environment()
            if not initialized:
                self._task_state.status = SimTaskStatus.ERROR
                self._task_state.error_message = f"Failed to initialize {self.env_type} environment"
                return

            # Get policy instance
            policy_instance = await self._get_policy(policy_port, policy_host, policy_provider)

            # Initialize policy with environment state keys
            if not await self._initialize_policy(policy_instance):
                self._task_state.status = SimTaskStatus.ERROR
                self._task_state.error_message = "Failed to initialize policy"
                return

            logger.info(f"🎯 Starting simulation: '{instruction}' on {self.tool_name_str}")
            logger.info(f"🧠 Using policy: {policy_provider} on {policy_host}:{policy_port}")
            logger.info(f"🎮 Max episodes: {max_episodes}, max steps per episode: {max_steps_per_episode}")

            self._task_state.status = SimTaskStatus.RUNNING

            # Initialize video recording per episode
            if record_video:
                logger.info("🎥 Video recording enabled")

            # Run episodes
            for episode in range(max_episodes):
                # Initialize per-episode video recording
                top_view_frames = []
                wrist_view_frames = []
                if self._task_state.status != SimTaskStatus.RUNNING or self._shutdown_event.is_set():
                    break

                # Reset environment for new episode
                observation = await self.sim_env.reset(task_name)
                episode_reward = 0.0
                episode_steps = 0
                episode_done = False  # Track episode termination

                logger.info(f"🎬 Episode {episode + 1}/{max_episodes} started")

                # Run episode
                for step in range(max_steps_per_episode):
                    if self._task_state.status != SimTaskStatus.RUNNING or episode_done:
                        break

                    # Record video frames if enabled (capture both views)
                    if record_video:
                        top_frame, wrist_frame = self._capture_video_frames(observation)
                        if top_frame is not None:
                            top_view_frames.append(top_frame)
                        if wrist_frame is not None:
                            wrist_view_frames.append(wrist_frame)

                    # Get actions from policy
                    robot_actions = await policy_instance.get_actions(observation, instruction)

                    # Execute actions from chunk
                    for action_dict in robot_actions[: self.action_horizon]:
                        if self._task_state.status != SimTaskStatus.RUNNING or episode_done:
                            break

                        # Execute step only if episode is not done
                        try:
                            observation, reward, done, info = await self.sim_env.step(action_dict)
                            episode_reward += reward
                            episode_steps += 1
                            self._task_state.step_count += 1

                            # Record frames after step if enabled
                            if record_video:
                                top_frame, wrist_frame = self._capture_video_frames(observation)
                                if top_frame is not None:
                                    top_view_frames.append(top_frame)
                                if wrist_frame is not None:
                                    wrist_view_frames.append(wrist_frame)

                            if done:
                                episode_done = True  # Mark episode as done
                                break

                        except Exception as step_error:
                            # Handle step errors (like executing action in terminated episode)
                            if "terminated episode" in str(step_error).lower():
                                logger.warning(f"⚠️ Episode already terminated, ending episode {episode + 1}")
                                episode_done = True
                                break
                            else:
                                # Re-raise other errors
                                raise step_error

                    if episode_done:
                        break

                    await asyncio.sleep(0.01)

                # Check episode success
                episode_success = True if done else False
                if episode_success:
                    self._task_state.success_count += 1

                self._task_state.episode_count += 1

                print(
                    f"🏁 Episode {episode + 1} completed: "
                    f"{'✅ Success' if episode_success else '❌ Failed'} "
                    f"(reward: {episode_reward:.2f}, steps: {episode_steps})"
                )

                # Save video for this episode if recording was enabled
                if record_video and (top_view_frames or wrist_view_frames):
                    try:
                        saved_path = self._save_rollout_video(
                            top_view_frames, wrist_view_frames, episode + 1, episode_success, instruction, video_path
                        )
                        logger.info(f"🎥 Episode video saved to: {saved_path}")
                    except Exception as video_error:
                        logger.error(f"❌ Failed to save episode video: {video_error}")

                if self._task_state.status != SimTaskStatus.RUNNING:
                    break

            # Update final state
            elapsed = time.time() - self._task_state.start_time
            self._task_state.duration = elapsed

            if self._task_state.status == SimTaskStatus.RUNNING:
                self._task_state.status = SimTaskStatus.COMPLETED
                success_rate = (
                    (self._task_state.success_count / self._task_state.episode_count * 100)
                    if self._task_state.episode_count > 0
                    else 0
                )
                logger.info(
                    f"✅ Simulation completed: '{instruction}' in {elapsed:.1f}s\n"
                    f"📊 Episodes: {self._task_state.episode_count}, Success rate: {success_rate:.1f}%\n"
                    f"🎯 Total steps: {self._task_state.step_count}"
                    + ("\n🎥 Videos saved per episode" if record_video else "")
                )

        except Exception as e:
            logger.error(f"❌ Simulation execution failed: {e}")
            self._task_state.status = SimTaskStatus.ERROR
            self._task_state.error_message = str(e)

    def _execute_task_sync(
        self,
        instruction: str,
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        max_episodes: int = 10,
        max_steps_per_episode: int = 500,
        task_name: Optional[str] = None,
        record_video: bool = False,
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute task synchronously in thread."""

        async def task_runner():
            await self._execute_task_async(
                instruction,
                policy_port,
                policy_host,
                policy_provider,
                max_episodes,
                max_steps_per_episode,
                task_name,
                record_video,
                video_path,
            )

        # Run task without creating new event loop
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as exec:
                future = exec.submit(lambda: asyncio.run(task_runner()))
                future.result()
        except RuntimeError:
            asyncio.run(task_runner())

        # Return final status
        success_rate = (
            (self._task_state.success_count / self._task_state.episode_count * 100)
            if self._task_state.episode_count > 0
            else 0
        )

        return {
            "status": "success" if self._task_state.status == SimTaskStatus.COMPLETED else "error",
            "content": [
                {
                    "text": f"✅ Simulation: '{instruction}' - {self._task_state.status.value}\n"
                    f"🎮 Environment: {self.env_type} ({self.task_suite})\n"
                    f"🧠 Policy: {policy_provider} on {policy_host}:{policy_port}\n"
                    f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
                    f"📊 Episodes: {self._task_state.episode_count}, Success rate: {success_rate:.1f}%\n"
                    f"🎯 Total steps: {self._task_state.step_count}"
                    + (f"\n❌ Error: {self._task_state.error_message}" if self._task_state.error_message else "")
                }
            ],
        }

    def start_task(
        self,
        instruction: str,
        policy_port: Optional[int] = 8000,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        max_episodes: int = 1,
        max_steps_per_episode: int = 500,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start simulation task asynchronously and return immediately."""

        if self._task_state.status == SimTaskStatus.RUNNING:
            return {
                "status": "error",
                "content": [{"text": f"❌ Task already running: {self._task_state.instruction}"}],
            }

        # Start task in background
        self._task_state.task_future = self._executor.submit(
            self._execute_task_sync,
            instruction,
            policy_port,
            policy_host,
            policy_provider,
            max_episodes,
            max_steps_per_episode,
            task_name,
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🚀 Simulation started: '{instruction}'\n"
                    f"🎮 Environment: {self.env_type} ({self.task_suite})\n"
                    f"🎯 Task: {task_name or 'random'}\n"
                    f"💡 Use action='status' to check progress\n"
                    f"💡 Use action='stop' to interrupt"
                }
            ],
        }

    def get_task_status(self) -> Dict[str, Any]:
        """Get current task execution status."""

        # Update duration for running tasks
        if self._task_state.status == SimTaskStatus.RUNNING:
            self._task_state.duration = time.time() - self._task_state.start_time

        status_text = f"📊 Simulation Status: {self._task_state.status.value.upper()}\n"

        if self._task_state.instruction:
            status_text += f"🎯 Task: {self._task_state.instruction}\n"
            status_text += f"🎮 Environment: {self.env_type} ({self.task_suite})\n"
            status_text += f"🎪 Current task: {self._task_state.current_task}\n"

        if self._task_state.status == SimTaskStatus.RUNNING:
            status_text += f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"📊 Episodes: {self._task_state.episode_count}\n"
            status_text += f"🔄 Steps: {self._task_state.step_count}\n"
            if self._task_state.episode_count > 0:
                success_rate = self._task_state.success_count / self._task_state.episode_count * 100
                status_text += f"✅ Success rate: {success_rate:.1f}%\n"
        elif self._task_state.status in [SimTaskStatus.COMPLETED, SimTaskStatus.STOPPED, SimTaskStatus.ERROR]:
            status_text += f"⏱️ Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"📊 Total Episodes: {self._task_state.episode_count}\n"
            status_text += f"🎯 Total Steps: {self._task_state.step_count}\n"
            if self._task_state.episode_count > 0:
                success_rate = self._task_state.success_count / self._task_state.episode_count * 100
                status_text += f"✅ Final Success rate: {success_rate:.1f}%\n"

        if self._task_state.error_message:
            status_text += f"❌ Error: {self._task_state.error_message}\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    def stop_task(self) -> Dict[str, Any]:
        """Stop currently running task."""

        if self._task_state.status != SimTaskStatus.RUNNING:
            return {
                "status": "success",
                "content": [{"text": f"💤 No task running to stop (current: {self._task_state.status.value})"}],
            }

        # Signal task to stop
        self._task_state.status = SimTaskStatus.STOPPED

        # Cancel future if it exists
        if self._task_state.task_future:
            self._task_state.task_future.cancel()

        logger.info(f"🛑 Simulation stopped: {self._task_state.instruction}")

        success_rate = (
            (self._task_state.success_count / self._task_state.episode_count * 100)
            if self._task_state.episode_count > 0
            else 0
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🛑 Simulation stopped: '{self._task_state.instruction}'\n"
                    f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
                    f"📊 Episodes completed: {self._task_state.episode_count}\n"
                    f"✅ Success rate: {success_rate:.1f}%\n"
                    f"🎯 Steps completed: {self._task_state.step_count}"
                }
            ],
        }

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "sim_env"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification with async actions."""
        return {
            "name": self.tool_name_str,
            "description": f"Universal simulated environment control with async task execution ({self.env_type}). "
            f"Actions: execute (blocking), start (async), status, stop. "
            f"For execute/start actions: instruction and policy_port are required. "
            f"For status/stop actions: no additional parameters needed.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform: execute (blocking), start (async), status, stop",
                            "enum": ["execute", "start", "status", "stop"],
                            "default": "execute",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start actions)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute/start actions)",
                            "default": 8000,
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider (groot, openai, etc.)",
                            "default": "groot",
                        },
                        "max_episodes": {
                            "type": "integer",
                            "description": "Maximum number of episodes to run",
                            "default": 1,
                        },
                        "max_steps_per_episode": {
                            "type": "integer",
                            "description": "Maximum steps per episode",
                            "default": 500,
                        },
                        "task_name": {
                            "type": "string",
                            "description": "Specific task name (optional, random if not specified)",
                        },
                        "record_video": {
                            "type": "boolean",
                            "description": "Whether to record video of the simulation",
                            "default": True,
                        },
                        "video_path": {
                            "type": "string",
                            "description": "Path to save the recorded video (optional, defaults to ./rollout/)",
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream simulation task execution with async actions."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})

            action = input_data.get("action", "execute")

            # Handle different actions
            if action == "execute":
                # Blocking execution (legacy behavior)
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port", 8000)
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                max_episodes = input_data.get("max_episodes", 1)
                max_steps_per_episode = input_data.get("max_steps_per_episode", 500)
                task_name = input_data.get("task_name")

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": "❌ instruction and policy_port are required for execute action"}],
                        }
                    )
                    return

                # Get video recording parameters
                record_video = input_data.get("record_video", True)
                video_path = input_data.get("video_path")

                # Execute task synchronously
                task_result = self._execute_task_sync(
                    instruction,
                    policy_port,
                    policy_host,
                    policy_provider,
                    max_episodes,
                    max_steps_per_episode,
                    task_name,
                    record_video,
                    video_path,
                )
                result = {"toolUseId": tool_use_id, **task_result}
                yield ToolResultEvent(result)

            elif action == "start":
                # Asynchronous execution start
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port", 8000)
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                max_episodes = input_data.get("max_episodes", 1)
                max_steps_per_episode = input_data.get("max_steps_per_episode", 500)
                task_name = input_data.get("task_name")

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": "❌ instruction and policy_port are required for start action"}],
                        }
                    )
                    return

                # Start task asynchronously
                start_result = self.start_task(
                    instruction,
                    policy_port,
                    policy_host,
                    policy_provider,
                    max_episodes,
                    max_steps_per_episode,
                    task_name,
                )
                result = {"toolUseId": tool_use_id, **start_result}
                yield ToolResultEvent(result)

            elif action == "status":
                # Get current task status
                status_result = self.get_task_status()
                result = {"toolUseId": tool_use_id, **status_result}
                yield ToolResultEvent(result)

            elif action == "stop":
                # Stop current task
                stop_result = self.stop_task()
                result = {"toolUseId": tool_use_id, **stop_result}
                yield ToolResultEvent(result)

            else:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [
                            {"text": f"❌ Unknown action: {action}. Valid actions: execute, start, status, stop"}
                        ],
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

    def cleanup(self):
        """Cleanup resources and stop any running tasks."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop any running task
            if self._task_state.status == SimTaskStatus.RUNNING:
                self.stop_task()

            # Cleanup environment
            if self.sim_env:
                try:
                    asyncio.run(self.sim_env.cleanup())
                except RuntimeError:
                    # During shutdown, event loop may not be available
                    pass

            # Shutdown executor
            self._executor.shutdown(wait=True)

            logger.info(f"🧹 {self.tool_name_str} cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error for {self.tool_name_str}: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:  # noqa: E722  # nosec B110 # Bare except acceptable in destructor
            pass  # Ignore errors in destructor

        # Handle async cleanup for simulation environment
        if hasattr(self, "sim_env") and self.sim_env:
            try:
                # Try to cleanup sim_env if it has cleanup method
                if hasattr(self.sim_env, "cleanup"):
                    import asyncio

                    try:
                        # Try to run cleanup if event loop exists
                        asyncio.get_event_loop().create_task(self.sim_env.cleanup())
                    except RuntimeError:  # nosec B110 # Specific exception type for event loop check
                        # Event loop not available during shutdown - ignore
                        pass
            except Exception:  # nosec B110 # Broad exception acceptable in destructor
                pass  # Ignore errors in destructor - can't propagate from __del__

    async def get_status(self) -> Dict[str, Any]:
        """Get simulation environment status including initialization and task state."""
        try:
            # Build status dict
            status_data = {
                "sim_env_name": self.tool_name_str,
                "env_type": self.env_type,
                "task_suite": self.task_suite,
                "data_config": self.data_config,
                "is_initialized": self.sim_env.is_initialized,
                "task_status": self._task_state.status.value,
                "current_instruction": self._task_state.instruction,
                "current_task": self._task_state.current_task,
                "task_duration": self._task_state.duration,
                "episode_count": self._task_state.episode_count,
                "success_count": self._task_state.success_count,
                "task_steps": self._task_state.step_count,
            }

            # Add success rate if episodes have been run
            if self._task_state.episode_count > 0:
                success_rate = self._task_state.success_count / self._task_state.episode_count * 100
                status_data["success_rate"] = success_rate

            # Add error info if present
            if self._task_state.error_message:
                status_data["task_error"] = self._task_state.error_message

            # Add available tasks if environment is initialized
            if self.sim_env.is_initialized and hasattr(self.sim_env, "available_tasks"):

                status_data["available_tasks"] = len(self.sim_env.available_tasks)
                status_data["task_examples"] = self.sim_env.available_tasks[:3]  # Show first 3 as examples

            return status_data

        except Exception as e:
            logger.error(f"❌ Error getting status for {self.tool_name_str}: {e}")
            return {
                "sim_env_name": self.tool_name_str,
                "error": str(e),
                "is_initialized": False,
                "task_status": "error",
            }

    def _capture_video_frame(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Capture a video frame from the observation.

        Args:
            observation: Robot observation containing camera data

        Returns:
            RGB frame as numpy array, or None if no camera found
        """
        try:
            # Priority order for camera selection
            camera_keys = [
                "video.webcam",  # Isaac-GR00T SO-100 style (our fix)
                "front_camera",  # Most common
                "agentview_image",  # Libero style
                "wrist_camera",  # Secondary view
                "robot0_eye_in_hand_image",  # Libero wrist
                "webcam",  # Generic
                "pixels",  # SO-100 style
            ]

            for camera_key in camera_keys:
                if camera_key in observation:
                    frame = observation[camera_key]
                    if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                        # Ensure frame is in correct format (H, W, C)
                        if len(frame.shape) == 4:
                            # Remove batch dimension if present (B, H, W, C) -> (H, W, C)
                            frame = frame[0]
                        elif len(frame.shape) == 2:
                            # Convert grayscale to RGB (H, W) -> (H, W, 3)
                            frame = np.stack([frame] * 3, axis=-1)

                        # Ensure frame is uint8
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                # Normalize [0,1] to [0,255]
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)

                        return frame

            logger.debug("⚠️ No suitable camera found in observation for video recording")
            return None

        except Exception as e:
            logger.debug(f"⚠️ Error capturing video frame: {e}")
            return None

    def _save_video(self, frames: List[np.ndarray], video_path: str, fps: int = 10) -> None:
        """Save video frames to file.

        Args:
            frames: List of RGB frames as numpy arrays
            video_path: Path to save the video file
            fps: Frames per second for the video
        """
        if not frames:
            logger.warning("⚠️ No frames to save for video")
            return

        try:
            # Try OpenCV first (preferred method)
            try:
                import cv2

                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    out.write(frame_bgr)

                out.release()
                logger.info(f"🎥 Video saved using OpenCV: {video_path} ({len(frames)} frames, {fps} fps)")
                return

            except ImportError:
                logger.debug("OpenCV not available, trying imageio")

            # Try imageio as fallback
            try:
                import imageio

                # Create video writer
                with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
                    for frame in frames:
                        writer.append_data(frame)

                logger.info(f"🎥 Video saved using imageio: {video_path} ({len(frames)} frames, {fps} fps)")
                return

            except ImportError:
                logger.debug("imageio not available, trying matplotlib")

            # Try matplotlib animation as last resort
            try:
                import matplotlib.animation as animation
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.axis("off")

                def animate(frame_idx):
                    ax.clear()
                    ax.imshow(frames[frame_idx])
                    ax.axis("off")
                    return []

                anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=True)

                # Save as MP4
                anim.save(video_path, writer="ffmpeg", fps=fps)
                plt.close(fig)

                logger.info(f"🎥 Video saved using matplotlib: {video_path} ({len(frames)} frames, {fps} fps)")
                return

            except ImportError:
                logger.warning("matplotlib not available")

            # Final fallback: save as GIF using PIL
            try:
                from PIL import Image

                # Convert numpy arrays to PIL Images
                pil_frames = []
                for frame in frames:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame))

                # Save as GIF instead of MP4
                gif_path = video_path.replace(".mp4", ".gif")
                pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=1000 // fps, loop=0)

                logger.info(f"🎥 Video saved as GIF using PIL: {gif_path} ({len(frames)} frames)")
                return

            except ImportError:
                logger.error("❌ No video libraries available (opencv, imageio, matplotlib, or PIL)")

        except Exception as e:
            logger.error(f"❌ Failed to save video: {e}")
            logger.error("💡 Try installing: pip install opencv-python imageio[ffmpeg] matplotlib")

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
            import time

            import imageio

            # Create rollout directory
            DATE = time.strftime("%Y_%m_%d")
            DATE_TIME = time.strftime("%Y_%m_%d_%H_%M_%S")
            rollout_dir = f"./rollouts/{DATE}"
            os.makedirs(rollout_dir, exist_ok=True)

            # Process task description for filename
            processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]

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
            print(f"Saved rollout MP4 at path {mp4_path}")
            return mp4_path

        except Exception as e:
            logger.warning(f"❌ Failed to save rollout video: {e}")
            raise

    async def stop(self):
        """Stop simulation environment and disconnect."""
        try:
            # Stop any running task first
            if self._task_state.status == SimTaskStatus.RUNNING:
                self.stop_task()

            # Cleanup environment
            await self.sim_env.cleanup()

            # Cleanup resources
            self.cleanup()

            logger.info(f"🛑 {self.tool_name_str} stopped and disconnected")

        except Exception as e:
            logger.error(f"❌ Error stopping simulation environment: {e}")
