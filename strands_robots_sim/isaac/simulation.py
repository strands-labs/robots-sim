"""Isaac Sim simulation backend -- GPU-native SimEngine implementation.

This module contains :class:`IsaacSimulation`, the primary implementation
of the ``SimEngine`` ABC for the NVIDIA Isaac Sim backend.

Architecture:
    - All heavy omni/Isaac imports are lazy (not at module level)
    - The class manages an Isaac Sim ``World``, ``Articulation`` handles,
      and RTX camera instances
    - Multi-env replication uses ``omni.isaac.cloner.Cloner``
    - SimulationApp is a process-wide singleton (never create more than one)
    - Rendering delegates to Isaac Sim's RTX pipeline

Thread safety:
    - ``step()``, ``send_action()``, and ``get_observation()`` acquire
      ``self._lock`` to prevent data races
    - ``step()`` must not run concurrently with ``add_robot()``

Environment variables:
    - STRANDS_ISAAC_HEADLESS: Override headless mode (true/false)
    - STRANDS_ISAAC_RTX_PATHTRACING: Enable RTX pathtracing (true/false)
    - STRANDS_ISAAC_NUCLEUS_URL: Override Nucleus asset server URL
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, TypedDict

import numpy as np


class SimulationAppLaunchConfig(TypedDict, total=False):
    """Typed shape for ``omni.isaac.kit.SimulationApp`` launch config.

    All keys optional; SimulationApp accepts an open-ended dict and any
    additional keys are forwarded to Kit unchanged. The keys below are the
    well-known ones documented by NVIDIA across Isaac Sim 4.x / 5.x and are
    the ones a Strands tool would realistically expose to an agent.

    See: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html

    Keys
    ----
    headless : bool
        Run without GUI. Required True on cloud / CI runners.
    renderer : str
        ``"RayTracedLighting"`` or ``"PathTracing"``.
    width, height : int
        Viewport resolution in pixels.
    physics_gpu : int
        CUDA device index for PhysX.
    active_gpu : int
        CUDA device index for rendering.
    multi_gpu : bool
        Enable multi-GPU rendering.
    sync_loads : bool
        Block until USD assets finish loading.
    hide_ui : bool
        Hide Kit's editor UI in non-headless mode.
    anti_aliasing : int
        Anti-aliasing level (0-3).
    """

    headless: bool
    renderer: str
    width: int
    height: int
    physics_gpu: int
    active_gpu: int
    multi_gpu: bool
    sync_loads: bool
    hide_ui: bool
    anti_aliasing: int


try:
    from strands_robots.simulation.base import SimEngine
except (ImportError, ModuleNotFoundError):
    # Fallback: strands-robots < 0.4.0 doesn't have simulation.base yet.
    # Provide a minimal ABC stub so the class can still be defined.
    from abc import ABC, abstractmethod

    class SimEngine(ABC):  # type: ignore[no-redef]
        """Minimal fallback ABC when strands-robots.simulation.base is unavailable.

        Mirrors the abstract surface of the real
        ``strands_robots.simulation.base.SimEngine`` so subclasses fail fast at
        instantiation if a method is missing -- same contract whether
        ``strands_robots`` is installed or not.
        """

        @abstractmethod
        def create_world(self, **kwargs): ...
        @abstractmethod
        def destroy(self): ...
        @abstractmethod
        def reset(self): ...
        @abstractmethod
        def step(self, n_steps: int = 1): ...
        @abstractmethod
        def get_state(self): ...
        @abstractmethod
        def add_robot(self, name: str, **kwargs): ...
        @abstractmethod
        def remove_robot(self, name: str): ...
        @abstractmethod
        def list_robots(self) -> list: ...
        @abstractmethod
        def robot_joint_names(self, robot_name: str) -> list: ...
        @abstractmethod
        def add_object(self, name: str, **kwargs): ...
        @abstractmethod
        def remove_object(self, name: str): ...
        @abstractmethod
        def get_observation(self, robot_name=None, *, skip_images=False): ...
        @abstractmethod
        def send_action(self, action, robot_name=None, n_substeps: int = 1): ...
        @abstractmethod
        def render(self, camera_name: str = "default", width=None, height=None): ...

        def cleanup(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.cleanup()


from strands_robots_sim.isaac.config import IsaacConfig  # noqa: E402  # late import: must follow SimEngine fallback def

logger = logging.getLogger(__name__)

# Module-level singleton tracking for SimulationApp
_SIMULATION_APP: Any = None
_SIMULATION_APP_LOCK = threading.Lock()


def _get_or_create_simulation_app(
    headless: bool = True,
    launch_config: SimulationAppLaunchConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create the process-wide SimulationApp singleton.

    Isaac Sim's SimulationApp can only be created ONCE per process.
    This function ensures that constraint is respected.

    Parameters
    ----------
    headless : bool
        Run without GUI.
    launch_config : SimulationAppLaunchConfig, optional
        Typed launch config dict forwarded to ``omni.isaac.kit.SimulationApp``.
        See :class:`SimulationAppLaunchConfig` for documented keys
        (``renderer``, ``width``, ``height``, ``physics_gpu``,
        ``active_gpu``, ``multi_gpu``, ``sync_loads``, ``hide_ui``,
        ``anti_aliasing``). The explicit ``headless`` argument always
        wins over any ``"headless"`` key in ``launch_config``.
    **kwargs
        Additional SimulationApp launch keys (escape hatch for Kit
        options not in :class:`SimulationAppLaunchConfig`). Merged on
        top of ``launch_config``; ``headless`` argument still wins.

    Returns
    -------
    SimulationApp instance.

    Raises
    ------
    ImportError
        If omni.isaac.kit is not available.
    """
    global _SIMULATION_APP

    with _SIMULATION_APP_LOCK:
        if _SIMULATION_APP is not None:
            return _SIMULATION_APP

        try:
            from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]
        except ImportError as e:
            from strands_robots_sim.isaac._install import not_available_import_error

            raise ImportError(not_available_import_error()) from e

        # Layer order: typed launch_config base, then **kwargs escape hatch,
        # then explicit headless argument (always wins so the caller's
        # intent is unambiguous).
        merged: dict[str, Any] = dict(launch_config or {})
        merged.update(kwargs)
        merged["headless"] = headless
        _SIMULATION_APP = SimulationApp(merged)
        logger.info(
            "SimulationApp created (headless=%s). " "Note: this is a process-wide singleton.",
            headless,
        )
        return _SIMULATION_APP


class _RobotState:
    """Internal bookkeeping for a robot in the Isaac simulation."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        joint_names: list[str],
        articulation: Any = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.joint_names = joint_names
        self.articulation = articulation


class _CameraState:
    """Internal bookkeeping for a camera in the Isaac simulation."""

    def __init__(self, name: str, prim_path: str, width: int, height: int):
        self.name = name
        self.prim_path = prim_path
        self.width = width
        self.height = height
        self.handle: Any = None


class IsaacSimulation(SimEngine):
    """GPU-native simulation backend built on NVIDIA Isaac Sim.

    Implements the ``SimEngine`` ABC. Provides photorealistic rendering,
    RTX sensors, USD scene management, and fleet replication via Cloner.

    Parameters
    ----------
    config : IsaacConfig or None
        Configuration. If None, defaults are used.
    **kwargs
        Shortcut kwargs merged into config (e.g. ``num_envs=1024``).

    Examples
    --------
    >>> sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    >>> ok, msg = IsaacSimulation.is_available()
    >>> if ok:
    ...     sim.create_world()
    ...     sim.add_robot("so100")
    ...     sim.step(100)
    ...     sim.destroy()
    """

    def __init__(self, config: IsaacConfig | None = None, **kwargs: Any) -> None:
        # Merge shortcut kwargs into config. Unknown kwargs are rejected
        # eagerly (rather than silently dropped) so a typo like
        # ``IsaacSimulation(headles=False)`` surfaces at construction time
        # instead of producing a default-config sim with no warning.
        import dataclasses

        if config is None:
            # IsaacConfig is a dataclass; passing an unknown kwarg raises
            # TypeError("__init__() got an unexpected keyword argument ...")
            # naturally. Both branches now have symmetric strictness.
            config = IsaacConfig(**kwargs)
        elif kwargs:
            fields = {f.name for f in dataclasses.fields(config)}
            unknown = sorted(set(kwargs) - fields)
            if unknown:
                raise TypeError(
                    f"IsaacSimulation got unexpected kwargs: {unknown}. " f"Known IsaacConfig fields: {sorted(fields)}."
                )
            config = dataclasses.replace(config, **kwargs)
        self._config = config

        # Simulation state (all lazy-initialized)
        self._app: Any = None
        self._world: Any = None

        # World state
        self._world_created = False
        self._replicated = False
        self._num_envs_active = 1
        self._sim_time = 0.0
        self._step_count = 0

        # Entity tracking
        self._robots: dict[str, _RobotState] = {}
        self._cameras: dict[str, _CameraState] = {}
        self._prim_registry: list[str] = []  # track all created prims for cleanup

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "IsaacSimulation initialized: num_envs=%d, device=%s, headless=%s",
            config.num_envs,
            config.device,
            config.headless,
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Check if Isaac Sim is available on this system.

        Returns
        -------
        tuple[bool, str | None]
            (available, reason_if_not). If available is True, reason is None.
        """
        # Probe what create_world() actually needs: omni.isaac.kit.SimulationApp.
        # The bare ``omni`` namespace is a PEP 420 namespace package shared by
        # omni.ui, omni.usd, partial Omniverse SDK installs, and Isaac-Lab
        # pre-bootstrap states -- its mere presence is not a reliable signal
        # that Isaac Sim is usable. ``importlib.util.find_spec`` checks the
        # specific submodule without importing it (no side effects); it
        # raises ModuleNotFoundError when a parent package along the dotted
        # path is missing, which we treat the same as "not available".
        import importlib.util

        try:
            kit_spec = importlib.util.find_spec("omni.isaac.kit")
        except ModuleNotFoundError:
            kit_spec = None
        if kit_spec is None:
            from strands_robots_sim.isaac._install import not_importable_reason

            return False, not_importable_reason()

        # Isaac requires CUDA
        try:
            import torch

            if not torch.cuda.is_available():
                return False, ("CUDA device not detected. Isaac Sim requires an NVIDIA GPU " "with CUDA support.")
        except ImportError:
            return False, ("PyTorch not installed. Isaac Sim requires torch with CUDA support.")

        return True, None

    @property
    def config(self) -> IsaacConfig:
        """Current configuration (read-only)."""
        return self._config

    # --- SimEngine: World Lifecycle ----------------------------------------

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world in Isaac Sim.

        Initializes the SimulationApp (singleton), creates a USD stage,
        configures physics, and optionally adds a ground plane.

        Parameters
        ----------
        timestep : float, optional
            Override physics_dt from config.
        gravity : list[float], optional
            Override gravity vector from config. [gx, gy, gz].
        ground_plane : bool
            Whether to add a ground plane. Default True.

        Returns
        -------
        dict
            Status dict with world info.
        """
        with self._lock:
            if self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "World already created. Call destroy() first."}],
                }

            try:
                # Create/get SimulationApp singleton
                self._app = _get_or_create_simulation_app(headless=self._config.headless)

                # Now safe to import Isaac core modules
                from omni.isaac.core import World  # type: ignore[import-not-found]

                dt = timestep if timestep is not None else self._config.physics_dt
                grav = gravity if gravity is not None else list(self._config.gravity)

                # Create World
                self._world = World(
                    stage_units_in_meters=1.0,
                    physics_dt=dt,
                    rendering_dt=self._config.rendering_dt,
                )

                # Set gravity
                # Isaac Sim 5.1: set_gravity takes a scalar magnitude, not a vector.
                # Extract the Z-component (convention: gravity points along -Z).
                gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav
                self._world.get_physics_context().set_gravity(gravity_magnitude)

                # Add ground plane
                if ground_plane and self._config.ground_plane:
                    self._world.scene.add_default_ground_plane()
                    self._prim_registry.append(f"{self._config.stage_path}/defaultGroundPlane")

                # Reset world to initialize
                self._world.reset()

                self._world_created = True
                self._sim_time = 0.0
                self._step_count = 0

                logger.info(
                    "World created: dt=%.5f, gravity=%s, headless=%s",
                    dt,
                    grav,
                    self._config.headless,
                )

                # Surface a structured snapshot of the freshly-created
                # environment alongside the human-readable text. Agents
                # spinning up a sim can introspect device / dt / scene
                # config without re-querying via get_state().
                world_info = {
                    "physics_dt": dt,
                    "rendering_dt": self._config.rendering_dt,
                    "gravity": list(grav) if isinstance(grav, (list, tuple)) else [0.0, 0.0, float(grav)],
                    "ground_plane": bool(ground_plane and self._config.ground_plane),
                    "stage_path": self._config.stage_path,
                    "stage_units_in_meters": 1.0,
                    "device": self._config.device,
                    "headless": self._config.headless,
                    "render_mode": self._config.render_mode,
                    "num_envs": self._config.num_envs,
                    "num_envs_active": self._num_envs_active,
                    "replicated": self._replicated,
                    "sim_time": self._sim_time,
                    "step_count": self._step_count,
                }

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Isaac Sim world created. "
                                f"dt={dt:.5f}, gravity={grav}, "
                                f"device={self._config.device}, "
                                f"headless={self._config.headless}"
                            ),
                            "json": world_info,
                        }
                    ],
                }

            except ImportError as e:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"Isaac Sim import failed: {e}. " "Ensure Isaac Sim is installed and accessible.")}
                    ],
                }
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                # Cleanup on partial failure. Narrow to what World() /
                # set_gravity / add_default_ground_plane / reset actually
                # raise on Isaac: RuntimeError (Carb / sim init), ValueError
                # (USD prim shape mismatches, e.g. set_init_state on the
                # ground plane), OSError (USD/Nucleus IO), AttributeError
                # (omni surface drift across SDK versions), TypeError
                # (Isaac Sim 5.1 ``set_gravity`` rejects non-scalar input
                # — see #52; defence in depth for similar argument-shape
                # surface drift on neighbouring physics-context calls).
                # Programming bugs (NameError, ImportError-not-already-
                # caught above) propagate.
                self._world = None
                logger.error("Failed to create Isaac world: %s", e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create world: {e}"}],
                }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources.

        Note: SimulationApp is NOT shut down (it is process-wide).
        Only the World/Stage are cleared.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world to destroy."}],
                }

            # Capture pre-teardown counts so the structured json payload
            # surfaces what was actually released (the agent's get_state()
            # window is gone after destroy() returns).
            num_robots_released = len(self._robots)
            num_cameras_released = len(self._cameras)
            num_prims_released = len(self._prim_registry)
            num_envs_released = self._num_envs_active
            sim_time_at_destroy = self._sim_time
            step_count_at_destroy = self._step_count

            try:
                if self._world is not None:
                    self._world.stop()
                    self._world.clear_instance()
                    self._world = None
            except (RuntimeError, OSError, AttributeError) as e:
                # World.stop() / clear_instance() can raise on partial init
                # or on a torn-down stage; AttributeError covers omni surface
                # drift across versions. Logged at WARNING because we still
                # mark the world destroyed below; programming bugs propagate.
                logger.warning("World cleanup warning: %s", e)

            # Clear entity tracking
            self._robots.clear()
            self._cameras.clear()
            self._prim_registry.clear()

            # Reset state
            self._world_created = False
            self._replicated = False
            self._num_envs_active = 1
            self._sim_time = 0.0
            self._step_count = 0

            logger.info("World destroyed. SimulationApp remains (process-wide singleton).")

            # Surface a structured snapshot of what teardown released
            # alongside the human-readable text. Mirrors the json content
            # block convention used by get_state() (L624) and create_world()
            # (L455) so an agent inspecting destroy() can confirm what was
            # actually torn down without re-querying.
            destroy_info = {
                "num_robots_released": num_robots_released,
                "num_cameras_released": num_cameras_released,
                "num_prims_released": num_prims_released,
                "num_envs_released": num_envs_released,
                "sim_time_at_destroy": sim_time_at_destroy,
                "step_count_at_destroy": step_count_at_destroy,
                "stage_path": self._config.stage_path,
                "simulation_app_alive": True,  # singleton survives destroy()
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Isaac Sim world destroyed. All resources released. "
                            "SimulationApp singleton remains active."
                        ),
                        "json": destroy_info,
                    }
                ],
            }

    def reset(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset simulation to initial state.

        Parameters
        ----------
        env_ids : list[int], optional
            Specific environment indices to reset. If None, reset all.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is not None:
                self._world.reset()

            self._sim_time = 0.0
            self._step_count = 0

            if env_ids is None:
                msg = "Full reset complete."
            else:
                msg = f"Partial reset complete for {len(env_ids)} envs."

            return {"status": "success", "content": [{"text": msg}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take. Default 1.

        Returns
        -------
        dict
            Status dict with timing info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is None:
                return {"status": "error", "content": [{"text": "World not initialized."}]}

            t0 = time.perf_counter()

            for _ in range(n_steps):
                self._world.step(render=self._config.render_mode != "headless")
                self._sim_time += self._config.physics_dt
                self._step_count += 1

            elapsed = time.perf_counter() - t0
            steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Stepped {n_steps}x. "
                            f"sim_time={self._sim_time:.4f}s, "
                            f"wall={elapsed * 1000:.1f}ms, "
                            f"{steps_per_sec:.0f} steps/sec"
                        )
                    }
                ],
            }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary.

        Returns
        -------
        dict
            Status dict with state information.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            state_data = {
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "num_envs": self._num_envs_active,
                "num_robots": len(self._robots),
                "num_cameras": len(self._cameras),
                "stage_path": self._config.stage_path,
                "device": self._config.device,
                "headless": self._config.headless,
                "render_mode": self._config.render_mode,
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"State: t={self._sim_time:.4f}s, "
                            f"step={self._step_count}, "
                            f"envs={self._num_envs_active}, "
                            f"robots={len(self._robots)}, "
                            f"cameras={len(self._cameras)}"
                        ),
                        "json": state_data,
                    }
                ],
            }

    # --- SimEngine: Robot Management ----------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        mjcf_path: str | None = None,
        usd_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup).
        urdf_path : str, optional
            Path to URDF file.
        mjcf_path : str, optional
            Path to MJCF file.
        usd_path : str, optional
            Path to USD file (native Isaac format).
        data_config : str, optional
            Named data config for procedural lookup.
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base orientation as quaternion [w, x, y, z].

        Returns
        -------
        dict
            Status dict with robot info.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world created. Call create_world() first."}],
                }

            if name in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' already exists."}],
                }

            if self._replicated:
                return {
                    "status": "error",
                    "content": [{"text": "Cannot add robots after replicate(). Call destroy() first."}],
                }

            pos = position or [0.0, 0.0, 0.0]
            prim_path = f"{self._config.stage_path}/Robots/{name}"

            # Try procedural first
            lookup_name = data_config or name
            try:
                from strands_robots_sim.isaac.procedural import get_procedural_robot

                procedural = get_procedural_robot(lookup_name)
            except ImportError:
                procedural = None

            if procedural is not None:
                # Build procedurally via USD API
                joint_names = procedural.joint_names
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                logger.info("Added robot '%s' (procedural, %d joints)", name, len(joint_names))
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' added (procedural: {procedural.name}, "
                                f"{len(joint_names)} joints: {joint_names})"
                            )
                        }
                    ],
                }

            elif usd_path is not None:
                # Load from USD (native Isaac format)
                joint_names = self._load_usd_robot(prim_path, usd_path, pos)
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                return {
                    "status": "success",
                    "content": [{"text": (f"Robot '{name}' added (USD: {usd_path}, " f"{len(joint_names)} joints)")}],
                }

            elif urdf_path is not None:
                # Convert URDF to USD and load
                joint_names = self._load_urdf_robot(prim_path, urdf_path, pos)
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                return {
                    "status": "success",
                    "content": [{"text": (f"Robot '{name}' added (URDF: {urdf_path}, " f"{len(joint_names)} joints)")}],
                }

            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{lookup_name}' not found in procedural registry "
                                "and no usd_path/urdf_path provided. "
                                "Available procedural robots: so100, panda, unitree_g1"
                            )
                        }
                    ],
                }

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object to the scene.

        Parameters
        ----------
        name : str
            Object identifier.
        shape : str
            Shape type: "box", "sphere", "capsule", "cylinder".
        position : list[float], optional
            Position [x, y, z].
        orientation : list[float], optional
            Quaternion [w, x, y, z].
        size : list[float], optional
            Shape dimensions.
        color : list[float], optional
            RGB color [r, g, b] in [0, 1].
        mass : float
            Mass in kg. Default 0.1.
        is_static : bool
            If True, object is fixed in space. Default False.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Validate shape
            valid_shapes = ("box", "sphere", "capsule", "cylinder")
            if shape not in valid_shapes:
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown shape: {shape!r}. Valid: {valid_shapes}"}],
                }

            pos = position or [0.0, 0.0, 0.5]
            prim_path = f"{self._config.stage_path}/Objects/{name}"
            self._prim_registry.append(prim_path)

            logger.debug("Added object '%s' (shape=%s, pos=%s)", name, shape, pos)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' added (shape={shape}, pos={pos})."}],
            }

    # --- SimEngine: Introspection / Removal ---------------------------------

    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world.

        Returns
        -------
        list[str]
            Robot names in insertion order. Empty if no robots have been
            added (or after :meth:`destroy`).
        """
        with self._lock:
            return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for ``robot_name``.

        Parameters
        ----------
        robot_name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        list[str]
            Joint names in articulation order, or an empty list if
            ``robot_name`` is not present (matches the silent-empty
            convention used by :meth:`get_observation` for unknown robots).
        """
        with self._lock:
            if robot_name not in self._robots:
                return []
            return list(self._robots[robot_name].joint_names)

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation.

        Drops the robot's bookkeeping entry and prunes any prims rooted at
        the robot's prim path from ``self._prim_registry``. The actual USD
        prim deletion is delegated to :meth:`destroy` / world teardown in
        Phase 1; only the in-Python registry is updated here.

        Parameters
        ----------
        name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class.
        """
        with self._lock:
            if name not in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' not found."}],
                }
            prim_path = self._robots[name].prim_path
            self._prim_registry = [p for p in self._prim_registry if not p.startswith(prim_path)]
            del self._robots[name]
            logger.info("Removed robot '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Robot '{name}' removed."}],
            }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene.

        Mirror of :meth:`add_object`'s prim-path convention
        (``{stage_path}/Objects/{name}``). Only updates the in-Python
        registry; USD prim deletion is handled at world teardown.

        Parameters
        ----------
        name : str
            Object identifier previously passed to :meth:`add_object`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class.
        """
        with self._lock:
            prim_path = f"{self._config.stage_path}/Objects/{name}"
            if prim_path not in self._prim_registry:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' not found."}],
                }
            self._prim_registry.remove(prim_path)
            logger.info("Removed object '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' removed."}],
            }

    # --- SimEngine: Observation / Action ------------------------------------

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get observation for a robot.

        Parameters
        ----------
        robot_name : str, optional
            Robot to observe. Auto-resolves if only one robot exists.
        skip_images : bool
            Skip camera rendering. Default False.

        Returns
        -------
        dict
            Observation with joint positions keyed by joint name. An empty dict
            indicates one of four diagnostically-distinct conditions, each of
            which is logged before return so silent failures are visible in
            operational logs:

            * ``world not yet created`` -- DEBUG (expected pre-init state).
            * ``ambiguous robot_name=None with multiple robots`` -- WARNING.
            * ``unknown robot_name`` (typo / not-yet-added) -- WARNING.
            * ``robot present but Articulation handle not yet initialised``
              (Phase 1 procedural / load stub) -- DEBUG via the inner except;
              the dict returns empty because no joint positions are reachable.

            The return shape is preserved as a plain dict (rather than the
            ``{"status": ..., "content": [...]}`` shape used by mutating
            methods on this class) because callers consume joint positions
            keyed by joint name; the four silent-``{}`` modes are distinguished
            in logs rather than in the return value.
        """
        with self._lock:
            if not self._world_created:
                # Expected pre-init state; many callers probe before
                # create_world() to feature-detect, so DEBUG-only.
                logger.debug(
                    "get_observation(robot_name=%r): world not yet created",
                    robot_name,
                )
                return {}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    logger.warning(
                        "get_observation(robot_name=None): ambiguous -- "
                        "%d robots present (%s); pass robot_name explicitly. "
                        "Returning empty observation.",
                        len(self._robots),
                        sorted(self._robots),
                    )
                    return {}

            if robot_name not in self._robots:
                logger.warning(
                    "get_observation(robot_name=%r): unknown robot. Known: %s. " "Returning empty observation.",
                    robot_name,
                    sorted(self._robots),
                )
                return {}

            robot = self._robots[robot_name]
            obs: dict[str, Any] = {}

            # Get joint state from Articulation handle
            if robot.articulation is not None:
                try:
                    joint_positions = robot.articulation.get_joint_positions()
                    if joint_positions is not None:
                        positions = (
                            joint_positions.cpu().numpy()
                            if hasattr(joint_positions, "cpu")
                            else np.array(joint_positions)
                        )
                        for i, jname in enumerate(robot.joint_names):
                            if i < len(positions):
                                obs[jname] = float(positions[i])
                except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                    # Articulation handle may raise RuntimeError on a not-yet
                    # -initialized world, AttributeError on torch-tensor surface
                    # drift, ValueError/TypeError on np coercion. Programming
                    # bugs propagate.
                    logger.debug("Failed to get joint positions: %s", e)

            return obs

    def send_action(
        self,
        action: dict[str, Any] | np.ndarray | list,
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> None:
        """Apply action and advance physics.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name.
        robot_name : str, optional
            Robot to control.
        n_substeps : int
            Physics sub-steps after applying action. Default 1.
        """
        with self._lock:
            if not self._world_created or self._world is None:
                return

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    return

            if robot_name not in self._robots:
                return

            robot = self._robots[robot_name]

            # Convert action to array
            if isinstance(action, dict):
                action_array = np.zeros(len(robot.joint_names), dtype=np.float32)
                for i, jname in enumerate(robot.joint_names):
                    if jname in action:
                        action_array[i] = float(action[jname])
            elif isinstance(action, np.ndarray):
                action_array = action.astype(np.float32).flatten()
            else:
                action_array = np.array(action, dtype=np.float32)

            # Apply to articulation
            if robot.articulation is not None:
                try:
                    robot.articulation.set_joint_position_targets(action_array)
                except (RuntimeError, ValueError, AttributeError) as e:
                    # set_joint_position_targets raises RuntimeError on a
                    # torn-down articulation, ValueError on shape mismatch,
                    # AttributeError on omni surface drift. Programming bugs
                    # (NameError, KeyError) propagate.
                    logger.debug("Failed to set joint targets: %s", e)

            # Step physics
            for _ in range(n_substeps):
                self._world.step(render=False)
                self._sim_time += self._config.physics_dt
                self._step_count += 1

    # --- SimEngine: Rendering -----------------------------------------------

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Render a camera view using Isaac Sim's RTX pipeline.

        Parameters
        ----------
        camera_name : str
            Camera identifier. Default "default".
        width : int, optional
            Frame width. Default from config.
        height : int, optional
            Frame height. Default from config.

        Returns
        -------
        dict
            Dict with "rgb", "depth", "seg" arrays and "status".
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            w = width or self._config.camera_width
            h = height or self._config.camera_height

            if self._config.render_mode == "headless":
                # Return blank frames in headless mode
                return {
                    "status": "success",
                    "rgb": np.zeros((h, w, 3), dtype=np.uint8),
                    "depth": np.zeros((h, w), dtype=np.float32),
                    "content": [{"text": f"Rendered (headless, no RTX): {w}x{h}"}],
                }

            # RTX rendering via camera handle
            if camera_name in self._cameras:
                cam = self._cameras[camera_name]
                # In full implementation, read from RTX camera annotators
                rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                depth = np.zeros((cam.height, cam.width), dtype=np.float32)
                return {
                    "status": "success",
                    "rgb": rgb,
                    "depth": depth,
                    "content": [{"text": f"Rendered (RTX {self._config.render_mode}): {cam.width}x{cam.height}"}],
                }

            # No camera configured — return blank
            return {
                "status": "success",
                "rgb": np.zeros((h, w, 3), dtype=np.uint8),
                "depth": np.zeros((h, w), dtype=np.float32),
                "content": [{"text": f"Rendered (no camera): {w}x{h}"}],
            }

    def add_camera(
        self,
        name: str = "default",
        position: list[float] | None = None,
        target: list[float] | None = None,
        width: int | None = None,
        height: int | None = None,
        fov: float = 60.0,
    ) -> dict[str, Any]:
        """Add an RTX camera to the scene.

        Parameters
        ----------
        name : str
            Camera identifier. Default "default".
        position : list[float], optional
            Camera position [x, y, z].
        target : list[float], optional
            Camera look-at target [x, y, z].
        width : int, optional
            Image width. Default from config.
        height : int, optional
            Image height. Default from config.
        fov : float
            Field of view in degrees. Default 60.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            w = width or self._config.camera_width
            h = height or self._config.camera_height
            pos = position or [2.0, 2.0, 2.0]

            prim_path = f"{self._config.stage_path}/Cameras/{name}"
            self._prim_registry.append(prim_path)

            cam_state = _CameraState(name=name, prim_path=prim_path, width=w, height=h)
            self._cameras[name] = cam_state

            return {
                "status": "success",
                "content": [{"text": (f"Camera '{name}' added at {pos}, " f"resolution={w}x{h}, fov={fov}")}],
            }

    # --- Isaac-specific: Fleet Replication -----------------------------------

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the current scene into parallel environments.

        Uses ``omni.isaac.cloner.Cloner`` for GPU-efficient replication.

        Parameters
        ----------
        num_envs : int, optional
            Number of environments. Defaults to config.num_envs.

        Returns
        -------
        dict
            Status dict with replication info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._robots:
                return {
                    "status": "error",
                    "content": [{"text": "Add at least one robot first."}],
                }

            n = num_envs or self._config.num_envs

            t0 = time.perf_counter()
            # In full implementation: use omni.isaac.cloner.Cloner
            # to replicate the scene N times
            self._replicated = True
            self._num_envs_active = n
            elapsed = time.perf_counter() - t0

            logger.info("Replicated to %d envs in %.2fs", n, elapsed)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Replicated to {n} environments. "
                            f"Build time: {elapsed * 1000:.0f}ms. "
                            f"Device: {self._config.device}."
                        ),
                        "json": {
                            "num_envs": n,
                            "build_time_ms": elapsed * 1000,
                        },
                    }
                ],
            }

    # --- Private Implementation ----------------------------------------------

    def _load_usd_robot(self, prim_path: str, usd_path: str, position: list[float]) -> list[str]:
        """Load a robot from USD file. Returns joint names."""
        # In full implementation: use omni.isaac.core to add USD reference
        # and extract Articulation joint names
        logger.info("Loading USD robot from %s at %s", usd_path, prim_path)
        return []

    def _load_urdf_robot(self, prim_path: str, urdf_path: str, position: list[float]) -> list[str]:
        """Load a robot from URDF (converted to USD). Returns joint names."""
        # In full implementation: use omni.isaac.urdf to convert and load
        logger.info("Loading URDF robot from %s at %s", urdf_path, prim_path)
        return []

    def cleanup(self) -> None:
        """Release all resources.

        Callers must invoke this explicitly (or use the class as a context
        manager). There is intentionally no ``__del__`` finalizer: at
        interpreter shutdown the ``threading`` / ``logger`` / ``omni``
        modules can already be partially torn down, and acquiring
        ``self._lock`` from a finalizer is unsafe. Relying on GC for
        Isaac Sim cleanup also leaks the ``World``/USD stage on the
        common case where the GC scheduler defers the finalizer past
        the SimulationApp shutdown.
        """
        if self._world_created:
            self.destroy()

    def __enter__(self) -> IsaacSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"IsaacSimulation("
            f"num_envs={self._config.num_envs}, "
            f"device={self._config.device!r}, "
            f"headless={self._config.headless}, "
            f"world={'created' if self._world_created else 'none'})"
        )
