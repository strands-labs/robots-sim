# API Reference

Hand-curated reference for the public surface of `strands-robots-sim`.
For the upstream `Simulation` AgentTool, `SimEngine` ABC, and policy
provider classes, see the
[`strands-robots` API docs](https://strands-labs.github.io/robots/api-reference/).

## Module layout

```
strands_robots_sim/
├── __init__.py            # PEP 562 lazy exports
└── isaac/
    ├── __init__.py        # IsaacConfig, IsaacSimulation lazy exports
    ├── _install.py        # Single source of truth for install metadata
    ├── config.py          # IsaacConfig dataclass + validation
    ├── simulation.py      # IsaacSimulation(SimEngine) -- main backend
    ├── procedural.py      # SO-100 / Panda / G1 builders + tree validator
    ├── loaders.py         # URDF / MJCF / USD -> ProceduralRobot
    └── tests/             # unit + GPU-integration tests
```

## `IsaacConfig`

```python
from strands_robots_sim.isaac import IsaacConfig
```

Dataclass owning all simulation-wide configuration. Constructor signature
(see [Simulation → Overview](simulation/overview.md) for the parameter
matrix):

```python
IsaacConfig(
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    physics_dt: float = 1.0 / 120.0,
    rendering_dt: float = 1.0 / 30.0,
    render_mode: str = "headless",          # "headless" / "rtx_realtime" / "rtx_pathtracing"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ground_plane: bool = True,
    stage_path: str = "/World",
    nucleus_url: str | None = None,
    camera_width: int = 640,
    camera_height: int = 480,
    enable_rtx_sensors: bool = True,
    verbose: bool = False,
    extra: dict[str, Any] = field(default_factory=dict),
)
```

Validation runs in `__post_init__`:

- `render_mode` must be one of `RENDER_MODES = ("headless", "rtx_realtime", "rtx_pathtracing")`.
- `num_envs >= 1`.
- `device` must look like `"cuda:N"` or be a string that
  `torch.device(...)` would accept.
- `physics_dt > 0` and `rendering_dt > 0`.

## `IsaacSimulation`

```python
from strands_robots_sim.isaac import IsaacSimulation
```

`SimEngine` subclass. Methods are split into lifecycle, scene authoring,
physics / observation, and rendering.

### Static / class methods

```python
IsaacSimulation.is_available() -> tuple[bool, str | None]
```

Pre-flight check. Returns `(True, None)` on a healthy machine, or
`(False, reason)` on a CPU-only / non-Isaac box. Critically, this does
**not** import `omni.*` — call it before constructing `IsaacSimulation` if
you want to fall back gracefully.

### Lifecycle

```python
IsaacSimulation(config: IsaacConfig | None = None, **kwargs: Any)
```

Boots the process-wide `SimulationApp` singleton. `**kwargs` are forwarded
into `IsaacConfig(...)` if `config` is `None`.

```python
sim.create_world(**kwargs) -> dict
sim.destroy() -> dict
sim.cleanup() -> None                       # idempotent; tears SimulationApp down
sim.reset(env_ids: list[int] | None = None) -> dict
sim.step(n_steps: int = 1) -> dict
sim.get_state() -> dict
```

`__enter__` / `__exit__` are wired to `cleanup()` so `with
IsaacSimulation(...) as sim:` works.

### Scene authoring

```python
sim.add_robot(
    name: str,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    usd_path: str | None = None,
    data_config: str | None = None,
    position: list[float] | None = None,
    orientation: list[float] | None = None,    # quaternion [w, x, y, z]
) -> dict

sim.remove_robot(name: str) -> dict
sim.list_robots() -> list[str]
sim.robot_joint_names(robot_name: str) -> list[str]

sim.add_object(
    name: str,
    shape: str = "box",                     # "box" (alias: "cuboid") / "sphere" / "cylinder" / "capsule"
    position: list[float] | None = None,
    orientation: list[float] | None = None,
    size: list[float] | None = None,        # alias: scale=
    color: list[float] | None = None,
    mass: float = 0.1,
    is_static: bool = False,
) -> dict

sim.remove_object(name: str) -> dict

sim.add_camera(
    name: str = "default",
    position: list[float] | None = None,    # default [2.0, 2.0, 2.0]
    target: list[float] | None = None,
    width: int | None = None,
    height: int | None = None,
    fov: float = 60.0,                       # horizontal FOV in degrees
) -> dict
```

See [Simulation → World Building](simulation/world-building.md) for
worked examples.

### Physics + observation

```python
sim.send_action(
    action: dict[str, float] | list[float],
    robot_name: str | None = None,
    n_substeps: int = 1,
) -> dict

sim.get_observation(
    robot_name: str | None = None,
    *,
    skip_images: bool = False,
) -> dict
```

`action` accepts a dict keyed by joint name or a flat list / array in
`robot_joint_names(robot_name)` order. `skip_images=True` skips camera
rendering when only joint state matters.

### Rendering

```python
sim.render(
    camera_name: str = "default",
    width: int | None = None,
    height: int | None = None,
) -> dict
```

Returns `{"rgb": ndarray (H, W, 3) uint8, "depth": ndarray (H, W) float32, ...}`
when a camera is attached and `render_mode != "headless"`. Returns blank
frames otherwise (headless / no camera attached) so calling code does not
have to special-case the no-render path.

## `ProceduralRobot` and the loaders

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd
```

All three return the same dataclass shape:

```python
@dataclass
class ProceduralRobot:
    name: str
    bodies: list[BodyDef]
    joints: list[JointDef]
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def num_joints(self) -> int: ...        # count of non-fixed joints

    @property
    def joint_names(self) -> list[str]: ... # names of non-fixed joints
```

Failure semantics are uniform across loaders:

| Condition | Exception |
|---|---|
| Path does not exist | `FileNotFoundError` |
| Document fails to parse | `ValueError` (with element + path) |
| Empty document (no links / joints / bodies) | `ValueError` |

The MJCF loader is verified against the seven robosuite-bundled MJCFs the
upstream LIBERO adapter consumes (`panda` / `iiwa` / `kinova3` / `jaco` /
`sawyer` / `ur5e` / `baxter`).

## Procedural builders

```python
from strands_robots_sim.isaac.procedural import (
    ProceduralRobot,
    BodyDef,
    JointDef,
    get_procedural_robot,                    # get_procedural_robot(name) -> ProceduralRobot | None
    list_procedural_robots,                  # list_procedural_robots() -> ["so100", "panda", "unitree_g1"]
    _validate_kinematic_tree,                # public-ish: used by tests
)
```

`get_procedural_robot(name)` looks up a `ProceduralRobot` by name or alias
(`so100` / `panda` / `unitree_g1`, plus aliases like `g1`, `franka`),
returning `None` if unknown. `list_procedural_robots()` returns the
canonical names. There are no `build_*` functions on the public surface —
the per-robot constructors (`_build_so100` etc.) are private; go through
`get_procedural_robot`. Each builds a `ProceduralRobot` instance without
any asset files. `_validate_kinematic_tree(robot)` raises `ValueError` if
the joint graph has a duplicate `(parent_body, child_body)` edge — the
validator runs at every builder's construction, fail-first.

## `_install.py` constants

```python
from strands_robots_sim.isaac._install import (
    ISAAC_SIM_MIN_VERSION,                  # "6.0"
    ISAAC_SIM_DOCKER_IMAGE,                 # "nvcr.io/nvidia/isaac-sim:6.0"
    ISAAC_LAB_BOOTSTRAP,                    # "git clone IsaacLab && ./isaaclab.sh -i"
    PIP_EXTRA,                              # "pip install 'strands-robots-sim[isaac]'"
)
```

Plus the helpers that compose these into user-facing strings:

```python
from strands_robots_sim.isaac._install import (
    install_options_block,                  # multi-line bullet block of install paths
    install_options_inline,                 # one-line variant
    not_importable_reason,                  # full reason string for is_available()
    not_available_import_error,             # ImportError message at runtime
)
```

These are the single source of truth for the `is_available()` reason
string and any `ImportError` rendered to the user. Updating Isaac Sim
versions = update one constant (`ISAAC_SIM_DOCKER_IMAGE` /
`ISAAC_SIM_MIN_VERSION`).

## Entry-point registration

```toml
# pyproject.toml
[project.entry-points."strands_robots.backends"]
isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"
```

`strands-robots>=0.4.1` walks the `strands_robots.backends` group from its
`create_simulation` factory (shipped via
[`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131)),
so `create_simulation("isaac", ...)` does:

```python
import importlib.metadata
ep = next(e for e in importlib.metadata.entry_points(group="strands_robots.backends")
          if e.name == "isaac")
cls = ep.load()
return cls(**kwargs)
```

This resolves to this repo's `IsaacSimulation` with nothing more than
`pip install strands-robots-sim`. If you want the `IsaacConfig` object in
hand, you can still construct the backend directly:

```python
from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig

sim = IsaacSimulation(IsaacConfig(**kwargs))
```

See [`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131)
for the upstream discovery work.

## See also

- [Architecture](architecture.md) — the plugin contract this surface implements.
- [Simulation → Overview](simulation/overview.md) — config + lifecycle in plain English.
- [Backends → Isaac Sim](backends/isaac.md) — the full backend reference,
  including procedural builders and tests.
