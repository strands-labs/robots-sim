# Quickstart

Bring up an Isaac Sim world, drop a robot in, render an RTX frame.

## Prerequisites

- Isaac Sim installed and verified (see [Installation](installation.md)).
- `strands-robots-sim[isaac]` installed in the same Python environment.

## Hello, RTX

```python
from strands_robots.simulation import create_simulation

sim = create_simulation(
    "isaac",
    render_mode="rtx_realtime",                # or "rtx_pathtracing" for path-traced
    headless=True,
)
sim.create_world()
sim.add_robot("so100")                         # procedural builder, no asset files
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])

sim.step(120)                                  # ~1 s of sim time
frame = sim.render(camera_name="front")        # {"rgb": (H, W, 3) uint8, "depth": ...}

sim.destroy()
```

!!! note "`create_simulation('isaac')` resolves via the entry-point group"

    `IsaacSimulation` is registered as a `strands_robots.backends` entry
    point. `strands-robots>=0.4.1` walks that group from `create_simulation`
    ([`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131)),
    so `create_simulation("isaac", ...)` resolves to this repo's
    `IsaacSimulation` — the same UX as `create_simulation("mujoco")`. The
    kwargs are forwarded into `IsaacConfig`. If you want the config object in
    hand you can still construct `IsaacSimulation(IsaacConfig(...))` directly.

What happened:

1. `create_simulation("isaac", ...)` resolves the backend through the
   `strands_robots.backends` entry point and forwards the kwargs into
   `IsaacConfig` — the recommended path.
2. `create_world()` spins up a `SimulationApp`, opens a USD stage, and
   adds a ground plane.
3. `add_robot("so100")` runs the procedural SO-100 builder — no asset
   files needed, no Nucleus required.
4. `add_object(...)` and `add_camera(...)` author scene primitives via
   the `isaacsim.core.api` / `isaacsim.sensors.camera` API.
5. `step(120)` advances PhysX 120 substeps.
6. `render(...)` returns an RGBA frame plus depth from the configured RTX
   sensor.

## Bring your own robot asset

The procedural builders ship a kinematically-approximate SO-100 / Panda /
G1 — useful for smoke tests, not enough for LIBERO-style manipulation. Use
the loaders / `add_robot(usd_path=...)` for real robots:

```python
sim.add_robot(name="panda", usd_path="/path/to/panda.usda")
# or:
sim.add_robot(name="panda", urdf_path="/path/to/panda.urdf")
```

For pre-loading description files into a `ProceduralRobot` dataclass
(useful when introspecting joint counts before adding the robot):

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")
print(panda.num_joints, panda.joint_names)
# 7 ['panda_joint1', 'panda_joint2', ...]
```

See [Simulation → World Building](../simulation/world-building.md) for the
full `add_robot` / `add_object` / `add_camera` reference.

## Run a benchmark

`strands-robots-sim` ships two end-to-end LIBERO drivers under
`examples/libero/`:

| Driver | Purpose |
|---|---|
| `examples/libero/run_isaac.py` | Programmatic — calls `evaluate_benchmark(...)` directly. CI / matrix-table input. |
| `examples/libero/run_isaac_agent.py` | Strands `Agent` + natural language — invokes the same eval through one `agent("...")` call. |

```bash
# Smoke test (mock policy, no GPU policy server):
python examples/libero/run_isaac.py --policy mock --n-episodes 5
```

!!! warning "`run_isaac.py` smoke test is gated on #116"

    The `--policy mock` smoke test currently fails inside
    `evaluate_benchmark` (the LIBERO `load_scene` gap tracked in
    [`#116`](https://github.com/strands-labs/robots-sim/issues/116)) and
    exits non-zero. Treat the commands below as the intended invocation
    shape; they will run clean once #116 lands.

```bash
# Bring your own robot asset:
python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd

# Real eval against an NVIDIA GR00T checkpoint (auto-orchestrates the GR00T
# Docker container; pass --no-auto-server to reuse one):
python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50
```

The GR00T checkpoint is cached under a **non-`/home`** path by default
(`/tmp/strands_robots/checkpoints`, overridable via
`$STRANDS_ROBOTS_CHECKPOINT_DIR` or `--checkpoint-dir`). This is required:
`gr00t_inference`'s `start_container` step refuses to bind-mount any path
under `/home`, so a `/home` cache would abort the lifecycle. The non-`/home`
default keeps `--policy groot` working out-of-the-box.

Both files print two grep-stable lines that the flagship
`libero_backend_matrix.py` driver subprocess-and-parses for the
side-by-side table:

```
benchmark_name=libero-spatial-pick_up_the_red_cube
policy=groot task=libero-spatial-pick_up_the_red_cube success_rate=1.00 wall_time=44.3s
```

See [Examples → Overview](../examples/overview.md) for the full driver
matrix and the LIBERO-specific gotchas.

## Driving from a Strands Agent

`IsaacSimulation` is **not** itself a Strands `AgentTool` yet — it has no
`tool_spec` / `__call__`, so `Agent(tools=[sim])` registers **0 tools**
(Strands logs `unrecognized tool specification` and the agent has nothing
to call). Until `IsaacSimulation` becomes an AgentTool (Phase-3 work on
[`#14`](https://github.com/strands-labs/robots-sim/issues/14)), wrap the
operations you want the agent to drive in a `@tool`-decorated function
that closes over the `sim` instance — the same pattern
`examples/libero/run_isaac_agent.py` uses for `evaluate_benchmark`:

```python
from strands import Agent, tool
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")


@tool(
    name="setup_camera_and_render",
    description=(
        "Add a camera at the given position looking at a target, step the "
        "world, then render a frame. Returns the simulation status dicts."
    ),
)
def setup_camera_and_render(
    position: list[float],
    target: list[float],
    n_steps: int = 100,
) -> dict:
    sim.add_camera(name="agent_cam", position=position, target=target)
    sim.step(n_steps)
    return sim.render(camera_name="agent_cam")


agent = Agent(tools=[setup_camera_and_render])
agent("Add a top-down camera at z=1.5 looking at the origin, "
      "step 100 frames, then render it")
```

The agent picks `setup_camera_and_render` and fills its arguments from the
prompt. Each underlying Isaac call returns the usual
`{"status", "content"}` payload, which the wrapper forwards back through
the tool boundary.

## Next

- [Architecture](../architecture.md) — how the plugin contract works.
- [Simulation → World Building](../simulation/world-building.md) — the
  full `add_robot` / `add_object` / `add_camera` reference.
- [Simulation → Domain Randomization](../simulation/domain-randomization.md) —
  Replicator synth-data pipeline.
- [Examples → Overview](../examples/overview.md) — runnable LIBERO drivers.
