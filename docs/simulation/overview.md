# Simulation overview

`IsaacSimulation` is the GPU-accelerated NVIDIA backend behind the upstream
`Simulation` AgentTool. It implements the `SimEngine` ABC from
`strands-robots`, so the same agent / policy code that drives MuJoCo
drives Isaac Sim — the only thing that changes is the backend string
passed to `create_simulation(...)`.

This page covers the **lifecycle** (config → world → step → render →
destroy) and the **render-mode tradeoffs**. The mechanics of authoring a
scene live in [World Building](world-building.md); the synth-data
pipeline in [Domain Randomization](domain-randomization.md).

## `IsaacConfig`

All simulation parameters flow through `IsaacConfig`, a dataclass with
sensible defaults:

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | `1` | Parallel envs (`1024+` for fleet RL). |
| `device` | `"cuda:0"` | CUDA device string. |
| `headless` | `True` | Run without GUI. Required for cloud / CI runners. |
| `physics_dt` | `1/120 s` | PhysX timestep. |
| `rendering_dt` | `1/30 s` | RTX render cadence. |
| `render_mode` | `"headless"` | `"headless"` / `"rtx_realtime"` / `"rtx_pathtracing"`. |
| `gravity` | `(0, 0, -9.81)` | Z-up. |
| `ground_plane` | `True` | Auto-add a ground plane on `create_world()`. |
| `stage_path` | `"/World"` | USD stage prefix. |
| `nucleus_url` | `None` | Override Omniverse Nucleus server URL. |
| `camera_width` / `camera_height` | `640` / `480` | Default camera resolution. |
| `enable_rtx_sensors` | `True` | RTX-accelerated camera / LiDAR. |
| `verbose` | `False` | Verbose logs from Isaac Sim / Kit. |
| `extra` | `{}` | Escape-hatch for experimental options. |

```python
from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

cfg = IsaacConfig(
    num_envs=1,
    headless=True,
    render_mode="rtx_pathtracing",
    physics_dt=1.0 / 240.0,
    camera_width=1280,
    camera_height=720,
)
sim = IsaacSimulation(cfg)
```

`create_simulation("isaac", headless=True, render_mode="rtx_realtime")`
upstream is shorthand for the same thing — kwargs are forwarded into
`IsaacConfig`.

### Environment-variable overrides

| Variable | Effect |
|---|---|
| `STRANDS_ISAAC_HEADLESS` | `"true"` / `"false"` — overrides `headless` |
| `STRANDS_ISAAC_RTX_PATHTRACING` | `"true"` / `"false"` — flip render mode |
| `STRANDS_ISAAC_NUCLEUS_URL` | Override Nucleus server URL |

These are useful for CI: a `Makefile` target can pin `STRANDS_ISAAC_HEADLESS=true`
without code changes.

## Render modes

| Mode | Speed | Use it for |
|---|---|---|
| `"headless"` | fastest | RL training; no rendering at all. |
| `"rtx_realtime"` | ~real-time | Smoke tests, demos, eval that doesn't need photoreal frames. |
| `"rtx_pathtracing"` | slow | Paper-grade visuals, sim2real renders, ground-truth synth-data. |

Switching modes is one `IsaacConfig` field — no code change beyond that:

```python
sim = IsaacSimulation(IsaacConfig(render_mode="rtx_pathtracing"))
```

`render_mode="headless"` skips the entire RTX pipeline; `render(...)`
returns a blank frame. Use it when only the physics + state matter (RL
training rollouts, BDDL eval that consults `mjData` rather than pixels).

## Lifecycle

```python
sim = IsaacSimulation(cfg)            # 1. boot SimulationApp (process singleton)
sim.create_world()                    # 2. open USD stage, add ground plane
sim.add_robot("so100")                # 3. populate the world
sim.add_camera(name="front", ...)
sim.step(100)                         # 4. advance physics (+ rendering at rendering_dt)
frame = sim.render(camera_name="front")
sim.destroy()                         # 5. clear the world (SimulationApp survives)
```

Five hard rules:

- **`SimulationApp` is a process-wide singleton.** Two `IsaacSimulation`
  instances in the same process share it. Re-bootstrapping it is not
  supported by Isaac Sim.
- **`destroy()` does not shut down `SimulationApp`.** Use `cleanup()` (or
  process exit) for that.
- **`step()` and any state-mutating call (`add_*`, `remove_*`,
  `set_joint_*`) cannot run concurrently.** An `RLock` serializes them;
  the second caller blocks.
- **`reset(env_ids=None)` resets all envs.** Pass an env-ID list for
  per-env reset (fleet mode).
- **Calling `add_robot(name=...)` with a duplicate name raises
  `ValueError`.** Robots, objects, and cameras share a single MuJoCo-style
  name table; keep them globally unique.

## `is_available()` — pre-flight check

`IsaacSimulation.is_available()` is a static method that returns
`(bool, str | None)` *without* importing `omni.*`. Use it to gate code
that conditionally creates an Isaac sim:

```python
from strands_robots_sim.isaac import IsaacSimulation

ok, reason = IsaacSimulation.is_available()
if not ok:
    print(f"Isaac Sim not available — falling back. Reason: {reason}")
    sim = create_simulation("mujoco")            # graceful degradation
else:
    sim = create_simulation("isaac", headless=True)
```

The reason string is structured (which `omni.*` import failed, expected
ImportError vs. unexpected error) and quotes the relevant install hint
from `_install.py` so users know exactly what to do next. See
[Troubleshooting](../troubleshooting.md).

## Thread safety

- All mutable simulation state is protected by an `RLock`.
- The `SimulationApp` singleton is initialized through a module-level
  guard (`_get_or_create_simulation_app`).
- `cleanup()` / `__exit__` are idempotent — calling them twice is safe.
- The simulation instance itself is **not** safe to share across processes;
  use `multiprocessing.spawn` and bring up Isaac inside each child.

## Next

- [World Building](world-building.md) — `add_robot` / `add_object` /
  `add_camera` / `render` reference.
- [Domain Randomization](domain-randomization.md) — Replicator pipeline.
- [Backends → Isaac Sim](../backends/isaac.md) — the full backend
  reference, including procedural builders and loader internals.
- [API Reference](../api-reference.md) — class signatures.
