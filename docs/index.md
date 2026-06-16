# Strands Robots Sim

> GPU-accelerated NVIDIA **Isaac Sim** backend for [`strands-robots`](https://github.com/strands-labs/robots).

`strands-robots-sim` is the GPU-accelerated Isaac Sim companion to `strands-robots`. It
ships an [`IsaacSimulation`](api-reference.md) that plugs into the same
`SimEngine` ABC the upstream MuJoCo backend implements, so a Strands Agent
that drives a MuJoCo world today can switch to Isaac Sim by changing a
single string:

=== "MuJoCo (lightweight, in `strands-robots`)"

    ```python
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("so100")
    sim.step(100)
    ```

=== "Isaac Sim (this repo, RTX, USD-native)"

    ```python
    import strands_robots_sim                      # registers "isaac" via entry points
    from strands_robots.simulation import create_simulation

    sim = create_simulation("isaac", render_mode="rtx_pathtracing", headless=True)
    sim.create_world()
    sim.add_robot("so100")                         # procedural; no asset files needed
    sim.step(100)
    frame = sim.render(camera_name="default")
    ```

The agent code, the policy interface, and the `Simulation` AgentTool surface
are all identical across backends.

## When you want this repo

| Pick this if you need... | ...because |
|---|---|
| **RTX path-traced rendering** | Photoreal observations / paper-grade frames / sim2real visuals |
| **USD-native scenes** | Real CAD / Nucleus assets, IsaacLab compatibility |
| **Replicator synthetic data** | Domain randomization at scale, ground-truth depth / segmentation |
| **Fleet RL with PhysX GPU** | IsaacLab-style training loops, 1024+ parallel envs |
| **Real-asset robots** | Bring your own URDF / MJCF / USD; load Franka, Panda, custom CAD |

If none of those fit, install the lightweight default at
[`strands-labs/robots`](https://github.com/strands-labs/robots) instead — it
runs everywhere (including Apple Silicon), boots in seconds, and the agent
contract is the same.

## How it works

`strands-robots-sim` is **discovered, not imported**. The package registers
`IsaacSimulation` as a `strands_robots.backends` entry point so
`create_simulation("isaac")` resolves to it without `strands-robots` ever
needing a hard dependency on Isaac Sim:

```mermaid
graph LR
    A[Strands Agent] --> B[Simulation<br/>AgentTool]
    B --> C[create_simulation 'isaac']
    C --> D[Entry-point lookup<br/>strands_robots.backends]
    D --> E[IsaacSimulation<br/>this repo]
    E --> F[Isaac Sim Kit<br/>SimulationApp]
    F --> G[PhysX + RTX]

    classDef agent fill:#0969da,stroke:#044289,color:#fff
    classDef glue fill:#8250df,stroke:#5a32a3,color:#fff
    classDef plugin fill:#bf8700,stroke:#875e00,color:#fff

    class A,B agent
    class C,D glue
    class E,F,G plugin
```

The same plugin shape is what makes the `mujoco` backend in `strands-robots`
and `isaac` here interchangeable: both are `SimEngine` subclasses; both ship
through entry points; the user-facing API is the `Simulation` AgentTool.

See [Architecture](architecture.md) for the full plugin contract.

## Install

System requirements: **NVIDIA RTX GPU, Ubuntu 22.04+, CUDA 12+, Isaac Sim 4.x**.
macOS / Apple Silicon contributors should install
[`strands-robots`](https://github.com/strands-labs/robots) directly and skip
this repo.

```bash
pip install 'strands-robots-sim[isaac]'
```

Isaac Sim itself is **not on PyPI** — it is an Omniverse Kit application
that must be installed separately via the Omniverse Launcher, Isaac Lab, or
the NGC Docker image. Full instructions in
[Getting Started → Installation](getting-started/installation.md).

## Quickstart

```python
import strands_robots_sim                      # register the "isaac" backend
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                         # procedural builder, no asset files
sim.add_object(name="cube", shape="cuboid", position=[0.4, 0.0, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.step(120)
frame = sim.render(camera_name="front")        # RTX RGBA + depth dict
sim.destroy()
```

For URDF / MJCF / USD ingestion, use the loader module:

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")        # stdlib XML, no extra deps
print(panda.num_joints, panda.joint_names)
```

See [Getting Started → Quickstart](getting-started/quickstart.md) for the
end-to-end LIBERO demo on RTX.

## Backend matrix

| Capability | MuJoCo<br/>(`strands-robots`) | **Isaac Sim**<br/>(this repo) |
|---|:---:|:---:|
| Native GPU | — | ✅ PhysX |
| Apple Silicon | ✅ | ❌ |
| Photoreal rendering | — | ✅ RTX path-traced |
| `num_envs` per GPU | 1–8 | 1024+ |
| USD scene format | — | ✅ native |
| Synthetic data (Replicator) | — | ✅ |
| Download size | ~50 MB | ~30 GB |
| Setup friction | low | high |

- **Pick MuJoCo** for fast iteration, debugging, macOS / Apple Silicon, CI.
- **Pick Isaac Sim** for RTX photoreal eval, USD-native scenes, IsaacLab
  fleet RL, or Replicator synth-data.

The full per-backend table with install hints lives in [Simulation →
Overview](simulation/overview.md).

## Where to next

- **First time?** → [Getting Started → Installation](getting-started/installation.md)
- **Plug-and-play example?** → [Getting Started → Quickstart](getting-started/quickstart.md)
- **The plugin model in detail?** → [Architecture](architecture.md)
- **Authoring a world?** → [Simulation → World Building](simulation/world-building.md)
- **Class / config reference?** → [API Reference](api-reference.md)
- **Something broken?** → [Troubleshooting](troubleshooting.md)

## Links

- [`strands-labs/robots`](https://github.com/strands-labs/robots) — default MuJoCo backend, `Simulation` AgentTool, LIBERO adapter
- [`strands-labs/robots` docs](https://strands-labs.github.io/robots/) — robot catalog, policy providers, hardware/device pages
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) ([IsaacLab](https://isaac-sim.github.io/IsaacLab/))
- [Project Board](https://github.com/orgs/strands-labs/projects/2)
