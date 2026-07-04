<div align="center">
  <h1>Strands Robots Sim</h1>

  <h2>The GPU-accelerated NVIDIA Isaac Sim backend for <a href="https://github.com/strands-labs/robots"><code>strands-robots</code></a></h2>

  <div align="center">
    <a href="https://pypi.org/project/strands-robots-sim/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/strands-robots-sim"/></a>
    <a href="https://github.com/strands-labs/robots-sim"><img alt="GitHub stars" src="https://img.shields.io/github/stars/strands-labs/robots-sim"/></a>
    <a href="https://github.com/strands-labs/robots-sim/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-labs/robots-sim"/></a>
    <a href="https://strands-labs.github.io/robots-sim/"><img alt="Docs" src="https://img.shields.io/badge/docs-strands--labs.github.io%2Frobots--sim-blue"/></a>
    <a href="https://developer.nvidia.com/isaac-sim"><img alt="Isaac Sim" src="https://img.shields.io/badge/Isaac%20Sim-6.0-76B900?logo=nvidia"/></a>
  </div>

  <p>
    <a href="https://strands-labs.github.io/robots-sim/">Docs</a>
    ◆ <a href="https://github.com/strands-labs/robots">strands-robots</a>
    ◆ <a href="https://strands-labs.github.io/robots/">Robots Docs</a>
    ◆ <a href="https://developer.nvidia.com/isaac-sim">NVIDIA Isaac Sim</a>
    ◆ <a href="https://github.com/orgs/strands-labs/projects/2">Project Board</a>
  </p>
</div>

`strands-robots-sim` is the GPU-accelerated Isaac Sim companion to
[`strands-robots`](https://github.com/strands-labs/robots). It ships an
**`IsaacSimulation`** that plugs into the same `SimEngine` ABC the upstream
MuJoCo backend implements, so a Strands Agent that drives a MuJoCo world
today switches to Isaac Sim by swapping the backend it constructs.

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                          # procedural; no asset files needed
sim.step(100)
frame = sim.render(camera_name="default")
```

> **Note:** `create_simulation("isaac", ...)` resolves to this repo's
> `IsaacSimulation` via the `strands_robots.backends` entry point —
> `strands-robots>=0.4.1` walks that group from its `create_simulation`
> factory ([`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131)),
> the same UX as `create_simulation("mujoco")`. The kwargs flow into
> `IsaacConfig`. You can still construct `IsaacSimulation(IsaacConfig(...))`
> directly when you want the config object in hand.

> **📚 Documentation:** <https://strands-labs.github.io/robots-sim/>
>
> Includes a [Quickstart](https://strands-labs.github.io/robots-sim/getting-started/quickstart/),
> the [Architecture](https://strands-labs.github.io/robots-sim/architecture/)
> behind the plugin contract, the
> [Backend reference](https://strands-labs.github.io/robots-sim/backends/isaac/),
> and [Troubleshooting](https://strands-labs.github.io/robots-sim/troubleshooting/).

## Why robots-sim

- **RTX path-traced rendering.** Photoreal observations, paper-grade
  frames, sim2real visuals — driven by the same agent code your MuJoCo
  smoke tests use.
- **USD-native scenes.** Real CAD assets, Nucleus support, IsaacLab
  compatibility.
- **Replicator synth-data.** Domain randomization at scale; ground-truth
  depth, segmentation, and bounding boxes alongside RGB.
- **Fleet RL on PhysX GPU.** 1024+ parallel envs with shared USD scenes.
- **Plugin shape, not a fork.** `strands-robots` has zero hard dependency
  on Isaac Sim — install this package and Isaac is discovered through
  entry points.

If none of those apply, install the lightweight default at
[`strands-labs/robots`](https://github.com/strands-labs/robots) instead —
it runs everywhere (including Apple Silicon), boots in seconds, and the
agent contract is identical.

## How it works

```mermaid
graph LR
    A[Strands Agent] --> B[Simulation<br/>AgentTool]
    B --> C[create_simulation 'isaac'<br/>walks entry points]
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

`strands-robots-sim` registers `IsaacSimulation` as a
`strands_robots.backends` entry point. `create_simulation("isaac")` walks the
entry-point group, imports the target string, and instantiates it —
`strands-robots>=0.4.1` ships that walker
([`strands-labs/robots#131`](https://github.com/strands-labs/robots/issues/131)),
so `create_simulation("isaac", ...)` resolves to this repo's `IsaacSimulation`
with nothing more than `pip install strands-robots-sim` (see
[Quick start](#quick-start)). The full plugin contract is documented in
[Architecture](https://strands-labs.github.io/robots-sim/architecture/).

## Installation

System requirements: **NVIDIA RTX GPU, Ubuntu 22.04+, CUDA 12+, Isaac Sim 6.0 (Python 3.12)**.
macOS / Apple Silicon contributors should install
[`strands-robots`](https://github.com/strands-labs/robots) directly and
skip this repo.

```bash
# Step 1 — install Isaac Sim itself (NOT on PyPI):
#   - Omniverse Launcher → Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - NGC Docker: docker pull nvcr.io/nvidia/isaac-sim:6.0

# Step 2 — install the Python plugin:
pip install 'strands-robots-sim[isaac]'
```

`strands-robots` (the upstream MuJoCo backend, `Simulation` AgentTool,
and policy providers) is pulled in transitively.

Full install matrix in
[Getting Started → Installation](https://strands-labs.github.io/robots-sim/getting-started/installation/).

## Quick start

### Single-env RTX render

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_pathtracing", headless=True)
sim.create_world()
sim.add_robot("so100")                          # procedural builder
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.step(120)
frame = sim.render(camera_name="front")          # {"rgb": ..., "depth": ...}
sim.destroy()
```

### IsaacLab-style fleet (preview)

```python
sim = create_simulation("isaac", num_envs=1024, headless=True,
                        render_mode="headless")
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
# ... RL training loop ...
```

### URDF / MJCF / USD ingestion

```python
from strands_robots_sim.isaac.loaders import load_urdf, load_mjcf, load_usd

panda = load_urdf("/path/to/panda.urdf")         # stdlib XML, no extra deps
print(panda.num_joints, panda.joint_names)
```

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

The full per-backend table with install hints lives in
[Simulation → Overview](https://strands-labs.github.io/robots-sim/simulation/overview/).

## Simulation surface

`IsaacSimulation` exposes the same `SimEngine` shape MuJoCo does:

```
World & lifecycle    create_world / destroy / reset / step / get_state / cleanup
Robots               add_robot (procedural | USD | URDF) / remove_robot / list_robots
                     robot_joint_names / send_action / get_observation
Objects              add_object (cuboid / sphere / cylinder / capsule, dynamic / static)
                     remove_object
Cameras & rendering  add_camera (look-at, FOV) / render (RGB + depth)
Loaders              load_urdf / load_mjcf / load_usd  → ProceduralRobot dataclass
```

Full reference: [Simulation → World Building](https://strands-labs.github.io/robots-sim/simulation/world-building/).

## Examples

The repo ships runnable LIBERO drivers under `examples/libero/`:

| File | Driver | Best for |
|---|---|---|
| `libero/run_isaac.py` | Programmatic — `evaluate_benchmark(...)` | CI / matrix tables |
| `libero/run_isaac_agent.py` | Strands `Agent` + natural language | Demos |
| `libero/libero_backend_matrix.py` | Flagship matrix runner — walks all backend rows ([R15 / #22](https://github.com/strands-labs/robots-sim/issues/22)) | Side-by-side backend comparison |

Plus visually-driven demos:

- [`examples/isaac_gs/`](examples/isaac_gs) — Isaac Sim + 3DGS hybrid-render digital twin
- [`examples/so101_curobo/isaac/`](examples/so101_curobo) — SO-101 synth-data with cuRobo on Isaac

```bash
# Smoke test (mock policy, no GPU policy server):
python examples/libero/run_isaac.py --policy mock --n-episodes 5

# Real eval against an NVIDIA GR00T checkpoint:
# (the checkpoint is cached under a non-`/home` path by default —
#  /tmp/strands_robots/checkpoints — so it clears gr00t_inference's
#  start_container mount guard; override with --checkpoint-dir.)
python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50
```

Full driver matrix: [Examples → Overview](https://strands-labs.github.io/robots-sim/examples/overview/).

## Project structure

```
strands_robots_sim/
├── __init__.py                # PEP 562 lazy exports
└── isaac/
    ├── __init__.py            # IsaacConfig, IsaacSimulation lazy exports
    ├── _install.py            # single source of truth for install metadata
    ├── config.py              # IsaacConfig dataclass + validation
    ├── simulation.py          # IsaacSimulation(SimEngine) — main backend
    ├── procedural.py          # SO-100 / Panda / G1 builders + tree validator
    ├── loaders.py             # URDF / MJCF / USD → ProceduralRobot
    └── tests/                 # unit + GPU-integration tests
docs/                          # MkDocs Material site (deployed to GitHub Pages)
examples/
├── libero/                    # LIBERO drivers (run_isaac.py + run_isaac_agent.py)
├── isaac_gs/                  # Isaac Sim + 3DGS hybrid-render demo
└── so101_curobo/              # SO-101 synth-data with cuRobo on Isaac
```

## Status

- **v0.2.0** — re-scoped foundation (Isaac plugin shape, MuJoCo back upstream).
- **v0.3.0** — Isaac Sim Phase 1 (entry-point registration, `IsaacConfig`,
  `IsaacSimulation` lifecycle, procedural builders, URDF / MJCF / USD
  loaders) shipped on `main`; data-plane wiring (`add_object`, `add_camera`,
  `render`, USD- and URDF-loaded articulations) shipping incrementally.
- **R9** — Replicator synth-data pipeline, tracked in [#16](https://github.com/strands-labs/robots-sim/issues/16).
- **R23** — IsaacLab-style fleet driver, tracked in [#27](https://github.com/strands-labs/robots-sim/issues/27).

Track the umbrella roadmap at [`#8`](https://github.com/strands-labs/robots-sim/issues/8).
Migration from the legacy 0.1.x API: [`examples/MIGRATION.md`](examples/MIGRATION.md).

## Development

```bash
pip install -e '.[isaac,dev]'

hatch run lint           # black --check + isort --check + flake8
hatch run format         # black + isort
hatch run test           # pytest strands_robots_sim/isaac/tests/

# GPU integration tests (requires Isaac Sim on the host):
STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py -v
```

Docs are MkDocs Material; build locally with:

```bash
pip install -r docs/requirements.txt
mkdocs serve                 # live-reload at http://127.0.0.1:8000
mkdocs build --strict        # CI's check
```

Python 3.12 required (mirroring Isaac Sim 6.0's bundled interpreter).
See [Contributing](https://strands-labs.github.io/robots-sim/contributing/) for the full guide.

## Security

Found a vulnerability? **Do not** open a public issue. Follow the
disclosure process in [SECURITY.md](SECURITY.md) (AWS VDP / HackerOne).

## Contributing

Issues and PRs welcome. Track work on the
[Strands Labs - Robots project board](https://github.com/orgs/strands-labs/projects/2);
it is the source of truth for roadmap and follow-ups.

- [GitHub Issues](https://github.com/strands-labs/robots-sim/issues)
- [Pull Requests](https://github.com/strands-labs/robots-sim/pulls)

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Links

<div align="center">
  <a href="https://github.com/strands-labs/robots-sim">GitHub</a>
  ◆ <a href="https://strands-labs.github.io/robots-sim/">Docs</a>
  ◆ <a href="https://pypi.org/project/strands-robots-sim/">PyPI</a>
  ◆ <a href="https://github.com/strands-labs/robots">strands-robots</a>
  ◆ <a href="https://strands-labs.github.io/robots/">Robots Docs</a>
  ◆ <a href="https://developer.nvidia.com/isaac-sim">NVIDIA Isaac Sim</a>
  ◆ <a href="https://isaac-sim.github.io/IsaacLab/">IsaacLab</a>
</div>
