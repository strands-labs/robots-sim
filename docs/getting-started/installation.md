# Installation

`strands-robots-sim` ships **two** install layers:

1. **Isaac Sim itself** — the Omniverse Kit application + Python runtime.
   Not on PyPI; install once per machine via Omniverse Launcher, Isaac Lab,
   or the NGC Docker image.
2. **The `strands-robots-sim` Python package** — a thin pip-installable
   plugin that registers `IsaacSimulation` as a `strands_robots.backends`
   entry point and adds a few light deps that Isaac uses internally
   (`usd-core` for USD authoring).

Order matters: install Isaac Sim **first**, then `pip install
'strands-robots-sim[isaac]'` into the same Python environment.

## System requirements

- **GPU** — NVIDIA RTX 2070+ (RTX 3090 / A100 / L4 / H100 for fleet training).
- **OS** — Ubuntu 22.04+ (Linux only; macOS / Apple Silicon are not
  supported by Isaac Sim).
- **CUDA** — 12.0+, with a recent NVIDIA driver matching the Isaac Sim release.
- **Python** — 3.12 (Isaac Sim 6.0 ships its own 3.12 embedded
  interpreter; mirror that version in your venv).
- **Disk** — ~30 GB for the Isaac Sim SDK on first run, plus ~5 GB for
  the cached USD assets.

If you only need fast iteration / Apple Silicon / CI smoke tests, install
[`strands-robots`](https://github.com/strands-labs/robots) directly and use
the MuJoCo backend; the agent contract is the same.

## Step 1 — install Isaac Sim

Pick one of three paths.

=== "Omniverse Launcher (recommended)"

    1. Download the [NVIDIA Omniverse Launcher](https://developer.nvidia.com/omniverse).
    2. Sign in with an NVIDIA developer account.
    3. From the **Exchange** tab, install **Isaac Sim 6.0**.
    4. Optionally launch the app once to confirm it boots.

    The launcher places Isaac Sim under `~/.local/share/ov/pkg/isaac-sim-6.0/`.

=== "Isaac Lab"

    [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) bundles Isaac Sim
    plus the IsaacLab fleet-RL framework:

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    ./isaaclab.sh -i              # downloads Isaac Sim + sets up the venv
    source _isaac_sim/setup_python_env.sh
    ```

    Pick this if you also want IsaacLab's vectorized envs / RL recipes.

=== "Docker (NGC)"

    The NVIDIA-published Isaac Sim image is the lowest-friction option for
    cloud / CI runners:

    ```bash
    docker pull nvcr.io/nvidia/isaac-sim:6.0
    docker run --gpus all -it --rm \
        -e ACCEPT_EULA=Y \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        nvcr.io/nvidia/isaac-sim:6.0
    ```

    Inside the container, Isaac Sim is at `/isaac-sim/` with a Python
    runtime at `/isaac-sim/python.sh`.

## Step 2 — install `strands-robots-sim`

Use the same Python environment that Isaac Sim's `python.sh` /
`setup_python_env.sh` activates. Then:

```bash
pip install 'strands-robots-sim[isaac]'
```

!!! info "The `[isaac]` extra does **not** install Isaac Sim"

    Isaac Sim itself is not on PyPI — it comes from the out-of-band install
    you did in Step 1 (Launcher / Isaac Lab / NGC Docker), which ships a
    complete, bootable Kit. The `[isaac]` extra therefore pulls in only the
    genuinely pip-installable companion dep — `usd-core` (the pure-Python
    USD runtime used by the procedural scene builders and loaders) — plus
    `strands-robots` transitively (the upstream `Simulation` AgentTool,
    `create_simulation()` factory, and policy providers).

    Do **not** try to `pip install isaacsim` into the environment yourself
    as a substitute for Step 1: NVIDIA's `isaacsim[all]` metapackage pulls
    the `isaacsim-*` packages but **not** the `isaacsim-extscache-*`
    packages, so `SimulationApp` aborts at boot with an `omni.ext`
    "Failed to resolve extension dependencies" error and `create_world()`
    never starts. The Launcher / Isaac Lab / Docker images bundle the
    complete extension set; use one of those.

## Step 3 — verify your install boots

After Steps 1 and 2, confirm `SimulationApp` actually boots end-to-end —
not just that the package imports. Run this with Isaac Sim's bundled Python
(`python.sh` / `setup_python_env.sh`-activated venv):

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()                 # boots SimulationApp; resolves all extensions
sim.add_robot("so100")
sim.add_object(name="cube", shape="cuboid", position=[0.4, 0.0, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.step(120)
frame = sim.render(camera_name="front")   # RTX RGBA + depth dict
print("rgb:", frame["rgb"].shape)          # e.g. (480, 640, 3)
sim.destroy()
```

If `create_world()` raises an `omni.ext` "Failed to resolve extension
dependencies" error, your Isaac Sim install is incomplete (typically a
bare `pip install isaacsim` without the `isaacsim-extscache-*` packages).
Use a Launcher / Isaac Lab / NGC Docker install instead — see Step 1. The
[Troubleshooting](../troubleshooting.md) page covers this and other
common boot failures.

## Step 4 — verify the package wiring

If you'd rather check availability without booting a full `SimulationApp`:

```python
import strands_robots_sim                      # registers entry points
from strands_robots_sim.isaac.simulation import IsaacSimulation

available, reason = IsaacSimulation.is_available()
print("isaac available:", available, "(reason:", reason, ")")
```

On a healthy machine this prints `isaac available: True (reason: None)`.
On a CPU-only / non-Isaac box it returns a structured diagnostic
explaining which `omni.*` import failed — see
[Troubleshooting](../troubleshooting.md).

You can also confirm the entry-point registration:

```python
from importlib.metadata import entry_points

for ep in entry_points(group="strands_robots.backends"):
    print(ep.name, "->", ep.value)
# isaac -> strands_robots_sim.isaac.simulation:IsaacSimulation
```

## From source

```bash
git clone https://github.com/strands-labs/robots-sim
cd robots-sim
pip install -e '.[isaac,dev]'
```

`hatch run lint` and `hatch run test` run the full unit-test slice (no GPU
required); `STRANDS_GPU_TEST=1 pytest strands_robots_sim/isaac/tests/test_gpu_integ.py`
runs the GPU integration tests.

## Next

- [Quickstart](quickstart.md) — bring up an Isaac world and step it.
- [Architecture](../architecture.md) — how the entry-point plugin contract works.
- [Backends → Isaac Sim](../backends/isaac.md) — the full backend reference.
