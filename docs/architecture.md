# Architecture

`strands-robots-sim` is a **plugin** for `strands-robots`, not a fork of it.
The upstream package ships the agent-facing `Simulation` AgentTool, the
`SimEngine` ABC, and a default MuJoCo backend. This repo adds an
`IsaacSimulation` that plugs into the same `SimEngine` ABC and registers
itself through Python entry points so the upstream package never needs a
hard dependency on Isaac Sim.

The result: an agent that drives a MuJoCo world today switches to Isaac Sim
by changing one string.

## The plugin contract

```mermaid
graph TB
    subgraph Upstream["strands-robots (upstream)"]
        AT[Simulation<br/>AgentTool]
        FAC[create_simulation 'name']
        ABC[SimEngine ABC]
        MUJ[MuJoCoSimEngine]
    end

    subgraph Plugin["strands-robots-sim (this repo)"]
        ISA[IsaacSimulation]
        CFG[IsaacConfig]
        LOAD[loaders.py<br/>URDF / MJCF / USD]
        PROC[procedural.py<br/>SO-100 / Panda / G1]
    end

    subgraph Isaac["Isaac Sim Kit"]
        APP[SimulationApp]
        WORLD[omni.isaac.core.World]
        PHYSX[PhysX GPU]
        RTX[RTX render]
    end

    AT --> FAC
    FAC -->|entry point lookup| ABC
    ABC <|.. MUJ
    ABC <|.. ISA
    ISA --> CFG
    ISA --> APP
    APP --> WORLD
    WORLD --> PHYSX
    APP --> RTX
    LOAD -.-> PROC
    PROC -.-> ISA

    classDef up fill:#0969da,stroke:#044289,color:#fff
    classDef plug fill:#bf8700,stroke:#875e00,color:#fff
    classDef isa fill:#76B900,stroke:#3e6800,color:#000

    class AT,FAC,ABC,MUJ up
    class ISA,CFG,LOAD,PROC plug
    class APP,WORLD,PHYSX,RTX isa
```

The contract has three pieces.

### 1. `SimEngine` ABC (upstream)

Every backend implements the same set of lifecycle + scene + physics +
render methods:

```python
class SimEngine(ABC):
    def create_world(self, **kwargs) -> dict: ...
    def destroy(self) -> dict: ...
    def reset(self, env_ids: list[int] | None = None) -> dict: ...
    def step(self, n_steps: int = 1) -> dict: ...
    def get_state(self) -> dict: ...

    def add_robot(self, name: str, **kwargs) -> dict: ...
    def remove_robot(self, name: str) -> dict: ...
    def list_robots(self) -> list[str]: ...
    def robot_joint_names(self, robot_name: str) -> list[str]: ...

    def add_object(self, name: str, **kwargs) -> dict: ...
    def remove_object(self, name: str) -> dict: ...
    def add_camera(self, name: str, **kwargs) -> dict: ...

    def get_observation(self, robot_name: str | None = None,
                        *, skip_images: bool = False) -> dict: ...
    def send_action(self, action, robot_name=None,
                    n_substeps: int = 1) -> dict: ...
    def render(self, camera_name: str = "default",
               width=None, height=None) -> dict: ...
```

The MuJoCo backend in `strands-robots` and `IsaacSimulation` here are both
`SimEngine` subclasses. The agent / policy loop talks to `SimEngine` and
never knows which backend is underneath.

### 2. Entry-point registration (this repo)

`strands-robots-sim` declares its backends as entry points in
[`pyproject.toml`](https://github.com/strands-labs/robots-sim/blob/main/pyproject.toml):

```toml
[project.entry-points."strands_robots.backends"]
isaac     = "strands_robots_sim.isaac.simulation:IsaacSimulation"
isaac_sim = "strands_robots_sim.isaac.simulation:IsaacSimulation"
```

When `strands-robots-sim` is installed, `importlib.metadata.entry_points`
sees both names. `create_simulation("isaac", ...)` upstream walks the
`strands_robots.backends` group, finds the `isaac` entry, imports the
target string, and instantiates it.

This is the same pattern other packages use to extend `strands-robots`
(future cuRobo / MoveIt2 backends, custom user backends). The upstream
package has zero hard dependency on `omni.*`.

### 3. Lazy `omni.*` imports (this repo)

Importing `strands_robots_sim` must not import `omni.*` — that would force
every `strands-robots` user to have Isaac Sim installed, defeating the
plugin design. We use **PEP 562 lazy module-level `__getattr__`** so
`from strands_robots_sim.isaac import IsaacSimulation` only triggers the
expensive `omni.*` import on first attribute access:

```python
# strands_robots_sim/isaac/__init__.py (sketch)
def __getattr__(name: str):
    if name == "IsaacSimulation":
        from .simulation import IsaacSimulation
        return IsaacSimulation
    if name == "IsaacConfig":
        from .config import IsaacConfig
        return IsaacConfig
    raise AttributeError(name)
```

Combined with `IsaacSimulation.is_available()`, this means a CPU-only dev
box can:

- `pip install 'strands-robots-sim[isaac]'` without errors,
- `import strands_robots_sim` (registers the entry point),
- call `IsaacSimulation.is_available()` and get a structured diagnostic
  back, **before** any `omni.*` import is attempted.

Only when the user asks for a real `IsaacSimulation(config)` do we boot
the `SimulationApp` singleton.

## Lifecycle

```mermaid
sequenceDiagram
    participant U as User code
    participant SR as create_simulation
    participant IS as IsaacSimulation
    participant APP as SimulationApp
    participant W as omni.isaac.core.World

    U->>SR: create_simulation('isaac', headless=True)
    SR->>IS: __init__(IsaacConfig(...))
    IS->>APP: _get_or_create_simulation_app(headless=True)
    Note right of APP: process-wide singleton
    APP-->>IS: SimulationApp instance
    U->>IS: create_world()
    IS->>W: World(physics_dt, rendering_dt, ...)
    U->>IS: add_robot('so100')
    IS->>W: USD prim + procedural builder
    U->>IS: step(100)
    IS->>W: world.step() x 100
    U->>IS: render('front')
    IS->>W: camera.get_rgba() / get_depth()
    U->>IS: destroy()
    IS->>W: world.clear()
    Note right of APP: SimulationApp not torn down
```

Two invariants you will hit if you ignore them:

- **`SimulationApp` is process-wide.** Creating a second `IsaacSimulation`
  reuses the same `SimulationApp`. `destroy()` clears the *world* but
  intentionally does **not** kill the `SimulationApp` — Kit cannot
  re-bootstrap inside a single process.
- **`step()` and `add_robot()` cannot run concurrently.** All mutable state
  is protected by an `RLock`; calls block, they don't race.

## What lives in this repo vs. upstream

| Concern | Lives in | Why |
|---|---|---|
| `Simulation` AgentTool, `SimEngine` ABC, `create_simulation` factory | `strands-robots` | Upstream owns the agent-facing surface |
| `MuJoCoSimEngine`, `MockPolicy`, LIBERO adapter | `strands-robots` | Default backend; runs everywhere |
| `IsaacSimulation`, `IsaacConfig`, procedural builders, URDF/MJCF/USD loaders | `strands-robots-sim` (here) | GPU-accelerated NVIDIA-only dependency |
| Robot catalog (`robots.json`), policy providers (GR00T, LeRobot, cuRobo) | `strands-robots` | Backend-agnostic |
| Hardware (`HardwareRobot`, mesh, IoT, device-connect) | `strands-robots` | Same |
| Replicator synth-data pipeline | `strands-robots-sim` (here) | Isaac-specific |
| LIBERO `run_isaac.py` / `run_isaac_agent.py` example drivers | `strands-robots-sim` (here) | Backend-specific entry points |

The split is deliberate: `strands-robots` is the agent's view of the
world; `strands-robots-sim` is the GPU-accelerated Isaac plumbing behind one of the
plug-in slots that view exposes.

## Next

- [Simulation → Overview](simulation/overview.md) — `IsaacConfig`, headless
  vs. RTX modes, world lifecycle.
- [Simulation → World Building](simulation/world-building.md) — the
  `add_robot` / `add_object` / `add_camera` / `render` surface.
- [API Reference](api-reference.md) — class signatures.
- [Backends → Isaac Sim](backends/isaac.md) — the full backend reference.
