# Domain randomization

Isaac Sim's [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
is the synthetic-data pipeline behind every NVIDIA sim2real demo: it
randomizes the scene (lighting, materials, object poses, camera intrinsics),
captures RGB + ground-truth annotations (depth, semantic / instance
segmentation, 3D bounding boxes, normals), and writes the data to disk in
a parquet / image / json layout your training loop can ingest directly.

`strands-robots-sim` exposes that pipeline behind one
[`replicate(...)`](../api-reference.md) call on `IsaacSimulation`.

!!! note "Status"
    The Replicator wiring lands as **R9 / [#16](https://github.com/strands-labs/robots-sim/issues/16)**.
    Today's `replicate(...)` is a Phase-2 stub; the runnable example
    `examples/isaac/isaac_replicator_synthdata.py` ships with R9. Track progress
    on [#16](https://github.com/strands-labs/robots-sim/issues/16) and
    on the umbrella [#8](https://github.com/strands-labs/robots-sim/issues/8).

## What Replicator gives you

For each step in a randomization recipe, Replicator produces:

- **RGB**, RTX path-traced or rasterized, at any resolution.
- **Depth** (perspective / orthographic), float32 meters.
- **Semantic segmentation** — per-pixel class labels resolved against the
  USD prim graph.
- **Instance segmentation** — per-pixel instance IDs.
- **2D / 3D bounding boxes** — axis-aligned and oriented, in pixel and
  world frames.
- **Normals** — surface-normal map.
- **Camera intrinsics + extrinsics** — JSON metadata per frame.

All of these are GPU-resident; writing to disk is the only host-bound op.

## The pipeline shape

```mermaid
graph LR
    A[Scene<br/>USD stage] --> B[Randomize<br/>poses / materials / lights]
    B --> C[Render<br/>RTX]
    C --> D[Annotators<br/>depth / segm / bbox]
    D --> E[Writers<br/>parquet / png / json]
    E -->|on disk| F[Dataset<br/>train your VLA]

    classDef stage fill:#0969da,stroke:#044289,color:#fff
    classDef rand fill:#bf8700,stroke:#875e00,color:#fff
    classDef render fill:#76B900,stroke:#3e6800,color:#000
    classDef out fill:#8250df,stroke:#5a32a3,color:#fff

    class A stage
    class B rand
    class C,D render
    class E,F out
```

## `IsaacSimulation.replicate(...)` — recipe-driven

A randomization recipe is a list of *randomizers* — declarative
descriptions of what to perturb each frame, expressed as keyword args to
`replicate(...)`:

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("isaac", render_mode="rtx_pathtracing", headless=True)
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0, 0.6], target=[0, 0, 0.1])

dataset = sim.replicate(
    n_frames=1000,
    randomize={
        "cube.position":     {"distribution": "uniform",
                              "min": [0.3, -0.2, 0.05], "max": [0.5, 0.2, 0.05]},
        "cube.color":        {"distribution": "uniform_rgb"},
        "lighting.intensity":{"distribution": "uniform", "min": 500, "max": 5000},
        "camera.position":   {"distribution": "gaussian",
                              "mean": [1.2, 0, 0.6], "stddev": [0.05, 0.05, 0.02]},
    },
    annotators=["rgb", "depth", "semantic_segmentation", "bounding_box_2d"],
    output_dir="/data/synth/cube-pickup-1k",
)
```

The output directory ends up structured like:

```
/data/synth/cube-pickup-1k/
├── camera_front/
│   ├── rgb/000000.png ... 000999.png
│   ├── depth/000000.npz ...
│   ├── semantic_segmentation/...
│   └── bounding_box_2d/...
├── metadata.parquet         # per-frame state + intrinsics + extrinsics
└── manifest.json            # dataset-level config (recipe, n_frames, seed)
```

Plug it into a LeRobotDataset / WebDataset / TFRecord training loop.

## Common randomizers

| Knob | Distribution | Effect |
|---|---|---|
| `<object>.position` | `uniform` / `gaussian` | Object spawn position jitter |
| `<object>.orientation` | `uniform_quat` | Random spawn rotation |
| `<object>.color` | `uniform_rgb` | Material color in [0, 1] sRGB |
| `<object>.material` | `choice` | Pick from a set of MDL materials |
| `<robot>.joint_positions` | `uniform_around_default` | Robot configuration jitter |
| `lighting.intensity` | `uniform` / `gaussian` | Per-light intensity |
| `lighting.position` | `uniform_sphere` | Per-light position |
| `lighting.temperature` | `uniform` (in Kelvin) | Color temperature |
| `camera.position` | `gaussian` | Camera-pose jitter |
| `camera.focal_length_mm` | `uniform` | Lens choice randomization |
| `background.skybox` | `choice` | HDR skybox swap |

Each randomizer maps to a Replicator `rep.modify.attribute(...)` /
`rep.randomizer.materials(...)` call under the hood; the dict shape is
the part of the API that's stable across Replicator versions.

## Headless vs. RTX

For RGB at scale, run `render_mode="rtx_pathtracing"` with `headless=True`
on a GPU-only host (L4 / A100 / H100). Path-traced takes ~100-500 ms /
frame at 720p; `rtx_realtime` is ~10 ms / frame and good enough when
photoreal lighting is not the goal.

For ground-truth-only datasets (depth + segmentation, no RGB),
`render_mode="rtx_realtime"` is almost always the right answer —
annotators run on the same RTX path even when you're not consuming RGB.

## Where the example lives

When R9 lands, the runnable demo is:

```bash
python examples/isaac/isaac_replicator_synthdata.py \
    --robot-usd /path/to/franka.usda \
    --n-frames 1000 \
    --output-dir /data/synth/cube-pickup-1k
```

It walks the same recipe shape this page documents and prints a
manifest + per-annotator file count when done. See
[Examples → Overview](../examples/overview.md) for the full driver list.

## Next

- [World Building](world-building.md) — `add_robot` / `add_object` /
  `add_camera` reference.
- [Backends → Isaac Sim](../backends/isaac.md) — full backend reference,
  including the `is_available()` diagnostic.
- [API Reference](../api-reference.md) — `replicate(...)` signature.
