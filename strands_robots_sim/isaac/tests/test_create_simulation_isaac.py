"""Regression tests for the documented Isaac quickstart path (#97, #139).

`docs/index.md`, `README.md`, `docs/getting-started/quickstart.md`, and
`docs/simulation/overview.md` all open with the factory path::

    from strands_robots.simulation import create_simulation
    sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)

That path now works: ``strands-robots>=0.4.1`` (the pinned floor, shipped via
robots#131) walks the ``strands_robots.backends`` entry-point group from its
``create_simulation`` factory, so ``create_simulation("isaac", ...)`` resolves
to this package's ``IsaacSimulation`` — the same UX as
``create_simulation("mujoco")``. The direct constructor stays supported for
callers that want the ``IsaacConfig`` object in hand::

    from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig
    sim = IsaacSimulation(IsaacConfig(render_mode="rtx_realtime", headless=True))

These tests pin both halves of that contract so neither can silently drift:

1. ``TestDocumentedDirectConstructor`` — the direct-constructor path must keep
   working on a CPU-only box (no ``omni.*`` import at construct time),
   accepting both the ``IsaacConfig`` and the kwargs forms.
2. ``TestCreateSimulationIsaacDiscovery`` — the documented factory path:
   ``create_simulation("isaac", ...)`` must resolve to ``IsaacSimulation``
   against the pinned ``strands-robots>=0.4.1`` floor.

Run with::

    pytest strands_robots_sim/isaac/tests/test_create_simulation_isaac.py -v
"""

from __future__ import annotations

import sys

import pytest


class TestDocumentedDirectConstructor:
    """The constructor the quickstart docs show must keep working (#97)."""

    def test_isaac_subpackage_exports_constructor_symbols(self):
        """``from strands_robots_sim.isaac import IsaacSimulation, IsaacConfig`` works."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        assert IsaacSimulation is not None
        assert IsaacConfig is not None

    def test_construct_from_isaac_config(self):
        """The documented ``IsaacSimulation(IsaacConfig(...))`` form constructs."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        sim = IsaacSimulation(IsaacConfig(render_mode="rtx_realtime", headless=True))
        assert sim._config.render_mode == "rtx_realtime"
        assert sim._config.headless is True

    def test_construct_from_kwargs(self):
        """The kwargs form ``IsaacSimulation(render_mode=..., headless=...)`` constructs.

        These are the same kwargs that flow through
        ``create_simulation("isaac", ...)`` into ``IsaacConfig``.
        """
        from strands_robots_sim.isaac import IsaacSimulation

        sim = IsaacSimulation(render_mode="rtx_pathtracing", headless=True)
        assert sim._config.render_mode == "rtx_pathtracing"
        assert sim._config.headless is True

    def test_constructor_is_cpu_safe(self):
        """Constructing ``IsaacSimulation`` must not import any ``omni.*`` module.

        The quickstart is meant to be copy-pasteable on a CPU-only dev box
        up to the ``create_world()`` call. If construction eagerly imported
        ``omni`` the snippet would explode at line 1 on every non-GPU host.
        """
        before = {k for k in sys.modules if k.startswith("omni")}

        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        IsaacSimulation(IsaacConfig(headless=True))

        added = {k for k in sys.modules if k.startswith("omni")} - before
        assert added == set(), f"Constructing IsaacSimulation imported omni modules: {sorted(added)}"

    def test_is_a_simengine_subclass(self):
        """``IsaacSimulation`` is a ``SimEngine`` so it's drop-in for the agent loop."""
        # ``strands-robots`` is the runtime dep but is intentionally NOT
        # installed in the lint/test hatch env (skip-install=true). Skip
        # cleanly there; the assertion runs in any env that has it.
        sim_mod = pytest.importorskip("strands_robots.simulation")

        from strands_robots_sim.isaac import IsaacSimulation

        assert issubclass(IsaacSimulation, sim_mod.SimEngine)


def _isaac_via_factory():
    """Call ``create_simulation('isaac')`` and return (sim, error).

    Exactly one of the two is non-None. Construction kwargs match the
    documented quickstart (headless so no display / GPU is needed for the
    resolution step itself).
    """
    from strands_robots.simulation import create_simulation

    try:
        sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
        return sim, None
    except Exception as exc:  # noqa: BLE001 - we classify it below
        return None, exc


def _factory_walks_entry_points() -> bool:
    """True if the installed ``strands-robots`` walks ``strands_robots.backends``.

    The entry-point walker landed in ``strands-robots>=0.4.1`` (robots#131);
    the pinned floor requires it. But the lint/test hatch env is
    ``skip-install=true`` and CI/dev boxes may still have an older
    ``strands-robots`` in the ambient Python, so this probe lets the positive
    assertion skip cleanly rather than fail on a stale, below-floor install.
    """
    try:
        from strands_robots.simulation import list_backends
    except Exception:  # pragma: no cover - strands-robots absent
        return False
    try:
        # The walker surfaces plugin backends (e.g. "isaac") in list_backends();
        # a pre-0.4.1 factory only lists the built-in mujoco aliases.
        return "isaac" in set(list_backends())
    except Exception:  # pragma: no cover - defensive
        return False


class TestCreateSimulationIsaacDiscovery:
    """Pin the documented ``create_simulation('isaac')`` contract (#97, #139).

    ``strands-robots>=0.4.1`` (the pinned floor, robots#131) walks the
    ``strands_robots.backends`` entry-point group, so ``create_simulation``
    resolves ``"isaac"`` to this package's ``IsaacSimulation``. This test
    asserts that documented behavior. It skips cleanly when ``strands-robots``
    is absent or below the floor (e.g. the ``skip-install=true`` hatch env or
    a stale ambient install) rather than failing spuriously.
    """

    def test_create_simulation_isaac_resolves_to_isaac_simulation(self):
        # ``strands-robots`` provides ``create_simulation`` but is not
        # installed in the lint/test hatch env (skip-install=true). Skip
        # cleanly there; runs anywhere the runtime dep is present.
        pytest.importorskip("strands_robots.simulation")

        if not _factory_walks_entry_points():
            pytest.skip(
                "Installed strands-robots is below the >=0.4.1 floor that walks "
                "strands_robots.backends (robots#131). Upgrade to validate the "
                "factory path: pip install 'strands-robots>=0.4.1'."
            )

        import strands_robots_sim  # noqa: F401 - parity with the doc snippet

        sim, err = _isaac_via_factory()

        assert err is None, (
            f"create_simulation('isaac') raised {type(err).__name__ if err else None}: {err!r}; "
            "expected it to resolve to IsaacSimulation against strands-robots>=0.4.1."
        )

        from strands_robots_sim.isaac import IsaacSimulation

        assert isinstance(sim, IsaacSimulation), (
            f"create_simulation('isaac') resolved to {type(sim)!r}; expected IsaacSimulation. "
            "strands-robots walked strands_robots.backends but routed 'isaac' to the wrong class."
        )
        # The documented kwargs must reach IsaacConfig.
        assert sim._config.render_mode == "rtx_realtime"
        assert sim._config.headless is True

    def test_entry_point_is_registered(self):
        """The ``isaac`` entry point is declared/discoverable.

        This is the plumbing the upstream walker consumes (see
        docs/architecture.md). Skips cleanly if the package isn't pip-installed
        in this env.
        """
        import importlib.metadata

        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            backend_eps = list(eps.select(group="strands_robots.backends"))
        else:  # pragma: no cover - Python <3.10 shape
            backend_eps = eps.get("strands_robots.backends", [])

        names = {ep.name for ep in backend_eps}
        if not backend_eps or "isaac" not in names:
            pytest.skip(
                "strands-robots-sim not pip-installed (or entry-point cache stale). "
                "Run `pip install -e .` to validate the entry point locally."
            )

        isaac_ep = next(ep for ep in backend_eps if ep.name == "isaac")
        assert isaac_ep.value == "strands_robots_sim.isaac.simulation:IsaacSimulation"
