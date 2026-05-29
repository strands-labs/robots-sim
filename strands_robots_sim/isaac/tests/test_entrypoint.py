"""Entry-point + lazy-import tests for the Isaac backend skeleton.

Pins the three pieces of packaging surface that the Isaac backend
relies on:

1. The ``[isaac]`` extra is declared in ``pyproject.toml`` and pulls in
   the pip-installable subset of Isaac Sim's runtime deps (``usd-core``,
   ``warp-lang``, ``pytest``).
2. The ``isaac`` and ``isaac_sim`` entry points under
   ``strands_robots.backends`` resolve to
   ``strands_robots_sim.isaac.simulation:IsaacSimulation``.
3. ``import strands_robots_sim.isaac`` is a PEP 562 lazy stub: it loads
   without pulling any ``omni.*`` modules into ``sys.modules``.

Contracts that depend on the simulation module itself (``SimEngine``
subclassing, abstract-method completeness, ``is_available()`` return
shape, no-GPU constructor) are covered in ``test_unit.py``.

Run with:: pytest strands_robots_sim/isaac/tests/test_entrypoint.py -v
"""

from __future__ import annotations

import importlib.metadata
import pathlib

import pytest

_PYPROJECT = pathlib.Path(__file__).resolve().parents[3] / "pyproject.toml"


class TestEntryPointDeclaration:
    """Validate that ``strands_robots.backends`` entry points are declared."""

    def test_pyproject_exists(self):
        assert _PYPROJECT.exists(), f"pyproject.toml not found at {_PYPROJECT}"

    def test_isaac_entry_point_declared_in_pyproject(self):
        """``isaac`` entry point points at the simulation module."""
        content = _PYPROJECT.read_text()
        assert 'isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content, (
            'Expected `isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"` '
            'under [project.entry-points."strands_robots.backends"] in pyproject.toml.'
        )

    def test_isaac_sim_alias_entry_point_declared_in_pyproject(self):
        """``isaac_sim`` alias is declared alongside ``isaac``."""
        content = _PYPROJECT.read_text()
        assert 'isaac_sim = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content, (
            "Expected `isaac_sim` alias entry point alongside `isaac` so users can write either "
            '`create_simulation("isaac")` or `create_simulation("isaac_sim")`.'
        )

    def test_isaac_extra_declared_in_pyproject(self):
        """``[project.optional-dependencies] isaac = [...]`` extra exists."""
        content = _PYPROJECT.read_text()
        assert "\nisaac = [" in content or "\nisaac=[" in content, (
            "Expected `isaac = [...]` under [project.optional-dependencies] declaring "
            "the pip-installable subset of Isaac Sim's runtime deps (usd-core, warp-lang, pytest)."
        )

    def test_isaac_extra_includes_pytest(self):
        """``[isaac]`` ships pytest so ``pip install '.[isaac]'`` is enough to run the suite."""
        content = _PYPROJECT.read_text()
        # crude but durable: locate the [isaac] block and check its body
        idx = content.find("\nisaac = [")
        assert idx != -1, "[isaac] extras block not found"
        block_end = content.find("]", idx)
        block = content[idx:block_end]
        assert "pytest" in block, (
            "[isaac] extras must include pytest so `pip install '.[isaac]'` covers the test deps "
            "without requiring a separate dev extra on CI hosts."
        )

    def test_entry_points_visible_via_importlib_metadata_when_installed(self):
        """If the package is pip-installed in this env, entry points are discoverable."""
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                backend_eps = list(eps.select(group="strands_robots.backends"))
            else:
                backend_eps = eps.get("strands_robots.backends", [])
        except Exception as exc:  # pragma: no cover - defensive
            pytest.skip(f"importlib.metadata unavailable: {exc}")

        if not backend_eps:
            pytest.skip(
                "Package not installed (no entry points discoverable). "
                "Run `pip install -e .` to validate this assertion locally."
            )

        names = {ep.name for ep in backend_eps}
        if "isaac" not in names and "isaac_sim" not in names:
            pytest.skip(
                "Package installed but entry-point cache is stale -- reinstall after "
                "pyproject.toml change: `pip install -e . --force-reinstall --no-deps`."
            )

        for ep in backend_eps:
            if ep.name in {"isaac", "isaac_sim"}:
                assert ep.value == "strands_robots_sim.isaac.simulation:IsaacSimulation", (
                    f"Entry point {ep.name!r} resolves to {ep.value!r}; expected "
                    "'strands_robots_sim.isaac.simulation:IsaacSimulation'."
                )


class TestLazyImportSurface:
    """Validate the PEP 562 lazy-import contract on the ``isaac`` subpackage."""

    def test_import_isaac_does_not_load_omni(self):
        """Importing ``strands_robots_sim.isaac`` adds zero ``omni.*`` modules."""
        import sys

        before = {k for k in sys.modules if k.startswith("omni")}
        import strands_robots_sim.isaac  # noqa: F401

        added = {k for k in sys.modules if k.startswith("omni")} - before
        assert added == set(), (
            f"Importing strands_robots_sim.isaac loaded omni modules: {sorted(added)}. "
            "The PEP 562 lazy stub must defer `omni` resolution until an attribute is accessed."
        )

    def test_isaac_subpackage_exposes_lazy_attrs_in___all__(self):
        """``__all__`` advertises the planned public surface."""
        import strands_robots_sim.isaac as isaac_pkg

        assert "IsaacSimulation" in isaac_pkg.__all__
        assert "IsaacConfig" in isaac_pkg.__all__

    def test_unknown_attr_raises_attributeerror(self):
        """Unknown attribute access raises AttributeError, not ImportError."""
        import strands_robots_sim.isaac as isaac_pkg

        with pytest.raises(AttributeError, match="no attribute 'NotARealClass'"):
            _ = isaac_pkg.NotARealClass

    def test_dunder_getattr_is_present(self):
        """The PEP 562 hook is defined at module level."""
        import strands_robots_sim.isaac as isaac_pkg

        assert hasattr(
            isaac_pkg, "__getattr__"
        ), "PEP 562 module-level __getattr__ must be defined for lazy import to work."
        assert callable(isaac_pkg.__getattr__)
