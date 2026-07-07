"""Deprecation-warning-on-import pin for the deprecated package (robots-sim#168).

The Isaac Sim backend now ships as an in-tree builtin of ``strands-robots``,
so ``strands-robots-sim`` is being deprecated and archived (epic #167).
Importing the top-level package must emit a :class:`DeprecationWarning`
pointing users at the concrete migration command
(``pip install strands-robots[isaac]``) so downstream code sees the notice
before the final release + PyPI deprecation land.

These tests pin three properties of that warning:

1. Importing ``strands_robots_sim`` (fresh, not from ``sys.modules`` cache)
   raises exactly one ``DeprecationWarning``.
2. The message names the concrete migration command so the fix is
   copy-pasteable.
3. The warning does not tear down the module's public surface — ``__version__``
   is still importable after the warning fires.

Run with:: pytest strands_robots_sim/isaac/tests/test_deprecation_warning.py -v
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


def _reimport_top_level():
    """Import ``strands_robots_sim`` fresh so the module-level warning fires.

    ``import`` is a no-op once a module sits in ``sys.modules`` (and its body,
    including ``warnings.warn(...)``, has already run at interpreter/pytest
    start-up). Drop the cached entry and re-import so the ``DeprecationWarning``
    is emitted inside the ``catch_warnings`` block under test.
    """
    sys.modules.pop("strands_robots_sim", None)
    return importlib.import_module("strands_robots_sim")


class TestDeprecationWarningOnImport:
    """Pin: importing the deprecated package warns with a migration hint."""

    def test_import_emits_deprecation_warning(self):
        """A fresh import raises exactly one ``DeprecationWarning``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _reimport_top_level()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1, (
            "Importing strands_robots_sim must emit exactly one "
            f"DeprecationWarning; got {len(deprecations)} "
            f"({[str(w.message) for w in deprecations]})."
        )

    def test_warning_names_migration_command(self):
        """The message points users at ``pip install strands-robots[isaac]``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _reimport_top_level()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "no DeprecationWarning emitted on import"
        message = str(deprecations[0].message)
        assert "pip install strands-robots[isaac]" in message, (
            "DeprecationWarning must name the concrete migration command "
            f"`pip install strands-robots[isaac]`; got: {message!r}"
        )
        assert "deprecated" in message.lower(), (
            "DeprecationWarning message should state the package is deprecated; " f"got: {message!r}"
        )

    def test_public_surface_survives_the_warning(self):
        """``__version__`` remains importable after the warning fires."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = _reimport_top_level()

        assert isinstance(module.__version__, str) and module.__version__, (
            "strands_robots_sim.__version__ must still resolve to a non-empty "
            "string after the deprecation warning is emitted."
        )
        assert "__version__" in module.__all__


@pytest.fixture(autouse=True)
def _restore_top_level_module():
    """Leave ``sys.modules`` as we found it so sibling tests import cleanly.

    Each test drops + re-imports ``strands_robots_sim``; ensure a clean,
    warning-suppressed instance is cached again afterwards so unrelated
    tests that ``import strands_robots_sim`` don't re-trigger the warning
    or observe a half-torn-down module.
    """
    yield
    sys.modules.pop("strands_robots_sim", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module("strands_robots_sim")
