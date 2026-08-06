"""strands-robots-sim — heavy NVIDIA simulation backends for strands-robots.

.. deprecated::
   This package is deprecated and will be archived (robots-sim#167). The
   Isaac Sim backend now ships as an in-tree builtin of ``strands-robots``;
   install it with ``pip install strands-robots[isaac]``. Importing this
   package emits a :class:`DeprecationWarning`.

As of 0.2.0 this package is a re-scoped plugin host. The legacy ``SimEnv``,
``SteppedSimEnv``, Libero-direct environment layer, GR00T policy client, and
``gr00t_inference`` AgentTool have all been removed — that lightweight
MuJoCo + LIBERO + GR00T code path now lives in
`strands-labs/robots <https://github.com/strands-labs/robots>`_, accessible
via the ``Simulation`` AgentTool, the ``LiberoAdapter`` benchmark plugin, and
``strands_robots.tools.gr00t_inference``.

This module is currently a no-op stub. The heavy GPU-only backend
(``IsaacSimulation``) registers itself through ``strands-robots`` entry
points; see the umbrella issue
https://github.com/strands-labs/robots-sim/issues/8.

See ``examples/MIGRATION.md`` for the old-API → new-API mapping.
"""

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("strands-robots-sim")
except PackageNotFoundError:
    # Editable install before metadata is generated, or running from a
    # working tree without ``pip install -e .`` having been run yet.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

# ---------------------------------------------------------------------------
# Deprecation notice (robots-sim#167 / #168)
#
# The Isaac Sim backend now lives in ``strands-labs/robots`` as an in-tree
# builtin. This package is being deprecated and will be archived; emit a
# ``DeprecationWarning`` on import so existing users are pointed at the new
# install path before the final release + PyPI deprecation land.
#
# ``stacklevel=2`` attributes the warning to the caller's ``import`` line
# rather than to this module, so ``python -W all`` / pytest ``-W`` output
# points at the user's code. The message names the concrete migration
# command from the issue so the fix is copy-pasteable.
# ---------------------------------------------------------------------------
_DEPRECATION_MESSAGE = (
    "strands-robots-sim is deprecated and will be archived. The Isaac Sim "
    "backend now ships as an in-tree builtin of strands-robots; install it "
    "with `pip install strands-robots[isaac]`. "
    "See https://github.com/strands-labs/robots-sim/issues/167 for the "
    "deprecation plan and examples/MIGRATION.md for the API mapping."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

_LEGACY_REMOVED = {
    "SimEnv": (
        "`SimEnv` was removed in strands-robots-sim 0.2.0. "
        "Use `Simulation(...).evaluate_benchmark(benchmark_name='libero-<suite>-<task>', ...)` "
        "from `strands-robots` instead. See examples/MIGRATION.md."
    ),
    "SteppedSimEnv": (
        "`SteppedSimEnv` was removed in strands-robots-sim 0.2.0. "
        "Use `Simulation.start_policy(...)` + poll `get_state` / `render` "
        "from `strands-robots` instead. See examples/MIGRATION.md."
    ),
    "gr00t_inference": (
        "`gr00t_inference` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.tools.gr00t_inference import gr00t_inference` instead. "
        "See examples/MIGRATION.md."
    ),
    "Gr00tPolicy": (
        "`Gr00tPolicy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies.groot import Gr00tPolicy` instead. "
        "See examples/MIGRATION.md."
    ),
    "Policy": (
        "`Policy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import Policy` instead. "
        "See examples/MIGRATION.md."
    ),
    "MockPolicy": (
        "`MockPolicy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import MockPolicy` instead. "
        "See examples/MIGRATION.md."
    ),
    "create_policy": (
        "`create_policy` was removed in strands-robots-sim 0.2.0. "
        "Use `from strands_robots.policies import create_policy` instead. "
        "See examples/MIGRATION.md."
    ),
}


def __getattr__(name):  # PEP 562 module-level __getattr__
    """Surface a clear, actionable error for legacy import names.

    Raises ``ImportError`` (not ``AttributeError`` + ``DeprecationWarning``)
    so the message survives ``-W error::DeprecationWarning`` test envs that
    would otherwise mask the actionable hint with the warning's traceback.
    """
    if name in _LEGACY_REMOVED:
        raise ImportError(_LEGACY_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
