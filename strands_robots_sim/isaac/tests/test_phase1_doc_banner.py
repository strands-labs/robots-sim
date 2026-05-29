"""Documentation honesty pin: Phase 1 status banner in docs/backends/isaac.md.

The Phase 1 skeleton silently no-ops the data plane: ``add_robot`` on
the procedural branch, ``_load_usd_robot``, ``_load_urdf_robot``,
``add_object``, ``add_camera``, and ``replicate`` all return
``status: "success"`` without instantiating the underlying USD prim or
articulation handle. ``docs/backends/isaac.md`` discloses this in a
``Phase 1 status`` banner before the Installation section, so the
disclosure precedes the documented call sequence rather than appearing
after a user would have already executed the silent no-ops.

This pin enforces three properties of the banner:

1. The banner exists.
2. It precedes the ``## Installation`` heading.
3. It enumerates the affected call surfaces (``add_robot``, ``replicate``,
   ``get_observation``).

Companion pin for the G1 DOF count lives in ``test_procedural_g1_dof.py``;
this file pins only the docs-side concerns.
"""

from __future__ import annotations

from pathlib import Path

_ISAAC_DOCS = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "backends" / "isaac.md"


class TestIsaacDocsPhase1Banner:
    """Pin: ``docs/backends/isaac.md`` must disclose Phase 1 data-plane no-ops."""

    def test_isaac_docs_file_exists(self) -> None:
        assert _ISAAC_DOCS.is_file(), f"missing Isaac doc page at {_ISAAC_DOCS}"

    def test_phase1_banner_present_before_installation(self) -> None:
        """Banner must appear before the Installation / Quick Start sections.

        The disclosure precedes any procedural docs the user would otherwise
        execute, so a maintainer who reads only the Quick Start sees the
        silent-no-op caveat first.
        """
        text = _ISAAC_DOCS.read_text(encoding="utf-8")

        # The banner must appear AND must appear before the Installation
        # section header (so the disclosure precedes any procedural docs the
        # user would otherwise execute).
        banner_marker = "Phase 1 status"
        install_marker = "## Installation"

        assert banner_marker in text, (
            "docs/backends/isaac.md missing the Phase 1 status disclosure "
            "banner. The doc's Quick Start otherwise executes a code path "
            "that silently no-ops on a real Isaac Sim install -- so the "
            "banner must precede the documented call sequence."
        )
        assert (
            install_marker in text
        ), "docs/backends/isaac.md missing the Installation section -- doc structure has changed; pin needs review."
        assert text.find(banner_marker) < text.find(install_marker), (
            "Phase 1 banner must precede the Installation section so the "
            "user sees the disclosure before following the install / quick-"
            "start steps."
        )

    def test_phase1_banner_names_the_silent_methods(self) -> None:
        """Banner must enumerate the Phase-1 silent-success methods.

        Without naming the methods, a future maintainer who reads only the
        banner won't know which API surfaces are affected, and the disclosure
        becomes a vague hedge.
        """
        text = _ISAAC_DOCS.read_text(encoding="utf-8")
        # Slice the banner block (`> **Phase 1 status...**` paragraph).
        banner_start = text.find("Phase 1 status")
        # Banner is one paragraph; cut at the next `## ` heading.
        banner_end = text.find("##", banner_start)
        assert banner_end > banner_start, "could not locate end of banner block"
        banner = text[banner_start:banner_end]

        for needed in ("add_robot", "replicate", "get_observation"):
            assert needed in banner, (
                f"Phase 1 banner does not mention `{needed}`; the disclosure "
                f"must enumerate the silent-success methods so users know "
                f"which call sites are affected."
            )
