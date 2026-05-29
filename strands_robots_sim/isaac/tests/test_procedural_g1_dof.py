"""Documentation honesty pin: G1 DOF count.

The ``unitree_g1`` procedural definition has 21 actuated joints (1 torso
+ 6 left leg + 6 right leg + 4 left arm + 4 right arm). This pin keeps
``procedural.py``'s G1 module docstring, the ``_build_unitree_g1`` builder
docstring, and the joint set itself in sync, so a future refactor that
changes one without the others fails loudly rather than letting the
docs drift back out of date.

Companion pins:

- ``test_phase1_doc_banner.py`` -- the Phase-1 status banner in
  ``docs/backends/isaac.md``.
- ``test_procedural_kinematic_guard.py`` -- the kinematic-tree topology
  invariant (each non-root link has exactly one inbound joint; G1 splits
  its 2-DOF compound joints through six intermediate massless link
  bodies). This file pins doc/comment honesty against the literal value;
  that file pins the invariant the literal value is asserting against.

Both companion pins run on every CI invocation; none are deferred.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROCEDURAL_PY = Path(__file__).resolve().parent.parent / "procedural.py"


class TestG1DOFCount:
    """Pin: G1 doc-string / inline-comment DOF count must match the joint set."""

    def test_g1_actual_joint_count_is_21(self) -> None:
        """The shipped G1 procedural definition has exactly 21 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("unitree_g1")
        assert robot.num_joints == 21, (
            f"unitree_g1 has {robot.num_joints} joints; if this changes, the "
            f"DOF count in procedural.py docstrings/comments and "
            f"docs/backends/isaac.md must be updated together."
        )

    def test_g1_module_docstring_advertises_21_not_29(self) -> None:
        """Module docstring must not claim 29-DOF (stale -- actual is 21)."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Look only at the module-level docstring (everything before the first
        # `from __future__` import, which is the canonical top-of-module
        # marker for this file).
        head = text.split("from __future__", 1)[0]
        assert "29-DOF" not in head and "29 DOF" not in head, (
            "procedural.py module docstring still claims 29-DOF for G1; the "
            "actual joint count is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in head or "21 DOF" in head, (
            "procedural.py module docstring no longer mentions the actual "
            "21-DOF count for G1; documentation must stay in sync with code."
        )

    def test_g1_builder_docstring_advertises_21_not_29(self) -> None:
        """``_build_unitree_g1`` docstring must not claim 29-DOF."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Find the def / docstring window for _build_unitree_g1 specifically.
        match = re.search(
            r"def\s+_build_unitree_g1\b.*?(?=\ndef |\Z)",
            text,
            flags=re.DOTALL,
        )
        assert match is not None, "could not locate _build_unitree_g1 in procedural.py"
        body = match.group(0)
        assert "29-DOF" not in body and "29 DOF" not in body, (
            "_build_unitree_g1 still mentions 29-DOF; the actual joint count "
            "is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in body or "21 DOF" in body, "_build_unitree_g1 no longer documents its actual 21-DOF count."
