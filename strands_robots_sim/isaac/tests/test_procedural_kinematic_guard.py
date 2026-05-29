"""Fail-first kinematic-tree pin: ``_validate_kinematic_tree``.

This pin enforces the fail-first contract for procedural-builder validation:

1. ``_validate_kinematic_tree`` runs unconditionally on every procedural
   builder. There is no env-var escape hatch -- a knowingly-broken robot
   has no good use case in this package, so callers cannot opt out.
2. All three shipped procedural robots (``so100``, ``panda``, ``unitree_g1``)
   build cleanly under that contract -- so the G1 builder must keep its
   topology a valid tree (intermediate massless link bodies between the
   2-DOF compound joints), not just punt the check off behind a flag.
3. A topology defect injected at builder time still surfaces with the body
   indices + joint names so the offender is obvious from the traceback alone.

A future refactor that re-introduces a duplicate ``(parent, child)`` edge
or relaxes the guard will fail this pin rather than silently regressing
into a robot that can't instantiate.
"""

from __future__ import annotations

import pytest

from strands_robots_sim.isaac import procedural
from strands_robots_sim.isaac.procedural import (
    BodyDef,
    JointDef,
    ProceduralRobot,
    _build_panda,
    _build_so100,
    _build_unitree_g1,
    _validate_kinematic_tree,
    get_procedural_robot,
)


class TestKinematicGuardFailFirst:
    """Pin: ``_validate_kinematic_tree`` validates by default; no env-var gating."""

    def test_g1_builds_cleanly_by_default(self) -> None:
        """G1 must build without raising under the default fail-first contract."""
        robot = get_procedural_robot("unitree_g1")
        assert robot is not None
        assert robot.name == "unitree_g1"
        # Six 2-DOF compound joints (hips, ankles, shoulder-yaw/elbow on each arm)
        # are split through massless intermediate ``*_link`` bodies, so the
        # actuated joint count stays 21.
        assert robot.num_joints == 21

    def test_g1_topology_has_no_duplicate_edges(self) -> None:
        """The shipped G1 kinematic graph must already be a valid tree."""
        robot = _build_unitree_g1()
        edges = [(j.parent_body, j.child_body) for j in robot.joints]
        assert len(edges) == len(set(edges)), (
            f"G1 kinematic graph has duplicate (parent, child) edges: "
            f"{[e for e in edges if edges.count(e) > 1]}. The builder must "
            f"insert intermediate massless link bodies for compound joints."
        )

    def test_so100_and_panda_build_cleanly(self) -> None:
        """SO-100 and Panda are valid trees under the default guard."""
        so100 = _build_so100()
        panda = _build_panda()
        assert so100.name == "so100"
        assert panda.name == "panda"

    def test_guard_raises_on_injected_duplicate_edge(self) -> None:
        """A robot with a duplicate edge must raise with diagnostic context."""
        bad = ProceduralRobot(
            name="broken_robot",
            bodies=[
                BodyDef(name="root"),
                BodyDef(name="child"),
            ],
            joints=[
                JointDef(name="joint_a", parent_body=0, child_body=1, axis=(1, 0, 0)),
                JointDef(name="joint_b", parent_body=0, child_body=1, axis=(0, 1, 0)),
            ],
        )
        with pytest.raises(ValueError) as excinfo:
            _validate_kinematic_tree(bad)
        msg = str(excinfo.value)
        assert "broken_robot" in msg
        assert "duplicate parent->child body edges" in msg
        # Both offending joint names must be named so the caller knows which
        # joints to split through an intermediate massless link body.
        assert "joint_a" in msg
        assert "joint_b" in msg

    def test_guard_has_no_env_var_escape_hatch(self) -> None:
        """The fail-first guard must not be gateable by an env var.

        Pin ensures ``procedural.py`` never re-introduces an opt-in switch:
        a knowingly-broken robot has no good use case in this package, so
        the validation is unconditional. A future refactor that adds an env
        var here will fail this test.
        """
        import inspect

        source = inspect.getsource(procedural)
        assert "STRANDS_ISAAC_VALIDATE_KINEMATICS" not in source, (
            "procedural.py re-introduced the STRANDS_ISAAC_VALIDATE_KINEMATICS "
            "env-var gate; the fail-first guard must run unconditionally."
        )
        assert "os.environ" not in source, (
            "procedural.py reads from os.environ; the kinematic-tree guard " "must not be conditioned on env state."
        )
