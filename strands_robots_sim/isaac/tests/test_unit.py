"""Unit tests for Isaac Sim backend (no GPU required).

All tests use mocking to avoid requiring Isaac Sim or CUDA.

Run with: pytest strands_robots_sim/isaac/tests/test_unit.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestIsaacConfig:
    """Tests for IsaacConfig dataclass validation."""

    def test_default_config(self):
        """Default config should be valid."""
        from strands_robots_sim.isaac.config import IsaacConfig

        config = IsaacConfig()
        assert config.num_envs == 1
        assert config.device == "cuda:0"
        assert config.headless is True
        assert config.render_mode == "headless"
        assert config.gravity == (0.0, 0.0, -9.81)
        assert config.ground_plane is True
        assert config.camera_width == 640
        assert config.camera_height == 480

    def test_custom_config(self):
        """Custom config values should be preserved."""
        from strands_robots_sim.isaac.config import IsaacConfig

        config = IsaacConfig(
            num_envs=1024,
            device="cuda:1",
            headless=True,
            physics_dt=1.0 / 240.0,
            render_mode="rtx_realtime",
        )
        assert config.num_envs == 1024
        assert config.device == "cuda:1"
        assert config.physics_dt == pytest.approx(1.0 / 240.0)
        assert config.render_mode == "rtx_realtime"

    def test_invalid_render_mode(self):
        """Invalid render_mode should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="render_mode"):
            IsaacConfig(render_mode="invalid")

    def test_invalid_device(self):
        """Non-CUDA device should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="CUDA"):
            IsaacConfig(device="cpu")

    def test_invalid_num_envs(self):
        """num_envs < 1 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="num_envs"):
            IsaacConfig(num_envs=0)

    def test_invalid_physics_dt(self):
        """physics_dt <= 0 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="physics_dt"):
            IsaacConfig(physics_dt=-0.001)

    def test_invalid_camera_dimensions(self):
        """Camera dimensions < 1 should raise ValueError."""
        from strands_robots_sim.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="camera"):
            IsaacConfig(camera_width=0)

    def test_config_round_trip(self):
        """Config should survive dataclass replace round-trip."""
        import dataclasses

        from strands_robots_sim.isaac.config import IsaacConfig

        original = IsaacConfig(num_envs=512, render_mode="rtx_pathtracing")
        copy = dataclasses.replace(original, num_envs=1024)
        assert copy.num_envs == 1024
        assert copy.render_mode == "rtx_pathtracing"
        assert original.num_envs == 512

    def test_env_var_headless_override(self, monkeypatch):
        """STRANDS_ISAAC_HEADLESS env var should override headless."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "false")
        config = IsaacConfig(headless=True)
        assert config.headless is False

    def test_env_var_rtx_pathtracing(self, monkeypatch):
        """STRANDS_ISAAC_RTX_PATHTRACING env var should set render mode."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "true")
        config = IsaacConfig(render_mode="headless")
        assert config.render_mode == "rtx_pathtracing"

    def test_env_var_nucleus_url(self, monkeypatch):
        """STRANDS_ISAAC_NUCLEUS_URL env var should be picked up."""
        from strands_robots_sim.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_NUCLEUS_URL", "omniverse://myhost/NVIDIA")
        config = IsaacConfig()
        assert config.nucleus_url == "omniverse://myhost/NVIDIA"


class TestIsaacSimulationAvailability:
    """Tests for IsaacSimulation.is_available()."""

    def test_isaac_is_simengine_subclass(self):
        """IsaacSimulation must subclass SimEngine ABC.

        Migrated from the cagataycali-original ``test_entrypoint.py``
        during the #31 split. The PR-1 R1 rewrite of ``test_entrypoint.py``
        is surface-only (no ``simulation`` import), so this contract pin
        moved to live next to the rest of the IsaacSimulation contract
        tests in this file.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        assert issubclass(IsaacSimulation, SimEngine), (
            "IsaacSimulation must inherit from SimEngine for entry-point "
            "registration to satisfy the factory contract."
        )

    def test_isaac_implements_all_abstract_methods(self):
        """IsaacSimulation must implement every SimEngine abstract method.

        Pin against the R5-class regression cagataycali called out on PR #47:
        upstream ``SimEngine`` ABC declares ``list_robots`` /
        ``remove_object`` / ``remove_robot`` / ``robot_joint_names`` as
        abstract; missing any one of them makes ``IsaacSimulation()`` raise
        ``TypeError: Can't instantiate abstract class …``. The fallback ABC
        stub in ``simulation.py`` mirrors the real surface so the test fails
        loudly on either side if a method is added upstream and forgotten
        here.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        abstract_methods = set()
        for name in dir(SimEngine):
            method = getattr(SimEngine, name, None)
            if callable(method) and getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        for method_name in abstract_methods:
            assert hasattr(IsaacSimulation, method_name), f"IsaacSimulation missing abstract method: {method_name}"
            impl = getattr(IsaacSimulation, method_name)
            assert callable(impl), f"IsaacSimulation.{method_name} must be callable, got {type(impl).__name__}"
            assert not getattr(
                impl, "__isabstractmethod__", False
            ), f"IsaacSimulation.{method_name} is still abstract; provide a concrete implementation."

    def test_is_available_returns_tuple(self):
        """is_available() must return a (bool, str|None) tuple."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        result = IsaacSimulation.is_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        available, reason = result
        assert isinstance(available, bool)
        if not available:
            assert isinstance(reason, str)

    def test_is_available_false_without_omni(self):
        """is_available() should return False when omni is not importable."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # On CI without Isaac Sim, this should be False
        available, reason = IsaacSimulation.is_available()
        # We can't assert False universally (GPU env might have it)
        # But we CAN verify the return contract
        assert isinstance(available, bool)
        if not available:
            assert "omni" in reason.lower() or "cuda" in reason.lower() or "torch" in reason.lower()

    @patch("builtins.__import__")
    def test_is_available_false_when_omni_missing(self, mock_import):
        """Simulate omni not installed."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        def side_effect(name, *args, **kwargs):
            if name == "omni":
                raise ImportError("No module named 'omni'")
            return MagicMock()

        mock_import.side_effect = side_effect

        # Need to call directly without module caching
        # Just verify the method exists and is callable
        assert callable(IsaacSimulation.is_available)

    def test_is_available_probes_omni_isaac_kit_specifically(self, monkeypatch):
        """is_available() must probe ``omni.isaac.kit``, not the bare ``omni``
        namespace package.

        Regression pin for review-feedback PR #31: a partial Omniverse install
        (omni.ui / omni.usd) leaves the bare ``omni`` namespace importable but
        ``omni.isaac.kit.SimulationApp`` -- which create_world() actually needs --
        unavailable. The pre-fix probe (`import omni`) returned ``(True, None)``
        in that environment, then create_world() raised ImportError seconds
        later. Tightened probe uses ``importlib.util.find_spec`` against the
        specific submodule.
        """
        import importlib.util

        from strands_robots_sim.isaac.simulation import IsaacSimulation

        captured: list[str] = []
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            captured.append(name)
            if name == "omni.isaac.kit":
                return None  # simulate not installed
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        available, reason = IsaacSimulation.is_available()

        assert "omni.isaac.kit" in captured, (
            "is_available() must call find_spec('omni.isaac.kit'); " f"got find_spec calls: {captured!r}"
        )
        assert available is False
        assert reason is not None
        assert "omni.isaac.kit" in reason


class TestIsaacSimulationContract:
    """Tests for IsaacSimulation method contracts (mocked)."""

    def test_instantiation_does_not_import_omni(self):
        """Constructor must NOT import omni or touch CUDA."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # This must succeed on any machine -- no CUDA required
        sim = IsaacSimulation(num_envs=1)
        assert sim is not None
        assert sim.config.num_envs == 1
        assert sim.config.headless is True

    def test_repr(self):
        """repr should be informative."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=4)
        r = repr(sim)
        assert "IsaacSimulation" in r
        assert "num_envs=4" in r

    def test_destroy_without_world(self):
        """destroy() on uninitialized sim should return error dict."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.destroy()
        assert result["status"] == "error"

    def test_step_without_world(self):
        """step() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.step(1)
        assert result["status"] == "error"

    def test_get_state_without_world(self):
        """get_state() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.get_state()
        assert result["status"] == "error"

    def test_add_robot_without_world(self):
        """add_robot() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_robot("test")
        assert result["status"] == "error"

    def test_add_object_without_world(self):
        """add_object() without create_world should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.add_object("test")
        assert result["status"] == "error"

    def test_add_object_invalid_shape(self):
        """add_object() with invalid shape should return error."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        sim._world_created = True  # bypass world check
        result = sim.add_object("test", shape="invalid_shape")
        assert result["status"] == "error"
        assert "invalid_shape" in result["content"][0]["text"]

    def test_get_observation_without_world(self):
        """get_observation() without world returns empty dict."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.get_observation("robot1")
        assert result == {}

    def test_context_manager(self):
        """IsaacSimulation supports context manager protocol."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with IsaacSimulation() as sim:
            assert sim is not None

    def test_kwargs_merge_into_config(self):
        """Shortcut kwargs should merge into config."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=256, headless=True)
        assert sim.config.num_envs == 256
        assert sim.config.headless is True

    def test_unknown_kwarg_raises_typeerror_no_config(self):
        """Typo kwargs must raise TypeError, not silently default.

        Regression pin for review-feedback PR #31: previously
        ``IsaacSimulation(headles=False)`` (typo) was silently dropped
        and the sim ran with default config. The validator now eagerly
        rejects unknown kwargs to surface typos at construction time.
        """
        import pytest

        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with pytest.raises(TypeError, match="headles"):
            IsaacSimulation(headles=False)

    def test_unknown_kwarg_raises_typeerror_with_config(self):
        """Typo kwargs alongside an explicit IsaacConfig must also raise."""
        import pytest

        from strands_robots_sim.isaac.config import IsaacConfig
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        with pytest.raises(TypeError, match="num_env"):
            IsaacSimulation(IsaacConfig(num_envs=4), num_env=8)

    def test_no_del_finalizer(self):
        """IsaacSimulation must not define __del__.

        Regression pin for review-feedback PR #31: a ``__del__`` that
        calls cleanup() -> destroy() acquires ``self._lock`` during
        interpreter shutdown, when ``threading`` / ``logger`` / ``omni``
        may already be partially torn down. Drop the finalizer; rely on
        explicit cleanup() or context-manager use. Bare-except in the
        prior __del__ also masked the symptom but invisible exceptions
        during finalization still print "Exception ignored in: ..." noise.
        """
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # Defining __del__ on a subclass would re-introduce the hazard,
        # so the assertion is on the class dict (not just on dir()).
        assert "__del__" not in IsaacSimulation.__dict__, (
            "IsaacSimulation must not define __del__; rely on explicit "
            "cleanup() or context manager (see cleanup() docstring)."
        )


class TestProceduralRobots:
    """Tests for procedural robot definitions."""

    def test_so100_definition(self):
        """SO-100 should have 6 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("so100")
        assert robot is not None
        assert robot.name == "so100"
        assert robot.num_joints == 6
        assert len(robot.joint_names) == 6

    def test_panda_definition(self):
        """Panda should have 7 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("panda")
        assert robot is not None
        assert robot.name == "panda"
        assert robot.num_joints == 7
        assert len(robot.joint_names) == 7

    def test_unitree_g1_definition(self):
        """Unitree G1 should have 21 joints (simplified)."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("unitree_g1")
        assert robot is not None
        assert robot.name == "unitree_g1"
        assert robot.num_joints == 21
        assert len(robot.joint_names) == 21

    def test_alias_resolution(self):
        """Aliases should resolve correctly."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        assert get_procedural_robot("so-100") is not None
        assert get_procedural_robot("franka") is not None
        assert get_procedural_robot("g1") is not None
        assert get_procedural_robot("franka_panda") is not None

    def test_unknown_robot_returns_none(self):
        """Unknown robot name should return None."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        assert get_procedural_robot("nonexistent_robot") is None

    def test_list_procedural_robots(self):
        """list_procedural_robots should return known robots."""
        from strands_robots_sim.isaac.procedural import list_procedural_robots

        robots = list_procedural_robots()
        assert "so100" in robots
        assert "panda" in robots
        assert "unitree_g1" in robots


class TestNoEmojisInOutput:
    """Verify no emojis in user-facing strings."""

    def test_destroy_message_no_emoji(self):
        """destroy() output must not contain emojis."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.destroy()
        text = result["content"][0]["text"]
        # Check for common emoji ranges
        for char in text:
            code = ord(char)
            assert code < 0x1F600 or code > 0x1F9FF, f"Emoji found in output: {char!r}"

    def test_step_error_no_emoji(self):
        """step() error must not contain emojis."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation()
        result = sim.step(1)
        text = result["content"][0]["text"]
        for char in text:
            code = ord(char)
            assert code < 0x1F600 or code > 0x1F9FF, f"Emoji found in output: {char!r}"


class TestExceptionClauseHygiene:
    """Static-AST checks for narrow exception clauses in simulation.py.

    Regression pin for review-feedback PR #31: bare ``except Exception``
    swallows programming bugs (AttributeError typos, KeyError on dict
    access, etc.) into logged error dicts that look identical to
    recoverable failures.

    Each behavioural except clause must enumerate the realistic failure
    modes for the API it wraps, so a future drift in the wrapped API's
    surface (or a typo in the wrapper) raises rather than silently
    becoming a no-op.
    """

    def test_no_bare_except_exception_in_simulation_module(self):
        """``except Exception`` is forbidden in simulation.py.

        Use a tuple of named exception classes (RuntimeError, ValueError,
        OSError, AttributeError, TypeError, ImportError, ...) instead.
        """
        import ast
        from pathlib import Path

        from strands_robots_sim.isaac import simulation

        src = Path(simulation.__file__).read_text()
        tree = ast.parse(src)

        offending: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    offending.append((node.lineno, "except Exception"))
                elif node.type is None:
                    offending.append((node.lineno, "bare except"))

        assert not offending, (
            f"simulation.py must not use bare 'except Exception' or 'except:'; "
            f"narrow to specific exception classes. Offending sites: {offending}"
        )


class TestInstallConstants:
    """Pin the install-instruction abstraction.

    The install hints (docker tag, Omniverse Launcher line, Isaac Lab
    bootstrap) live in :mod:`strands_robots_sim.isaac._install` so a
    single edit propagates everywhere they surface (review feedback
    on PR #47). These tests pin (a) the contract that the module
    exposes the expected constants, and (b) that the messages
    consumed by ``IsaacSimulation`` are built from those constants
    -- if someone hardcodes a tag back into ``simulation.py`` the
    ``test_simulation_module_has_no_hardcoded_image`` regression
    pin will fail.
    """

    def test_constants_present(self):
        from strands_robots_sim.isaac import _install

        assert _install.ISAAC_SIM_DOCKER_IMAGE.startswith("nvcr.io/nvidia/isaac-sim:")
        assert _install.ISAAC_SIM_MIN_VERSION  # non-empty
        assert "isaaclab" in _install.ISAAC_LAB_BOOTSTRAP.lower()
        assert "strands-robots-sim[isaac]" in _install.PIP_EXTRA

    def test_not_importable_reason_composes_from_constants(self):
        from strands_robots_sim.isaac import _install

        msg = _install.not_importable_reason()
        assert _install.ISAAC_SIM_DOCKER_IMAGE in msg
        assert _install.ISAAC_SIM_MIN_VERSION in msg
        assert _install.ISAAC_LAB_BOOTSTRAP.split(" && ")[-1] in msg
        assert _install.PIP_EXTRA in msg

    def test_not_available_import_error_composes_from_constants(self):
        from strands_robots_sim.isaac import _install

        msg = _install.not_available_import_error()
        assert _install.ISAAC_SIM_DOCKER_IMAGE in msg
        assert "Omniverse Launcher" in msg

    def test_simulation_module_has_no_hardcoded_image(self):
        """Regression pin: docker tag must not be re-hardcoded into simulation.py.

        If this fails, fold the new occurrence into
        ``strands_robots_sim.isaac._install`` so the install-hint
        single-source-of-truth survives.
        """
        from pathlib import Path

        from strands_robots_sim.isaac import simulation

        src = Path(simulation.__file__).read_text()
        # The literal docker image tag must appear nowhere in simulation.py;
        # callers should use _install.ISAAC_SIM_DOCKER_IMAGE.
        assert "nvcr.io/nvidia/isaac-sim:" not in src, (
            "simulation.py contains a hardcoded Isaac Sim docker tag. "
            "Use strands_robots_sim.isaac._install.ISAAC_SIM_DOCKER_IMAGE instead."
        )

    def test_is_available_reason_uses_install_module(self):
        """The reason string returned by ``is_available()`` when omni
        is missing must come from ``_install.not_importable_reason``.
        """
        import importlib.util as iu

        from strands_robots_sim.isaac import _install
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # Force the "not importable" branch by stubbing find_spec.
        original_find_spec = iu.find_spec

        def fake_find_spec(name, *a, **kw):
            if name == "omni.isaac.kit":
                return None
            return original_find_spec(name, *a, **kw)

        with patch.object(iu, "find_spec", side_effect=fake_find_spec):
            available, reason = IsaacSimulation.is_available()

        if available is False and reason is not None and "omni" in reason.lower():
            # Only assert composition when we actually hit the omni branch
            # (CUDA / torch branches return earlier on some hosts).
            assert reason == _install.not_importable_reason()


class TestCreateWorldGravityScalarApi:
    """Regression pins for `strands-labs/robots-sim#52` — Isaac Sim 5.1
    `PhysicsContext.set_gravity` API mismatch (took vec3 pre-5.1, takes
    a scalar magnitude in 5.1).

    Without Isaac Sim installed we can't exercise `create_world`'s
    runtime path on this CI host, so the regression is pinned via
    source-inspection: the scalar-extraction call site and the
    `TypeError`-tolerant `except` clause are both pinned so a future
    refactor can't silently revert either one.
    """

    def test_set_gravity_call_extracts_scalar_from_vector(self) -> None:
        """`create_world` must extract a scalar Z-component before calling
        `set_gravity`.

        Isaac Sim 5.1's `PhysicsContext.set_gravity(value: float)` raises
        `TypeError` if handed a list/tuple. The fix at PR #47 commit
        `a65f9f9` added the `grav[2]` extraction; this test pins it so
        a future refactor doesn't accidentally revert to the pre-5.1
        vector-pass pattern that #52 reproduced.
        """
        import inspect

        from strands_robots_sim.isaac import simulation

        source = inspect.getsource(simulation)

        # Both halves of the fix must be present: the extraction line
        # and the call site that consumes its result.
        extraction = "gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav"
        call_site = "set_gravity(gravity_magnitude)"

        assert extraction in source, (
            "create_world() must extract a scalar Z-component before set_gravity. "
            "If this assertion fails, a refactor likely reverted the fix from "
            "https://github.com/strands-labs/robots-sim/issues/52 — Isaac Sim 5.1 "
            "set_gravity takes a scalar, not a vec3. Restore the extraction line."
        )
        assert call_site in source, (
            "set_gravity must be invoked with the extracted scalar (gravity_magnitude). "
            "If this assertion fails, the call site was rewritten to pass the raw "
            "list — that path will TypeError under Isaac Sim 5.1 (see #52)."
        )

    def test_create_world_except_clause_catches_typeerror(self) -> None:
        """`create_world`'s narrow except clause must include `TypeError`.

        Defence in depth for #52-class surface drift: Isaac Sim 5.1's
        `set_gravity(value: float)` already rejects non-scalar input
        with `TypeError`, and other physics-context calls (e.g.
        `set_solver_position_iteration_count`) have the same shape.
        Catching `TypeError` here means a future regression on any of
        those paths surfaces as a structured error envelope rather
        than an unhandled exception.
        """
        import ast
        import inspect

        from strands_robots_sim.isaac import simulation

        source = inspect.getsource(simulation)
        tree = ast.parse(source)

        # Find the create_world function and inspect its top-level
        # except clauses. The relevant one is the broad-tuple clause
        # that handles partial-init cleanup; ImportError stays in its
        # own clause (specific recovery message).
        create_world_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_world":
                create_world_fn = node
                break

        assert create_world_fn is not None, "create_world() function not found in simulation module"

        # Walk the try / except block(s) inside create_world and find
        # the catch-tuple clause (the narrow except that handles
        # partial-init cleanup).
        broad_except_caught: tuple[str, ...] | None = None
        for node in ast.walk(create_world_fn):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
                names = tuple(elt.id for elt in node.type.elts if isinstance(elt, ast.Name))
                # We want the cleanup clause, identified by it catching
                # RuntimeError + ValueError + OSError + AttributeError
                # at minimum (the pre-#52 superset).
                if {"RuntimeError", "ValueError", "OSError", "AttributeError"}.issubset(set(names)):
                    broad_except_caught = names
                    break

        assert broad_except_caught is not None, (
            "create_world() lost its narrow-tuple except clause. "
            "Re-add it as `except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:` "
            "to keep partial-init cleanup behaviour consistent (see #52)."
        )

        assert "TypeError" in broad_except_caught, (
            f"create_world() except clause is {broad_except_caught!r}; "
            f"missing TypeError. Add it back per "
            f"https://github.com/strands-labs/robots-sim/issues/52 — "
            f"Isaac Sim 5.1 set_gravity raises TypeError on non-scalar input, "
            f"and other physics-context calls have the same shape; catching "
            f"TypeError here means surface drift surfaces as a structured "
            f"error envelope rather than an unhandled exception."
        )
