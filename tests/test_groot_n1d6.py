"""Tests for GR00T multi-protocol support (sim_wrapper + direct).

Verifies that observation formatting adapts to the active protocol
while data configs remain unchanged. All tests are mock-only.
"""

import numpy as np
import pytest

from strands_robots_sim.policies.groot.data_config import PROTOCOLS, load_data_config

pytestmark = pytest.mark.mock


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def libero_obs():
    return {
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
        "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    }


@pytest.fixture
def state_keys():
    return ["robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def _make_policy(protocol, state_keys):
    from strands_robots_sim.policies.groot import Gr00tPolicy

    p = Gr00tPolicy(data_config="libero", host="localhost", port=9999, protocol=protocol)
    p.set_robot_state_keys(state_keys)
    return p


# -- DataConfig tests --------------------------------------------------------


class TestDataConfig:
    """Configs are unchanged from original; protocol is additive."""

    def test_original_configs_unchanged(self):
        """Every original config still has its original keys."""
        for name in ("libero", "libero_spatial", "libero_goal", "libero_meanstd"):
            cfg = load_data_config(name)
            assert "video.image" in cfg["video"]
            assert cfg["action"][0].startswith("action.robot0_")
            assert cfg["state"] == ["state"]

    def test_libero_has_sim_wrapper_protocol(self):
        assert load_data_config("libero").get("protocol") == "sim_wrapper"

    def test_meanstd_has_no_protocol(self):
        assert "protocol" not in load_data_config("libero_meanstd")

    def test_protocol_override(self):
        cfg = load_data_config("libero", protocol="direct")
        assert cfg["protocol"] == "direct"

    def test_colon_syntax(self):
        cfg = load_data_config("libero:direct")
        assert cfg["protocol"] == "direct"

    def test_dict_passthrough(self):
        d = {"video": ["v"], "state": ["s"], "action": ["a"], "language": ["l"]}
        assert load_data_config(d) == d

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            load_data_config("nonexistent")

    def test_fuzzy_match(self):
        assert load_data_config("libero_custom") is not None

    def test_protocols_complete(self):
        for name, p in PROTOCOLS.items():
            for field in ("video_ndim", "state_dtype", "request_wrap", "response_batch_dim", "language_type"):
                assert field in p, f"{name} missing {field}"


# -- sim_wrapper observation --------------------------------------------------


class TestSimWrapperObservation:

    def test_video_5d(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "test")
        assert obs["video.image"].ndim == 5
        assert obs["video.image"].shape == (1, 1, 256, 256, 3)
        assert obs["video.image"].dtype == np.uint8

    def test_state_float32(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "test")
        assert obs["state.x"].dtype == np.float32

    def test_state_3d(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "test")
        assert obs["state.x"].ndim == 3

    def test_state_values(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "test")
        assert float(obs["state.x"].flat[0]) == pytest.approx(0.1)
        assert float(obs["state.y"].flat[0]) == pytest.approx(0.2)
        assert float(obs["state.z"].flat[0]) == pytest.approx(0.3)

    def test_gripper_multi_dim(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "test")
        assert obs["state.gripper"].shape[-1] == 2  # robot0_gripper_qpos is 2-dim

    def test_language_tuple(self, libero_obs, state_keys):
        obs = _make_policy("sim_wrapper", state_keys)._build_observation(libero_obs, "hi")
        assert obs["annotation.human.action.task_description"] == ("hi",)


# -- direct observation -------------------------------------------------------


class TestDirectObservation:

    def test_video_4d(self, libero_obs, state_keys):
        obs = _make_policy("direct", state_keys)._build_observation(libero_obs, "test")
        assert obs["video.image"].ndim == 4

    def test_state_float64(self, libero_obs, state_keys):
        obs = _make_policy("direct", state_keys)._build_observation(libero_obs, "test")
        assert obs["state.x"].dtype == np.float64

    def test_language_list(self, libero_obs, state_keys):
        obs = _make_policy("direct", state_keys)._build_observation(libero_obs, "hi")
        assert obs["annotation.human.action.task_description"] == ["hi"]


# -- Action conversion --------------------------------------------------------


class TestActionConversion:

    def test_libero_7dim(self, state_keys):
        p = _make_policy("sim_wrapper", state_keys)
        chunk = {
            f"action.{k}": np.random.randn(16, 1).astype(np.float32)
            for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        }
        actions = p._to_robot_actions(chunk)
        assert len(actions) == 16
        assert all(len(a["action"]) == 7 for a in actions)

    def test_batch_dim_stripped(self, state_keys):
        p = _make_policy("sim_wrapper", state_keys)
        chunk = {f"action.{k}": np.zeros((1, 8, 1)) for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")}
        actions = p._to_robot_actions(chunk)
        assert len(actions) == 8

    def test_fallback(self, state_keys):
        chunk = _make_policy("sim_wrapper", state_keys)._create_fallback_actions()
        assert all(isinstance(v, np.ndarray) for v in chunk.values())


# -- Protocol selection -------------------------------------------------------


class TestProtocolSelection:

    def test_colon_sim_wrapper(self):
        from strands_robots_sim.policies.groot import Gr00tPolicy

        assert Gr00tPolicy(data_config="libero:sim_wrapper", host="localhost", port=9999).protocol_name == "sim_wrapper"

    def test_colon_direct(self):
        from strands_robots_sim.policies.groot import Gr00tPolicy

        assert Gr00tPolicy(data_config="libero:direct", host="localhost", port=9999).protocol_name == "direct"

    def test_legacy_n1d6(self):
        from strands_robots_sim.policies.groot import Gr00tPolicy

        assert (
            Gr00tPolicy(data_config="libero", host="localhost", port=9999, groot_version="n1d6").protocol_name
            == "sim_wrapper"
        )

    def test_legacy_n1d5(self):
        from strands_robots_sim.policies.groot import Gr00tPolicy

        assert (
            Gr00tPolicy(data_config="libero", host="localhost", port=9999, groot_version="n1d5").protocol_name
            == "direct"
        )

    def test_default_from_config(self):
        from strands_robots_sim.policies.groot import Gr00tPolicy

        assert Gr00tPolicy(data_config="libero", host="localhost", port=9999).protocol_name == "sim_wrapper"
