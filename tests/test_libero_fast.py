#!/usr/bin/env python3
"""
Pytest tests for fast Libero simulation with zero actions.

This test suite validates the fast example with real Libero + zero actions (no policy).
Requires: pip install strands-robots[sim]
"""

import asyncio

# ruff: noqa: B101 # Assert statements are expected in tests
import pytest

pytestmark = pytest.mark.libero


@pytest.fixture(scope="module")
def skip_if_no_libero():
    """Skip tests if Libero is not installed."""
    try:
        import libero  # noqa: F401

        return True
    except ImportError:
        pytest.skip("Libero not installed. Install with: pip install strands-robots[sim]")


def test_fast_libero_simulation(skip_if_no_libero):
    """
    Test fast SimEnv example with real Libero + zero actions.

    This eliminates policy overhead while using SimEnv wrapper.
    """
    import time

    from strands_robots_sim import SimEnv

    async def run_fast_test():
        print("Creating SimEnv with Libero...")
        # Create SimEnv
        sim_env = SimEnv(tool_name="fast_libero_sim", env_type="libero", task_suite="libero_spatial")

        print("Initializing environment...")
        start_time = time.time()
        success = await sim_env.sim_env.initialize()
        init_time = time.time() - start_time

        assert success, "Environment initialization should succeed"  # nosec B101
        assert sim_env.sim_env.is_initialized  # nosec B101
        print(f"Initialized in {init_time:.1f}s")

        print("Resetting environment...")
        start_time = time.time()
        obs = await sim_env.sim_env.reset()
        reset_time = time.time() - start_time

        assert obs is not None  # nosec B101
        assert "robot0_joint_pos" in obs  # nosec B101
        print(f"Reset in {reset_time:.1f}s")

        print("Running fast simulation...")
        start_time = time.time()

        total_reward = 0
        for i in range(5):  # 5 steps - fast!
            # Zero actions - no policy calls
            action = {
                "robot0_joint_pos": [0.0] * 7,
            }

            obs, reward, done, info = await sim_env.sim_env.step(action)
            total_reward += reward

            assert obs is not None  # nosec B101
            assert isinstance(reward, (int, float))  # nosec B101
            assert isinstance(done, bool)  # nosec B101

            if done:
                print(f"Episode ended at step {i+1}")
                break

        step_time = time.time() - start_time
        print(f"Simulation completed in {step_time:.1f}s")
        print(f"Total reward: {total_reward:.3f}")

        # Cleanup
        await sim_env.sim_env.cleanup()

        return True

    result = asyncio.run(run_fast_test())
    assert result  # nosec B101


def test_libero_video_frame_capture(skip_if_no_libero):
    """Test that video frames can be captured from Libero observations."""
    from strands_robots_sim import SimEnv

    async def test_capture():
        sim_env = SimEnv(tool_name="test_sim", env_type="libero", task_suite="libero_spatial")

        await sim_env.sim_env.initialize()
        obs = await sim_env.sim_env.reset()

        # Check for camera observations
        has_agentview = "agentview_image" in obs or "front_camera" in obs
        has_wrist = "robot0_eye_in_hand_image" in obs or "wrist_camera" in obs

        assert has_agentview, "Should have agent/front camera view"  # nosec B101
        assert has_wrist, "Should have wrist camera view"  # nosec B101

        # Get frames
        if "agentview_image" in obs:
            frame = obs["agentview_image"]
        elif "front_camera" in obs:
            frame = obs["front_camera"]

        # Verify frame properties
        assert frame is not None  # nosec B101
        assert len(frame.shape) == 3, "Frame should be (H, W, C)"  # nosec B101
        assert frame.shape[2] == 3, "Frame should have 3 color channels"  # nosec B101

        await sim_env.sim_env.cleanup()

    asyncio.run(test_capture())


def test_libero_action_format(skip_if_no_libero):
    """Test that Libero accepts GR00T action format."""
    from strands_robots_sim import SimEnv

    async def test_action():
        sim_env = SimEnv(tool_name="test_sim", env_type="libero", task_suite="libero_spatial")

        await sim_env.sim_env.initialize()
        await sim_env.sim_env.reset()

        # Test GR00T format: 7-dim delta pose [dx,dy,dz,dr,dp,dy,gripper]
        action = {"action": [0.0] * 7}
        obs, reward, done, info = await sim_env.sim_env.step(action)

        assert obs is not None, "Step should return observation"  # nosec B101
        assert not done or "success" in info, "Info should contain success status"  # nosec B101

        await sim_env.sim_env.cleanup()

    asyncio.run(test_action())


def test_libero_task_suite_loading(skip_if_no_libero):
    """Test that different Libero task suites can be loaded."""
    from strands_robots_sim import SimEnv

    task_suites = ["libero_spatial", "libero_goal", "libero_10"]

    async def test_suite(suite_name):
        sim_env = SimEnv(tool_name="test_sim", env_type="libero", task_suite=suite_name)

        success = await sim_env.sim_env.initialize()
        if success:
            assert sim_env.sim_env.is_initialized  # nosec B101
            assert len(sim_env.sim_env.available_tasks) > 0  # nosec B101
            print(f"{suite_name}: {len(sim_env.sim_env.available_tasks)} tasks")
            await sim_env.sim_env.cleanup()
            return True
        return False

    # Test at least one suite loads successfully
    results = []
    for suite in task_suites:
        try:
            result = asyncio.run(test_suite(suite))
            results.append(result)
        except Exception as e:
            print(f"Suite {suite} failed: {e}")
            results.append(False)

    assert any(results), "At least one task suite should load successfully"  # nosec B101


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "libero"])
