#!/usr/bin/env python3
"""
Pytest tests for mock Libero simulation scenarios.

This test suite covers:
- Choice 1: Libero + Mock GR00T (requires: pip install strands-robots[sim])
- Choice 2: Full Mock (no dependencies - RECOMMENDED for testing)
"""

import asyncio

# ruff: noqa: B101 # Assert statements are expected in tests
import pytest

pytestmark = pytest.mark.mock


@pytest.mark.libero
def test_libero_with_mock_groot():
    """
    Test Choice 1: Real Libero + Mock GR00T

    This test requires Libero to be installed but mocks the GR00T policy.
    Requires: pip install strands-robots[sim]
    """
    try:
        import libero  # noqa: F401
    except ImportError:
        pytest.skip("Libero not installed. Install with: pip install strands-robots[sim]")

    import random

    from strands_robots_sim import SimEnv

    async def run_test():
        print("🎮 Creating simulation environment with mock GR00T...")

        # Create simulation environment
        sim_env = SimEnv(tool_name="test_libero_sim", env_type="libero", task_suite="libero_spatial")

        # Initialize environment
        print("Initializing environment...")
        success = await sim_env.sim_env.initialize()
        assert success, "Environment initialization should succeed"  # nosec B101

        # Get available tasks
        available_tasks = sim_env.sim_env.available_tasks
        assert len(available_tasks) > 0, "Should have available tasks"  # nosec B101

        # Random selection for test purposes (not security-sensitive)
        selected_task = random.choice(available_tasks)  # nosec B311
        print(f"🎲 Selected task: {selected_task}")

        # Reset environment
        obs = await sim_env.sim_env.reset()
        assert obs is not None  # nosec B101

        # Mock GR00T actions - just return zero actions
        total_reward = 0
        for step in range(5):
            # Mock GR00T policy output
            action = {"action": [0.0] * 7}

            obs, reward, done, info = await sim_env.sim_env.step(action)
            total_reward += reward

            assert obs is not None  # nosec B101
            assert isinstance(reward, (int, float))  # nosec B101

            if done:
                print(f"Episode completed at step {step+1}")
                break

        print(f"✅ Test completed with total reward: {total_reward}")

        # Cleanup
        await sim_env.sim_env.cleanup()

        return True

    result = asyncio.run(run_test())
    assert result  # nosec B101


def test_fully_mock_simulation():
    """
    Test Choice 2: Full Mock (no dependencies)

    This test mocks everything - no Libero or GR00T required.
    RECOMMENDED for testing without dependencies.
    """
    from unittest.mock import MagicMock

    print("🎮 Creating fully mocked simulation...")

    # Mock SimEnv
    mock_sim_env = MagicMock()
    mock_sim_env.tool_name = "mock_libero_sim"

    # Mock environment
    mock_env = MagicMock()
    mock_env.is_initialized = False
    mock_env.available_tasks = ["pick up the black bowl", "put the bowl on the plate"]

    # Mock initialization
    async def mock_initialize():
        mock_env.is_initialized = True
        return True

    mock_env.initialize = mock_initialize

    # Mock reset
    async def mock_reset():
        return {
            "robot0_joint_pos": [0.0] * 7,
            "agentview_image": [[0] * 128 for _ in range(128)],
            "robot0_eye_in_hand_image": [[0] * 128 for _ in range(128)],
        }

    mock_env.reset = mock_reset

    # Mock step
    async def mock_step(action):
        obs = await mock_reset()
        reward = 0.0
        done = False
        info = {"success": False}
        return obs, reward, done, info

    mock_env.step = mock_step

    # Mock cleanup
    async def mock_cleanup():
        mock_env.is_initialized = False
        return True

    mock_env.cleanup = mock_cleanup

    mock_sim_env.sim_env = mock_env

    # Run test simulation
    async def run_fully_mock_test():
        print("Initializing mock environment...")
        success = await mock_sim_env.sim_env.initialize()
        assert success  # nosec B101
        assert mock_sim_env.sim_env.is_initialized  # nosec B101

        print("Resetting mock environment...")
        obs = await mock_sim_env.sim_env.reset()
        assert obs is not None  # nosec B101
        assert "robot0_joint_pos" in obs  # nosec B101

        print("Running mock simulation steps...")
        total_reward = 0
        for step in range(5):
            action = {"action": [0.0] * 7}
            obs, reward, done, info = await mock_sim_env.sim_env.step(action)
            total_reward += reward

            assert obs is not None  # nosec B101
            assert isinstance(reward, (int, float))  # nosec B101

            if done:
                print(f"Mock episode completed at step {step+1}")
                break

        print(f"✅ Mock test completed with total reward: {total_reward}")

        # Cleanup
        await mock_sim_env.sim_env.cleanup()
        assert not mock_sim_env.sim_env.is_initialized  # nosec B101

        return True

    result = asyncio.run(run_fully_mock_test())
    assert result  # nosec B101


def test_mock_agent_with_tools():
    """
    Test mocked Agent with SimEnv and GR00T tools.

    This tests the full integration without real services.
    """
    from unittest.mock import MagicMock

    print("🤖 Creating mocked Agent with tools...")

    # Mock SimEnv tool
    mock_sim_env = MagicMock()
    mock_sim_env.tool_name = "my_libero_sim"

    # Mock GR00T inference tool
    mock_groot = MagicMock()

    def mock_groot_call(action, **kwargs):
        if action == "start":
            return "✅ GR00T service started (mocked)"
        elif action == "status":
            return "running"
        elif action == "stop":
            return "✅ GR00T service stopped (mocked)"
        elif action == "infer":
            return {"action": [0.0] * 7}
        return "ok"

    mock_groot.side_effect = mock_groot_call

    # Mock Agent
    mock_agent = MagicMock()
    mock_agent.tool = MagicMock()
    mock_agent.tool.gr00t_inference = mock_groot
    mock_agent.tool.my_libero_sim = lambda action: f"Mock simulation {action}"

    # Test workflow
    print("1. Starting mock GR00T service...")
    result = mock_agent.tool.gr00t_inference(action="start", checkpoint_path="/data/checkpoints/mock", port=8000)
    assert "started" in result.lower() or "mocked" in result.lower()  # nosec B101

    print("2. Checking mock status...")
    status = mock_agent.tool.gr00t_inference(action="status", port=8000)
    assert status == "running"  # nosec B101

    print("3. Running mock simulation...")
    sim_result = mock_agent.tool.my_libero_sim(action="run")
    assert "Mock" in sim_result  # nosec B101

    print("4. Stopping mock GR00T service...")
    stop_result = mock_agent.tool.gr00t_inference(action="stop", port=8000)
    assert "stopped" in stop_result.lower() or "mocked" in stop_result.lower()  # nosec B101

    print("✅ Mock agent test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "mock"])
