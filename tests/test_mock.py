#!/usr/bin/env python3
"""
Pytest tests for mock simulation environment.

This test suite validates the full mock example with no dependencies.
Uses Mock Libero + Mock GR00T.
"""

# ruff: noqa: B101 # Assert statements are expected in tests
# flake8: noqa: B101
# nosec B101

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class MockGR00THandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for GR00T inference service."""

    def do_POST(self):
        """Handle POST requests for inference."""
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            # Mock response - return random actions
            import numpy as np

            # Generate mock robot actions
            mock_actions = []
            for _ in range(8):  # Action horizon
                action = {
                    "robot0_joint_pos": np.random.uniform(-0.1, 0.1, 7).tolist(),
                    "robot0_eef_pos": np.random.uniform(-0.01, 0.01, 3).tolist(),
                    "robot0_eef_quat": [0, 0, 0, 1],  # Identity quaternion
                    "robot0_gripper_qpos": [np.random.uniform(-0.1, 0.1)],
                }
                mock_actions.append(action)

            response = {"actions": mock_actions, "status": "success"}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            error_response = {"error": str(e), "status": "error"}
            self.wfile.write(json.dumps(error_response).encode("utf-8"))

    def do_GET(self):
        """Handle GET requests for status."""
        if self.path == "/status":
            response = {"status": "running", "model": "mock-gr00t-v1.5", "port": self.server.server_port}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress server logs."""
        pass


class MockGR00TServer:
    """Mock GR00T inference server for testing."""

    def __init__(self, port=8000):
        self.port = port
        self.server = None
        self.thread = None
        self.running = False

    def _find_available_port(self, start_port=8000, max_attempts=100):
        """Find an available port starting from start_port."""
        import socket

        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

    def start(self):
        """Start the mock server."""
        try:
            # Find available port if the default is in use
            try:
                self.server = HTTPServer(("localhost", self.port), MockGR00THandler)
            except OSError as e:
                if "Address already in use" in str(e):
                    available_port = self._find_available_port(self.port)
                    self.port = available_port
                    self.server = HTTPServer(("localhost", self.port), MockGR00THandler)
                else:
                    raise

            self.running = True
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            # Intentional delay: Allow server thread to start and bind to port before returning
            time.sleep(0.1)  # nosec B311
            return True
        except Exception as e:
            print(f"Failed to start mock server: {e}")
            return False

    def _run_server(self):
        """Run the server loop."""
        while self.running:
            self.server.handle_request()

    def stop(self):
        """Stop the mock server."""
        self.running = False
        if self.server:
            self.server.server_close()

    def is_running(self):
        """Check if server is running."""
        return self.running and self.thread and self.thread.is_alive()


@pytest.fixture
def mock_groot_server():
    """Fixture to start and stop mock GR00T server."""
    server = MockGR00TServer(port=8000)
    assert server.start(), "Failed to start mock server"  # nosec B101
    yield server
    server.stop()


def test_fully_mock_simulation(mock_groot_server):
    """Test full mock simulation with no dependencies - Mock Libero + Mock GR00T."""
    from strands import Agent

    from strands_robots_sim import SimEnv

    # Create simulation environment with mock backend
    sim_env = SimEnv(
        tool_name="my_mock_sim",
        env_type="mock_libero",  # Use mock environment - no dependencies
        task_suite="libero_spatial",
    )

    # Create agent
    agent = Agent(tools=[sim_env])

    # Verify SimEnv created correctly
    assert sim_env.tool_name == "my_mock_sim"  # nosec B101
    assert sim_env.env_type == "mock_libero"  # nosec B101
    assert sim_env.task_suite == "libero_spatial"  # nosec B101

    # Test with mock policy
    result = agent.tool.my_mock_sim(
        action="execute",
        instruction="pick up the red block using mock actions",
        policy_port=mock_groot_server.port,
        policy_provider="mock",  # Use mock policy
        max_episodes=2,
        max_steps_per_episode=10,
    )

    # Verify result
    assert result is not None  # nosec B101
    assert "status" in result  # nosec B101

    # Check final status
    status = agent.tool.my_mock_sim(action="status")
    assert status is not None  # nosec B101
    assert "status" in status  # nosec B101


def test_mock_environment_initialization():
    """Test that mock environment initializes correctly."""
    import asyncio

    from strands_robots_sim import SimEnv

    sim_env = SimEnv(tool_name="test_mock", env_type="mock_libero", task_suite="libero_spatial")

    # Test initialization
    async def test_init():
        success = await sim_env.sim_env.initialize()
        assert success, "Mock environment initialization should succeed"  # nosec B101
        assert sim_env.sim_env.is_initialized  # nosec B101
        assert len(sim_env.sim_env.available_tasks) > 0  # nosec B101

    asyncio.run(test_init())


def test_mock_environment_reset():
    """Test that mock environment can reset."""
    import asyncio

    from strands_robots_sim import SimEnv

    sim_env = SimEnv(tool_name="test_mock", env_type="mock_libero", task_suite="libero_spatial")

    async def test_reset():
        await sim_env.sim_env.initialize()
        obs = await sim_env.sim_env.reset()

        # Verify observation contains expected keys
        assert "robot0_joint_pos" in obs  # nosec B101
        assert "robot0_eef_pos" in obs  # nosec B101
        assert "front_camera" in obs or "agentview_image" in obs  # nosec B101

    asyncio.run(test_reset())


def test_mock_environment_step():
    """Test that mock environment can execute steps."""
    import asyncio

    from strands_robots_sim import SimEnv

    sim_env = SimEnv(tool_name="test_mock", env_type="mock_libero", task_suite="libero_spatial")

    async def test_step():
        await sim_env.sim_env.initialize()
        await sim_env.sim_env.reset()

        # Execute a step with zero action
        action = {"robot0_joint_pos": [0.0] * 7}
        obs, reward, done, info = await sim_env.sim_env.step(action)

        # Verify step results
        assert obs is not None  # nosec B101
        assert isinstance(reward, (int, float))  # nosec B101
        assert isinstance(done, bool)  # nosec B101
        assert isinstance(info, dict)  # nosec B101

    asyncio.run(test_step())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
