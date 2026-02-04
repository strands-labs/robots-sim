#!/usr/bin/env python3
"""
GR00T Inference Service Management Tool

Manages GR00T policy inference services running in Docker containers.
Uses Isaac-GR00T's native inference service for proper ZMQ communication.
"""

import re
import shutil
import socket
import subprocess  # nosec B404 # subprocess is required for Docker operations
import time
from typing import Any, Dict, Optional

from strands import tool

# Cache for command paths to avoid repeated lookups
_COMMAND_CACHE: Dict[str, Optional[str]] = {}


def _get_command_path(command: str) -> str:
    """Get full path to system command using shutil.which().

    Args:
        command: Command name (e.g., 'docker', 'kill')

    Returns:
        Full path to command

    Raises:
        RuntimeError: If command not found in PATH
    """
    if command not in _COMMAND_CACHE:
        path = shutil.which(command)
        if path is None:
            raise RuntimeError(f"Command '{command}' not found in PATH")
        _COMMAND_CACHE[command] = path
    return _COMMAND_CACHE[command]


def _validate_port(port: int) -> bool:
    """Validate port number is in valid range."""
    return isinstance(port, int) and 1 <= port <= 65535


def _validate_pid(pid: str) -> bool:
    """Validate PID is numeric."""
    return pid.isdigit() and int(pid) > 0


def _validate_container_name(name: str) -> bool:
    """Validate container name follows Docker naming rules."""
    if not name:
        return False
    # Docker container names: alphanumeric, underscore, period, hyphen
    # Must start with alphanumeric
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$", name))


def _validate_path(path: str) -> bool:
    """Validate path doesn't contain shell metacharacters."""
    if not path:
        return False
    # Allow alphanumeric, forward slash, underscore, hyphen, period, colon (for paths)
    # Disallow common shell metacharacters: $, `, ;, &, |, <, >, etc.
    dangerous_chars = ["$", "`", ";", "&", "|", "<", ">", "(", ")", "{", "}", "[", "]", "!", "*", "?", "\n", "\r"]
    return not any(char in path for char in dangerous_chars)


def _validate_hostname(host: str) -> bool:
    """Validate hostname/IP address."""
    if not host:
        return False
    # Allow 0.0.0.0, localhost, or valid hostnames
    if host in ["0.0.0.0", "localhost", "127.0.0.1"]:  # nosec B104 # Explicitly allow binding options
        return True
    # Basic hostname validation
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9.-]*$", host))


def _validate_identifier(identifier: str) -> bool:
    """Validate identifier (embodiment_tag, data_config, etc.).

    Allows:
    - Simple identifiers: "so100_dualcam", "new_embodiment"
    - Module paths: "examples.Libero.custom_data_config:LiberoDataConfig"
    """
    if not identifier:
        return False
    # Allow alphanumeric, underscore, hyphen, period (for module paths), colon (for class separator)
    return bool(re.match(r"^[a-zA-Z0-9_.:/-]+$", identifier))


@tool
def gr00t_inference(
    action: str,
    checkpoint_path: str = None,
    policy_name: str = None,
    port: int = None,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
    denoising_steps: int = 2,
    host: str = "127.0.0.1",
    container_name: str = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Manage GR00T inference services in Docker containers using Isaac-GR00T native scripts.

    Args:
        action: Action to perform
            - "start": Start inference service with checkpoint
            - "stop": Stop inference service on port
            - "status": Check status of service on port
            - "list": List all running services
            - "restart": Restart service with new checkpoint
            - "find_containers": Find available isaac-gr00t containers
        checkpoint_path: Path to model checkpoint (for start/restart)
        policy_name: Name for the policy service (for registration)
        port: Port for inference service
        data_config: GR00T data config (so100_dualcam, so100, fourier_gr1_arms_only, etc.)
        embodiment_tag: Embodiment tag for model
        denoising_steps: Number of denoising steps
        host: Host to bind service to (default: 127.0.0.1 for localhost only).
            Use "0.0.0.0" to bind to all interfaces if external access is required.
            Security: Only use 0.0.0.0 in trusted networks or with proper firewall rules.
        container_name: Specific container name
        timeout: Timeout for operations

    Returns:
        Dict with status and information about the operation
    """
    # Validate inputs to prevent command injection
    if port is not None and not _validate_port(port):
        return {"status": "error", "message": f"Invalid port: {port}"}

    if checkpoint_path is not None and not _validate_path(checkpoint_path):
        return {"status": "error", "message": "Invalid checkpoint path: contains dangerous characters"}

    if container_name is not None and not _validate_container_name(container_name):
        return {"status": "error", "message": f"Invalid container name: {container_name}"}

    if host and not _validate_hostname(host):
        return {"status": "error", "message": f"Invalid hostname: {host}"}

    if data_config and not _validate_identifier(data_config):
        return {"status": "error", "message": f"Invalid data_config: {data_config}"}

    if embodiment_tag and not _validate_identifier(embodiment_tag):
        return {"status": "error", "message": f"Invalid embodiment_tag: {embodiment_tag}"}

    if action == "find_containers":
        return _find_gr00t_containers()
    elif action == "list":
        return _list_running_services()
    elif action == "status":
        if port is None:
            return {"status": "error", "message": "Port required for status check"}
        return _check_service_status(port)
    elif action == "stop":
        if port is None:
            return {"status": "error", "message": "Port required to stop service"}
        return _stop_service(port)
    elif action == "start":
        if checkpoint_path is None:
            return {"status": "error", "message": "Checkpoint path required to start service"}
        if port is None:
            return {"status": "error", "message": "Port required to start service"}
        return _start_service(
            checkpoint_path,
            port,
            data_config,
            embodiment_tag,
            denoising_steps,
            host,
            container_name,
            policy_name,
            timeout,
        )
    elif action == "restart":
        if checkpoint_path is None or port is None:
            return {"status": "error", "message": "Checkpoint path and port required for restart"}
        # Stop existing service and start new one
        _stop_service(port)
        # Intentional delay: Allow time for service cleanup and port release before restart
        time.sleep(2)  # nosec B311
        return _start_service(
            checkpoint_path,
            port,
            data_config,
            embodiment_tag,
            denoising_steps,
            host,
            container_name,
            policy_name,
            timeout,
        )
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _find_gr00t_containers() -> Dict[str, Any]:
    """Find available Isaac-GR00T containers.

    Looks for containers with Isaac-GR00T images by checking:
    1. Direct match: "isaac-gr00t" in image name
    2. Fallback: "isaac" in image AND ("gr00t" in image OR "jetson" in container name)

    Returns both running and stopped containers.
    """
    try:
        docker_cmd = _get_command_path("docker")
        result = subprocess.run(  # nosec B603 # Command path validated, no user input in command
            [docker_cmd, "ps", "-a", "--format", "{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        containers = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    name, image, status = parts[0], parts[1], parts[2]
                    ports = parts[3] if len(parts) > 3 else ""

                    # Check if this is an Isaac-GR00T container by image name
                    # Look for isaac-gr00t in image name, or isaac-sim with gr00t context
                    is_gr00t_container = "isaac-gr00t" in image.lower() or (
                        "isaac" in image.lower() and ("gr00t" in image.lower() or "jetson" in name.lower())
                    )

                    if is_gr00t_container:
                        containers.append({"name": name, "image": image, "status": status, "ports": ports})

        return {"status": "success", "containers": containers, "message": f"Found {len(containers)} GR00T containers"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to find containers: {e}"}


def _list_running_services() -> Dict[str, Any]:
    """List all running GR00T inference services by checking ZMQ ports."""
    try:
        services = []
        common_ports = [5555, 5556, 5557, 5558, 8000, 8001, 8002, 8003]

        for port in common_ports:
            if _is_zmq_service_running(port):
                services.append({"port": port, "protocol": "ZMQ", "status": "running"})

        return {"status": "success", "services": services, "message": f"Found {len(services)} running ZMQ services"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to list services: {e}"}


def _is_zmq_service_running(port: int) -> bool:
    """Check if ZMQ service is running on port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _check_service_status(port: int) -> Dict[str, Any]:
    """Check status of ZMQ service on specific port."""
    if _is_zmq_service_running(port):
        return {"status": "success", "port": port, "service_status": "running", "protocol": "ZMQ"}
    else:
        return {
            "status": "error",
            "port": port,
            "service_status": "not_running",
            "message": f"No ZMQ service running on port {port}",
        }


def _stop_service(port: int) -> Dict[str, Any]:
    """Stop GR00T inference service running on specific port."""
    try:
        # First try to find and kill processes in Docker containers
        containers_result = _find_gr00t_containers()
        if containers_result["status"] == "success":
            running_containers = [c for c in containers_result["containers"] if "Up" in c["status"]]

            for container in running_containers:
                container_name = container["name"]
                # Validate container name before use
                if not _validate_container_name(container_name):
                    continue

                try:
                    # Find inference service processes in this container using the specific port
                    # Validate port is numeric to prevent injection
                    port_str = str(int(port))  # Ensure port is numeric
                    docker_cmd = _get_command_path("docker")
                    pgrep_cmd = _get_command_path("pgrep")
                    result = subprocess.run(  # nosec B603 # All inputs validated: container_name, port
                        [
                            docker_cmd,
                            "exec",
                            container_name,
                            pgrep_cmd,
                            "-f",
                            f"inference_service.py.*--port {port_str}",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split("\n")
                        for pid in pids:
                            if pid and _validate_pid(pid):
                                # Kill the process inside the container
                                kill_cmd = _get_command_path("kill")
                                subprocess.run(
                                    [docker_cmd, "exec", container_name, kill_cmd, "-TERM", pid], check=True
                                )  # nosec B603 # PID validated

                        # Intentional delay: Wait for graceful SIGTERM shutdown before force kill check
                        time.sleep(2)  # nosec B311

                        # Force kill if still running
                        result = subprocess.run(  # nosec B603 # All inputs validated
                            [
                                docker_cmd,
                                "exec",
                                container_name,
                                pgrep_cmd,
                                "-f",
                                f"inference_service.py.*--port {port_str}",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )

                        if result.returncode == 0 and result.stdout.strip():
                            pids = result.stdout.strip().split("\n")
                            for pid in pids:
                                if pid and _validate_pid(pid):
                                    subprocess.run(
                                        [docker_cmd, "exec", container_name, kill_cmd, "-KILL", pid], check=True
                                    )  # nosec B603 # PID validated

                        return {
                            "status": "success",
                            "port": port,
                            "container": container_name,
                            "message": f"GR00T service on port {port} stopped in container {container_name}",
                        }

                except subprocess.CalledProcessError:
                    continue  # Try next container

        # Fallback: try to find processes on host system
        # Validate port is numeric to prevent injection
        port_str = str(int(port))  # Ensure port is numeric
        lsof_cmd = _get_command_path("lsof")
        kill_cmd = _get_command_path("kill")
        result = subprocess.run(
            [lsof_cmd, "-t", f"-i:{port_str}"], capture_output=True, text=True
        )  # nosec B603 # Port validated

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid and _validate_pid(pid):
                    subprocess.run([kill_cmd, "-TERM", pid], check=True)  # nosec B603 # PID validated

            # Intentional delay: Wait for graceful SIGTERM shutdown before force kill check
            time.sleep(2)  # nosec B311

            # Force kill if still running
            result = subprocess.run(
                [lsof_cmd, "-t", f"-i:{port_str}"], capture_output=True, text=True
            )  # nosec B603 # Port validated

            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid and _validate_pid(pid):
                        subprocess.run([kill_cmd, "-KILL", pid], check=True)  # nosec B603 # PID validated

            return {"status": "success", "port": port, "message": f"Service on port {port} stopped"}
        else:
            return {"status": "success", "port": port, "message": f"No service running on port {port}"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to stop service: {e}"}


def _start_service(
    checkpoint_path: str,
    port: int,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    host: str,
    container_name: str,
    policy_name: str,
    timeout: int,
) -> Dict[str, Any]:
    """Start GR00T inference service using Isaac-GR00T's native inference service."""
    try:
        # Find container if not specified
        if container_name is None:
            containers = _find_gr00t_containers()
            if containers["status"] == "error":
                return containers

            running_containers = [c for c in containers["containers"] if "Up" in c["status"]]
            if not running_containers:
                return {"status": "error", "message": "No running GR00T containers found"}

            container_name = running_containers[0]["name"]

        # Validate container_name to prevent command injection
        # Note: This validation is done at top level, but also here for container_name from docker ps
        if not _validate_container_name(container_name):
            return {"status": "error", "message": f"Invalid container name: {container_name}"}

        # Build Isaac-GR00T inference service command
        cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "/opt/Isaac-GR00T/scripts/inference_service.py",
            "--server",
            "--model-path",
            checkpoint_path,
            "--port",
            str(port),
            "--host",
            host,
            "--data-config",
            data_config,
            "--embodiment-tag",
            embodiment_tag,
            "--denoising-steps",
            str(denoising_steps),
        ]

        # Start service
        subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603

        # Wait for ZMQ service to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _is_zmq_service_running(port):
                return {
                    "status": "success",
                    "port": port,
                    "checkpoint_path": checkpoint_path,
                    "container_name": container_name,
                    "policy_name": policy_name,
                    "protocol": "ZMQ",
                    "data_config": data_config,
                    "embodiment_tag": embodiment_tag,
                    "message": f"GR00T ZMQ service started on port {port}",
                }
            # Intentional delay: Polling interval while waiting for service initialization
            time.sleep(1)  # nosec B311

        return {"status": "error", "message": f"ZMQ service failed to start within {timeout} seconds"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to start service: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


if __name__ == "__main__":
    print("🐳 GR00T Inference Service Manager (Isaac-GR00T Native)")
    print("Uses Isaac-GR00T's ZMQ-based inference service")

    # Example usage
    examples = [
        "gr00t_inference(action='find_containers')",
        (
            "gr00t_inference(action='start', checkpoint_path='/data/checkpoints/gr00t-wave/checkpoint-300000', "
            "port=5555, policy_name='wave_model')"
        ),
        "gr00t_inference(action='list')",
        "gr00t_inference(action='status', port=5555)",
        "gr00t_inference(action='stop', port=5555)",
    ]

    for example in examples:
        print(f"  {example}")
