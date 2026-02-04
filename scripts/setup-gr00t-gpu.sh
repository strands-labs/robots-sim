#!/bin/bash

# Isaac-GR00T Docker Setup Script for GPU Systems
# Follows /Users/yinsong/workspace_cline/strands-robots-sim/docker-setup-guide.md
# Step 3 Option C + Step 5 Option A

set -e  # Exit on error

# Resolve script directory once before any cd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on GPU system
check_gpu() {
    log "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        success "NVIDIA GPU detected"
    else
        error "nvidia-smi not found. This script is for GPU systems."
        echo "Install NVIDIA drivers and nvidia-docker2 first."
        exit 1
    fi
}

# Check Docker installation
check_docker() {
    log "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check if user can run docker without sudo
    if ! docker info &> /dev/null; then
        error "Cannot run docker commands. Add user to docker group or run with sudo."
        exit 1
    fi
    
    success "Docker is available"
}

# Step 1: Get Isaac-GR00T Repository
clone_isaac_groot() {
    log "Step 1: Getting Isaac-GR00T Repository..."
    
    if [ -d "Isaac-GR00T" ]; then
        warning "Isaac-GR00T directory already exists"
        read -p "Remove and re-clone? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf Isaac-GR00T
        else
            log "Using existing Isaac-GR00T directory"
            cd Isaac-GR00T
            return
        fi
    fi
    
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
    success "Cloned Isaac-GR00T repository"
    cd Isaac-GR00T
    
    log "Repository contents:"
    ls -la
}

# Step 2: Examine Docker Setup
examine_docker_setup() {
    log "Step 2: Examining Docker Setup..."
    
    log "Looking for Docker-related files..."
    find . -name "Dockerfile*" -o -name "docker*" -o -name "*.docker" || true
    
    log "Checking for docker directory..."
    ls -la docker/ 2>/dev/null || log "No docker directory found"
    
    log "Looking for build scripts..."
    find . -name "*build*" -name "*.sh" || true
}

# Step 3 Option C: Copy Dockerfile into Isaac-GR00T checkout
create_dockerfile() {
    log "Step 3 Option C: Setting up Dockerfile..."

    DOCKERFILE_SRC="$SCRIPT_DIR/Dockerfile.gr00t-gpu"

    if [ ! -f "$DOCKERFILE_SRC" ]; then
        error "Dockerfile.gr00t-gpu not found at $DOCKERFILE_SRC"
        exit 1
    fi

    if [ -f "Dockerfile" ]; then
        warning "Dockerfile already exists"
        read -p "Overwrite with compatible version? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Keeping existing Dockerfile"
            return
        fi
    fi

    cp "$DOCKERFILE_SRC" Dockerfile
    success "Copied Dockerfile.gr00t-gpu into Isaac-GR00T as Dockerfile"
}

# Build Docker container
build_container() {
    log "Building Docker container..."
    
    docker build -t isaac-gr00t .
    
    success "Docker container built successfully"
}

# Step 4: Test Container Structure
test_container() {
    log "Step 4: Testing Container Structure..."
    
    log "Testing if container builds successfully..."
    docker run --rm isaac-gr00t echo "Container built successfully!"
    
    log "Checking Python and packages..."
    docker run --rm isaac-gr00t python --version
    docker run --rm isaac-gr00t pip list | grep -E "(torch|transformers|zmq)" || true
    
    log "Finding inference scripts..."
    docker run --rm isaac-gr00t find /opt/Isaac-GR00T -name "*inference*" -type f || true
    
    log "Checking scripts directory..."
    docker run --rm isaac-gr00t ls -la /opt/Isaac-GR00T/scripts/ 2>/dev/null || log "No scripts directory found"
    
    success "Container structure verified"
}

# Step 5B: Download NVIDIA GR00T-N1.5-3B Model
download_nvidia_groot_model() {
    log "Step 5B: Downloading NVIDIA GR00T-N1.5-3B Model..."
    
    # Create checkpoints directory on host if not exists
    mkdir -p ./checkpoints
    log "Checkpoint directory: $(pwd)/checkpoints"
    
    # Use absolute path for volume mount to avoid issues
    CHECKPOINTS_DIR="$(pwd)/checkpoints"
    log "Using absolute path for volume mount: $CHECKPOINTS_DIR"
    
    log "Downloading nvidia/GR00T-N1.5-3B..."
    
    docker run --rm -v "$CHECKPOINTS_DIR":/data/checkpoints isaac-gr00t bash -c "
echo 'Starting NVIDIA GR00T-N1.5-3B model download...'
pip install huggingface_hub && python -c \"
from huggingface_hub import snapshot_download
import os
import sys

print('Downloading NVIDIA GR00T-N1.5-3B model...')
print(f'Target directory: /data/checkpoints/GR00T-N1.5-3B')

try:
    # Ensure target directory exists
    os.makedirs('/data/checkpoints/GR00T-N1.5-3B', exist_ok=True)
    
    print('Downloading NVIDIA GR00T-N1.5-3B model...')
    repo_path = snapshot_download(
        repo_id='nvidia/GR00T-N1.5-3B',
        local_dir='/data/checkpoints/GR00T-N1.5-3B',
        local_dir_use_symlinks=False,  # Ensure real files, not symlinks
        force_download=False  # Don't re-download if already exists
    )
    print(f'NVIDIA GR00T-N1.5-3B model downloaded to: {repo_path}')
    
    # Ensure permissions are correct
    import subprocess
    subprocess.run(['chmod', '-R', '755', '/data/checkpoints'], check=False)
    
    # Verify download by listing files
    print('Verifying NVIDIA GR00T-N1.5-3B download...')
    total_files = 0
    sample_files = []
    for root, dirs, files in os.walk('/data/checkpoints/GR00T-N1.5-3B'):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            if total_files <= 5:  # Show first 5 files
                sample_files.append(file_path)
    
    print(f'NVIDIA GR00T-N1.5-3B model files: {total_files}')
    print('Sample files:')
    for f in sample_files:
        print(f'  {f}')
    
    if total_files == 0:
        print('ERROR: No NVIDIA GR00T-N1.5-3B files downloaded!')
        sys.exit(1)
    else:
        print('NVIDIA GR00T-N1.5-3B model download completed successfully!')
        
except Exception as e:
    print(f'NVIDIA GR00T-N1.5-3B download failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\"
"
    
    # Verify NVIDIA GR00T-N1.5-3B model on host
    nvidia_file_count=$(find ./checkpoints/GR00T-N1.5-3B -type f 2>/dev/null | wc -l)
    log "NVIDIA GR00T-N1.5-3B files found on host: $nvidia_file_count"
    
    if [ "$nvidia_file_count" -gt 0 ]; then
        success "NVIDIA GR00T-N1.5-3B model downloaded successfully ($nvidia_file_count files)"
    else
        warning "NVIDIA GR00T-N1.5-3B model download may have failed"
        return 1
    fi
}

# Step 5C: Download Libero Spatial Checkpoint
download_libero_checkpoint() {
    log "Step 5 Option A: Downloading Libero Spatial Checkpoint..."
    
    # Create checkpoints directory on host
    mkdir -p ./checkpoints
    log "Created checkpoints directory: $(pwd)/checkpoints"
    
    log "Testing volume mount..."
    docker run --rm -v $(pwd)/checkpoints:/data/checkpoints isaac-gr00t bash -c "
echo 'Testing volume mount...'
touch /data/checkpoints/test_file.txt
ls -la /data/checkpoints/
"
    
    if [ -f "./checkpoints/test_file.txt" ]; then
        log "Volume mount working correctly"
        rm ./checkpoints/test_file.txt
    else
        error "Volume mount failed - checkpoints won't be saved to host"
        return 1
    fi
    
    log "Downloading youliangtan/gr00t-n1.5-libero-spatial-posttrain..."
    
    # Use absolute path for volume mount to avoid issues
    CHECKPOINTS_DIR="$(pwd)/checkpoints"
    log "Using absolute path for volume mount: $CHECKPOINTS_DIR"
    
    docker run --rm -v "$CHECKPOINTS_DIR":/data/checkpoints isaac-gr00t bash -c "
echo 'Confirming volume mount before download...'
ls -la /data/checkpoints/
pwd
echo 'Starting download...'

pip install huggingface_hub && python -c \"
from huggingface_hub import snapshot_download
import os
import sys

print('Downloading GR00T Libero Spatial checkpoint...')
print(f'Container working directory: {os.getcwd()}')
print(f'Target directory: /data/checkpoints/gr00t-libero-spatial')
print('Contents of /data/checkpoints before download:')
if os.path.exists('/data/checkpoints'):
    print(os.listdir('/data/checkpoints'))
else:
    print('Directory does not exist, creating...')
    os.makedirs('/data/checkpoints', exist_ok=True)

try:
    # Ensure target directory exists
    os.makedirs('/data/checkpoints/gr00t-libero-spatial', exist_ok=True)
    
    # Download the entire repository to the mounted volume
    # Download to a temporary location first, then copy to avoid issues
    print('Downloading checkpoint...')
    repo_path = snapshot_download(
        repo_id='youliangtan/gr00t-n1.5-libero-spatial-posttrain',
        local_dir='/data/checkpoints/gr00t-libero-spatial',
        local_dir_use_symlinks=False,  # Ensure real files, not symlinks
        force_download=False  # Don't re-download if already exists
    )
    print(f'Download completed to: {repo_path}')
    
    # Ensure permissions are correct
    import subprocess
    subprocess.run(['chmod', '-R', '755', '/data/checkpoints'], check=False)
    
    # Verify download by listing files
    print('Verifying download in container...')
    total_files = 0
    sample_files = []
    for root, dirs, files in os.walk('/data/checkpoints'):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            if total_files <= 10:  # Show first 10 files
                sample_files.append(file_path)
    
    print(f'Total files downloaded: {total_files}')
    print('Sample files:')
    for f in sample_files:
        print(f'  {f}')
    
    if total_files == 0:
        print('ERROR: No files downloaded!')
        sys.exit(1)
    else:
        print('Download completed successfully!')
        
except Exception as e:
    print(f'Download failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\"

echo 'Post-download verification in container:'
ls -la /data/checkpoints/
find /data/checkpoints -type f | head -5
"
    
    log "Verifying downloaded checkpoints on host..."
    
    # Check if files appear on host (volume mount worked)
    host_file_count=$(find ./checkpoints -type f 2>/dev/null | wc -l)
    log "Files found on host: $host_file_count"
    
    if [ "$host_file_count" -eq 0 ]; then
        warning "Volume mount didn't work during download. Manually copying files from container..."
        
        # Start a temporary container to copy files
        log "Starting temporary container to copy checkpoints..."
        docker run -d --name temp-gr00t isaac-gr00t tail -f /dev/null
        
        # Copy files from container to host
        log "Copying checkpoints from container to host..."
        docker cp temp-gr00t:/root/.cache/huggingface/hub/models--youliangtan--gr00t-n1.5-libero-spatial-posttrain/. ./checkpoints/gr00t-libero-spatial/ 2>/dev/null || \
        docker cp temp-gr00t:/data/checkpoints/. ./checkpoints/ 2>/dev/null || \
        {
            # Run the download in a new container and copy directly
            log "Alternative: Re-downloading and copying directly..."
            docker exec temp-gr00t bash -c "
                pip install huggingface_hub > /dev/null 2>&1
                python -c \"
from huggingface_hub import snapshot_download
repo_path = snapshot_download('youliangtan/gr00t-n1.5-libero-spatial-posttrain', local_dir='/tmp/checkpoint')
print(f'Downloaded to: {repo_path}')
\"
            "
            docker cp temp-gr00t:/tmp/checkpoint/. ./checkpoints/gr00t-libero-spatial/
        }
        
        # Cleanup temporary container
        docker stop temp-gr00t
        docker rm temp-gr00t
        
        log "Manual copy completed"
    fi
    
    # Verify final result
    final_file_count=$(find ./checkpoints -type f 2>/dev/null | wc -l)
    log "Final file count on host: $final_file_count"
    
    if [ "$final_file_count" -gt 0 ]; then
        log "Sample files on host:"
        find ./checkpoints -type f | head -5
        
        log "Looking for model files..."
        find ./checkpoints -name "*.bin" -o -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" || log "No standard model files found"
        
        success "Checkpoint downloaded successfully ($final_file_count files)"
    else
        error "Failed to get checkpoint files to host!"
        error "Files exist in container but not accessible on host"
        error "Try running: docker cp gr00t-inference:/data/checkpoints/. ./checkpoints/"
        return 1
    fi
}

# Step 5D: Download Libero-90 Posttrain Checkpoint
download_libero_90_checkpoint() {
    log "Step 5D: Downloading Libero-90 Posttrain Checkpoint..."
    
    # Create checkpoints directory on host if not exists
    mkdir -p ./checkpoints
    log "Checkpoint directory: $(pwd)/checkpoints"
    
    # Use absolute path for volume mount to avoid issues
    CHECKPOINTS_DIR="$(pwd)/checkpoints"
    log "Using absolute path for volume mount: $CHECKPOINTS_DIR"
    
    log "Downloading youliangtan/gr00t-n1.5-libero-90-posttrain..."
    
    docker run --rm -v "$CHECKPOINTS_DIR":/data/checkpoints isaac-gr00t bash -c "
echo 'Starting Libero-90 Posttrain checkpoint download...'
pip install huggingface_hub && python -c \"
from huggingface_hub import snapshot_download
import os
import sys

print('Downloading GR00T Libero-90 Posttrain checkpoint...')
print(f'Target directory: /data/checkpoints/gr00t-n1.5-libero-90-posttrain')

try:
    # Ensure target directory exists
    os.makedirs('/data/checkpoints/gr00t-n1.5-libero-90-posttrain', exist_ok=True)
    
    print('Downloading Libero-90 Posttrain checkpoint...')
    repo_path = snapshot_download(
        repo_id='youliangtan/gr00t-n1.5-libero-90-posttrain',
        local_dir='/data/checkpoints/gr00t-n1.5-libero-90-posttrain',
        local_dir_use_symlinks=False,  # Ensure real files, not symlinks
        force_download=False  # Don't re-download if already exists
    )
    print(f'Libero-90 Posttrain checkpoint downloaded to: {repo_path}')
    
    # Ensure permissions are correct
    import subprocess
    subprocess.run(['chmod', '-R', '755', '/data/checkpoints'], check=False)
    
    # Verify download by listing files
    print('Verifying Libero-90 Posttrain download...')
    total_files = 0
    sample_files = []
    for root, dirs, files in os.walk('/data/checkpoints/gr00t-n1.5-libero-90-posttrain'):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            if total_files <= 5:  # Show first 5 files
                sample_files.append(file_path)
    
    print(f'Libero-90 Posttrain checkpoint files: {total_files}')
    print('Sample files:')
    for f in sample_files:
        print(f'  {f}')
    
    if total_files == 0:
        print('ERROR: No Libero-90 Posttrain files downloaded!')
        sys.exit(1)
    else:
        print('Libero-90 Posttrain checkpoint download completed successfully!')
        
except Exception as e:
    print(f'Libero-90 Posttrain download failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\"
"
    
    # Verify Libero-90 Posttrain checkpoint on host
    libero90_file_count=$(find ./checkpoints/gr00t-n1.5-libero-90-posttrain -type f 2>/dev/null | wc -l)
    log "Libero-90 Posttrain files found on host: $libero90_file_count"
    
    if [ "$libero90_file_count" -gt 0 ]; then
        success "Libero-90 Posttrain checkpoint downloaded successfully ($libero90_file_count files)"
    else
        warning "Libero-90 Posttrain checkpoint download may have failed"
        return 1
    fi
}

# Step 5E: Download Libero-Long Posttrain Checkpoint
download_libero_long_checkpoint() {
    log "Step 5E: Downloading Libero-Long Posttrain Checkpoint..."

    # Create checkpoints directory on host if not exists
    mkdir -p ./checkpoints
    log "Checkpoint directory: $(pwd)/checkpoints"

    # Use absolute path for volume mount to avoid issues
    CHECKPOINTS_DIR="$(pwd)/checkpoints"
    log "Using absolute path for volume mount: $CHECKPOINTS_DIR"

    log "Downloading youliangtan/gr00t-n1.5-libero-long-posttrain..."

    docker run --rm -v "$CHECKPOINTS_DIR":/data/checkpoints isaac-gr00t bash -c "
echo 'Starting Libero-Long Posttrain checkpoint download...'
pip install huggingface_hub && python -c \"
from huggingface_hub import snapshot_download
import os
import sys

print('Downloading GR00T Libero-Long Posttrain checkpoint...')
print(f'Target directory: /data/checkpoints/gr00t-n1.5-libero-long-posttrain')

try:
    # Ensure target directory exists
    os.makedirs('/data/checkpoints/gr00t-n1.5-libero-long-posttrain', exist_ok=True)

    print('Downloading Libero-Long Posttrain checkpoint...')
    repo_path = snapshot_download(
        repo_id='youliangtan/gr00t-n1.5-libero-long-posttrain',
        local_dir='/data/checkpoints/gr00t-n1.5-libero-long-posttrain',
        local_dir_use_symlinks=False,  # Ensure real files, not symlinks
        force_download=False  # Don't re-download if already exists
    )
    print(f'Libero-Long Posttrain checkpoint downloaded to: {repo_path}')

    # Ensure permissions are correct
    import subprocess
    subprocess.run(['chmod', '-R', '755', '/data/checkpoints'], check=False)

    # Verify download by listing files
    print('Verifying Libero-Long Posttrain download...')
    total_files = 0
    sample_files = []
    for root, dirs, files in os.walk('/data/checkpoints/gr00t-n1.5-libero-long-posttrain'):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            if total_files <= 5:  # Show first 5 files
                sample_files.append(file_path)

    print(f'Libero-Long Posttrain checkpoint files: {total_files}')
    print('Sample files:')
    for f in sample_files:
        print(f'  {f}')

    if total_files == 0:
        print('ERROR: No Libero-Long Posttrain files downloaded!')
        sys.exit(1)
    else:
        print('Libero-Long Posttrain checkpoint download completed successfully!')

except Exception as e:
    print(f'Libero-Long Posttrain download failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\"
"

    # Verify Libero-Long Posttrain checkpoint on host
    liberolong_file_count=$(find ./checkpoints/gr00t-n1.5-libero-long-posttrain -type f 2>/dev/null | wc -l)
    log "Libero-Long Posttrain files found on host: $liberolong_file_count"

    if [ "$liberolong_file_count" -gt 0 ]; then
        success "Libero-Long Posttrain checkpoint downloaded successfully ($liberolong_file_count files)"
    else
        warning "Libero-Long Posttrain checkpoint download may have failed"
        return 1
    fi
}

# Step 6: Run Container with GPU Support
run_container() {
    log "Step 6: Running Container with GPU Support..."
    
    # Stop and remove existing container if it exists (running or stopped)
    if docker ps -a -q -f name=gr00t-inference | grep -q .; then
        warning "Found existing gr00t-inference container, removing it..."
        # Stop if running
        if docker ps -q -f name=gr00t-inference | grep -q .; then
            log "Stopping running container..."
            docker stop gr00t-inference
        fi
        # Remove container
        log "Removing existing container..."
        docker rm gr00t-inference
        success "Existing container removed"
    fi
    
    log "Starting GPU-enabled container..."
    docker run -d --name gr00t-inference \
      --gpus all \
      -v "$(pwd)/checkpoints":/data/checkpoints \
      -p 5555:5555 -p 8000:8000 \
      isaac-gr00t tail -f /dev/null
    
    log "Verifying GPU access in container..."
    docker exec gr00t-inference nvidia-smi
    
    success "Container running with GPU support"
}

# Step 7: Test Inference Service
test_inference() {
    log "Step 7: Testing Inference Service..."
    
    log "Checking if inference service script exists..."
    docker exec gr00t-inference find /opt/Isaac-GR00T -name "*inference*.py" -type f || true
    
    log "Trying inference service help..."
    if docker exec gr00t-inference python /opt/Isaac-GR00T/scripts/inference_service.py --help 2>/dev/null; then
        success "Inference service script found and working"
    else
        warning "Standard inference script not found, checking alternatives..."
        docker exec gr00t-inference find /opt/Isaac-GR00T -name "*.py" -exec grep -l "inference" {} \; || true
    fi
    
    log "Testing basic Python imports in container..."
    docker exec gr00t-inference python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name()}')
"
    
    success "Container is ready for inference"
}

# Step 8: Create verification script
create_verification() {
    log "Step 8: Creating verification script..."
    
    cat > verify_gr00t.py << 'EOF'
#!/usr/bin/env python3
"""
Verification script for GR00T setup
Run this from the strands-robots-sim directory
"""

import subprocess
import requests
import json

def check_container():
    """Check if gr00t-inference container is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'gr00t-inference' in result.stdout:
            print("✅ gr00t-inference container is running")
            return True
        else:
            print("❌ gr00t-inference container is not running")
            return False
    except Exception as e:
        print(f"❌ Error checking container: {e}")
        return False

def check_checkpoints():
    """Check if checkpoints are available"""
    try:
        result = subprocess.run(['docker', 'exec', 'gr00t-inference', 'ls', '-la', '/data/checkpoints/'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            checkpoints_found = []
            if 'GR00T-N1.5-3B' in result.stdout:
                checkpoints_found.append("NVIDIA GR00T-N1.5-3B")
            if 'gr00t-libero-spatial' in result.stdout:
                checkpoints_found.append("Libero Spatial")
            if 'gr00t-n1.5-libero-90-posttrain' in result.stdout:
                checkpoints_found.append("Libero-90")
            if 'gr00t-n1.5-libero-long-posttrain' in result.stdout:
                checkpoints_found.append("Libero-Long")

            if checkpoints_found:
                print(f"✅ Checkpoints available: {', '.join(checkpoints_found)}")
                return True
            else:
                print("❌ No checkpoints found")
                return False
        else:
            print("❌ Cannot access checkpoints directory")
            return False
    except Exception as e:
        print(f"❌ Error checking checkpoints: {e}")
        return False

def check_gpu():
    """Check GPU access in container"""
    try:
        result = subprocess.run(['docker', 'exec', 'gr00t-inference', 'nvidia-smi'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU access confirmed")
            return True
        else:
            print("❌ GPU not accessible in container")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    print("🔍 Verifying GR00T setup...")
    print()
    
    container_ok = check_container()
    checkpoints_ok = check_checkpoints()
    gpu_ok = check_gpu()
    
    print()
    if container_ok and checkpoints_ok and gpu_ok:
        print("🎉 GR00T setup verification PASSED!")
        print("You can now use the strands-robots-sim gr00t_inference tool")
    else:
        print("⚠️  GR00T setup verification FAILED")
        print("Check the logs above for issues")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x verify_gr00t.py
    success "Created verification script: verify_gr00t.py"
}

# Main execution
main() {
    log "Starting Isaac-GR00T GPU Setup Script"
    log "Following docker-setup-guide.md: Step 3 Option C + Step 5 Option A"
    echo
    
    # Prerequisites
    check_gpu
    check_docker
    echo
    
    # Main setup steps
    clone_isaac_groot
    echo
    
    examine_docker_setup
    echo
    
    create_dockerfile
    echo
    
    build_container
    echo
    
    test_container
    echo
    
    download_nvidia_groot_model
    echo
    
    download_libero_checkpoint
    echo
    
    download_libero_90_checkpoint
    echo

    download_libero_long_checkpoint
    echo

    run_container
    echo
    
    test_inference
    echo
    
    create_verification
    echo
    
    success "🎉 Isaac-GR00T GPU setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Run './verify_gr00t.py' to verify the setup"
    echo "2. Use the strands-robots-sim gr00t_inference tool"
    echo "3. Test with SimEnv integration"
    echo
    echo "Container info:"
    echo "  Name: gr00t-inference"
    echo "  Ports: 5555, 8000"
    echo "  Checkpoints: ./checkpoints/"
    echo "  GPU: Enabled"
}

# Run main function
main "$@"
