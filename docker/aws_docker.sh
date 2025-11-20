#!/bin/bash

# Check if all necessary arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <script> <checkpoint_file>"
    exit 1
fi

# Assign command-line arguments to variables
SCRIPT=$1                # Path to your training script (input from terminal)
CHECKPOINT_FILE=$2       # Specific checkpoint file to check for (input from terminal)

# Memory and GPU thresholds
MEMORY_THRESHOLD=85      # Memory usage threshold in percentage
GPU_THRESHOLD=21000      # GPU memory threshold in MB (21 GB)

# Function to check GPU usage
check_gpu_usage() {
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
    if [ "$GPU_USED" -gt "$GPU_THRESHOLD" ]; then
        echo "GPU memory usage is too high: $GPU_USED MB"
        return 1  # GPU memory exceeds threshold
    else
        return 0  # GPU memory is within acceptable range
    fi
}

# Function to check memory usage
check_memory_usage() {
    MEMORY_USED=$(free | awk '/Mem/ { printf("%.2f"), $3/$2 * 100.0 }')
    MEMORY_THRESHOLD=80  # Set your memory threshold here
    
    echo "Memory usage: $MEMORY_USED%"
    # Check if memory usage exceeds the threshold
    if (( ${MEMORY_USED%.*} > MEMORY_THRESHOLD )); then
        echo "Warning: Memory usage is too high!"
        return 1
    else
        return 0  # Memory usage is within acceptable range
    fi
}

# Function to check if the specified checkpoint file exists
check_checkpoint() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        return 0  # File exists, no need to restart
    else
        echo "Checkpoint $CHECKPOINT_FILE is missing."
        return 1  # File is missing, need to restart
    fi
}

# Function to clean up and reset the system (Memory/GPU) when script fails or resources are high
cleanup_and_reset() {
    echo "Cleaning up resources..."
    
    # Kill processes using high memory
    echo "Killing high memory usage processes..."
    # pkill -f "python"  # Adjust this to target relevant processes
    
    # Wait before retrying
    # sleep 10
}

# Function to run the Docker container and training script
run_script() {
    echo "Running Docker container and script..."
    
    sudo docker run --name deform --entrypoint bash -it --gpus all \
    -e "ACCEPT_EULA=Y" --rm --network=host -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /data/IsaacLab:/data/IsaacLab:rw \
    docker.io/lemonla/deform:latest -c "cd /root/IsaacLab ; source /home/lme/.local/share/ov/pkg/miniconda/etc/profile.d/conda.sh ; conda init ; conda activate isaaclab \
    ; git config --global --add safe.directory /data/IsaacLab ;./isaaclab.sh -i;python tools/carb_setting.py /isaac-sim/kit/kernel/config/kit-core.json fatal && $SCRIPT ; exit" #$SCRIPT
    
    # Capture the exit status of the Docker command
    SCRIPT_EXIT_STATUS=$?
    
    sudo docker stop deform
    
    
    # Check if the checkpoint file exists after training
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Checkpoint $CHECKPOINT_FILE found."
        if [ $SCRIPT_EXIT_STATUS -eq 0 ]; then
            echo "Training finished successfully and checkpoint is present. No restart needed."
            exit 0  # Exit the loop and script if training finished successfully and checkpoint is present
        else
            echo "Training interrupted but checkpoint is present. No restart needed."
            exit 0  # Exit the loop and script if training was interrupted but the checkpoint is present
        fi
    else
        echo "Checkpoint $CHECKPOINT_FILE is missing."
        if [ $SCRIPT_EXIT_STATUS -eq 0 ]; then
            echo "Training finished successfully but checkpoint is missing. Restarting..."
        else
            echo "Training interrupted and checkpoint is missing. Restarting..."
        fi
        cleanup_and_reset  # Call the cleanup function and restart if the checkpoint is missing
    fi
}

# Main loop
while true; do
    # Check for GPU and memory usage
    check_gpu_usage
    GPU_STATUS=$?
    check_memory_usage
    MEMORY_STATUS=$?
    
    if [ "$GPU_STATUS" -ne 0 ] || [ "$MEMORY_STATUS" -ne 0 ]; then
        echo "High resource usage detected. Cleaning up and restarting the script..."
        cleanup_and_reset  # Call cleanup if resources are high
    fi
    
    # Check if the specific checkpoint file exists
    check_checkpoint
    CHECKPOINT_STATUS=$?
    
    if [ $CHECKPOINT_STATUS -ne 0 ]; then
        echo "Checkpoint is missing. Restarting the training..."
        run_script
    else
        echo "Checkpoint $CHECKPOINT_FILE is present. No need to restart."
        exit 0  # Exit the loop if the checkpoint exists
    fi
    
    # Sleep for a while before checking again (adjust as needed)
    sleep 30
done
