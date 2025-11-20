#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
    export BASH_SOURCE=$SCRIPT_DIR/setup_python_env.sh
fi
MY_DIR="$(realpath -s "$SCRIPT_DIR")"

# Set Isaac paths
export CARB_APP_PATH=$SCRIPT_DIR/kit
export EXP_PATH=$MY_DIR/apps
export ISAAC_PATH=$MY_DIR

# --- Backup conda CUDA/cuDNN paths BEFORE sourcing Isaac Sim's setup ---
CONDA_CUDA_PATHS=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep "$CONDA_PREFIX" | tr '\n' ':' | sed 's/:$//')

# Source Isaac Lab's python environment setup
. ${MY_DIR}/setup_python_env.sh

# Remove Kit's Python path to avoid interpreter conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "$SCRIPT_DIR/kit/python/lib/python3.10" | tr '\n' ':' | sed 's/:$//')

# --- Remove Kit CUDA libs from LD_LIBRARY_PATH ---
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" \
    | tr ':' '\n' \
    | grep -v "$SCRIPT_DIR/kit/lib" \
    | grep -v "$SCRIPT_DIR/kit/omni" \
    | tr '\n' ':' | sed 's/:$//')

# --- Prepend conda's CUDA/cuDNN paths back to the front ---
if [ -n "$CONDA_CUDA_PATHS" ]; then
    export LD_LIBRARY_PATH="$CONDA_CUDA_PATHS:$LD_LIBRARY_PATH"
fi
