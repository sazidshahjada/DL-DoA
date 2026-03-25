#!/bin/bash
# This script sets up a Conda environment for the DL-DoA project and installs the required packages.
# Usage: ./setup_env.sh

# Take input for environment name
read -p "Enter the name for the Conda environment: " ENV_NAME
PYTHON_VERSION="3.12"

echo "Creating Conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing packages..."
pip install -r requirements.txt

if [[ "$(uname -s)" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "No NVIDIA GPU detected. Installing CPU-only version of PyTorch..."
        pip install torch torchvision torchaudio
    fi
else
    echo "Non-Linux OS detected. Installing CPU-only version of PyTorch..."
    pip install torch torchvision torchaudio
fi

echo "Environment setup completed!"

echo "Activate the environment using:"
echo "conda activate $ENV_NAME"
