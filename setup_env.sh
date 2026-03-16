#!/bin/bash

# Environment name
ENV_NAME="dl-doa"

# Python version
PYTHON_VERSION="3.12"

echo "Creating Conda environment: $ENV_NAME"

# Create environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate environment

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing core scientific packages..."
conda install -y numpy scipy pandas matplotlib seaborn scikit-learn tqdm pyyaml jupyter ipykernel ipywidgets iprogress

echo "Installing deep learning libraries (PyTorch)..."
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch


# If you want GPU support instead, replace the above line with:
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Environment setup completed!"

echo "Activate the environment using:"
echo "conda activate $ENV_NAME"
