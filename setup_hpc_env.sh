#!/bin/bash
# One-time setup: run interactively on the CX3 login node.
#
# Why interactive (not a job)?
#   1. Compute nodes have no outbound internet — model weights must be
#      downloaded here, where internet access is available.
#   2. Avoids wasting GPU-billed time on installs and downloads.
#   3. The conda env and cached weights are stored on shared filesystems
#      ($HOME and $EPHEMERAL) so every subsequent job can use them directly.
#
# Usage:
#   ssh your_username@login.cx3.hpc.ic.ac.uk
#   git clone https://github.com/nballou/tiktok-classifier.git
#   cd tiktok-classifier
#   bash setup_hpc_env.sh

set -e

MODEL="google/gemma-4-26B-A4B-it"   # Must match the MODEL variable in run_*.pbs

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"

if conda env list | grep -q "^tiktok "; then
    echo "Conda env 'tiktok' already exists — skipping creation."
else
    echo "Creating conda env 'tiktok'..."
    conda create -n tiktok python=3.11 -y
fi

conda activate tiktok
pip install --quiet vllm pandas pyarrow "openai>=1.0" httpx

echo "Installed:"
pip show vllm openai pandas | grep -E "^(Name|Version)"

# ---------------------------------------------------------------------------
# 2. Download model weights to $EPHEMERAL
#    Uses huggingface_hub (installed with vllm) — no GPU required.
# ---------------------------------------------------------------------------
export HF_HOME=$EPHEMERAL/huggingface
mkdir -p $HF_HOME

echo ""
echo "Downloading model weights for $MODEL to \$EPHEMERAL/huggingface..."
echo "(~8GB for a 4B model — takes a few minutes)"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', cache_dir='$HF_HOME/hub')
print('Download complete.')
"

echo ""
echo "Setup complete. Run the test job next:"
echo "  mkdir -p ~/tiktok-classifier/logs"
echo "  qsub run_test.pbs"
