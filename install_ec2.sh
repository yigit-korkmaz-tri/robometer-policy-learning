# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Set Cache Directories
mkdir /opt/dlami/nvme/cache
echo 'export UV_CACHE_DIR="/opt/dlami/nvme/cache/uv"' >> ~/.bashrc
echo 'export HF_HOME="/opt/dlami/nvme/cache/huggingface"' >> ~/.bashrc
echo 'export OPENPI_DATA_HOME="/opt/dlami/nvme/cache/openpi"' >> ~/.bashrc
source ~/.bashrc

# Update submodules
git submodule update --init --recursive

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync