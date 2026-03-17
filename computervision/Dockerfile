# syntax=docker/dockerfile:1
# Use official PyTorch image: torch 2.6.0 + torchaudio 2.6.0 + torchvision 0.21.0
# + CUDA 12.4 + cuDNN 9 are pre-installed, avoiding ~3 GB of downloads on cache misses.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl \
    # Required for mysqlclient
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    # Required for mysqldump
    default-mysql-client \
    # Required for opencv-python
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy code (only used in production via normal docker-compose)
# In development, this will be overridden by a volume mount
COPY . .

# Not required for computervision service
# EXPOSE ${API_DOCKER_PORT:-5555}

CMD ["python", "JobExecuter.py"]

