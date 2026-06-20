# torch 2.6.0 + torchvision + torchaudio + CUDA 12.6 + cuDNN 9 pre-installed.
# Requires NVIDIA driver >= 525 (local: driver 580 / CUDA 13 is compatible).
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

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
    libglib2.0-0 \
    # Venv required
    python3-venv \
    # Moviepy requires ffmpeg
    ffmpeg
    # Required for pyodbc (likely, so comment for now)
    # unixodbc-dev

WORKDIR /app

# Copy and install dependencies
COPY requirements.in .

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel pip-tools && \
    pip-compile requirements.in --output-file requirements.txt && \
    pip install -r requirements.txt

# Copy code (only used in production via normal docker-compose)
# In development, this will be overridden by a volume mount
COPY . .

# Not required for computervision service
# EXPOSE ${API_DOCKER_PORT:-5555}

CMD ["python", "JobExecuter.py"]

