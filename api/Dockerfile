# syntax=docker/dockerfile:1
FROM python:3.12-slim

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

EXPOSE ${API_DOCKER_PORT:-5555}

CMD ["python", "app.py"]

