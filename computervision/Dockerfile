FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code (only used in production via normal docker-compose)
# In development, this will be overridden by a volume mount
COPY . .

# Not required for computervision service
# EXPOSE ${API_DOCKER_PORT:-5555}

CMD ["python", "JobExecuter.py"]

