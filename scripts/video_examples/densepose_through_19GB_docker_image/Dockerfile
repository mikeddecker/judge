# Use official Ubuntu 24.04 image
FROM ubuntu:24.04

# Set environment variables to non-interactive (this avoids some prompts during package installation)
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev \
    libomp-dev \
    libnccl2 \
    libnccl-dev \
    ca-certificates \
    software-properties-common \
    gnupg2 \
    lsb-release

# Add CUDA repository and install CUDA 12.6
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-6

# Install Python 3.12 (or any compatible version)
RUN apt-get install -y python3.12

# Set up pip for the correct Python version
RUN apt-get install -y python3-pip

RUN pip install opencv-python --break-system-packages
# Install PyTorch 2.1.0 with the necessary CUDA version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Clone Detectron2 repository
RUN git clone https://github.com/facebookresearch/detectron2.git /home/ubuntu/detectron2

# Install DensePose from Detectron2 repository
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose --break-system-packages

RUN pip install gradio --break-system-packages

# Clean up unnecessary apt-get files to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/ubuntu/detectron2

# RUN --mount=type=bind,target=/home/ubuntu/app.py,source=app.py

COPY 2024-pka-juniors-sipiro-elena.mp4 2024-pka-juniors-sipiro-elena.mp4

# Set entrypoint (you can override this with docker run command)
ENTRYPOINT [ "bash" ]
