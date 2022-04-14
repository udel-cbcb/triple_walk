FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ssh-client \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir -p /root/workdir && mkdir -p /root/.ssh && mkdir -p /root/.vscode-server && pip install ninja
WORKDIR /root/workdir

# install nsights compute
RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2020.3.0 && \
     rm -rf /var/lib/apt/lists/*

# update the path
ENV PATH="/opt/nvidia/nsight-compute/2020.3.0/:${PATH}"

# Set the default command to python3
CMD ["bash"]