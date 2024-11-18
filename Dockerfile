FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install additional packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY pyproject.toml .
ENV UV_LINK_MODE=copy
RUN pip install uv && \
    uv sync
