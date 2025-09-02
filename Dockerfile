FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy the repo code into the image
COPY . /workspace

# Install project dependencies
RUN pip install --upgrade pip \
 && pip install -e . \
 && pip install flash-attn --no-build-isolation