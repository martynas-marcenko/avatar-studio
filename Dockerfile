FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Clone InfiniteTalk
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install -q \
    runpod \
    diffusers==0.36.0 \
    transformers==4.54.0 \
    accelerate \
    peft \
    safetensors \
    omegaconf \
    hydra-core \
    loguru \
    pydantic \
    huggingface_hub \
    opencv-python \
    soundfile \
    librosa \
    torchdiffeq \
    tensordict \
    einops \
    easydict \
    xfuser
# Install flash-attn separately (requires torch from previous step)
RUN pip install -q flash-attn

# Copy handler
COPY runpod_handler.py /app/runpod_handler.py

# Models will be downloaded on first request and cached
# (avoids 44GB download during build)
RUN mkdir -p /root/.avatar-studio/models

# Start RunPod handler
CMD ["python", "-u", "/app/runpod_handler.py"]
