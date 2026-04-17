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
    xfuser \
    flash-attn

# Copy handler
COPY runpod_handler.py /app/runpod_handler.py

# Download models (this will take a while)
RUN mkdir -p ~/.avatar-studio/models && \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Wan-AI/Wan2.1-I2V-14B-480P', local_dir='/root/.avatar-studio/models/Wan2.1-I2V-14B-480P'); \
    snapshot_download('TencentGameMate/chinese-wav2vec2-base', local_dir='/root/.avatar-studio/models/chinese-wav2vec2-base'); \
    snapshot_download('MeiGen-AI/InfiniteTalk', local_dir='/root/.avatar-studio/models/InfiniteTalk')"

# Start RunPod handler
CMD ["python", "-u", "/app/runpod_handler.py"]
