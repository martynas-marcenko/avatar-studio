"""RunPod serverless handler for Wan2.2-S2V audio-driven video generation via Diffusers.

Reference implementation: https://github.com/inference-sh/grid/blob/main/native/generative/video/wan-2-2-s2v/inference.py
"""

import base64
import gc
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import runpod
import torch
from diffusers import AutoencoderKLWan, WanSpeechToVideoPipeline
from diffusers.utils import export_to_video, load_audio, load_image
from transformers import Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "tolgacangoz/Wan2.2-S2V-14B-Diffusers"
_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    logger.info(f"Loading components from {MODEL_ID}...")
    audio_encoder = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, subfolder="audio_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )

    logger.info("Assembling pipeline...")
    _pipe = WanSpeechToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        audio_encoder=audio_encoder,
        torch_dtype=torch.bfloat16,
    )
    _pipe.to("cuda")

    try:
        _pipe.transformer.set_attention_backend("flash")
    except Exception as e:
        logger.warning(f"Could not set flash attention: {e}")

    logger.info("Pipeline ready")
    return _pipe


def _aspect_ratio_resize(image, max_area):
    h, w = image.size[1], image.size[0]
    aspect = w / h
    target_w = int(((max_area * aspect) ** 0.5) // 64 * 64)
    target_h = int(((max_area / aspect) ** 0.5) // 64 * 64)
    image = image.resize((target_w, target_h))
    return image, target_h, target_w


def _merge_audio(video_path: str, audio_path: str):
    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_merged{ext}"
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", temp_output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    shutil.move(temp_output, video_path)


def _save_base64(data: str, path: Path) -> Path:
    path.write_bytes(base64.b64decode(data))
    return path


def handler(job):
    try:
        pipe = _get_pipeline()
        job_input = job["input"]
        logger.info(f"Processing job: {job['id']}")

        if "image" not in job_input or "audio" not in job_input:
            return {"error": "'image' and 'audio' are required"}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = _save_base64(job_input["image"], tmpdir / "image.png")
            audio_path = _save_base64(job_input["audio"], tmpdir / "audio.wav")
            video_path = tmpdir / "output.mp4"

            prompt = job_input.get("prompt", "A person talking naturally")
            resolution = job_input.get("resolution", "480p")
            num_frames_per_chunk = int(job_input.get("num_frames_per_chunk", 81))
            fps = int(job_input.get("fps", 16))
            seed = job_input.get("seed")

            max_area = 480 * 832 if resolution == "480p" else 720 * 1280

            logger.info("Loading and resizing image...")
            first_frame = load_image(str(image_path))
            first_frame, height, width = _aspect_ratio_resize(first_frame, max_area)

            logger.info("Loading audio...")
            audio, sampling_rate = load_audio(str(audio_path))

            generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None

            logger.info(f"Generating video {width}x{height}, {num_frames_per_chunk} frames/chunk...")
            output = pipe(
                image=first_frame,
                audio=audio,
                sampling_rate=sampling_rate,
                prompt=prompt,
                height=height,
                width=width,
                num_frames_per_chunk=num_frames_per_chunk,
                generator=generator,
            ).frames[0]

            export_to_video(output, str(video_path), fps=fps)
            _merge_audio(str(video_path), str(audio_path))

            video_b64 = base64.b64encode(video_path.read_bytes()).decode()

            del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "status": "success",
                "video": video_b64,
                "resolution": f"{width}x{height}",
                "fps": fps,
            }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
