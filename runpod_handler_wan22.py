"""RunPod serverless handler for Wan2.2-S2V audio-driven video generation via Diffusers."""

import base64
import logging
import tempfile
from pathlib import Path

import runpod
import torch
from diffusers import WanSpeechToVideoPipeline
from diffusers.utils import export_to_video, load_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "tolgacangoz/Wan2.2-S2V-14B-Diffusers"
_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    logger.info(f"Loading {MODEL_ID}...")
    _pipe = WanSpeechToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    _pipe.to("cuda")
    logger.info("Pipeline ready")
    return _pipe


def _save_base64(data: str, path: Path) -> Path:
    path.write_bytes(base64.b64decode(data))
    return path


def handler(job):
    try:
        pipe = _get_pipeline()
        job_input = job["input"]
        logger.info(f"Processing job: {job['id']}")

        if "image" not in job_input:
            return {"error": "'image' is required"}
        if "audio" not in job_input:
            return {"error": "'audio' is required"}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            image_path = _save_base64(job_input["image"], tmpdir / "image.png")
            audio_path = _save_base64(job_input["audio"], tmpdir / "audio.wav")
            output_path = tmpdir / "output.mp4"

            prompt = job_input.get("prompt", "A person talking naturally")
            num_frames = int(job_input.get("duration", 5)) * 24
            height = int(job_input.get("height", 480))
            width = int(job_input.get("width", 832))
            steps = int(job_input.get("steps", 40))
            guidance = float(job_input.get("guidance_scale", 5.0))

            image = load_image(str(image_path))

            logger.info(f"Generating {num_frames} frames at {width}x{height}...")
            output = pipe(
                image=image,
                audio=str(audio_path),
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).frames[0]

            export_to_video(output, str(output_path), fps=24)

            video_b64 = base64.b64encode(output_path.read_bytes()).decode()
            return {
                "status": "success",
                "video": video_b64,
                "frames": num_frames,
                "resolution": f"{width}x{height}",
            }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
