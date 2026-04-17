"""RunPod serverless handler for InfiniteTalk video generation."""

import json
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any
import logging
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(job):
    """RunPod serverless handler function.

    Expects input JSON:
    {
        "image": "base64-encoded-image or URL",
        "audio": "base64-encoded-audio or URL",
        "duration": 5,
        "resolution": "480p",
        "steps": 40,
        "audio_cfg": 4.0,
        "text_cfg": 5.0,
        "prompt": "optional description"
    }
    """
    try:
        job_input = job["input"]
        logger.info(f"Processing job: {job['id']}")

        # Validate required inputs
        if "image" not in job_input and "video" not in job_input:
            return {"error": "Either 'image' or 'video' is required"}
        if "audio" not in job_input:
            return {"error": "'audio' is required"}

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Handle image input
            image_path = None
            if "image" in job_input:
                image_path = _save_base64_file(
                    job_input["image"],
                    tmpdir / "input_image.png"
                )

            # Handle video input
            video_path = None
            if "video" in job_input:
                video_path = _save_base64_file(
                    job_input["video"],
                    tmpdir / "input_video.mp4"
                )

            # Handle audio input
            audio_path = _save_base64_file(
                job_input["audio"],
                tmpdir / "input_audio.wav"
            )

            # Prepare generation parameters
            duration = job_input.get("duration", 5)
            resolution = job_input.get("resolution", "480p")
            steps = job_input.get("steps", 40)
            audio_cfg = job_input.get("audio_cfg", 4.0)
            text_cfg = job_input.get("text_cfg", 5.0)
            prompt = job_input.get("prompt", "A professional talking avatar")

            # Output path
            output_path = tmpdir / "output.mp4"

            # Build command
            cmd = [
                "python",
                str(Path.home() / "Documents/web-projects/InfiniteTalk/generate_infinitetalk.py"),
                "--ckpt_dir", str(Path.home() / ".avatar-studio/models/Wan2.1-I2V-14B-480P"),
                "--wav2vec_dir", str(Path.home() / ".avatar-studio/models/chinese-wav2vec2-base"),
                "--infinitetalk_dir", str(Path.home() / ".avatar-studio/models/InfiniteTalk/single/infinitetalk.safetensors"),
                "--input_json", str(_create_input_json(tmpdir, image_path, video_path, audio_path, prompt)),
                "--size", f"infinitetalk-{resolution.rstrip('p')}",
                "--sample_steps", str(steps),
                "--mode", "streaming",
                "--motion_frame", "9",
                "--sample_audio_guide_scale", str(audio_cfg),
                "--sample_text_guide_scale", str(text_cfg),
                "--max_frame_num", str(int(duration * 25)),
                "--save_file", str(output_path.parent / output_path.stem),
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            # Execute generation
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Generation failed: {result.stderr}")
                return {"error": f"Generation failed: {result.stderr}"}

            # Read output video
            if not output_path.exists():
                return {"error": "Output video not found"}

            # Encode output as base64
            with open(output_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()

            return {
                "status": "success",
                "video": video_b64,
                "duration": duration,
                "resolution": resolution
            }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}


def _save_base64_file(data: str, output_path: Path) -> Path:
    """Save base64-encoded data to file."""
    try:
        # Try to decode as base64
        file_data = base64.b64decode(data)
    except Exception:
        # If not base64, assume it's raw binary data
        file_data = data.encode() if isinstance(data, str) else data

    with open(output_path, "wb") as f:
        f.write(file_data)

    return output_path


def _create_input_json(tmpdir: Path, image_path, video_path, audio_path, prompt) -> Path:
    """Create input JSON for InfiniteTalk."""
    input_json = {
        "prompt": prompt,
        "cond_video": str(image_path or video_path),
        "cond_audio": {"person1": str(audio_path)}
    }

    json_path = tmpdir / "input_config.json"
    with open(json_path, "w") as f:
        json.dump(input_json, f)

    return json_path


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
