"""Video generation logic using Infinite Talk."""

import json
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger
from huggingface_hub import snapshot_download
from .config import Config


class AvatarGenerator:
    """Handles video generation using Infinite Talk."""

    def __init__(self, config: Config, gpu_mem: str = 'full', download_models: bool = True):
        self.config = config
        self.gpu_mem = gpu_mem
        self.infinitetalk_repo = Path.home() / 'Documents/web-projects/InfiniteTalk'

        if not self.infinitetalk_repo.exists():
            raise RuntimeError(
                f'InfiniteTalk repo not found at {self.infinitetalk_repo}\n'
                'Clone it first: git clone https://github.com/MeiGen-AI/InfiniteTalk.git'
            )

        if download_models:
            self._ensure_models()

    def _ensure_models(self):
        """Download models if not already present."""
        logger.info('Checking models...')

        models = [
            ('wan_i2v', 'Wan-AI/Wan2.1-I2V-14B-480P'),
            ('wav2vec', 'TencentGameMate/chinese-wav2vec2-base'),
            ('infinitetalk', 'MeiGen-AI/InfiniteTalk'),
        ]

        for key, repo in models:
            model_path = self.config.get_model_path(key)
            if not model_path.exists():
                logger.info(f'Downloading {key} from {repo}...')
                try:
                    snapshot_download(
                        repo_id=repo,
                        local_dir=str(model_path),
                        repo_type='model'
                    )
                except Exception as e:
                    logger.warning(f'Failed to download {repo}: {e}')

    def generate(
        self,
        input_path: str,
        audio_path: str,
        output_path: str,
        duration: int = 5,
        resolution: str = '480p',
        steps: int = 40,
        motion_frames: int = 9,
        audio_cfg: float = 4.0,
        text_cfg: float = 5.0,
        prompt: Optional[str] = None,
        is_image: bool = True
    ) -> str:
        """Generate avatar video using Infinite Talk.

        Args:
            input_path: Path to input image or video
            audio_path: Path to input audio file
            output_path: Path for output video
            duration: Video duration in seconds
            resolution: Output resolution (480p or 720p)
            steps: Number of diffusion steps
            motion_frames: Number of motion frames
            audio_cfg: Audio guidance scale
            text_cfg: Text guidance scale
            prompt: Optional text description
            is_image: True for image-to-video, False for video-to-video

        Returns:
            Path to generated video
        """

        # Create input JSON config
        if not prompt:
            prompt = 'A professional talking avatar'

        input_json = {
            'prompt': prompt,
            'cond_video': input_path,
            'cond_audio': {'person1': audio_path}
        }

        input_json_path = Path(output_path).parent / 'input_config.json'
        with open(input_json_path, 'w') as f:
            json.dump(input_json, f)

        # Build command
        infinitetalk_weights = self.config.get_model_path('infinitetalk') / 'single/infinitetalk.safetensors'

        cmd = [
            'python', str(self.infinitetalk_repo / 'generate_infinitetalk.py'),
            '--ckpt_dir', str(self.config.get_model_path('wan_i2v')),
            '--wav2vec_dir', str(self.config.get_model_path('wav2vec')),
            '--infinitetalk_dir', str(infinitetalk_weights),
            '--input_json', str(input_json_path),
            '--size', f'infinitetalk-{resolution.rstrip("p")}',
            '--sample_steps', str(steps),
            '--mode', 'streaming',
            '--motion_frame', str(motion_frames),
            '--sample_audio_guide_scale', str(audio_cfg),
            '--sample_text_guide_scale', str(text_cfg),
            '--max_frame_num', str(int(duration * 25)),  # 25 fps
            '--save_file', str(Path(output_path).parent / Path(output_path).stem),
        ]

        # Add memory optimization for low GPU memory
        if self.gpu_mem == 'low':
            cmd.append('--num_persistent_param_in_dit')
            cmd.append('0')

        logger.info(f'Running: {" ".join(cmd)}')

        try:
            subprocess.run(cmd, cwd=str(self.infinitetalk_repo), check=True)
            logger.info(f'✓ Generated video saved')
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Generation failed with exit code {e.returncode}')
