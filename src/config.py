"""Configuration management for Avatar Studio."""

import json
from pathlib import Path
from typing import Dict, Any
import os


class Config:
    """Manages Avatar Studio configuration and model paths."""

    def __init__(self):
        self.home_dir = Path.home()
        self.config_dir = self.home_dir / '.avatar-studio'
        self.models_dir = self.config_dir / 'models'
        self.config_path = self.config_dir / 'config.json'

        # Create directories
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        # Load or create config
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file or create default."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)

        # Default config
        default = {
            'models': {
                'wan_i2v': 'Wan2.1-I2V-14B-480P',
                'wav2vec': 'chinese-wav2vec2-base',
                'infinitetalk': 'InfiniteTalk'
            },
            'huggingface_org': 'MeiGen-AI',
            'inference': {
                'default_steps': 40,
                'default_resolution': '480p',
                'default_audio_cfg': 4.0,
                'default_text_cfg': 5.0
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(default, f, indent=2)

        return default

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_model_path(self, model_key: str) -> Path:
        """Get local path for a model."""
        model_name = self.get(f'models.{model_key}')
        return self.models_dir / model_name if model_name else None

    def get_huggingface_repo(self, model_key: str) -> str:
        """Get Hugging Face repo ID for a model."""
        model_name = self.get(f'models.{model_key}')
        org = self.get('huggingface_org')
        return f'{org}/{model_name}' if model_name else None
