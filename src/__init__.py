"""Avatar Studio - Audio-driven avatar video generation CLI."""

__version__ = '0.1.0'

from .cli import cli
from .generator import AvatarGenerator
from .config import Config

__all__ = ['cli', 'AvatarGenerator', 'Config']
