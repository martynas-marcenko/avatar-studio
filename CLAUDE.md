# Avatar Studio

CLI tool for generating talking avatar videos using Infinite Talk.

## Overview

Avatar Studio is a command-line interface for creating audio-driven video content. It wraps MeiGen's InfiniteTalk model to generate synchronized avatar videos from images/video + audio input.

## Project Structure

```
avatar-studio/
├── src/
│   ├── cli.py              # Main CLI entry point
│   ├── generator.py        # Video generation logic
│   └── config.py           # Configuration management
├── tests/                  # Test suite
├── config/
│   └── models.json         # Model paths and settings
├── outputs/                # Generated videos
└── .claude/                # Claude Code configuration
```

## Dependencies

- Python 3.10+
- PyTorch with CUDA support
- Infinite Talk models (auto-downloaded)
- Click (CLI framework)

## Usage

```bash
# Generate a 5-second avatar video
avatar-studio generate \
  --image path/to/image.png \
  --audio path/to/audio.wav \
  --duration 5 \
  --output outputs/video.mp4
```

## Model Management

Models are stored in `~/.avatar-studio/models/` and auto-downloaded on first use.

## Development Notes

- Keep CLI simple and focused
- Leverage Infinite Talk's existing inference code
- Support both image-to-video and video-to-video modes
