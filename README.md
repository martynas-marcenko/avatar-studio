# Avatar Studio

A CLI tool for generating talking avatar videos using MeiGen's Infinite Talk model.

## Quick Start

### 1. Clone Infinite Talk (required dependency)

```bash
cd ~/Documents/web-projects
git clone https://github.com/MeiGen-AI/InfiniteTalk.git
```

### 2. Install Avatar Studio

```bash
cd avatar-studio
pip install -e .
```

### 3. Generate a video

```bash
avatar-studio generate \
  --image path/to/image.png \
  --audio path/to/audio.wav \
  --duration 5 \
  --output outputs/video.mp4
```

## Commands

### generate

Generate an avatar video from image/video + audio.

```bash
avatar-studio generate \
  --image INPUT_IMAGE \
  --audio INPUT_AUDIO.wav \
  --duration 5 \
  --resolution 480p \
  --output outputs/output.mp4
```

**Options:**
- `--image`: Input image (for image-to-video mode)
- `--video`: Input video (for video-to-video mode)
- `--audio`: Input audio file (WAV format) **required**
- `--output`: Output video path (default: `outputs/output.mp4`)
- `--duration`: Video duration in seconds (default: 5)
- `--resolution`: Output resolution - `480p` or `720p` (default: `480p`)
- `--steps`: Diffusion steps, 8-40 (lower = faster, less quality)
- `--audio-cfg`: Audio guidance scale, 3-5 recommended (default: 4.0)
- `--prompt`: Optional text description of the avatar

### info

Show model status and configuration.

```bash
avatar-studio info
```

## Model Management

Models are automatically downloaded to `~/.avatar-studio/models/` on first use:
- `Wan2.1-I2V-14B-480P` - Base image-to-video model
- `chinese-wav2vec2-base` - Audio encoder
- `InfiniteTalk` - Audio conditioning weights

## Memory Optimization

For GPUs with limited VRAM, use `--gpu-mem low`:

```bash
avatar-studio generate --image image.png --audio audio.wav --gpu-mem low
```

## Performance Tips

- **Quality**: Increase `--steps` (20-40 recommended, 40 = best quality)
- **Speed**: Decrease `--steps` (8-12 for quick generation)
- **Lip sync**: Adjust `--audio-cfg` between 3-5 (4 = default, 5 = more precise)
- **Duration**: Keep under 1 minute for best results; use video-to-video mode for longer content

## Troubleshooting

**Models not downloading?**
```bash
avatar-studio generate --no-download  # skip download check
```

**Out of memory?**
```bash
avatar-studio generate --gpu-mem low --steps 20
```

## Project Structure

```
avatar-studio/
├── src/
│   ├── cli.py           # Main CLI commands
│   ├── generator.py     # Infinite Talk integration
│   └── config.py        # Configuration management
├── outputs/             # Generated videos
├── requirements.txt     # Dependencies
└── CLAUDE.md           # Project documentation
```
