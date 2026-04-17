#!/usr/bin/env python3
"""Avatar Studio - CLI tool for audio-driven avatar video generation."""

import click
import os
from pathlib import Path
from loguru import logger
from .generator import AvatarGenerator
from .config import Config

@click.group()
def cli():
    """Avatar Studio - Generate talking avatar videos with Infinite Talk."""
    pass


@cli.command()
@click.option('--image', type=click.Path(exists=True), help='Input image path (for image-to-video)')
@click.option('--video', type=click.Path(exists=True), help='Input video path (for video-to-video)')
@click.option('--audio', type=click.Path(exists=True), required=True, help='Input audio file (WAV format)')
@click.option('--output', type=click.Path(), default='outputs/output.mp4', help='Output video path')
@click.option('--duration', type=int, default=5, help='Video duration in seconds')
@click.option('--resolution', type=click.Choice(['480p', '720p']), default='480p', help='Output resolution')
@click.option('--steps', type=int, default=40, help='Diffusion steps (lower = faster, lower quality)')
@click.option('--motion-frames', type=int, default=9, help='Number of motion frames')
@click.option('--audio-cfg', type=float, default=4.0, help='Audio guidance scale (3-5 recommended)')
@click.option('--text-cfg', type=float, default=5.0, help='Text guidance scale')
@click.option('--prompt', type=str, default='', help='Description of the avatar (optional)')
@click.option('--gpu-mem', type=click.Choice(['full', 'low']), default='full', help='GPU memory optimization')
@click.option('--no-download', is_flag=True, help='Skip model download check')
def generate(image, video, audio, output, duration, resolution, steps, motion_frames,
             audio_cfg, text_cfg, prompt, gpu_mem, no_download):
    """Generate an avatar video from image/video + audio."""

    if not image and not video:
        click.echo(click.style('Error: Provide either --image or --video', fg='red'), err=True)
        raise click.Exit(1)

    if image and video:
        click.echo(click.style('Error: Provide either --image or --video, not both', fg='red'), err=True)
        raise click.Exit(1)

    # Setup
    config = Config()
    logger.info(f'Avatar Studio v0.1.0')
    logger.info(f'Models directory: {config.models_dir}')

    # Create output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = AvatarGenerator(config, gpu_mem=gpu_mem, download_models=(not no_download))

    # Generate
    try:
        logger.info(f'Generating {"image-to-video" if image else "video-to-video"} content')
        logger.info(f'Duration: {duration}s | Resolution: {resolution}')
        logger.info(f'Input: {image or video}')
        logger.info(f'Audio: {audio}')

        generator.generate(
            input_path=image or video,
            audio_path=audio,
            output_path=str(output_path),
            duration=duration,
            resolution=resolution,
            steps=steps,
            motion_frames=motion_frames,
            audio_cfg=audio_cfg,
            text_cfg=text_cfg,
            prompt=prompt or None,
            is_image=(image is not None)
        )

        logger.info(click.style(f'✓ Video saved to {output_path}', fg='green'))

    except Exception as e:
        logger.error(f'Generation failed: {str(e)}')
        raise click.Exit(1)


@cli.command()
def info():
    """Show Avatar Studio configuration and model status."""
    config = Config()

    click.echo(click.style('Avatar Studio', bold=True))
    click.echo(f'Models directory: {config.models_dir}')
    click.echo(f'Config file: {config.config_path}')

    # Check model status
    models = ['Wan2.1-I2V-14B-480P', 'chinese-wav2vec2-base', 'InfiniteTalk']
    click.echo(click.style('\nModel Status:', bold=True))
    for model in models:
        model_path = config.models_dir / model
        status = '✓' if model_path.exists() else '✗'
        click.echo(f'  {status} {model}')


if __name__ == '__main__':
    cli()
