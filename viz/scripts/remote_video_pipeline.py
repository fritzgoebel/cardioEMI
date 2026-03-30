#!/usr/bin/env python3
"""
Remote video pipeline: generate viz data + render video on Karolina.

Designed to run inside the DOLFINx container on Karolina.
Installs pyvista/imageio to a writable path, generates viz data from
simulation output, then renders an MP4 video.

Usage:
    python3 remote_video_pipeline.py <sim_name> [options]

Output protocol (parsed by the webapp):
    PROGRESS:<percent>:<message>
    VIDEO_FILE:<filename>
    ERROR:<message>
"""
import sys
import os
import json
import subprocess
import argparse
from pathlib import Path


def report(percent, message):
    print(f"PROGRESS:{percent}:{message}", flush=True)


def ensure_dependencies(pylibs):
    """Install pyvista and imageio-ffmpeg to writable path if not available."""
    try:
        sys.path.insert(0, pylibs)
        import pyvista
        import imageio
        report(5, "Dependencies already available")
        return True
    except ImportError:
        pass

    report(2, "Installing pyvista, matplotlib, and imageio-ffmpeg...")
    os.makedirs(pylibs, exist_ok=True)
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install',
         '--target', pylibs, '-q',
         'pyvista', 'matplotlib', 'imageio[ffmpeg]'],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"ERROR:Failed to install dependencies: {result.stderr}", flush=True)
        return False

    sys.path.insert(0, pylibs)
    report(5, "Dependencies installed")
    return True


def generate_viz_data(sim_dir, viz_dir):
    """Generate visualization binary files from simulation output."""
    # Import the generate script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from generate_viz_from_output import generate_viz_data as _generate

    def progress(percent, message):
        # Scale to 10-50% range
        scaled = 10 + int(percent * 0.4)
        report(scaled, message)

    return _generate(sim_dir, viz_dir, progress_callback=progress)


def render_video(viz_dir, sim_dir, output_dir, camera_config=None,
                 resolution=(1920, 1080), fps=30, colormap='coolwarm'):
    """Render video from viz data."""
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from video_exporter import VideoExporter

    def progress(frame, total, message):
        # Scale to 50-95% range
        scaled = 50 + int((frame / max(total, 1)) * 45)
        report(scaled, message)

    exporter = VideoExporter(viz_dir, sim_dir, output_dir)
    return exporter.export(
        camera_config=camera_config,
        resolution=resolution,
        fps=fps,
        progress_callback=progress,
        colormap=colormap,
    )


def main():
    parser = argparse.ArgumentParser(description='Remote video generation pipeline')
    parser.add_argument('sim_name', help='Simulation directory name')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--camera', type=str, help='Camera config as JSON')
    parser.add_argument('--colormap', type=str, default='coolwarm',
                        help='Colormap (coolwarm, viridis, plasma, inferno)')
    parser.add_argument('--pylibs', type=str, default='/home/fenics/.pylibs',
                        help='Writable path for pip installs')
    parser.add_argument('--project-root', type=str, default='/home/fenics',
                        help='Project root directory')
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sim_dir = project_root / args.sim_name
    viz_dir = project_root / 'viz' / 'data' / args.sim_name
    video_dir = project_root / 'viz' / 'videos'

    if not sim_dir.exists():
        print(f"ERROR:Simulation directory not found: {sim_dir}", flush=True)
        sys.exit(1)

    report(0, "Starting remote video pipeline...")

    # Step 1: Ensure dependencies
    if not ensure_dependencies(args.pylibs):
        sys.exit(1)

    # Step 2: Generate viz data (if not already present)
    metadata_file = viz_dir / 'mesh_metadata.json'
    if metadata_file.exists():
        report(10, "Viz data already exists, skipping generation")
    else:
        report(10, "Generating visualization data...")
        try:
            generate_viz_data(sim_dir, viz_dir)
        except Exception as e:
            print(f"ERROR:Viz generation failed: {e}", flush=True)
            sys.exit(1)

    report(50, "Rendering video...")

    # Step 3: Render video
    camera_config = None
    if args.camera:
        camera_config = json.loads(args.camera)

    try:
        video_path = render_video(
            viz_dir, sim_dir, video_dir,
            camera_config=camera_config,
            resolution=(args.width, args.height),
            fps=args.fps,
            colormap=args.colormap,
        )
        filename = Path(video_path).name
        report(100, "Video export complete!")
        print(f"VIDEO_FILE:{filename}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR:Video rendering failed: {e}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
