#!/usr/bin/env python3
"""
Server-side video export for simulation results.
Uses PyVista for headless 3D rendering and imageio-ffmpeg for encoding.

Supports the per-facet voltage visualization system where each membrane facet
displays the correct voltage for its specific cell pair interface.
"""
import numpy as np
import json
import h5py
from pathlib import Path
from datetime import datetime

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class VideoExporter:
    """Export simulation animation as video."""

    def __init__(self, viz_data_dir: Path, sim_output_dir: Path, output_dir: Path):
        """
        Initialize video exporter.

        Args:
            viz_data_dir: Directory containing visualization binary files (expanded mesh)
            sim_output_dir: Directory containing simulation output (v_i_j.h5 files)
            output_dir: Directory to write output video
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("pyvista is required for video export. Install with: pip install pyvista")
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is required for video export. Install with: pip install imageio[ffmpeg]")

        self.viz_data_dir = Path(viz_data_dir)
        self.sim_output_dir = Path(sim_output_dir)
        self.output_dir = Path(output_dir)

        # Load expanded mesh data
        self.vertices = np.fromfile(self.viz_data_dir / 'mesh_vertices.bin', dtype=np.float32).reshape(-1, 3)
        self.facets = np.fromfile(self.viz_data_dir / 'membrane_facets.bin', dtype=np.uint32).reshape(-1, 3)

        with open(self.viz_data_dir / 'mesh_metadata.json') as f:
            self.metadata = json.load(f)

        # Load per-facet mapping data
        facet_orig_vertices_path = self.viz_data_dir / 'facet_orig_vertices.bin'
        facet_pair_indices_path = self.viz_data_dir / 'facet_pair_indices.bin'

        if facet_orig_vertices_path.exists() and facet_pair_indices_path.exists():
            self.facet_orig_vertices = np.fromfile(facet_orig_vertices_path, dtype=np.uint32).reshape(-1, 3)
            self.facet_pair_indices = np.fromfile(facet_pair_indices_path, dtype=np.int32)
            self.unique_pairs = [tuple(p) for p in self.metadata.get('unique_pairs', [])]
            self.use_per_facet = True
        else:
            self.facet_orig_vertices = None
            self.facet_pair_indices = None
            self.unique_pairs = []
            self.use_per_facet = False

        # Find available vij files
        self.vij_files = {}
        for vij_path in self.sim_output_dir.glob('v_*_*.h5'):
            parts = vij_path.stem.split('_')
            if len(parts) == 3:
                try:
                    i, j = int(parts[1]), int(parts[2])
                    self.vij_files[(i, j)] = vij_path
                except ValueError:
                    pass

        # Use per-facet if we have both the mapping and vij files
        self.use_per_facet = self.use_per_facet and len(self.vij_files) > 0

    def _create_pyvista_mesh(self):
        """Create PyVista mesh from binary data."""
        # Create faces array in VTK format: [n_vertices, v0, v1, v2, ...]
        n_faces = len(self.facets)
        faces = np.zeros((n_faces, 4), dtype=np.int64)
        faces[:, 0] = 3  # Triangle
        faces[:, 1:] = self.facets
        faces = faces.ravel()

        mesh = pv.PolyData(self.vertices, faces)
        return mesh

    def _voltage_to_color(self, v, v_min, v_max):
        """Convert voltage to RGB color (blue -> white -> red)."""
        t = np.clip((v - v_min) / (v_max - v_min + 1e-10), 0, 1)

        # Blue (cold) -> White (mid) -> Red (hot)
        colors = np.zeros((len(t), 3))

        # Blue to White (t < 0.5)
        mask = t < 0.5
        s = t[mask] * 2
        colors[mask, 0] = s  # R
        colors[mask, 1] = s  # G
        colors[mask, 2] = 1  # B

        # White to Red (t >= 0.5)
        mask = t >= 0.5
        s = (t[mask] - 0.5) * 2
        colors[mask, 0] = 1      # R
        colors[mask, 1] = 1 - s  # G
        colors[mask, 2] = 1 - s  # B

        return (colors * 255).astype(np.uint8)

    def _parse_time(self, key):
        """Parse HDF5 time key to float."""
        return float(key.replace('_', '.'))

    def _load_all_timesteps(self):
        """Load all timesteps using per-facet voltage mapping."""
        if self.use_per_facet:
            return self._load_timesteps_per_facet()
        else:
            return self._load_timesteps_fallback()

    def _load_timesteps_per_facet(self):
        """Load timesteps using per-membrane voltage files (v_i_j.h5)."""
        # Get timestep keys from first vij file
        first_vij_path = list(self.vij_files.values())[0]
        with h5py.File(first_vij_path, 'r') as f:
            func_name = list(f['Function'].keys())[0]
            v_group = f['Function'][func_name]
            timestep_keys = sorted(v_group.keys(), key=self._parse_time)

        # Load all vij data
        vij_data = {}
        for pair, vij_path in self.vij_files.items():
            with h5py.File(vij_path, 'r') as f:
                func_name = list(f['Function'].keys())[0]
                v_group = f['Function'][func_name]
                vij_data[pair] = {}
                for key in timestep_keys:
                    if key in v_group:
                        vij_data[pair][key] = v_group[key][:].flatten()

        # Build per-vertex voltages for expanded mesh
        num_facets = len(self.facet_orig_vertices)
        timesteps = []

        for key in timestep_keys:
            # For each facet, get voltage from the correct vij
            expanded_voltages = np.zeros(num_facets * 3, dtype=np.float32)

            for facet_idx in range(num_facets):
                pair_idx = self.facet_pair_indices[facet_idx]
                orig_verts = self.facet_orig_vertices[facet_idx]

                if pair_idx < len(self.unique_pairs):
                    pair = self.unique_pairs[pair_idx]
                    if pair in vij_data and key in vij_data[pair]:
                        v_data = vij_data[pair][key]
                        for local_v, orig_v in enumerate(orig_verts):
                            if orig_v < len(v_data):
                                expanded_voltages[facet_idx * 3 + local_v] = v_data[orig_v]

            time = self._parse_time(key)
            timesteps.append((time, expanded_voltages))

        return timesteps

    def _load_timesteps_fallback(self):
        """Load timesteps using summed v.h5 file (old method).

        Maps original vertex voltages to the expanded mesh format.
        """
        v_h5 = self.sim_output_dir / 'v.h5'
        if not v_h5.exists():
            raise FileNotFoundError(f"v.h5 not found in {self.sim_output_dir}")

        with h5py.File(v_h5, 'r') as f:
            v_group = f['Function']['v']
            timestep_keys = sorted(v_group.keys(), key=self._parse_time)

            timesteps = []
            for key in timestep_keys:
                v_data = v_group[key][:].flatten()

                # If we have facet_orig_vertices, expand voltages to match expanded mesh
                if self.facet_orig_vertices is not None:
                    num_facets = len(self.facet_orig_vertices)
                    expanded_voltages = np.zeros(num_facets * 3, dtype=np.float32)

                    for facet_idx, orig_verts in enumerate(self.facet_orig_vertices):
                        for local_v, orig_v in enumerate(orig_verts):
                            if orig_v < len(v_data):
                                expanded_voltages[facet_idx * 3 + local_v] = v_data[orig_v]

                    v_data = expanded_voltages

                time = self._parse_time(key)
                timesteps.append((time, v_data))

            return timesteps

    def export(self, camera_config: dict = None, resolution: tuple = (1920, 1080),
               fps: int = 30, progress_callback=None) -> str:
        """
        Export simulation to video.

        Args:
            camera_config: Dict with position, target, up, fov (from Three.js camera)
            resolution: (width, height) tuple
            fps: Frames per second
            progress_callback: Optional callback(frame, total, message)

        Returns:
            Path to output video file
        """
        def report(frame, total, message):
            if progress_callback:
                progress_callback(frame, total, message)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load timesteps
        report(0, 1, "Loading simulation data...")
        timesteps = self._load_all_timesteps()
        total_frames = len(timesteps)

        if total_frames == 0:
            raise ValueError("No timesteps found in results file")

        # Calculate voltage range
        all_v = np.concatenate([t[1] for t in timesteps])
        v_min = float(np.min(all_v))
        v_max = float(np.max(all_v))

        report(0, total_frames, f"Rendering {total_frames} frames...")

        # Create PyVista mesh
        mesh = self._create_pyvista_mesh()

        # Setup offscreen rendering - required for macOS and headless environments
        pv.OFF_SCREEN = True

        # Try to start virtual framebuffer (Linux only, ignored on macOS)
        try:
            pv.start_xvfb()
        except Exception:
            pass  # Expected on macOS/Windows

        # Create offscreen plotter with proper configuration
        plotter = pv.Plotter(off_screen=True, window_size=resolution, lighting='three_lights')

        # Configure camera if provided
        if camera_config:
            pos = camera_config.get('position', [200, 150, 200])
            target = camera_config.get('target', [50, 30, 40])
            up = camera_config.get('up', [0, 1, 0])

            plotter.camera_position = [pos, target, up]
        else:
            # Default camera based on mesh bounds
            bounds = self.metadata.get('bounds', {})
            center = [
                (bounds.get('x', [0, 100])[0] + bounds.get('x', [0, 100])[1]) / 2,
                (bounds.get('y', [0, 100])[0] + bounds.get('y', [0, 100])[1]) / 2,
                (bounds.get('z', [0, 100])[0] + bounds.get('z', [0, 100])[1]) / 2,
            ]
            extent = max(
                bounds.get('x', [0, 100])[1] - bounds.get('x', [0, 100])[0],
                bounds.get('y', [0, 100])[1] - bounds.get('y', [0, 100])[0],
                bounds.get('z', [0, 100])[1] - bounds.get('z', [0, 100])[0],
            )
            plotter.camera_position = [
                [center[0] + extent, center[1] + extent * 0.5, center[2] + extent],
                center,
                [0, 1, 0]
            ]

        # Prepare video writer
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        video_path = self.output_dir / f'simulation_{timestamp}.mp4'
        writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)

        try:
            for i, (time, voltages) in enumerate(timesteps):
                # Update mesh colors
                colors = self._voltage_to_color(voltages, v_min, v_max)
                mesh['colors'] = colors

                # Render
                plotter.clear()
                plotter.add_mesh(mesh, scalars='colors', rgb=True, show_scalar_bar=False)

                # Add time annotation
                plotter.add_text(f"t = {time:.3f} ms", position='upper_left',
                               font_size=14, color='white')

                # Capture frame
                img = plotter.screenshot(return_img=True)
                writer.append_data(img)

                report(i + 1, total_frames, f"Frame {i + 1}/{total_frames}")

        finally:
            writer.close()
            plotter.close()

        report(total_frames, total_frames, "Video export complete!")
        return str(video_path)


def main():
    """Command-line entry point with argument parsing."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Export simulation animation as video')
    parser.add_argument('sim_name', nargs='?', default=None,
                        help='Simulation name (legacy positional argument)')
    parser.add_argument('--viz-data', type=str, help='Visualization data directory')
    parser.add_argument('--sim-output', type=str, help='Simulation output directory')
    parser.add_argument('--video-output', type=str, help='Video output directory')
    parser.add_argument('--width', type=int, default=1920, help='Video width')
    parser.add_argument('--height', type=int, default=1080, help='Video height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--camera', type=str, help='Camera config as JSON')

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    # Determine paths from arguments
    if args.viz_data and args.sim_output:
        viz_data_dir = Path(args.viz_data)
        sim_output_dir = Path(args.sim_output)
        video_output_dir = Path(args.video_output) if args.video_output else SCRIPT_DIR.parent / 'videos'
    elif args.sim_name:
        # Legacy mode: single sim_name argument
        sim_name = args.sim_name
        viz_data_dir = SCRIPT_DIR.parent / 'data' / sim_name
        sim_output_dir = PROJECT_ROOT / sim_name
        video_output_dir = SCRIPT_DIR.parent / 'videos'
    else:
        # Default
        sim_name = 'pepe36_colored_sim'
        viz_data_dir = SCRIPT_DIR.parent / 'data' / sim_name
        sim_output_dir = PROJECT_ROOT / sim_name
        video_output_dir = SCRIPT_DIR.parent / 'videos'

    # Parse camera config if provided
    camera_config = None
    if args.camera:
        camera_config = json.loads(args.camera)

    print(f"Exporting video...", flush=True)
    print(f"  Viz data: {viz_data_dir}", flush=True)
    print(f"  Simulation output: {sim_output_dir}", flush=True)
    print(f"  Video output: {video_output_dir}", flush=True)

    try:
        exporter = VideoExporter(viz_data_dir, sim_output_dir, video_output_dir)

        def progress(frame, total, message):
            # Output in format expected by server: PROGRESS:percent:message
            percent = int((frame / max(total, 1)) * 100)
            print(f"PROGRESS:{percent}:{message}", flush=True)

        video_path = exporter.export(
            camera_config=camera_config,
            resolution=(args.width, args.height),
            fps=args.fps,
            progress_callback=progress
        )

        # Output video filename for server to parse
        print(f"VIDEO_FILE:{Path(video_path).name}", flush=True)
        print(f"Video saved to: {video_path}", flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR:{str(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
