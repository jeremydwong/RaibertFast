#!/usr/bin/env python3
"""
Visualize multiple C++ hopper trajectories using Matplotlib 3D with automatic video export.

This script provides automated video export without requiring browser interaction.

Usage:
    python src/visualize_multi_hopper_matplotlib.py [output_dir] [prefix]

Example:
    python src/visualize_multi_hopper_matplotlib.py ../RaibertFastBuild trajectory_cuda
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import colorsys

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from hopper import hopperParameters


def get_hopper_color(idx, total):
    """Generate rainbow color based on index."""
    hue = idx / max(1, total - 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (r, g, b)


def load_trajectory(filename):
    """Load trajectory CSV exported from C++ simulation."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    tout = data[:, 0]
    yout = data[:, 1:11]  # state columns
    return tout, yout


def find_trajectory_files(directory, prefix):
    """Find all trajectory files matching the prefix pattern."""
    pattern = os.path.join(directory, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    # Filter out summary file
    files = [f for f in files if not f.endswith('_summary.csv')]
    return files


def create_hopper_geometry(x_foot, z_foot, phi_leg, phi_body, leg_length, y_offset, color):
    """Create 3D geometry for a single hopper at given state."""
    # Compute hip position
    cos_leg = np.cos(phi_leg)
    sin_leg = np.sin(phi_leg)
    x_hip = x_foot + leg_length * sin_leg
    z_hip = z_foot + leg_length * cos_leg

    polys = []

    # Body (simplified box at hip)
    body_size = 0.3
    body_height = 0.15
    # Rotate body by phi_body
    cos_b = np.cos(phi_body)
    sin_b = np.sin(phi_body)

    # Body corners in local frame (x, z)
    body_corners_local = np.array([
        [-body_size, -body_height],
        [body_size, -body_height],
        [body_size, body_height],
        [-body_size, body_height]
    ])

    # Rotate and translate
    body_corners = []
    for bx, bz in body_corners_local:
        rx = cos_b * bx - sin_b * bz + x_hip
        rz = sin_b * bx + cos_b * bz + z_hip
        body_corners.append([rx, y_offset, rz])

    polys.append(body_corners)

    # Leg (thin rectangle from hip down)
    leg_width = 0.03
    leg_top = 0.5
    leg_bottom = -0.3

    leg_corners_local = np.array([
        [-leg_width, leg_bottom],
        [leg_width, leg_bottom],
        [leg_width, leg_top],
        [-leg_width, leg_top]
    ])

    leg_corners = []
    for lx, lz in leg_corners_local:
        rx = cos_leg * lx - sin_leg * lz + x_hip
        rz = sin_leg * lx + cos_leg * lz + z_hip
        leg_corners.append([rx, y_offset, rz])

    polys.append(leg_corners)

    # Foot (thin vertical from foot position up)
    foot_width = 0.04
    foot_height = leg_length  # extends up to roughly where hip is

    foot_corners = [
        [x_foot - foot_width, y_offset, z_foot],
        [x_foot + foot_width, y_offset, z_foot],
        [x_foot + foot_width, y_offset, z_foot + foot_height * 0.9],
        [x_foot - foot_width, y_offset, z_foot + foot_height * 0.9]
    ]

    polys.append(foot_corners)

    return polys


def animate_multi_hopper_matplotlib(trajectories, p, output_path, fps=30):
    """
    Animate multiple hoppers using matplotlib and export to video.

    Args:
        trajectories: List of (tout, yout) tuples, one per hopper
        p: Parameters object
        output_path: Path for output video file
        fps: Animation framerate
    """
    N = len(trajectories)
    print(f"Animating {N} hoppers with matplotlib...")

    # Subsample for target fps
    tout_ref = trajectories[0][0]
    ds = max(1, round(len(tout_ref) / (fps * (tout_ref[-1] - tout_ref[0]))))

    # Subsample all trajectories
    trajectories_sub = []
    for tout, yout in trajectories:
        trajectories_sub.append((tout[::ds], yout[::ds, :]))

    num_frames = len(trajectories_sub[0][0])
    print(f"Rendering {num_frames} frames at {fps} fps...")

    # Compute bounds
    x_min, x_max = float('inf'), float('-inf')
    z_min, z_max = 0, 3
    for tout, yout in trajectories_sub:
        x_min = min(x_min, yout[:, 0].min() - 2)
        x_max = max(x_max, yout[:, 0].max() + 2)

    y_extent = N * 0.5 / 2 + 2

    # Create figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Pre-compute colors
    colors = [get_hopper_color(i, N) for i in range(N)]

    def init():
        ax.clear()
        return []

    def update(frame_idx):
        ax.clear()

        # Compute camera center (average x of all hoppers)
        avg_x = 0
        count = 0
        for i, (tout, yout) in enumerate(trajectories_sub):
            if frame_idx < len(tout):
                avg_x += yout[frame_idx, 0]
                count += 1
        if count > 0:
            avg_x /= count

        # Set view limits (camera follows)
        view_width = 15
        ax.set_xlim(avg_x - view_width/2, avg_x + view_width)
        ax.set_ylim(-y_extent, y_extent)
        ax.set_zlim(-0.5, 3)

        # Draw ground
        ground_x = np.array([avg_x - view_width, avg_x + view_width*2])
        ground_y = np.array([-y_extent, y_extent])
        ground_X, ground_Y = np.meshgrid(ground_x, ground_y)
        ground_Z = np.zeros_like(ground_X)
        ax.plot_surface(ground_X, ground_Y, ground_Z, alpha=0.3, color='gray')

        # Draw each hopper
        for i, (tout, yout) in enumerate(trajectories_sub):
            if frame_idx >= len(tout):
                continue

            q = yout[frame_idx]
            x_foot = q[0]
            z_foot = q[1]
            phi_leg = q[2]
            phi_body = q[3]
            leg_length = q[4]
            y_offset = (i - N/2) * 0.5

            polys = create_hopper_geometry(
                x_foot, z_foot, phi_leg, phi_body, leg_length,
                y_offset, colors[i]
            )

            # Add polygons
            for poly in polys:
                collection = Poly3DCollection([poly], alpha=0.8)
                collection.set_facecolor(colors[i])
                collection.set_edgecolor('black')
                collection.set_linewidth(0.5)
                ax.add_collection3d(collection)

        # Set view angle
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        t = trajectories_sub[0][0][frame_idx] if frame_idx < len(trajectories_sub[0][0]) else 0
        ax.set_title(f'{N} Hoppers - t = {t:.2f}s')

        return []

    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=num_frames, interval=1000/fps, blit=False
    )

    print(f"Saving video to {output_path}...")
    print("This may take several minutes...")

    # Try different writers
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(output_path, writer=writer, dpi=100)
        print(f"Video saved: {output_path}")
    except Exception as e:
        print(f"FFMpeg writer failed: {e}")
        try:
            # Fallback to pillow for GIF
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"Trying GIF export to {gif_path}...")
            writer = animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=80)
            print(f"GIF saved: {gif_path}")
        except Exception as e2:
            print(f"GIF export also failed: {e2}")
            print("Please install ffmpeg or pillow for video export")

    plt.close(fig)


def main():
    # Parse arguments
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../RaibertFastBuild"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "trajectory_cuda"

    # Convert to absolute path
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(root_dir, output_dir)

    video_output = os.path.join(output_dir, "multi_hopper_animation.mp4")

    # Find trajectory files
    files = find_trajectory_files(output_dir, prefix)

    if not files:
        print(f"ERROR: No trajectory files found matching pattern: {prefix}_*.csv in {output_dir}")
        print("\nTo generate trajectories, run:")
        print("  cd ../RaibertFastBuild && ./test_cuda.exe --multi -n 64 -t 5.0 -o trajectory_cuda")
        return 1

    print(f"Found {len(files)} trajectory files in {output_dir}")

    # Load all trajectories
    trajectories = []
    for f in files:
        try:
            tout, yout = load_trajectory(f)
            trajectories.append((tout, yout))
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if not trajectories:
        print("ERROR: No valid trajectories loaded")
        return 1

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"  Time range: {trajectories[0][0][0]:.2f} to {trajectories[0][0][-1]:.2f} s")
    print(f"  Frames per trajectory: {len(trajectories[0][0])}")

    # Get parameters
    p = hopperParameters()

    # Run animation
    animate_multi_hopper_matplotlib(trajectories, p, video_output, fps=30)

    return 0


if __name__ == '__main__':
    sys.exit(main())
