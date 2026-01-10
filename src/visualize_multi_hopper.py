#!/usr/bin/env python3
"""
Visualize multiple C++ hopper trajectories using MeshCat.

Loads trajectory data from parallel CUDA simulation and animates all hoppers
together in a 3D scene.

Usage:
    python src/visualize_multi_hopper.py [output_prefix] [num_hoppers]

Example:
    python src/visualize_multi_hopper.py trajectory_cuda 100

Default prefix: trajectory_cuda
"""

import sys
import os
import glob
import numpy as np

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from hopper import hopperParameters

# Color palette for hoppers (rainbow)
def get_hopper_color(idx, total):
    """Generate rainbow color based on index."""
    import colorsys
    hue = idx / max(1, total - 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return int(r * 255) << 16 | int(g * 255) << 8 | int(b * 255)


def load_trajectory(filename):
    """Load trajectory CSV exported from C++ simulation."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    tout = data[:, 0]
    yout = data[:, 1:11]  # state columns
    return tout, yout


def find_trajectory_files(prefix):
    """Find all trajectory files matching the prefix pattern."""
    pattern = f"{prefix}_*.csv"
    files = sorted(glob.glob(pattern))
    # Filter out summary file
    files = [f for f in files if not f.endswith('_summary.csv')]
    return files


def animate_multi_hopper(trajectories, p, fps=30, depth=0.15):
    """
    Animate multiple hoppers in MeshCat.

    Args:
        trajectories: List of (tout, yout) tuples, one per hopper
        p: Parameters object
        fps: Animation framerate
        depth: 3D extrusion depth

    Returns:
        vis: MeshCat Visualizer object
    """
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
        from meshcat.animation import Animation
    except ImportError:
        print("MeshCat not installed. Install with: pip install meshcat")
        return None

    def polygon_to_mesh(vertices_in, depth, color):
        """Convert 2D polygon to 3D extruded mesh."""
        n = vertices_in.shape[1]

        vertices = []
        for i in range(n):
            vertices.append([vertices_in[0, i], -depth/2, vertices_in[2, i]])
        for i in range(n):
            vertices.append([vertices_in[0, i], depth/2, vertices_in[2, i]])

        vertices = np.array(vertices)
        faces = []

        for i in range(1, n-1):
            faces.append([0, i, i+1])
        for i in range(1, n-1):
            faces.append([n, n+i+1, n+i])
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, i+n])
            faces.append([next_i, next_i+n, i+n])

        return g.TriangularMeshGeometry(vertices, np.array(faces)), g.MeshLambertMaterial(color=color)

    N = len(trajectories)
    print(f"Animating {N} hoppers...")

    # Create visualizer
    vis = meshcat.Visualizer()
    print(f"MeshCat visualizer: {vis.url()}")
    vis.delete()

    # Set camera
    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([0, 1.5, 8]) @ tf.rotation_matrix(-0.2, [1, 0, 0])
    )

    # Geometry templates (simplified for multi-hopper - just body box, leg, and foot)
    body_box_x = np.array([-0.3, 0.3, 0.3, -0.3, -0.3])
    body_box_z = np.array([-0.15, -0.15, 0.15, 0.15, -0.15])
    body_shape = np.array([body_box_x, np.zeros_like(body_box_x), body_box_z])

    leg_x = np.array([-0.02, 0.02, 0.02, -0.02, -0.02])
    leg_z = np.array([-0.3, -0.3, 0.5, 0.5, -0.3])
    leg_shape = np.array([leg_x, np.zeros_like(leg_x), leg_z])

    foot_x = np.array([-0.04, 0.04, 0.04, -0.04, -0.04])
    foot_z = np.array([0.0, 0.0, 1.7, 1.7, 0.0])
    foot_shape = np.array([foot_x, np.zeros_like(foot_x), foot_z])

    # Create hopper geometry for each robot
    for i in range(N):
        color = get_hopper_color(i, N)
        y_offset = (i - N/2) * 0.5  # Spread hoppers along Y axis

        body_geom, body_mat = polygon_to_mesh(body_shape, depth, color)
        leg_geom, leg_mat = polygon_to_mesh(leg_shape, depth, color)
        foot_geom, foot_mat = polygon_to_mesh(foot_shape, depth, color)

        vis[f"hopper_{i}"]["body"].set_object(body_geom, body_mat)
        vis[f"hopper_{i}"]["leg"].set_object(leg_geom, leg_mat)
        vis[f"hopper_{i}"]["foot"].set_object(foot_geom, foot_mat)

    # Ground
    num_squares = 40
    square_size = 2.0
    for i in range(num_squares):
        for j in range(num_squares // 2):
            color = 0xffffff if (i + j) % 2 == 0 else 0x505050
            vis[f"ground/square_{i}_{j}"].set_object(
                g.Box([square_size, square_size, 0.05]),
                g.MeshLambertMaterial(color=color)
            )
            vis[f"ground/square_{i}_{j}"].set_transform(
                tf.translation_matrix([
                    (i - num_squares/2) * square_size + square_size/2,
                    (j - num_squares/4) * square_size + square_size/2,
                    0
                ])
            )
    vis["ground"].set_transform(tf.translation_matrix([0, 0, -0.25]))

    # Prepare animation - use first trajectory for timing reference
    tout_ref = trajectories[0][0]
    ds = max(1, round(len(tout_ref) / (fps * (tout_ref[-1] - tout_ref[0]))))

    # Subsample all trajectories
    trajectories_sub = []
    for tout, yout in trajectories:
        trajectories_sub.append((tout[::ds], yout[::ds, :]))

    num_frames = len(trajectories_sub[0][0])
    print(f"Recording {num_frames} frames at {fps} fps...")

    anim = Animation(default_framerate=fps)

    # Animate all hoppers
    for frame_idx in range(num_frames):
        with anim.at_frame(vis, frame_idx) as frame:
            for i, (tout, yout) in enumerate(trajectories_sub):
                if frame_idx >= len(tout):
                    continue

                q = yout[frame_idx]
                x_foot = q[0]
                z_foot = q[1]
                phi_leg = q[2]
                phi_body = q[3]
                leg_length = q[4]

                # Spread hoppers along Y axis
                y_offset = (i - N/2) * 0.5

                cos_leg = np.cos(phi_leg)
                sin_leg = np.sin(phi_leg)
                x_hip = x_foot + leg_length * sin_leg
                z_hip = z_foot + leg_length * cos_leg

                # Foot
                foot_tf = tf.translation_matrix([x_foot, y_offset, z_foot]) @ \
                          tf.rotation_matrix(phi_leg, [0, 1, 0])
                frame[f"hopper_{i}"]["foot"].set_transform(foot_tf)

                # Leg
                leg_tf = tf.translation_matrix([x_hip, y_offset, z_hip]) @ \
                         tf.rotation_matrix(phi_leg, [0, 1, 0])
                frame[f"hopper_{i}"]["leg"].set_transform(leg_tf)

                # Body
                body_tf = tf.translation_matrix([x_hip, y_offset, z_hip]) @ \
                          tf.rotation_matrix(phi_body, [0, 1, 0])
                frame[f"hopper_{i}"]["body"].set_transform(body_tf)

    print(f"Recorded {num_frames} frames for {N} hoppers")

    vis.set_animation(anim, play=False, repetitions=0)
    vis["/Grid"].set_property("visible", False)

    print("\n" + "="*60)
    print("Multi-Hopper Animation ready!")
    print("="*60)
    print(f"Open in browser: {vis.url()}")
    print("Use animation controls in the Animations panel")
    print("="*60)

    return vis


def main():
    # Parse arguments
    prefix = sys.argv[1] if len(sys.argv) > 1 else "trajectory_cuda"

    # Find trajectory files
    files = find_trajectory_files(prefix)

    if not files:
        print(f"ERROR: No trajectory files found matching pattern: {prefix}_*.csv")
        print("\nTo generate trajectories, run:")
        print("  nvcc -O2 cpp/cuda/test_cuda.cu -o test_cuda.exe")
        print("  ./test_cuda.exe --multi -n 100 -t 5.0 -o trajectory_cuda")
        return 1

    print(f"Found {len(files)} trajectory files")

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
    vis = animate_multi_hopper(trajectories, p, fps=30)

    if vis is not None:
        print("\nPress Ctrl+C to exit...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
