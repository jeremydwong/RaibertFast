#!/usr/bin/env python3
"""
Visualize C++ hopper trajectory using MeshCat.

Loads trajectory data from C++ simulation and animates it using the
proper 3D MeshCat visualization from meshcat_viz.py.

Usage:
    python src/visualize_cpp_trajectory.py [trajectory_file]

Default trajectory file: cpp/trajectory.csv
"""

import sys
import os
import numpy as np

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from hopper import hopperParameters
from meshcat_viz import animate_meshcat


def load_cpp_trajectory(filename):
    """Load trajectory CSV exported from C++ simulation."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Columns: t, x_foot, z_foot, phi_leg, phi_body, len_leg,
    #          ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
    #          fsm_state, u1, u2
    tout = data[:, 0]
    yout = data[:, 1:11]  # state columns

    return tout, yout


def main():
    # Default trajectory file
    default_trajectory = os.path.join(root_dir, 'cpp', 'trajectory.csv')
    trajectory_file = sys.argv[1] if len(sys.argv) > 1 else default_trajectory

    if not os.path.exists(trajectory_file):
        print(f"ERROR: Trajectory file not found: {trajectory_file}")
        print("\nTo generate a trajectory, run the C++ simulation:")
        print("  cd cpp && ./build.sh && ./hopper")
        return 1

    print(f"Loading C++ trajectory: {trajectory_file}")
    tout, yout = load_cpp_trajectory(trajectory_file)

    print(f"  Time: {tout[0]:.4f} to {tout[-1]:.4f} s")
    print(f"  Frames: {len(tout)}")

    # Get parameters
    p = hopperParameters()

    # Run MeshCat animation
    vis = animate_meshcat(tout, yout, p, fps=30, depth=0.2)

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
