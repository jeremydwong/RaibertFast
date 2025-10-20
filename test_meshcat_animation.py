"""
Test MeshCat animation with hopper simulation
"""
import sys
sys.path.insert(0, 'src')

from hopper import call_hopper, hopperParameters
import numpy as np
import time

def animate_meshcat(tout, yout, p, speed=1.0):
    """
    Animate the hopper simulation using MeshCat 3D visualization

    Args:
        tout: Time array from simulation
        yout: State array from simulation
        p: Parameters object
        speed: Playback speed multiplier (default: 1.0)
    """
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
    except ImportError:
        print("MeshCat not installed. Install with: pip install meshcat")
        return None

    # Create visualizer
    vis = meshcat.Visualizer()
    print(f"MeshCat visualizer available at: {vis.url()}")

    # Clear any existing visualizations
    vis.delete()

    # Create ground plane
    vis["ground"].set_object(
        g.Box([20, 0.05, 2]),
        g.MeshLambertMaterial(color=0x808080)
    )
    vis["ground"].set_transform(
        tf.translation_matrix([0, -0.025, 0])
    )

    # Create hopper body (main body as a box)
    body_width = 0.4
    body_height = 0.3
    body_depth = 0.3
    vis["hopper"]["body"].set_object(
        g.Box([body_width, body_height, body_depth]),
        g.MeshLambertMaterial(color=0x3498db)
    )

    # Create leg (cylinder)
    leg_radius = 0.03
    vis["hopper"]["leg"].set_object(
        g.Cylinder(1.0, leg_radius),  # height will be scaled dynamically
        g.MeshLambertMaterial(color=0xe74c3c)
    )

    # Create foot (sphere)
    foot_radius = 0.08
    vis["hopper"]["foot"].set_object(
        g.Sphere(foot_radius),
        g.MeshLambertMaterial(color=0x2ecc71)
    )

    # Animation parameters
    sr_video = 30
    ds = max(1, round(len(tout) / (sr_video * (tout[-1] - tout[0]))))
    tout_vid = tout[::ds]
    yout_vid = yout[::ds, :]

    print(f"Animating {len(tout_vid)} frames at {sr_video} fps...")

    # Animate
    for i, (t, q) in enumerate(zip(tout_vid, yout_vid)):
        x_foot = q[0]
        z_foot = q[1]
        phi_leg = q[2]
        phi_body = q[3]
        leg_length = q[4]

        # Compute positions
        # Foot position (2D -> 3D: x->x, z->y, add z=0 for depth)
        foot_pos = [x_foot, z_foot, 0]

        # Hip position
        hip_x = x_foot + leg_length * np.sin(phi_leg)
        hip_y = z_foot + leg_length * np.cos(phi_leg)
        hip_pos = [hip_x, hip_y, 0]

        # Body COM position
        body_x = hip_x + p.l_2 * np.sin(phi_body)
        body_y = hip_y + p.l_2 * np.cos(phi_body)
        body_pos = [body_x, body_y, 0]

        # Set foot transform
        vis["hopper"]["foot"].set_transform(
            tf.translation_matrix(foot_pos)
        )

        # Set leg transform (position and orientation)
        leg_center_x = (x_foot + hip_x) / 2
        leg_center_y = (z_foot + hip_y) / 2
        leg_transform = tf.translation_matrix([leg_center_x, leg_center_y, 0])
        # Rotate leg - meshcat cylinder is along z-axis, we need it at angle phi_leg
        leg_rotation = tf.rotation_matrix(phi_leg, [0, 0, 1])
        leg_rotation = leg_rotation @ tf.rotation_matrix(np.pi/2, [1, 0, 0])

        # Scale leg length
        leg_scale = tf.scale_matrix(leg_length, direction=[0, 1, 0])
        vis["hopper"]["leg"].set_transform(leg_transform @ leg_rotation @ leg_scale)

        # Set body transform (position and orientation)
        body_transform = tf.translation_matrix(body_pos)
        body_rotation = tf.rotation_matrix(phi_body, [0, 0, 1])
        vis["hopper"]["body"].set_transform(body_transform @ body_rotation)

        # Sleep to control playback speed
        if i > 0:
            dt = tout_vid[i] - tout_vid[i-1]
            time.sleep(dt / speed)

    print("Animation complete!")
    return vis


if __name__ == "__main__":
    print("Running quick hopper simulation (2 seconds)...")

    # Run a short simulation
    tout, yout, State, p, anim = call_hopper(tfinal=2, x_dot_des=2.0, save_figures=False)

    print("\nStarting MeshCat animation...")
    vis = animate_meshcat(tout, yout, p, speed=1.0)

    if vis is not None:
        print("\n" + "="*60)
        print("SUCCESS! MeshCat animation is working!")
        print("="*60)
        print(f"\nView the animation at: {vis.url()}")
        print("\nYou can:")
        print("  - Rotate: Left-click and drag")
        print("  - Pan: Right-click and drag")
        print("  - Zoom: Scroll wheel")
        print("\nPress Ctrl+C to exit...")

        try:
            # Keep the script running so the server stays up
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
