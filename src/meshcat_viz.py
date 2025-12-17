"""
MeshCat 3D visualization for Raibert Hopper.

This module provides the animate_meshcat function that creates a proper 3D
visualization matching the 2D matplotlib animation geometry.

Extracted from hopper_demo.ipynb for reuse by both Python and C++ trajectories.
"""

import numpy as np


def animate_meshcat(tout, yout, p, fps=30, depth=0.2):
    """
    Animate the hopper simulation using MeshCat 3D visualization

    Args:
        tout: Time array from simulation
        yout: State array from simulation (Nx10 or Nx5, first 5 cols are positions)
        p: Parameters object (needs p.l_1, p.l_2)
        fps: Frames per second for animation (default: 30)
        depth: Depth of the 3D extrusion (default: 0.2)

    Returns:
        vis: MeshCat Visualizer object, or None if meshcat not installed
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
        # Add all front vertices
        for i in range(n):
            vertices.append([vertices_in[0, i], -depth/2, vertices_in[2, i]])
        # Add all back vertices
        for i in range(n):
            vertices.append([vertices_in[0, i], depth/2, vertices_in[2, i]])

        vertices = np.array(vertices)
        faces = []

        # Front: fan from vertex 0
        for i in range(1, n-1):
            faces.append([0, i, i+1])

        # Back: fan from vertex n (reversed)
        for i in range(1, n-1):
            faces.append([n, n+i+1, n+i])

        # Sides
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, i+n])
            faces.append([next_i, next_i+n, i+n])

        return g.TriangularMeshGeometry(vertices, np.array(faces)), g.MeshLambertMaterial(color=color)

    # Create visualizer
    vis = meshcat.Visualizer()
    print(f"MeshCat visualizer: {vis.url()}")

    vis.delete()

    # Set camera for better initial view
    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([0, 1.5, 5]) @ tf.rotation_matrix(-0.3, [1, 0, 0])
    )

    # Create geometry - EXACT same as 2D animation from hopper_demo.ipynb
    brick1x = np.array([0.9, 1.05, 1.05, 0.9, 0.9])
    brick1z = np.array([0.15, 0.15, 0.0, 0.0, 0.15])
    brick1 = np.array([brick1x,
                      np.zeros_like(brick1x),
                      brick1z])

    brick2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ brick1

    beamx = np.array([-1, -1, 1, 1, -1])
    beamz = np.array([0.15, 0.175, 0.175, 0.15, 0.15])
    beam = np.array([beamx,
                     np.zeros_like(beamx),
                     beamz])

    compx = np.array([0.25, 0.25, 0.025, 0.025, -0.025, -0.025, -0.25, -0.25, 0.25])
    compz = np.array([0.15, 0.3, 0.3, 0.15, 0.15, 0.3, 0.3, 0.15, 0.15])
    comp = np.array([compx,
                    np.zeros_like(compx),
                    compz])

    body_shape = np.hstack([brick1, brick2, beam, comp])
    body_shape = body_shape - np.array([[0], [0], [0.16]]) @ np.ones((1, body_shape.shape[1]))

    legx = np.array([-0.015, -0.04, -0.04, 0.04, 0.04, 0.015, -0.015])
    legz = np.array([-0.25, -0.25, 0.525, 0.525, -0.25, -0.25, -0.25])
    leg_shape = np.array([legx,
                          np.zeros_like(legx),
                          legz])
    leg_shape = leg_shape - np.array([[0], [0], [0.16]]) @ np.ones((1, leg_shape.shape[1]))

    footx = np.array([0.0, -0.03, -0.03, 0.015, 0.015, -0.025, -0.025, 0.025, 0.025, -0.015, -0.015, 0.03, 0.03, 0.0])
    footz = np.array([0.0, 0.05, 0.14, 0.14, 1.675, 1.675, 1.725, 1.725, 1.675, 1.675, 0.14, 0.14, 0.05, 0.0])
    foot_shape = np.array([footx,
                           np.zeros_like(footx),
                           footz])

    print(f"Creating 3D meshes (depth={depth})...")

    # Create 3D meshes from 2D polygons
    brick1_geom, brick1_mat = polygon_to_mesh(brick1, depth, 0xBFBFBF)
    brick2_geom, brick2_mat = polygon_to_mesh(brick2, depth, 0xBFBFBF)
    beam_geom, beam_mat = polygon_to_mesh(beam, depth, 0xBFBFBF)
    comp_geom, comp_mat = polygon_to_mesh(comp, depth, 0xBFBFBF)
    leg_geom, leg_mat = polygon_to_mesh(leg_shape, depth, 0xBFBFBF)
    foot_geom, foot_mat = polygon_to_mesh(foot_shape, depth, 0xBFBFBF)

    vis["hopper"]["body"]["brick1"].set_object(brick1_geom, brick1_mat)
    vis["hopper"]["body"]["brick2"].set_object(brick2_geom, brick2_mat)
    vis["hopper"]["body"]["beam"].set_object(beam_geom, beam_mat)
    vis["hopper"]["body"]["comp"].set_object(comp_geom, comp_mat)
    vis["hopper"]["leg"].set_object(leg_geom, leg_mat)
    vis["hopper"]["foot"].set_object(foot_geom, foot_mat)

    # Ground - checkerboard pattern
    num_squares = 20
    square_size = 2.0
    for i in range(num_squares):
        for j in range(num_squares):
            color = 0xffffff if (i + j) % 2 == 0 else 0x505050
            vis[f"ground/square_{i}_{j}"].set_object(
                g.Box([square_size, square_size, 0.05]),
                g.MeshLambertMaterial(color=color)
            )
            vis[f"ground/square_{i}_{j}"].set_transform(
                tf.translation_matrix([
                    (i - num_squares/2) * square_size + square_size/2,
                    (j - num_squares/2) * square_size + square_size/2,
                    0
                ])
            )
    vis["ground"].set_transform(tf.translation_matrix([0, 0, -0.25]))

    print("Geometry created")

    # Prepare animation - subsample to target fps
    ds = max(1, round(len(tout) / (fps * (tout[-1] - tout[0]))))
    tout_vid = tout[::ds]
    yout_vid = yout[::ds, :]

    print(f"Recording {len(tout_vid)} frames at {fps} fps...")

    anim = Animation(default_framerate=fps)

    # Animate - EXACT same transforms as 2D code
    for i, (t, q) in enumerate(zip(tout_vid, yout_vid)):
        x_foot = q[0]
        z_foot = q[1]
        phi_leg = q[2]
        phi_body = q[3]
        leg_length = q[4]

        # Compute hip position EXACTLY as 2D code does
        cos_leg = np.cos(phi_leg)
        sin_leg = np.sin(phi_leg)
        x_hip = x_foot + leg_length * sin_leg
        z_hip = z_foot + leg_length * cos_leg

        with anim.at_frame(vis, i) as frame:
            # Foot: rotate by phi_leg around origin, translate to foot position
            foot_tf = tf.translation_matrix([x_foot, 0, z_foot]) @ \
                      tf.rotation_matrix(phi_leg, [0, 1, 0])
            frame["hopper"]["foot"].set_transform(foot_tf)

            # Leg: rotate by phi_leg around origin, translate to hip position
            leg_tf = tf.translation_matrix([x_hip, 0, z_hip]) @ \
                     tf.rotation_matrix(phi_leg, [0, 1, 0])
            frame["hopper"]["leg"].set_transform(leg_tf)

            # Body: rotate by phi_body around origin, translate to hip position
            body_tf = tf.translation_matrix([x_hip, 0, z_hip]) @ \
                      tf.rotation_matrix(phi_body, [0, 1, 0])
            frame["hopper"]["body"].set_transform(body_tf)

    print(f"Recorded {len(tout_vid)} frames")

    # Set animation
    vis.set_animation(anim, play=False, repetitions=0)
    vis["/Grid"].set_property("visible", False)
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0, -3, 10])
    vis["/Cameras/default"].set_property("target", [0, 0, 0])

    print("\n" + "="*60)
    print("Animation ready!")
    print("="*60)
    print(f"Open in browser: {vis.url()}")
    print("Use animation controls in the Animations panel")
    print("="*60)

    return vis
