import numpy as np

# Exact shapes from the 2D code
brick1 = np.array([[0.9, 1.05, 1.05, 0.9, 0.9], [0.15, 0.15, 0.0, 0.0, 0.15]])
brick2 = np.array([[-1, 0], [0, 1]]) @ brick1
beam = np.array([[-1, -1, 1, 1], [0.15, 0.175, 0.175, 0.15]])
comp = np.array([[0.25, 0.25, 0.025, 0.025, -0.025, -0.025, -0.25, -0.25],
                 [0.15, 0.3, 0.3, 0.15, 0.15, 0.3, 0.3, 0.15]])
body_shape = np.hstack([brick1, brick2, beam, comp])
body_shape = body_shape - np.array([[0], [0.16]]) @ np.ones((1, body_shape.shape[1]))

leg_shape = np.array([[-0.015, -0.04, -0.04, 0.04, 0.04, 0.015, -0.015],
                      [-0.25, -0.25, 0.525, 0.525, -0.25, -0.25, -0.25]])
leg_shape = leg_shape - np.array([[0], [0.16]]) @ np.ones((1, leg_shape.shape[1]))

foot_shape = np.array([[0.0, -0.03, -0.03, 0.015, 0.015, -0.025, -0.025, 0.025, 0.025, -0.015, -0.015, 0.03, 0.03, 0.0],
                       [0.0, 0.05, 0.14, 0.14, 1.675, 1.675, 1.725, 1.725, 1.675, 1.675, 0.14, 0.14, 0.05, 0.0]])

print("Foot shape bounds:")
print(f"  X: [{foot_shape[0,:].min():.3f}, {foot_shape[0,:].max():.3f}]")
print(f"  Y: [{foot_shape[1,:].min():.3f}, {foot_shape[1,:].max():.3f}]")
print(f"  Origin should be at foot contact point")

print("\nLeg shape bounds (after -0.16 offset):")
print(f"  X: [{leg_shape[0,:].min():.3f}, {leg_shape[0,:].max():.3f}]")
print(f"  Y: [{leg_shape[1,:].min():.3f}, {leg_shape[1,:].max():.3f}]")
print(f"  Origin should be at hip")

print("\nBody shape bounds (after -0.16 offset):")
print(f"  X: [{body_shape[0,:].min():.3f}, {body_shape[0,:].max():.3f}]")
print(f"  Y: [{body_shape[1,:].min():.3f}, {body_shape[1,:].max():.3f}]")
print(f"  Origin should be at hip")

# Test transformation for a sample state
q0 = 0.0  # x_foot
q1 = 0.4  # y_foot  
q2 = 0.01  # phi_leg
q3 = 0.0  # phi_body
q4 = 1.0  # leg_length

print(f"\n\nTest state: foot=({q0}, {q1}), phi_leg={q2}, phi_body={q3}, leg_length={q4}")

# Compute hip position
hip = np.array([0, q4])
rot = np.array([[np.cos(q2), np.sin(q2)], [-np.sin(q2), np.cos(q2)]])
hip_world = rot @ hip + np.array([q0, q1])

print(f"\nHip position in world: ({hip_world[0]:.3f}, {hip_world[1]:.3f})")

# Transform foot
Foot = rot @ foot_shape + np.array([[q0], [q1]])
print(f"\nFoot after transform:")
print(f"  Bottom vertex: ({Foot[0,0]:.3f}, {Foot[1,0]:.3f})")
print(f"  Top vertex: ({Foot[0,5]:.3f}, {Foot[1,5]:.3f})")

# Transform leg  
Leg = rot @ leg_shape + hip_world.reshape(2, 1)
print(f"\nLeg after transform:")
print(f"  Min Y: {Leg[1,:].min():.3f}")
print(f"  Max Y: {Leg[1,:].max():.3f}")

# Transform body
rot_body = np.array([[np.cos(q3), np.sin(q3)], [-np.sin(q3), np.cos(q3)]])
Body = rot_body @ body_shape + hip_world.reshape(2, 1)
print(f"\nBody after transform:")
print(f"  Min Y: {Body[1,:].min():.3f}")
print(f"  Max Y: {Body[1,:].max():.3f}")

