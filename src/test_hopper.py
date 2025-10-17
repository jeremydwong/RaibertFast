import numpy as np
from hopper import hopperParameters, hopperDynamicsFwd, hopperDynamics

# Test with initial conditions matching MATLAB
p = hopperParameters()
p.x_dot_des = 3.0

# y0 = [0.0; 0.4; 0.01; 0.0; 1.0; zeros(5,1)];
y0 = np.array([0.0, 0.4, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
t = 0.0

print("Testing hopperDynamics at t=0...")
print(f"Initial state: {y0}")
print(f"FSM state: {p.fsm_state}")

try:
    # Test the dynamics function
    qdot = hopperDynamics(t, y0, p)
    print(f"Success! qdot = {qdot}")
except Exception as e:
    print(f"Error: {e}")

    # Let's debug by calling hopperDynamicsFwd directly
    print("\nDebugging hopperDynamicsFwd...")
    try:
        structFwd = hopperDynamicsFwd(t, y0, p)
        print(f"structFwd['stated'] = {structFwd['stated']}")
    except Exception as e2:
        print(f"Error in hopperDynamicsFwd: {e2}")

        # Let's check R value
        R = y0[4] - p.l_1
        print(f"\nR = q[4] - p.l_1 = {y0[4]} - {p.l_1} = {R}")
        print(f"This should be 0.5, not 0")
