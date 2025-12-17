# C++ Hopper Implementation - Design Document

## Overview

A C++ port of the Raibert hopper simulation, designed for:
1. **Educational contrast** with Python (static typing, explicit memory, performance)
2. **Testing fundamentals** (unit tests for dynamics, comparison with Python/MATLAB)
3. **Performance demonstration** (expected 10-100x speedup over Python)

## Philosophy

Following Casey Muratori / Jonathan Blow principles:
- **Unity build**: Single compilation unit, fast builds
- **No Boost**: Simple, direct dependencies only
- **Readable code**: Students can read top-to-bottom
- **Explicit over implicit**: No hidden magic

---

## Project Structure

```
cpp/
├── build.sh                    # One-line build script
├── build.cpp                   # Unity build: #includes everything, contains main()
├── hopper.hpp                  # All simulation code (~400 lines)
├── ode.hpp                     # RK45 integrator + event detection (~150 lines)
├── test_hopper.cpp             # Unit tests (separate build target)
├── compare_with_matlab.cpp     # Comparison against reference data
├── export_trajectory.cpp       # Exports trajectory for Python visualization
└── test_data/
    └── reference_cases.csv     # Test cases exported from MATLAB .mat
```

Python additions:
```
src/
├── export_test_cases.py        # Converts .mat → CSV for C++ testing
└── visualize_cpp_trajectory.py # MeshCat visualization of C++ output
```

---

## Data Structures

### State (10 elements, named)

```cpp
struct State {
    // Positions (5)
    double x_foot;      // q[0] - x position of foot
    double z_foot;      // q[1] - y/z position of foot (height)
    double phi_leg;     // q[2] - absolute angle of leg from vertical
    double phi_body;    // q[3] - absolute angle of body from vertical
    double len_leg;     // q[4] - leg length

    // Velocities (5)
    double ddt_x_foot;      // q[5]
    double ddt_z_foot;      // q[6]
    double ddt_phi_leg;     // q[7]
    double ddt_phi_body;    // q[8]
    double ddt_len_leg;     // q[9]

    // Array access for ODE solver compatibility
    double& operator[](int i);
    const double& operator[](int i) const;
    static constexpr int SIZE = 10;
};
```

### StateDot (10 elements)

```cpp
struct StateDot {
    // Velocity copies (for ODE: dy/dt where y = [pos, vel])
    double ddt_x_foot;
    double ddt_z_foot;
    double ddt_phi_leg;
    double ddt_phi_body;
    double ddt_len_leg;

    // Accelerations
    double dddt_x_foot;
    double dddt_z_foot;
    double dddt_phi_leg;
    double dddt_phi_body;
    double dddt_len_leg;

    double& operator[](int i);
    static constexpr int SIZE = 10;
};
```

### Parameters

```cpp
struct Parameters {
    // Physical constants
    double m      = 10.0;    // body mass
    double m_l    = 1.0;     // leg mass
    double J      = 10.0;    // body moment of inertia
    double J_l    = 1.0;     // leg moment of inertia
    double g      = 9.8;     // gravity
    double k_l    = 1e3;     // leg spring constant
    double k_stop = 1e5;     // leg stop spring constant
    double b_stop = 1e3;     // leg stop damping
    double k_g    = 1e4;     // ground spring constant
    double b_g    = 300.0;   // ground damping
    double r_s0   = 1.0;     // leg spring rest length
    double l_1    = 0.5;     // foot to leg COM distance
    double l_2    = 0.4;     // hip to body COM distance

    // FSM state constants
    static constexpr int FSM_COMPRESSION = 0;
    static constexpr int FSM_THRUST      = 1;
    static constexpr int FSM_FLIGHT      = 2;
    static constexpr int FSM_NUM_STATES  = 3;

    // Mutable FSM state (changes during simulation)
    int    fsm_state       = FSM_FLIGHT;
    double t_state_switch  = 0.0;
    double x_dot_des       = 0.0;
    double T_s             = 0.425;
    double T_compression   = 0.0;
    double t_thrust_on     = 0.0;
    double T_MAX_THRUST_DUR = 0.425 * 0.35;
};
```

### Control Output

```cpp
struct ControlOutput {
    double u1;      // Leg spring actuator force
    double u2;      // Hip torque
    double a_des;   // Desired leg angle (for logging)
};
```

### Dynamics Output

```cpp
struct DynamicsOutput {
    StateDot state_dot;
    ControlOutput control;
    double r_sd;        // Spring deflection
    int fsm_state;      // Current FSM state
};
```

---

## Core Functions

### Control

```cpp
// Returns control forces based on FSM state
ControlOutput hopper_control(double t, const State& q, const Parameters& p);
```

### Dynamics

```cpp
// Full dynamics computation (returns all outputs)
DynamicsOutput hopper_dynamics_fwd(double t, const State& q, const Parameters& p);

// ODE-compatible wrapper (returns just state derivative)
StateDot hopper_dynamics(double t, const State& q, const Parameters& p);
```

### Matrix Solvers

```cpp
// Hand-written 5x5 Gaussian elimination (educational, transparent)
void solve_5x5_gaussian(const double M[5][5], const double b[5], double x[5]);

// Eigen-based solver (default, faster for larger systems)
void solve_5x5_eigen(const double M[5][5], const double b[5], double x[5]);

// Function pointer to select solver (default: Eigen)
using MatrixSolver = void(*)(const double[5][5], const double[5], double[5]);
extern MatrixSolver g_matrix_solver;  // = solve_5x5_eigen
```

### Events

```cpp
// Returns event function value (zero-crossing triggers state transition)
double hopper_event(double t, const State& q, const Parameters& p);
```

---

## ODE Integration (ode.hpp)

Custom implementation, no external dependencies.

### Dormand-Prince RK45 (Adaptive Step)

```cpp
struct RK45Config {
    double rtol = 1e-6;
    double atol = 1e-8;
    double max_step = 0.05;
    double min_step = 1e-10;
};

// Single step with error estimate
// Returns: (y_new, error_estimate, suggested_next_step)
std::tuple<State, double, double> rk45_step(
    std::function<StateDot(double, const State&)> f,
    double t, const State& y, double h
);
```

### Event Detection (Bisection)

```cpp
// Find zero-crossing of event function within [t0, t1]
// Precondition: event(t0) and event(t1) have opposite signs
double find_event_time(
    std::function<StateDot(double, const State&)> dynamics,
    std::function<double(double, const State&)> event,
    double t0, const State& y0,
    double t1, const State& y1,
    double tol = 1e-10
);
```

### Main Integration Loop

```cpp
struct IntegrateResult {
    std::vector<double> t;
    std::vector<State> y;
    bool event_occurred;
    double t_event;
    State y_event;
};

IntegrateResult integrate_with_events(
    std::function<StateDot(double, const State&)> dynamics,
    std::function<double(double, const State&)> event,
    double t0, double tf,
    const State& y0,
    RK45Config config = {}
);
```

---

## Testing Strategy

### test_hopper.cpp

```cpp
// ============================================
// TEST 1: Known input produces expected output
// ============================================
void test_dynamics_known_input() {
    Parameters p;
    State q = {0.0, 0.5, 0.1, 0.05, 1.0,   // positions
               1.0, 0.0, 0.0, 0.0, 0.0};   // velocities
    p.fsm_state = Parameters::FSM_FLIGHT;

    auto result = hopper_dynamics_fwd(0.0, q, p);

    // Compare against pre-computed Python reference
    ASSERT_NEAR(result.state_dot.dddt_x_foot, expected_ddx, 1e-10);
    // ... etc
}

// ============================================
// TEST 2: Zero gravity, stationary → no acceleration
// ============================================
void test_zero_gravity_equilibrium() {
    Parameters p;
    p.g = 0.0;

    // State at rest, foot on ground, leg vertical
    State q = {0.0, 0.0, 0.0, 0.0, 1.0,   // positions
               0.0, 0.0, 0.0, 0.0, 0.0};   // velocities (all zero)
    p.fsm_state = Parameters::FSM_FLIGHT;  // in air, no ground contact

    auto result = hopper_dynamics_fwd(0.0, q, p);

    // All accelerations should be zero (or near-zero)
    ASSERT_NEAR(result.state_dot.dddt_x_foot, 0.0, 1e-10);
    ASSERT_NEAR(result.state_dot.dddt_z_foot, 0.0, 1e-10);
    // ... etc
}

// ============================================
// TEST 3: Both matrix solvers produce same result
// ============================================
void test_matrix_solvers_equivalent() {
    double M[5][5] = { /* test matrix */ };
    double b[5] = { /* test vector */ };
    double x_gauss[5], x_eigen[5];

    solve_5x5_gaussian(M, b, x_gauss);
    solve_5x5_eigen(M, b, x_eigen);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(x_gauss[i], x_eigen[i], 1e-12);
    }
}

// ============================================
// TEST 4: Control outputs in each FSM state
// ============================================
void test_control_fsm_states() {
    Parameters p;
    State q = { /* representative state */ };

    // Test THRUST state produces non-zero u1
    p.fsm_state = Parameters::FSM_THRUST;
    auto ctrl = hopper_control(0.0, q, p);
    ASSERT(ctrl.u1 > 0.0);  // thrust force should be positive

    // Test COMPRESSION state has zero u1
    p.fsm_state = Parameters::FSM_COMPRESSION;
    ctrl = hopper_control(0.0, q, p);
    ASSERT_NEAR(ctrl.u1, 0.0, 1e-10);
}
```

### compare_with_matlab.cpp

```cpp
int main() {
    // Load reference cases from CSV
    auto test_cases = load_test_cases("test_data/reference_cases.csv");

    double max_error = 0.0;
    int worst_case = -1;

    for (int i = 0; i < test_cases.size(); i++) {
        auto& tc = test_cases[i];

        // Set up parameters to match test case
        Parameters p;
        p.fsm_state = tc.fsm_state;
        p.t_state_switch = tc.t_state_switch;
        p.x_dot_des = tc.x_dot_des;
        p.T_s = tc.T_s;
        p.T_compression = tc.T_compression;
        p.t_thrust_on = tc.t_thrust_on;

        // Compute dynamics
        auto result = hopper_dynamics_fwd(tc.t, tc.state, p);

        // Compare
        double error = max_abs_diff(result.state_dot, tc.expected_state_dot);
        if (error > max_error) {
            max_error = error;
            worst_case = i;
        }
    }

    printf("Max error: %.6e (case %d)\n", max_error, worst_case);
    return (max_error < 1e-10) ? 0 : 1;
}
```

---

## Visualization (Hybrid Approach)

### C++ Side: export_trajectory.cpp

```cpp
int main() {
    Parameters p;
    p.x_dot_des = 3.0;

    State y0 = {0.0, 0.4, 0.01, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0};

    // Run simulation
    auto [tout, yout] = run_simulation(0.0, 5.0, y0, p);

    // Export to CSV
    FILE* f = fopen("trajectory.csv", "w");
    fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,...\n");
    for (int i = 0; i < tout.size(); i++) {
        fprintf(f, "%.10e,%.10e,%.10e,...\n",
                tout[i], yout[i].x_foot, yout[i].z_foot, ...);
    }
    fclose(f);

    printf("Exported %zu frames to trajectory.csv\n", tout.size());
    return 0;
}
```

### Python Side: visualize_cpp_trajectory.py

```python
"""Load C++ trajectory and visualize with MeshCat"""
import pandas as pd
from test_meshcat_animation import animate_meshcat
from hopper import hopperParameters

def main():
    # Load C++ output
    df = pd.read_csv('cpp/trajectory.csv')
    tout = df['t'].values
    yout = df[['x_foot','z_foot','phi_leg','phi_body','len_leg',
               'ddt_x_foot','ddt_z_foot','ddt_phi_leg','ddt_phi_body','ddt_len_leg']].values

    p = hopperParameters()
    animate_meshcat(tout, yout, p, speed=1.0)

if __name__ == '__main__':
    main()
```

---

## Build System

### build.sh

```bash
#!/bin/bash
set -e

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    CXX=clang++
    EIGEN_PATH="/opt/homebrew/include/eigen3"  # or /usr/local/include/eigen3
else
    CXX=g++
    EIGEN_PATH="/usr/include/eigen3"
fi

# Unity build - compile everything in one shot
$CXX -std=c++17 -O2 -DNDEBUG \
    -I"$EIGEN_PATH" \
    build.cpp \
    -o hopper

echo "Built: ./hopper"
```

### build.cpp (Unity Build Entry Point)

```cpp
// Unity build: all code compiled as single translation unit
// This is the Casey Muratori / Jonathan Blow style

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>

// External dependency: Eigen (header-only)
#include <Eigen/Dense>

// Our code
#include "hopper.hpp"
#include "ode.hpp"

int main(int argc, char** argv) {
    printf("Raibert Hopper Simulation (C++)\n");
    printf("================================\n\n");

    Parameters p;
    p.x_dot_des = 3.0;

    State y0 = {};
    y0.x_foot = 0.0;
    y0.z_foot = 0.4;
    y0.phi_leg = 0.01;
    y0.phi_body = 0.0;
    y0.len_leg = 1.0;
    // velocities default to 0

    auto start = std::chrono::high_resolution_clock::now();

    // Run simulation (event-driven loop similar to Python)
    auto [tout, yout, fsm_history] = run_hopper_simulation(0.0, 5.0, y0, p);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("Simulation complete.\n");
    printf("  Time: %.4f s simulated in %.4f s (%.1fx realtime)\n",
           tout.back(), elapsed, tout.back() / elapsed);
    printf("  Final position: (%.4f, %.4f)\n",
           yout.back().x_foot, yout.back().z_foot);

    // Export trajectory
    export_trajectory("trajectory.csv", tout, yout);
    printf("  Exported to trajectory.csv\n");

    return 0;
}
```

### Test Build

```bash
#!/bin/bash
# build_tests.sh
clang++ -std=c++17 -O0 -g \
    -I"/opt/homebrew/include/eigen3" \
    test_hopper.cpp \
    -o test_hopper

./test_hopper
```

---

## Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **C++17 compiler** | Language | Xcode Command Line Tools |
| **Eigen** | Matrix solve | `brew install eigen` |

That's it. No Boost. No CMake (just shell scripts). No package managers beyond Homebrew for Eigen.

---

## Educational Highlights for Students

### Static Typing Contrast

**Python:**
```python
def hopper_dynamics(t, q, p):
    u, a_des = hopperStateControl(t, q, p)  # types unclear
    # q[0] - what is this? have to remember
```

**C++:**
```cpp
StateDot hopper_dynamics(double t, const State& q, const Parameters& p) {
    ControlOutput ctrl = hopper_control(t, q, p);  // types explicit
    // q.x_foot - self-documenting
}
```

### Memory and Performance

- No garbage collection pauses
- Data locality (structs vs Python dicts)
- Compiler optimizations visible in assembly

### Explicit Error Handling

```cpp
if (std::isnan(a_des)) {
    fprintf(stderr, "ERROR: NaN in a_des at t=%.6f\n", t);
    // In Python this might silently propagate
}
```

---

## Implementation Order

1. **hopper.hpp** - Data structures and dynamics (core logic)
2. **test_hopper.cpp** - Basic unit tests
3. **ode.hpp** - RK45 integrator with events
4. **build.cpp** - Main simulation loop
5. **compare_with_matlab.cpp** - Validation against reference
6. **Python export scripts** - Test data + visualization

---

## Approval Checklist

- [ ] Data structures (State, StateDot, Parameters) look correct?
- [ ] Testing approach covers what you need?
- [ ] Unity build style acceptable?
- [ ] Hybrid visualization approach works for your class?
- [ ] Implementation order makes sense?

Please review and let me know if you'd like any changes before I start implementation.
