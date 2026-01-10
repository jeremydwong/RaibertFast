# CUDA Hopper Implementation - Design Document

## Overview

A CUDA port of the Raibert hopper simulation for massively parallel execution (~10k simultaneous hoppers). Primary goal: parameter sweeps and reinforcement learning for optimal feedback gains.

---

## Code Style: Casey Muratori / Jonathan Blow Principles

1. **Simple build script** - One line to build. No CMake, no makefiles, no build system complexity.
2. **Structs not objects** - No member methods. Data and functions are separate.
3. **Clarity in syntax** - Redefinition of contextualized static types where it aids readability.
4. **Unity builds** - Single compilation unit. All code `#include`d together. No spaghetti imports, no partial builds.

### Build Commands (Windows)

```bat
REM CPU build (from cpp/)
cl /std:c++17 /O2 /EHsc /I"C:\libs\eigen-3.4.1" build.cpp /Fe:hopper.exe

REM CUDA build (from cpp/cuda/)
nvcc -O2 test_cuda.cu -o test_cuda.exe

REM CPU implicit midpoint test (from cpp/)
cl /std:c++17 /O2 /EHsc test_implicit_cpu.cpp /Fe:test_implicit_cpu.exe
```

### Tool Paths (Windows)

```
cl.exe:   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
nvcc.exe: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe
Eigen:    C:\libs\eigen-3.4.1
vcvars:   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
```

---

### Key Constraints

1. **No warp divergence**: All hoppers in a warp must execute the same instructions
2. **Fixed-step integration**: Adaptive stepping causes divergence
3. **Stiff dynamics**: Ground contact spring (k_g=1e4) requires careful integrator choice
4. **Approximate events OK**: FSM transitions can have ~dt slop for simplicity

### Non-Goals

- MATLAB-matching precision (this is for learning, not validation)
- Sophisticated event detection (bisection, interpolation)
- Variable step sizes

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Host (CPU)                           │
├─────────────────────────────────────────────────────────────┤
│  - Initialize N hopper states (randomized ICs)              │
│  - Initialize N parameter sets (for sweeps)                 │
│  - Allocate device memory                                   │
│  - Launch kernels                                           │
│  - Collect results (final states, metrics)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Device (GPU)                          │
├─────────────────────────────────────────────────────────────┤
│  Kernel: step_all_hoppers<<<blocks, threads>>>              │
│                                                             │
│  for each timestep:                                         │
│    1. Compute control (u1, u2) based on FSM state           │
│    2. Compute dynamics (accelerations)                      │
│    3. Semi-implicit Euler step                              │
│    4. Check event conditions, update FSM if triggered       │
│    5. (Optional) Store trajectory sample                    │
│                                                             │
│  Memory layout: Structure of Arrays (SoA) for coalescing    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integrator Design: Flexible Architecture

We design for easy swapping between integrator methods. **Implicit Midpoint** is the primary choice for its combination of stability, accuracy, and energy preservation.

### Integrator Interface

All integrators implement the same device function signature:

```cpp
// Takes current state (q, qdot), returns new state after dt
// All state variables passed by pointer for in-place update
__device__ void integrator_step(
    // Positions (in/out)
    double* x_foot, double* z_foot, double* phi_leg, double* phi_body, double* len_leg,
    // Velocities (in/out)
    double* ddt_x_foot, double* ddt_z_foot, double* ddt_phi_leg, double* ddt_phi_body, double* ddt_len_leg,
    // Control inputs (computed externally)
    double u1, double u2,
    // Timestep
    double dt
);
```

The main simulation kernel calls this function without knowing which integrator is compiled in. Switching integrators = recompile with different `#define` or template parameter.

```cpp
// In hopper_cuda.cuh
#define INTEGRATOR_IMPLICIT_MIDPOINT    1
#define INTEGRATOR_SEMI_IMPLICIT_EULER  2
#define INTEGRATOR_IMPLICIT_EULER       3

#ifndef HOPPER_INTEGRATOR
#define HOPPER_INTEGRATOR INTEGRATOR_IMPLICIT_MIDPOINT
#endif
```

---

## Primary: Implicit Midpoint (Second-Order, Symplectic, Stable)

The implicit midpoint method evaluates dynamics at the midpoint between current and next state:

```
y_mid = (y + y_new) / 2
y_new = y + dt * f(y_mid)
```

Rearranging, we solve the nonlinear system:
```
G(y_new) = y_new - y - dt * f((y + y_new) / 2) = 0
```

**Why Implicit Midpoint?**

| Property | Implicit Midpoint | Implicit Euler | Semi-Implicit Euler |
|----------|-------------------|----------------|---------------------|
| Order | 2 | 1 | 1 |
| Symplectic | Yes | No | Yes |
| Stability | Unconditional | Unconditional | Conditional |
| Energy drift | Minimal | Dissipative | Minimal (if stable) |
| Newton iterations | Yes | Yes | No |

- **Second-order**: Error scales as O(dt²), not O(dt). At dt=1e-4, this is 1e-8 vs 1e-4 local error.
- **Symplectic**: Preserves the geometric structure of Hamiltonian systems. Energy oscillates but doesn't drift monotonically.
- **No artificial damping**: Unlike implicit Euler, which damps high frequencies, midpoint preserves oscillatory behavior.

For a hopper where energy injection/dissipation balance is critical, these properties matter.

**Newton Iteration for Implicit Midpoint:**

Let y = [q; qdot] be the full 10D state. We solve:

```
G(y_new) = y_new - y - dt * f((y + y_new) / 2) = 0
```

Newton step:
```
y_new^{k+1} = y_new^{k} - J^{-1} * G(y_new^{k})
```

The Jacobian is:
```
J = dG/dy_new = I - (dt/2) * df/dy|_{y_mid}
```

Note the factor of 1/2 compared to implicit Euler (which has J = I - dt * df/dy).

**Implementation:**

```cpp
__device__ void implicit_midpoint_step(
    double* x_foot, double* z_foot, double* phi_leg, double* phi_body, double* len_leg,
    double* ddt_x_foot, double* ddt_z_foot, double* ddt_phi_leg, double* ddt_phi_body, double* ddt_len_leg,
    double u1, double u2, double dt
) {
    // Pack current state
    double y[10] = {*x_foot, *z_foot, *phi_leg, *phi_body, *len_leg,
                    *ddt_x_foot, *ddt_z_foot, *ddt_phi_leg, *ddt_phi_body, *ddt_len_leg};
    double y_new[10];

    // Initial guess: explicit Euler step
    double f_curr[10];
    compute_state_derivative(y, u1, u2, f_curr);
    for (int i = 0; i < 10; i++) {
        y_new[i] = y[i] + dt * f_curr[i];
    }

    // Newton iterations (fixed count for GPU uniformity)
    constexpr int NEWTON_ITERS = 4;

    for (int iter = 0; iter < NEWTON_ITERS; iter++) {
        // Compute midpoint
        double y_mid[10];
        for (int i = 0; i < 10; i++) {
            y_mid[i] = 0.5 * (y[i] + y_new[i]);
        }

        // Evaluate dynamics at midpoint
        double f_mid[10];
        compute_state_derivative(y_mid, u1, u2, f_mid);

        // Compute residual: G = y_new - y - dt * f(y_mid)
        double G[10];
        for (int i = 0; i < 10; i++) {
            G[i] = y_new[i] - y[i] - dt * f_mid[i];
        }

        // Compute Jacobian J = I - (dt/2) * df/dy at y_mid
        double J[10][10];
        compute_jacobian_midpoint(y_mid, u1, u2, dt, J);  // Note: includes dt/2 factor

        // Solve J * delta = G
        double delta[10];
        solve_10x10(J, G, delta);

        // Update: y_new = y_new - delta
        for (int i = 0; i < 10; i++) {
            y_new[i] -= delta[i];
        }
    }

    // Unpack result
    *x_foot = y_new[0]; *z_foot = y_new[1]; *phi_leg = y_new[2];
    *phi_body = y_new[3]; *len_leg = y_new[4];
    *ddt_x_foot = y_new[5]; *ddt_z_foot = y_new[6]; *ddt_phi_leg = y_new[7];
    *ddt_phi_body = y_new[8]; *ddt_len_leg = y_new[9];
}
```

---

## Jacobian Computation

The Jacobian df/dy for our system has structure. Let y = [q; qdot] where q and qdot are each 5D.

The state derivative is:
```
f(y) = [qdot; qddot(q, qdot)]
```

So:
```
df/dy = [ 0        I      ]
        [ dqddot/dq  dqddot/dqdot ]
```

The top-left is zero (positions don't directly affect velocity derivatives - those are just copied).
The top-right is identity (qdot directly becomes the position derivative).
The bottom row contains the actual dynamics Jacobian.

Let A = dqddot/dq (5x5) and B = dqddot/dqdot (5x5).

For implicit midpoint, the full Jacobian of G is:
```
J = I - (dt/2) * [ 0  I ]  =  [ I      -dt/2 * I    ]
                [ A  B ]     [ -dt/2 * A   I - dt/2 * B ]
```

### Computing A and B

**Option 1: Finite Difference** (implement first)

```cpp
__device__ void compute_jacobian_midpoint(
    const double y_mid[10], double u1, double u2, double dt,
    double J[10][10]
) {
    constexpr double eps = 1e-7;

    // Compute f at y_mid
    double f0[10];
    compute_state_derivative(y_mid, u1, u2, f0);

    // For each state variable, perturb and compute derivative
    for (int j = 0; j < 10; j++) {
        double y_pert[10];
        for (int i = 0; i < 10; i++) y_pert[i] = y_mid[i];
        y_pert[j] += eps;

        double f_pert[10];
        compute_state_derivative(y_pert, u1, u2, f_pert);

        // df/dy_j ≈ (f_pert - f0) / eps
        for (int i = 0; i < 10; i++) {
            double df_dy = (f_pert[i] - f0[i]) / eps;
            // J = I - (dt/2) * df/dy
            if (i == j) {
                J[i][j] = 1.0 - 0.5 * dt * df_dy;
            } else {
                J[i][j] = -0.5 * dt * df_dy;
            }
        }
    }
}
```

This requires 10 extra dynamics evaluations per Newton iteration. With 4 Newton iterations, that's 40 dynamics evals per timestep. At dt=1e-4, this may be acceptable, but could become a bottleneck.

**Option 2: Analytical Jacobian** (optimize later if needed)

Derive A = dqddot/dq and B = dqddot/dqdot symbolically from the dynamics equations. This is tedious but eliminates the finite-difference overhead. The dynamics involve:
- Trig functions of angles (sin, cos)
- The 5x5 mass matrix M(q)
- Forces that depend on q and qdot

Since qddot = M(q)^{-1} * forces(q, qdot), the Jacobian involves:
- dM^{-1}/dq (product rule with M^{-1} * dM/dq * M^{-1})
- dforces/dq and dforces/dqdot

**Recommendation**: Start with finite difference. If profiling shows Jacobian computation is >50% of step time, invest in analytical derivation.

---

## 10x10 Linear Solve

We can exploit the block structure of J to reduce to 5x5 solves.

Given:
```
J = [ I           -dt/2 * I    ]
    [ -dt/2 * A   I - dt/2 * B ]
```

And solving J * [dq; dqdot] = [G_q; G_qdot]:

From the first row:
```
dq - (dt/2) * dqdot = G_q
=> dq = G_q + (dt/2) * dqdot
```

Substitute into second row:
```
-dt/2 * A * (G_q + dt/2 * dqdot) + (I - dt/2 * B) * dqdot = G_qdot
(I - dt/2 * B - (dt/2)² * A) * dqdot = G_qdot + dt/2 * A * G_q
```

Let S = I - dt/2 * B - (dt/2)² * A (the Schur complement, 5x5).

1. Solve S * dqdot = G_qdot + dt/2 * A * G_q  (5x5 solve)
2. Compute dq = G_q + dt/2 * dqdot

This reduces to one 5x5 solve per Newton iteration (plus matrix multiplications).

```cpp
__device__ void solve_10x10_block(
    const double A[5][5], const double B[5][5],
    const double G_q[5], const double G_qdot[5],
    double dt,
    double dq[5], double dqdot[5]
) {
    double half_dt = 0.5 * dt;
    double half_dt_sq = half_dt * half_dt;

    // Build Schur complement S = I - dt/2 * B - (dt/2)^2 * A
    double S[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            S[i][j] = -half_dt * B[i][j] - half_dt_sq * A[i][j];
            if (i == j) S[i][j] += 1.0;
        }
    }

    // Build RHS: G_qdot + dt/2 * A * G_q
    double rhs[5];
    for (int i = 0; i < 5; i++) {
        rhs[i] = G_qdot[i];
        for (int j = 0; j < 5; j++) {
            rhs[i] += half_dt * A[i][j] * G_q[j];
        }
    }

    // Solve S * dqdot = rhs
    solve_5x5(S, rhs, dqdot);

    // Back-substitute: dq = G_q + dt/2 * dqdot
    for (int i = 0; i < 5; i++) {
        dq[i] = G_q[i] + half_dt * dqdot[i];
    }
}
```

---

## Fallback: Semi-Implicit Euler (If Midpoint Too Slow)

If profiling shows implicit midpoint is too expensive for the target throughput, semi-implicit Euler is a fallback. It's much cheaper (1 dynamics eval, no iteration) but may require smaller dt for stability.

```cpp
__device__ void semi_implicit_euler_step(
    double* x_foot, double* z_foot, double* phi_leg, double* phi_body, double* len_leg,
    double* ddt_x_foot, double* ddt_z_foot, double* ddt_phi_leg, double* ddt_phi_body, double* ddt_len_leg,
    double u1, double u2, double dt
) {
    // 1. Compute accelerations at current state
    double qdd[5];
    compute_accelerations(
        *x_foot, *z_foot, *phi_leg, *phi_body, *len_leg,
        *ddt_x_foot, *ddt_z_foot, *ddt_phi_leg, *ddt_phi_body, *ddt_len_leg,
        u1, u2, qdd
    );

    // 2. Update velocities (explicit in acceleration)
    *ddt_x_foot   += dt * qdd[0];
    *ddt_z_foot   += dt * qdd[1];
    *ddt_phi_leg  += dt * qdd[2];
    *ddt_phi_body += dt * qdd[3];
    *ddt_len_leg  += dt * qdd[4];

    // 3. Update positions (using NEW velocities - this is the "semi-implicit" part)
    *x_foot   += dt * (*ddt_x_foot);
    *z_foot   += dt * (*ddt_z_foot);
    *phi_leg  += dt * (*ddt_phi_leg);
    *phi_body += dt * (*ddt_phi_body);
    *len_leg  += dt * (*ddt_len_leg);
}
```

---

## Fallback: Implicit Euler (Debugging / Maximum Stability)

Implicit Euler is first-order and dissipative, but maximally robust. Useful for debugging if midpoint has convergence issues.

Same Newton machinery as midpoint, but simpler residual:
```
F(y_new) = y_new - y - dt * f(y_new) = 0
J = I - dt * df/dy
```

Not recommended for production due to artificial energy damping.

---

## Stability Comparison

| Method | Order | Stability | Energy | Evals/step | 5x5 Solves/step |
|--------|-------|-----------|--------|------------|-----------------|
| Semi-Implicit Euler | 1 | Conditional | Conserved | 1 | 1 |
| Implicit Euler | 1 | Unconditional | Dissipated | 4-5 | 4-5 |
| **Implicit Midpoint** | **2** | **Unconditional** | **Conserved** | **4-5** | **4-5** |
| RK4 | 4 | Conditional | Varies | 4 | 4 |

**Primary choice: Implicit Midpoint** - best combination of accuracy, stability, and physical fidelity.

---

## Integrator Selection at Compile Time

```cpp
// hopper_integrator.cuh

template<int IntegratorType>
__device__ void integrator_step(
    double* x_foot, double* z_foot, double* phi_leg, double* phi_body, double* len_leg,
    double* ddt_x_foot, double* ddt_z_foot, double* ddt_phi_leg, double* ddt_phi_body, double* ddt_len_leg,
    double u1, double u2, double dt
);

template<>
__device__ void integrator_step<INTEGRATOR_IMPLICIT_MIDPOINT>(...) {
    implicit_midpoint_step(...);
}

template<>
__device__ void integrator_step<INTEGRATOR_SEMI_IMPLICIT_EULER>(...) {
    semi_implicit_euler_step(...);
}

template<>
__device__ void integrator_step<INTEGRATOR_IMPLICIT_EULER>(...) {
    implicit_euler_step(...);
}

// Main kernel uses:
integrator_step<HOPPER_INTEGRATOR>(...);
```

Switch integrators by changing one `#define` and recompiling.

---

## Data Layout: Structure of Arrays (SoA)

### Why SoA?

GPU memory accesses are coalesced when adjacent threads access adjacent memory. With Array of Structures (AoS), threads accessing the same field read scattered memory:

```cpp
// AoS (bad for GPU):
struct Hopper { double x, z, phi_leg, ...; };  // 80 bytes per hopper
Hopper hoppers[N];
// Thread 0 reads hoppers[0].x at byte 0
// Thread 1 reads hoppers[1].x at byte 80  <- 80-byte stride, poor coalescing

// SoA (good for GPU):
struct HopperArrays {
    double* x_foot;      // N doubles, contiguous
    double* z_foot;      // N doubles, contiguous
    ...
};
// Thread 0 reads x_foot[0] at byte 0
// Thread 1 reads x_foot[1] at byte 8   <- 8-byte stride, perfect coalescing
```

### Memory Layout

```cpp
// Device memory structure (SoA)
struct HopperStateArrays {
    // Positions (5 arrays)
    double* x_foot;       // [N]
    double* z_foot;       // [N]
    double* phi_leg;      // [N]
    double* phi_body;     // [N]
    double* len_leg;      // [N]

    // Velocities (5 arrays)
    double* ddt_x_foot;   // [N]
    double* ddt_z_foot;   // [N]
    double* ddt_phi_leg;  // [N]
    double* ddt_phi_body; // [N]
    double* ddt_len_leg;  // [N]

    // FSM state (integer)
    int* fsm_state;       // [N]

    // FSM timing
    double* T_s;          // [N] - stance time estimate
    double* T_compression;// [N]
    double* t_thrust_on;  // [N]
};

struct HopperParamArrays {
    // Physical params (could be shared or per-hopper for sweeps)
    double* k_fp;         // [N] - foot placement gain
    double* b_fp;         // [N] - foot placement damping
    double* k_att;        // [N] - attitude gain
    double* b_att;        // [N] - attitude damping
    double* k_xdot;       // [N] - velocity control gain
    double* thrust;       // [N] - thrust force
    double* x_dot_des;    // [N] - desired velocity

    // Fixed physical params (shared, not per-hopper)
    // These go in constant memory
};

// Constant memory for shared parameters
__constant__ struct {
    double m;             // body mass
    double m_l;           // leg mass
    double J;             // body inertia
    double J_l;           // leg inertia
    double g;             // gravity
    double k_l;           // leg spring constant
    double k_stop;        // leg stop spring
    double b_stop;        // leg stop damping
    double k_g;           // ground spring
    double b_g;           // ground damping
    double r_s0;          // leg rest length
    double l_1;           // foot to leg COM
    double l_2;           // hip to body COM
    double dt;            // timestep
} d_const_params;
```

### Memory Estimates

Per hopper:
- State: 10 doubles + 1 int + 3 doubles (timing) = 112 bytes
- Params (for sweeps): 7 doubles = 56 bytes
- Total per hopper: ~168 bytes

For N=10,000 hoppers: ~1.7 MB (fits easily in GPU memory)

For trajectory storage (optional):
- If storing every 0.001s for 5s: 5000 samples
- Per sample: 10 doubles = 80 bytes
- Per hopper trajectory: 400 KB
- For 10k hoppers: 4 GB (might need selective storage or streaming)

---

## FSM Event Detection

### Original (CPU) Approach
- Integrate until event function crosses zero
- Bisection to find precise crossing time
- Interpolate state at crossing

### CUDA Approach: Check Every Step

Since dt=1e-4s is small, we accept ~0.1ms timing slop:

```cpp
__device__ void check_and_update_fsm(
    int idx,
    HopperStateArrays* states,
    double t
) {
    int fsm = states->fsm_state[idx];

    if (fsm == FSM_COMPRESSION) {
        // Transition when leg stops compressing (ddt_len_leg crosses zero upward)
        // Check: was negative or zero, now positive
        double dlen = states->ddt_len_leg[idx];
        if (dlen > 0.0) {
            // Transition to THRUST
            states->fsm_state[idx] = FSM_THRUST;
            states->T_compression[idx] = t - /* segment start time */;
            states->t_thrust_on[idx] = t;
        }
    }
    else if (fsm == FSM_THRUST) {
        // Transition when leg fully extended
        double r_sd = d_const_params.r_s0 - states->len_leg[idx];
        if (r_sd < -0.0001) {  // leg extended past rest length
            // Transition to FLIGHT
            states->fsm_state[idx] = FSM_FLIGHT;
            // Update T_s estimate
        }
    }
    else if (fsm == FSM_FLIGHT) {
        // Touchdown when foot hits ground
        double z = states->z_foot[idx];
        if (z < 0.0) {
            // Transition to COMPRESSION
            states->fsm_state[idx] = FSM_COMPRESSION;
        }
    }
}
```

### Handling FSM Divergence

Different hoppers will be in different FSM states. This causes *some* warp divergence, but:

1. The control law branches are small (~10-20 instructions each)
2. Dynamics computation is the same for all states
3. We can't avoid this without artificial synchronization

Acceptable tradeoff: FSM branches are fast, dynamics are uniform.

---

## 5x5 Matrix Solve on GPU

The dynamics require solving M * qdd = eta where M is 5x5.

### Option A: Inline Gaussian Elimination

```cpp
__device__ void solve_5x5(
    const double M[5][5],
    const double b[5],
    double x[5]
) {
    // Copy to local (registers)
    double A[5][5], rhs[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        rhs[i] = b[i];
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            A[i][j] = M[i][j];
        }
    }

    // Forward elimination with partial pivoting
    #pragma unroll
    for (int col = 0; col < 5; col++) {
        // Find pivot (unrolled for 5x5)
        int pivot = col;
        double pivot_val = fabs(A[col][col]);
        #pragma unroll
        for (int row = col + 1; row < 5; row++) {
            if (fabs(A[row][col]) > pivot_val) {
                pivot_val = fabs(A[row][col]);
                pivot = row;
            }
        }

        // Swap rows if needed
        if (pivot != col) {
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                double tmp = A[col][j];
                A[col][j] = A[pivot][j];
                A[pivot][j] = tmp;
            }
            double tmp = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = tmp;
        }

        // Eliminate
        #pragma unroll
        for (int row = col + 1; row < 5; row++) {
            double factor = A[row][col] / A[col][col];
            #pragma unroll
            for (int j = col; j < 5; j++) {
                A[row][j] -= factor * A[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    #pragma unroll
    for (int i = 4; i >= 0; i--) {
        x[i] = rhs[i];
        #pragma unroll
        for (int j = i + 1; j < 5; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}
```

### Option B: Analytical Inverse (Precompute Structure)

The mass matrix M has a specific sparsity pattern. We could derive analytical expressions for the inverse, avoiding runtime pivoting. This is more complex but eliminates branches.

**Recommendation**: Start with Option A (Gaussian elimination). Profile, optimize if needed.

---

## Kernel Structure

### Main Simulation Kernel

```cpp
__global__ void simulate_hoppers(
    HopperStateArrays states,
    HopperParamArrays params,
    int N,
    int num_steps,
    double t_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load state into registers
    double x_foot = states.x_foot[idx];
    double z_foot = states.z_foot[idx];
    double phi_leg = states.phi_leg[idx];
    double phi_body = states.phi_body[idx];
    double len_leg = states.len_leg[idx];
    double ddt_x_foot = states.ddt_x_foot[idx];
    double ddt_z_foot = states.ddt_z_foot[idx];
    double ddt_phi_leg = states.ddt_phi_leg[idx];
    double ddt_phi_body = states.ddt_phi_body[idx];
    double ddt_len_leg = states.ddt_len_leg[idx];
    int fsm_state = states.fsm_state[idx];

    // Load per-hopper params
    double k_fp = params.k_fp[idx];
    double x_dot_des = params.x_dot_des[idx];
    // ... etc

    double t = t_start;
    double dt = d_const_params.dt;

    for (int step = 0; step < num_steps; step++) {
        // 1. Compute control
        double u1, u2;
        compute_control(
            t, x_foot, z_foot, phi_leg, phi_body, len_leg,
            ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
            fsm_state, k_fp, x_dot_des, /* other params */,
            &u1, &u2
        );

        // 2. Compute accelerations
        double qdd[5];
        compute_accelerations(
            x_foot, z_foot, phi_leg, phi_body, len_leg,
            ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
            u1, u2, qdd
        );

        // 3. Semi-implicit Euler step
        ddt_x_foot   += dt * qdd[0];
        ddt_z_foot   += dt * qdd[1];
        ddt_phi_leg  += dt * qdd[2];
        ddt_phi_body += dt * qdd[3];
        ddt_len_leg  += dt * qdd[4];

        x_foot   += dt * ddt_x_foot;
        z_foot   += dt * ddt_z_foot;
        phi_leg  += dt * ddt_phi_leg;
        phi_body += dt * ddt_phi_body;
        len_leg  += dt * ddt_len_leg;

        // 4. Check FSM transitions
        fsm_state = check_fsm_transition(
            z_foot, len_leg, ddt_len_leg, fsm_state
        );

        t += dt;
    }

    // Write state back to global memory
    states.x_foot[idx] = x_foot;
    states.z_foot[idx] = z_foot;
    // ... etc
    states.fsm_state[idx] = fsm_state;
}
```

### Launch Configuration

```cpp
// For N=10,000 hoppers:
int threads_per_block = 256;  // Typical choice
int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // = 40

simulate_hoppers<<<num_blocks, threads_per_block>>>(
    d_states, d_params, N, num_steps, t_start
);
```

---

## File Structure

```
cpp/
├── cuda/
│   ├── hopper_cuda.cuh       # Device code: dynamics, control, integrator
│   ├── hopper_cuda.cu        # Kernels and host-side wrappers
│   ├── hopper_types.cuh      # SoA structures, constants
│   ├── build.sh              # nvcc build script
│   └── main.cu               # Entry point, benchmarking
│
├── hopper.hpp                # Original CPU implementation (reference)
├── ode.hpp                   # Original CPU integrator (reference)
└── build.cpp                 # Original CPU main
```

### Build Script

```bash
#!/bin/bash
# cuda/build.sh

nvcc -std=c++17 -O3 \
    -arch=sm_86 \              # Adjust for your GPU (e.g., sm_75 for Turing)
    -Xcompiler -Wall \
    main.cu \
    -o hopper_cuda

echo "Built: ./hopper_cuda"
```

---

## Validation Strategy

### 1. Single-Hopper CPU vs GPU

Run one hopper on both CPU (RK45) and GPU (semi-implicit Euler) with same ICs:
- Compare trajectories
- Expect small differences due to integrator
- Check: same FSM transition sequence, similar final state

### 2. Energy Conservation

Semi-implicit Euler should approximately conserve energy during flight (no ground contact, no control). Test:
- Set FSM to FLIGHT, disable ground contact
- Run for many cycles
- Check total energy drift

### 3. Ground Contact Stability

Stress test the stiff ground contact:
- Drop hopper from various heights
- Check for numerical blowup
- Verify ground penetration stays small (z_foot > -0.01m or similar)

### 4. Mass Parallel Correctness

Run N=10,000 hoppers with identical ICs and params:
- All should produce identical results
- Verifies no race conditions or indexing bugs

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Hoppers | 10,000 | Single GPU |
| Sim time | 5 seconds | Typical episode |
| dt | 1e-4 s | 50,000 steps |
| Wall time | < 1 second | For 10k x 5s |
| Throughput | > 50k hopper-seconds/sec | |

Expected performance based on similar workloads:
- Each step: ~200 FLOPs (control) + ~500 FLOPs (dynamics) + ~500 FLOPs (5x5 solve) ≈ 1200 FLOPs
- 50,000 steps × 10,000 hoppers × 1200 FLOPs = 600 GFLOPs total
- RTX 3080: ~30 TFLOPS FP32 → theoretical minimum ~0.02s
- With memory overhead, expect ~0.1-0.5s actual

---

## Testing Strategy

### Philosophy

Debug on CPU first, then port to GPU. GPU debugging is painful; CPU debugging has printf, gdb, and fast iteration. By validating each layer against the C++ reference, GPU bugs are isolated to parallelization issues only.

### Test Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Full Trajectory (qualitative only)                │
│  - Different integrators = different paths                  │
│  - FSM timing differs due to event detection slop           │
│  - Compare: "does it hop reasonably?"                       │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Integrator Step (CPU implicit midpoint)           │
│  - Single step comparison to RK45 (expect small diff)       │
│  - Energy conservation over many steps (flight only)        │
│  - Stability through ground contact                         │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Jacobian & Linear Solve                           │
│  - Finite-diff Jacobian vs analytical (if implemented)      │
│  - 5x5 solve correctness                                    │
│  - Schur complement 10x10 solve correctness                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Dynamics (exact match to C++ reference)           │
│  - compute_accelerations: state → qdd[5]                    │
│  - compute_control: state + FSM → u1, u2                    │
│  - These are pure functions, must match exactly             │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Dynamics Unit Tests (Exact Match)

These test the physics port. Must match C++ `hopper.hpp` to machine precision.

```cpp
// test_dynamics.cpp

// Test case structure for dynamics
struct DynamicsTestCase {
    State q;
    int fsm_state;
    double t;
    double expected_qdd[5];
    double expected_u1, expected_u2;
};

// Generate test cases from C++ reference
void generate_dynamics_test_cases() {
    // Flight state, typical values
    test_dynamics({
        .q = {0.0, 0.5, 0.1, 0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        .fsm_state = FSM_FLIGHT,
        .t = 0.0
    });

    // Ground contact (foot below ground)
    test_dynamics({
        .q = {0.0, -0.01, 0.1, 0.05, 0.95, 1.0, -0.5, 0.0, 0.0, -0.1},
        .fsm_state = FSM_COMPRESSION,
        .t = 0.5
    });

    // Thrust phase
    test_dynamics({
        .q = {1.0, -0.005, 0.05, 0.02, 0.92, 2.0, 0.1, 0.0, 0.0, 0.5},
        .fsm_state = FSM_THRUST,
        .t = 1.0
    });

    // Leg at stop (extended past rest length)
    test_dynamics({
        .q = {2.0, 0.3, -0.1, 0.0, 1.02, 2.5, 1.0, 0.0, 0.0, 0.2},
        .fsm_state = FSM_FLIGHT,
        .t = 1.5
    });

    // Edge case: vertical leg, no angle
    test_dynamics({
        .q = {0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        .fsm_state = FSM_FLIGHT,
        .t = 0.0
    });

    // Edge case: large leg angle
    test_dynamics({
        .q = {0.0, 0.3, 0.5, 0.1, 1.0, 2.0, 0.0, 1.0, 0.5, 0.0},
        .fsm_state = FSM_FLIGHT,
        .t = 0.0
    });
}

void test_dynamics(const DynamicsTestCase& tc) {
    Parameters p;
    p.fsm_state = tc.fsm_state;

    auto result = hopper_dynamics_fwd(tc.t, tc.q, p);

    // Accelerations must match exactly (same computation)
    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(result.state_dot[5+i], tc.expected_qdd[i], 1e-12);
    }

    // Control outputs must match
    ASSERT_NEAR(result.control.u1, tc.expected_u1, 1e-12);
    ASSERT_NEAR(result.control.u2, tc.expected_u2, 1e-12);
}
```

### Layer 2: Linear Algebra Unit Tests

```cpp
// test_linalg.cpp

void test_solve_5x5_known_system() {
    // Simple system with known solution
    double M[5][5] = {
        {2, 1, 0, 0, 0},
        {1, 3, 1, 0, 0},
        {0, 1, 4, 1, 0},
        {0, 0, 1, 5, 1},
        {0, 0, 0, 1, 6}
    };
    double b[5] = {1, 2, 3, 4, 5};
    double x[5];

    solve_5x5(M, b, x);

    // Verify M*x = b
    for (int i = 0; i < 5; i++) {
        double sum = 0;
        for (int j = 0; j < 5; j++) sum += M[i][j] * x[j];
        ASSERT_NEAR(sum, b[i], 1e-12);
    }
}

void test_solve_5x5_ill_conditioned() {
    // Near-singular matrix (high condition number)
    // Verify we don't blow up, solution is reasonable
}

void test_solve_5x5_matches_eigen() {
    // Compare our Gaussian elimination to Eigen
    // Already have this in C++ test_hopper.cpp
}

void test_schur_complement_solve() {
    // Create known A, B matrices
    // Solve full 10x10 system with naive Gaussian
    // Solve with Schur complement method
    // Compare solutions
}
```

### Layer 3: Integrator Unit Tests (CPU First)

```cpp
// test_integrator.cpp

void test_implicit_midpoint_single_step() {
    // Compare single step of implicit midpoint to RK45
    // Expect small but nonzero difference (different methods)
    State q0 = { /* initial state */ };
    Parameters p;
    p.fsm_state = FSM_FLIGHT;

    // RK45 step
    State q_rk45 = rk45_single_step(q0, p, dt);

    // Implicit midpoint step
    State q_midpoint = implicit_midpoint_step_cpu(q0, p, dt);

    // Should be close but not identical
    for (int i = 0; i < 10; i++) {
        double diff = std::abs(q_rk45[i] - q_midpoint[i]);
        ASSERT(diff < 1e-6);  // Same ballpark
        // Don't require exact match
    }
}

void test_implicit_midpoint_newton_convergence() {
    // Verify Newton iteration converges
    // Check residual decreases each iteration
    State q0 = { /* state in ground contact - stiff */ };

    for (int n_iters = 1; n_iters <= 6; n_iters++) {
        double residual = implicit_midpoint_residual(q0, p, dt, n_iters);
        printf("Iterations: %d, Residual: %.2e\n", n_iters, residual);
    }
    // Expect residual to decrease, plateau around 4 iterations
}

void test_energy_conservation_flight() {
    // Flight phase only: no ground contact, no control (u1=u2=0)
    // Energy should oscillate but not drift

    State q0 = {0.0, 1.0, 0.1, 0.0, 1.0,  // In air
                0.0, 0.0, 0.5, 0.0, 0.0}; // Some rotation
    Parameters p;
    p.fsm_state = FSM_FLIGHT;
    p.g = 0.0;  // Zero gravity for pure oscillation test

    double E0 = compute_energy(q0, p);

    State q = q0;
    for (int step = 0; step < 10000; step++) {
        q = implicit_midpoint_step_cpu(q, p, 1e-4);
    }

    double E_final = compute_energy(q, p);
    double drift = std::abs(E_final - E0) / E0;

    printf("Energy drift over 10k steps: %.2e\n", drift);
    ASSERT(drift < 0.01);  // Less than 1% drift
}

void test_ground_contact_stability() {
    // Drop hopper from height, verify no blowup
    State q0 = {0.0, 0.5, 0.0, 0.0, 1.0,   // Above ground
                0.0, -2.0, 0.0, 0.0, 0.0}; // Falling
    Parameters p;
    p.fsm_state = FSM_FLIGHT;

    State q = q0;
    double min_z = 0.0;

    for (int step = 0; step < 50000; step++) {  // 5 seconds
        q = implicit_midpoint_step_cpu(q, p, 1e-4);

        // Check for blowup
        ASSERT(!std::isnan(q.z_foot));
        ASSERT(!std::isinf(q.z_foot));
        ASSERT(std::abs(q.z_foot) < 100);  // Reasonable bound

        min_z = std::min(min_z, q.z_foot);
    }

    printf("Maximum ground penetration: %.4f m\n", -min_z);
    ASSERT(-min_z < 0.02);  // Less than 2cm penetration
}

void test_stiff_spring_stability() {
    // Test with higher ground stiffness
    Parameters p;
    p.k_g = 1e5;  // 10x stiffer than default

    // Run same ground contact test
    // Verify still stable
}
```

### Layer 4: FSM and Full Trajectory Tests (Qualitative)

```cpp
// test_trajectory.cpp

void test_fsm_transitions_occur() {
    // Run simulation, verify we see all FSM states
    State q0 = default_initial_state();
    Parameters p;
    p.x_dot_des = 2.0;

    std::set<int> states_seen;

    State q = q0;
    for (int step = 0; step < 50000; step++) {
        q = implicit_midpoint_step_cpu(q, p, 1e-4);
        update_fsm(q, p);  // Per-step FSM check
        states_seen.insert(p.fsm_state);
    }

    ASSERT(states_seen.count(FSM_FLIGHT) > 0);
    ASSERT(states_seen.count(FSM_COMPRESSION) > 0);
    ASSERT(states_seen.count(FSM_THRUST) > 0);
}

void test_hopper_moves_forward() {
    // Basic sanity: hopper should travel in +x direction
    State q0 = default_initial_state();
    Parameters p;
    p.x_dot_des = 3.0;

    State q = run_simulation_cpu(q0, p, 5.0);  // 5 seconds

    printf("Final x position: %.2f m\n", q.x_foot);
    ASSERT(q.x_foot > 5.0);  // Should have traveled forward
}

void test_velocity_tracking() {
    // Hopper should roughly track desired velocity
    Parameters p;
    p.x_dot_des = 2.0;

    State q = run_simulation_cpu(default_initial_state(), p, 5.0);
    double final_velocity = compute_body_velocity(q, p);

    printf("Desired: %.2f m/s, Actual: %.2f m/s\n", p.x_dot_des, final_velocity);
    ASSERT(std::abs(final_velocity - p.x_dot_des) < 1.0);  // Within 1 m/s
}
```

### GPU-Specific Tests

```cpp
// test_cuda.cu

void test_gpu_dynamics_matches_cpu() {
    // Run dynamics on GPU, compare to CPU
    // Must match exactly (same computation)

    State q = { /* test state */ };
    Parameters p;

    // CPU
    auto cpu_result = hopper_dynamics_fwd(0.0, q, p);

    // GPU (single thread)
    auto gpu_result = run_dynamics_kernel_single(q, p);

    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(cpu_result.state_dot[i], gpu_result.state_dot[i], 1e-12);
    }
}

void test_gpu_integrator_matches_cpu() {
    // Run integrator on GPU, compare to CPU implicit midpoint
    // Must match exactly

    State q0 = { /* test state */ };
    State q_cpu = implicit_midpoint_step_cpu(q0, p, dt);
    State q_gpu = run_integrator_kernel_single(q0, p, dt);

    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(q_cpu[i], q_gpu[i], 1e-12);
    }
}

void test_parallel_identical_hoppers() {
    // Run N hoppers with identical ICs
    // All should produce identical results
    // Catches race conditions, indexing bugs

    int N = 1024;
    State q0 = { /* initial state */ };

    std::vector<State> results = run_parallel_simulation(N, q0, p, 1.0);

    for (int i = 1; i < N; i++) {
        for (int j = 0; j < 10; j++) {
            ASSERT_EQ(results[0][j], results[i][j]);
        }
    }
}

void test_parallel_different_hoppers() {
    // Run N hoppers with different ICs
    // Verify results are different (no accidental sharing)

    int N = 1024;
    std::vector<State> q0s = generate_random_ics(N);

    std::vector<State> results = run_parallel_simulation(q0s, p, 1.0);

    // Not all the same
    bool all_same = true;
    for (int i = 1; i < N; i++) {
        if (results[0].x_foot != results[i].x_foot) {
            all_same = false;
            break;
        }
    }
    ASSERT(!all_same);
}
```

### Test Data Generation

Export reference values from C++ to avoid recomputing:

```cpp
// generate_test_data.cpp

void export_dynamics_test_cases() {
    FILE* f = fopen("test_data/dynamics_cases.csv", "w");
    fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
               "dx,dz,dphi_leg,dphi_body,dlen,"
               "fsm,qdd0,qdd1,qdd2,qdd3,qdd4,u1,u2\n");

    // Generate many test cases
    for (auto& tc : test_cases) {
        Parameters p;
        p.fsm_state = tc.fsm;
        auto result = hopper_dynamics_fwd(tc.t, tc.q, p);

        fprintf(f, "%.10e,...\n", ...);
    }
    fclose(f);
}
```

### File Structure

```
cpp/
├── hopper.hpp                    # Reference dynamics (unchanged)
├── ode.hpp                       # Reference RK45 (unchanged)
├── implicit_midpoint.hpp         # NEW: CPU implicit midpoint
├── test_dynamics.cpp             # Layer 1 tests
├── test_linalg.cpp               # Layer 2 tests
├── test_integrator.cpp           # Layer 3 tests
├── test_trajectory.cpp           # Layer 4 tests
├── generate_test_data.cpp        # Export reference values
├── test_data/
│   ├── dynamics_cases.csv        # Reference dynamics outputs
│   └── integrator_cases.csv      # Reference integrator outputs
└── cuda/
    ├── hopper_cuda.cuh           # GPU dynamics
    ├── integrator_cuda.cuh       # GPU integrator
    └── test_cuda.cu              # GPU-specific tests
```

---

## Implementation Phases

### Phase 1: Core CUDA Infrastructure
- [ ] Define SoA data structures (`hopper_types.cuh`)
- [ ] Port `compute_accelerations` to device function (dynamics without integration)
- [ ] Port `compute_control` to device function
- [ ] Implement inline 5x5 Gaussian elimination (`solve_5x5`)
- [ ] Basic test kernel: single dynamics evaluation, compare output to CPU

### Phase 2: Implicit Midpoint Integrator
- [ ] Implement `compute_state_derivative` (wraps accelerations into 10D derivative)
- [ ] Implement finite-difference Jacobian (`compute_jacobian_midpoint`)
- [ ] Implement block-structured 10x10 solve using Schur complement
- [ ] Implement `implicit_midpoint_step` with Newton iteration
- [ ] Test: single step, compare to CPU RK45 step (expect small differences)

### Phase 3: Integrator Validation
- [ ] Run single hopper through flight phase only (no ground contact)
- [ ] Verify energy conservation (should oscillate but not drift)
- [ ] Run single hopper through ground contact sequence
- [ ] Verify stability: no blowup, reasonable ground penetration (<1cm)
- [ ] Compare multi-hop trajectory to CPU qualitatively
- [ ] If issues: tune Newton iterations, dt, or fall back to semi-implicit

### Phase 4: Full Simulation Loop
- [ ] Implement FSM transition checking (per-step, no bisection)
- [ ] Full simulation kernel (50k steps for 5s at dt=1e-4)
- [ ] Host-side memory allocation and teardown
- [ ] Test: single hopper, full 5s sim, verify FSM transitions occur correctly
- [ ] Compare final state to CPU (expect small differences due to integrator)

### Phase 5: Massive Parallelism
- [ ] Initialize N hoppers with random ICs (host-side or cuRAND)
- [ ] Parameter arrays for per-hopper control gains
- [ ] Benchmark: 1k, 10k, 100k hoppers
- [ ] Profile with Nsight Compute, identify bottlenecks
- [ ] Optimize: register pressure, memory coalescing, occupancy

### Phase 6: Results Collection & Sweeps
- [ ] Define fitness metrics (velocity tracking error, energy efficiency, stability)
- [ ] Reduction kernel for statistics across hoppers
- [ ] Parameter sweep infrastructure (grid over k_fp, k_att, k_xdot, thrust)
- [ ] Optional: trajectory export for visualization of select hoppers
- [ ] Export best parameters found

---

## Later Enhancements

### Split Flight Phase: AIR_PREPARE → SPEED_MATCH

Current flight control applies foot placement throughout the aerial phase. A potential improvement:

**AIR_PREPARE**: Immediately after liftoff
- Retract leg slightly
- Begin rotating leg toward touchdown angle
- Goal: achieve target leg angular velocity before ground approach

**SPEED_MATCH**: Approaching ground
- Leg rotating at velocity matched to expected ground speed
- Minimizes energy loss at touchdown (leg doesn't suddenly accelerate/decelerate)
- May improve efficiency at high speeds

FSM would become:
```
COMPRESSION → THRUST → AIR_PREPARE → SPEED_MATCH → COMPRESSION → ...
```

**Implementation considerations:**
- Need height/time-to-touchdown estimation for AIR_PREPARE → SPEED_MATCH transition
- Leg angular velocity target depends on expected ground contact geometry
- May violate Raibert's symmetric stance assumption but could enable better high-speed tracking

**Deferred**: Implement basic 3-state FSM first, validate against original, then experiment with 4-state variant.

### Other Future Work

- **Terrain**: Non-flat ground (height map)
- **Disturbances**: Random pushes, wind
- **Multiple legs**: Extend to biped/quadruped
- **RL integration**: Gym-style interface
- **Batched parameter optimization**: CMA-ES or similar on GPU

---

## Questions / Decisions Needed

1. **Double vs Float**: Using double throughout for now (matches CPU). Could switch to float for 2x memory bandwidth and potentially 2x compute on consumer GPUs. Profile first.

2. **Trajectory storage**: Store every N-th step? Stream to host? Only store final state + metrics? Depends on use case.

3. **Random IC generation**: Use cuRAND on device, or generate on host and copy?

4. **Build system**: Plain nvcc script for now. CMake if it gets complex.

---

## Approval Checklist

- [ ] Semi-implicit Euler acceptable as integrator?
- [ ] SoA memory layout makes sense?
- [ ] FSM event checking (per-step, with slop) acceptable?
- [ ] Phase ordering looks right?
- [ ] Deferred items captured correctly?

