// hopper_cuda.cuh - CUDA port of hopper dynamics and control
//
// Direct port from hopper.hpp, adapted for GPU execution.
// All functions are __host__ __device__ so they can run on both CPU and GPU.

#ifndef HOPPER_CUDA_CUH
#define HOPPER_CUDA_CUH

#include <cmath>
#include "hopper_types.cuh"

// ============================================================================
// 5x5 GAUSSIAN ELIMINATION (inline, no external dependencies)
// ============================================================================

__host__ __device__ inline void solve_5x5(
    const Scalar M_in[5][5],
    const Scalar b_in[5],
    Scalar x[5]
) {
    // Copy inputs (we modify during elimination)
    Scalar M[5][5];
    Scalar b[5];

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        b[i] = b_in[i];
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            M[i][j] = M_in[i][j];
        }
    }

    // Forward elimination with partial pivoting
    #pragma unroll
    for (int col = 0; col < 5; col++) {
        // Find pivot (largest absolute value in column)
        int pivot_row = col;
        Scalar pivot_val = fabs(M[col][col]);

        #pragma unroll
        for (int row = col + 1; row < 5; row++) {
            Scalar val = fabs(M[row][col]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = row;
            }
        }

        // Swap rows if needed
        if (pivot_row != col) {
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                Scalar tmp = M[col][j];
                M[col][j] = M[pivot_row][j];
                M[pivot_row][j] = tmp;
            }
            Scalar tmp = b[col];
            b[col] = b[pivot_row];
            b[pivot_row] = tmp;
        }

        // Eliminate column below pivot
        #pragma unroll
        for (int row = col + 1; row < 5; row++) {
            Scalar factor = M[row][col] / M[col][col];
            #pragma unroll
            for (int j = col; j < 5; j++) {
                M[row][j] -= factor * M[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    #pragma unroll
    for (int i = 4; i >= 0; i--) {
        x[i] = b[i];
        #pragma unroll
        for (int j = i + 1; j < 5; j++) {
            x[i] -= M[i][j] * x[j];
        }
        x[i] /= M[i][i];
    }
}

// ============================================================================
// CONTROL (from hopper.hpp hopper_control)
// ============================================================================

struct ControlOutput {
    Scalar u1;      // Leg spring actuator force [N]
    Scalar u2;      // Hip torque [N*m]
    Scalar a_des;   // Desired leg angle (for logging) [rad]
};

__host__ __device__ inline ControlOutput compute_control(
    Scalar t,
    Scalar x_foot, Scalar z_foot, Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    Scalar ddt_x_foot, Scalar ddt_z_foot, Scalar ddt_phi_leg, Scalar ddt_phi_body, Scalar ddt_len_leg,
    int fsm_state,
    const ControlParams& ctrl,
    const PhysicsParams& phys,
    Scalar t_thrust_on,
    Scalar T_s
) {
    ControlOutput out;
    out.u1 = 0.0;
    out.u2 = 0.0;
    out.a_des = 0.0;

    // Control gains from params
    Scalar k_fp = ctrl.k_fp;
    Scalar b_fp = ctrl.b_fp;
    Scalar k_att = ctrl.k_att;
    Scalar b_att = ctrl.b_att;
    Scalar k_xdot = ctrl.k_xdot;
    Scalar thrust = ctrl.thrust_scale * phys.k_l;
    Scalar x_dot_des = ctrl.x_dot_des;

    constexpr Scalar thr_z_low = 0.01;

    Scalar T_s_use = (T_s == 0.0) ? 0.425 : T_s;

    // Extract state variables for readability
    Scalar a = phi_leg;
    Scalar dadt = ddt_phi_leg;
    Scalar b = phi_body;
    Scalar dbdt = ddt_phi_body;
    Scalar l = len_leg;
    Scalar dldt = ddt_len_leg;
    Scalar stance_ang_des = a / 2.0;

    if (fsm_state == FSM_THRUST) {
        out.u1 = thrust;
        out.u2 = -k_att * (b - stance_ang_des) - b_att * dbdt;
    }
    else if (fsm_state == FSM_COMPRESSION) {
        out.u1 = 0.0;
        out.u2 = -k_att * (b - stance_ang_des) - b_att * dbdt;
    }
    else if (fsm_state == FSM_FLIGHT) {
        // Compute body velocity
        Scalar d_xbody_dt = ddt_x_foot + dldt * sin(a) + l * cos(a) * dadt
                         + phys.l_2 * cos(b) * dbdt;

        // Desired leg angle for foot placement
        Scalar arg = (d_xbody_dt * T_s_use / 2.0 + k_xdot * (d_xbody_dt - x_dot_des)) / l;

        // Clamp argument to valid range for asin
        if (arg > 1.0) arg = 1.0;
        if (arg < -1.0) arg = -1.0;
        out.a_des = -asin(arg);

        // Check for NaN
        if (isnan(out.a_des)) {
            out.a_des = 0.0;
        }

        // Apply foot placement control if above ground
        if (z_foot > thr_z_low) {
            out.u2 = k_fp * (a - out.a_des) + b_fp * dadt;
        }
        out.u1 = 0.0;
    }

    // Thrust duration limiting
    if (fsm_state == FSM_THRUST) {
        if (t - t_thrust_on > ctrl.T_MAX_THRUST_DUR) {
            out.u1 = 0.0;
        }
    }

    return out;
}

// Overload using HopperState struct
__host__ __device__ inline ControlOutput compute_control(
    Scalar t,
    const HopperState& state,
    const ControlParams& ctrl,
    const PhysicsParams& phys
) {
    return compute_control(
        t,
        state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
        state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
        state.fsm_state,
        ctrl, phys,
        state.t_thrust_on,
        state.T_s
    );
}

// ============================================================================
// DYNAMICS (from hopper.hpp hopper_dynamics_fwd)
// ============================================================================

__host__ __device__ inline void compute_accelerations(
    Scalar x_foot, Scalar z_foot, Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    Scalar ddt_x_foot, Scalar ddt_z_foot, Scalar ddt_phi_leg, Scalar ddt_phi_body, Scalar ddt_len_leg,
    Scalar u1, Scalar u2,
    const PhysicsParams& phys,
    Scalar qdd[5]
) {
    // Geometric quantities
    Scalar R = len_leg - phys.l_1;
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);

    // Spring deflection (positive when compressed)
    Scalar r_sd = phys.r_s0 - len_leg;

    // Leg spring force
    Scalar F_k;
    if (r_sd > 0) {
        // Spring compressed: linear spring + actuator
        F_k = phys.k_l * r_sd + u1;
    } else {
        // Spring extended past rest: hard stop with damping
        F_k = phys.k_stop * r_sd + u1 - phys.b_stop * ddt_len_leg;
    }

    // Ground reaction forces
    Scalar F_x, F_z;
    if (z_foot < 0) {
        // Foot below ground: spring-damper contact
        F_x = -phys.b_g * ddt_x_foot;
        F_z = phys.k_g * (-z_foot);
        Scalar damping = -phys.b_g * ddt_z_foot;
        if (damping > 0) {
            F_z = F_z + damping;
        }
    } else {
        F_x = 0.0;
        F_z = 0.0;
    }

    // Torque about foot due to ground reaction and hip
    Scalar a_torque = phys.l_1 * F_z * s1 - phys.l_1 * F_x * c1 - u2;

    // Mass matrix M (5x5)
    Scalar M[5][5];
    M[0][0] = -phys.m_l * R;
    M[0][1] = 0;
    M[0][2] = (phys.J_l - phys.m_l * R * phys.l_1) * c1;
    M[0][3] = 0;
    M[0][4] = 0;

    M[1][0] = 0;
    M[1][1] = phys.m_l * R;
    M[1][2] = (phys.J_l - phys.m_l * R * phys.l_1) * s1;
    M[1][3] = 0;
    M[1][4] = 0;

    M[2][0] = phys.m * R;
    M[2][1] = 0;
    M[2][2] = (phys.J_l + phys.m * R * len_leg) * c1;
    M[2][3] = phys.m * R * phys.l_2 * c2;
    M[2][4] = phys.m * R * s1;

    M[3][0] = 0;
    M[3][1] = -phys.m * R;
    M[3][2] = (phys.J_l + phys.m * R * len_leg) * s1;
    M[3][3] = phys.m * R * phys.l_2 * s2;
    M[3][4] = -phys.m * R * c1;

    M[4][0] = 0;
    M[4][1] = 0;
    M[4][2] = phys.J_l * phys.l_2 * cos(phi_leg - phi_body);
    M[4][3] = -phys.J * R;
    M[4][4] = 0;

    // RHS vector eta (5x1)
    Scalar eta[5];
    Scalar phi_leg_dot_sq = ddt_phi_leg * ddt_phi_leg;
    Scalar phi_body_dot_sq = ddt_phi_body * ddt_phi_body;

    eta[0] = a_torque * c1 - R * (F_x - F_k * s1 - phys.m_l * phys.l_1 * phi_leg_dot_sq * s1);

    eta[1] = a_torque * s1 + R * (phys.m_l * phys.l_1 * phi_leg_dot_sq * c1 + F_z - F_k * c1 - phys.m_l * phys.g);

    eta[2] = a_torque * c1 + R * F_k * s1
           + phys.m * R * (len_leg * phi_leg_dot_sq * s1 + phys.l_2 * phi_body_dot_sq * s2
                          - 2.0 * ddt_len_leg * ddt_phi_leg * c1);

    eta[3] = a_torque * s1 - R * (F_k * c1 - phys.m * phys.g)
           - phys.m * R * (2.0 * ddt_len_leg * ddt_phi_leg * s1
                          + len_leg * phi_leg_dot_sq * c1 + phys.l_2 * phi_body_dot_sq * c2);

    eta[4] = a_torque * phys.l_2 * cos(phi_leg - phi_body)
           - R * (phys.l_2 * F_k * sin(phi_body - phi_leg) + u2);

    // Solve for accelerations: M * qdd = eta
    solve_5x5(M, eta, qdd);
}

// Overload using HopperState struct
__host__ __device__ inline void compute_accelerations(
    const HopperState& state,
    Scalar u1, Scalar u2,
    const PhysicsParams& phys,
    Scalar qdd[5]
) {
    compute_accelerations(
        state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
        state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
        u1, u2, phys, qdd
    );
}

// ============================================================================
// STATE DERIVATIVE (for integrators)
// Computes f(y) = [qdot; qddot] given y = [q; qdot]
// ============================================================================

__host__ __device__ inline void compute_state_derivative(
    const Scalar y[10],  // [x_foot, z_foot, phi_leg, phi_body, len_leg, dx, dz, dphi_leg, dphi_body, dlen]
    Scalar u1, Scalar u2,
    const PhysicsParams& phys,
    Scalar f[10]
) {
    // First 5 elements of derivative are just velocities
    f[0] = y[5];   // ddt_x_foot
    f[1] = y[6];   // ddt_z_foot
    f[2] = y[7];   // ddt_phi_leg
    f[3] = y[8];   // ddt_phi_body
    f[4] = y[9];   // ddt_len_leg

    // Last 5 elements are accelerations
    Scalar qdd[5];
    compute_accelerations(
        y[0], y[1], y[2], y[3], y[4],  // positions
        y[5], y[6], y[7], y[8], y[9],  // velocities
        u1, u2, phys, qdd
    );

    f[5] = qdd[0];
    f[6] = qdd[1];
    f[7] = qdd[2];
    f[8] = qdd[3];
    f[9] = qdd[4];
}

// ============================================================================
// FSM EVENT DETECTION (per-step check, no bisection)
// ============================================================================

__host__ __device__ inline int check_fsm_transition(
    Scalar z_foot, Scalar len_leg, Scalar ddt_len_leg,
    int fsm_state,
    Scalar r_s0
) {
    constexpr Scalar thresh_leg_extended = 0.0001;

    if (fsm_state == FSM_COMPRESSION) {
        // Transition when leg stops compressing (ddt_len_leg crosses zero upward)
        if (ddt_len_leg > 0.0) {
            return FSM_THRUST;
        }
    }
    else if (fsm_state == FSM_THRUST) {
        // Transition when leg fully extended
        Scalar r_sd = r_s0 - len_leg;
        if (r_sd < -thresh_leg_extended) {
            return FSM_FLIGHT;
        }
    }
    else if (fsm_state == FSM_FLIGHT) {
        // Touchdown: when foot hits ground
        if (z_foot < 0.0) {
            return FSM_COMPRESSION;
        }
    }

    return fsm_state;  // No transition
}

// ============================================================================
// FULL DYNAMICS OUTPUT (for testing/comparison with CPU)
// ============================================================================

struct DynamicsOutput {
    Scalar state_dot[10];
    ControlOutput control;
    Scalar r_sd;
    int fsm_state;
};

__host__ __device__ inline DynamicsOutput hopper_dynamics_fwd_cuda(
    Scalar t,
    const HopperState& state,
    const ControlParams& ctrl,
    const PhysicsParams& phys
) {
    DynamicsOutput out;
    out.fsm_state = state.fsm_state;

    // Get control
    out.control = compute_control(t, state, ctrl, phys);

    // Get accelerations
    Scalar qdd[5];
    compute_accelerations(state, out.control.u1, out.control.u2, phys, qdd);

    // Assemble state derivative
    out.state_dot[0] = state.ddt_x_foot;
    out.state_dot[1] = state.ddt_z_foot;
    out.state_dot[2] = state.ddt_phi_leg;
    out.state_dot[3] = state.ddt_phi_body;
    out.state_dot[4] = state.ddt_len_leg;
    out.state_dot[5] = qdd[0];
    out.state_dot[6] = qdd[1];
    out.state_dot[7] = qdd[2];
    out.state_dot[8] = qdd[3];
    out.state_dot[9] = qdd[4];

    // Spring deflection for logging
    out.r_sd = phys.r_s0 - state.len_leg;

    return out;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

__host__ __device__ inline const char* fsm_state_name(int state) {
    switch (state) {
        case FSM_COMPRESSION: return "COMPRESSION";
        case FSM_THRUST:      return "THRUST";
        case FSM_FLIGHT:      return "FLIGHT";
        default: return "UNKNOWN";
    }
}

#endif // HOPPER_CUDA_CUH
