// integrator_cuda.cuh - Integrators for CUDA hopper simulation
//
// Implements implicit midpoint (primary), semi-implicit Euler (fallback)
// Supports both finite-difference and analytical Jacobian for implicit midpoint

#ifndef INTEGRATOR_CUDA_CUH
#define INTEGRATOR_CUDA_CUH

#include "hopper_cuda.cuh"
#include "jacobian_analytical.cuh"

// ============================================================================
// 10x10 LINEAR SOLVE USING BLOCK STRUCTURE
// ============================================================================
//
// The Jacobian has structure:
//   J = [ I           -dt/2 * I    ]
//       [ -dt/2 * A   I - dt/2 * B ]
//
// We use Schur complement to reduce to 5x5 solve.

__host__ __device__ inline void solve_10x10_block(
    const double A[5][5],      // dqddot/dq (5x5)
    const double B[5][5],      // dqddot/dqdot (5x5)
    const double G_q[5],       // residual for positions
    const double G_qdot[5],    // residual for velocities
    double dt,
    double dq[5],              // output: position correction
    double dqdot[5]            // output: velocity correction
) {
    double half_dt = 0.5 * dt;
    double half_dt_sq = half_dt * half_dt;

    // Build Schur complement S = I - dt/2 * B - (dt/2)^2 * A
    double S[5][5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            S[i][j] = -half_dt * B[i][j] - half_dt_sq * A[i][j];
            if (i == j) S[i][j] += 1.0;
        }
    }

    // Build RHS: G_qdot + dt/2 * A * G_q
    double rhs[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        rhs[i] = G_qdot[i];
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            rhs[i] += half_dt * A[i][j] * G_q[j];
        }
    }

    // Solve S * dqdot = rhs
    solve_5x5(S, rhs, dqdot);

    // Back-substitute: dq = G_q + dt/2 * dqdot
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        dq[i] = G_q[i] + half_dt * dqdot[i];
    }
}

// ============================================================================
// FINITE DIFFERENCE JACOBIAN
// ============================================================================

__host__ __device__ inline void compute_jacobian_blocks(
    const double y[10],
    double u1, double u2,
    const PhysicsParams& phys,
    double A[5][5],    // dqddot/dq
    double B[5][5]     // dqddot/dqdot
) {
    constexpr double eps = 1e-7;

    // Compute qddot at current state
    double qdd0[5];
    compute_accelerations(
        y[0], y[1], y[2], y[3], y[4],
        y[5], y[6], y[7], y[8], y[9],
        u1, u2, phys, qdd0
    );

    // Compute A = dqddot/dq (perturb positions, indices 0-4)
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        double y_pert[10];
        #pragma unroll
        for (int i = 0; i < 10; i++) y_pert[i] = y[i];
        y_pert[j] += eps;

        double qdd_pert[5];
        compute_accelerations(
            y_pert[0], y_pert[1], y_pert[2], y_pert[3], y_pert[4],
            y_pert[5], y_pert[6], y_pert[7], y_pert[8], y_pert[9],
            u1, u2, phys, qdd_pert
        );

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            A[i][j] = (qdd_pert[i] - qdd0[i]) / eps;
        }
    }

    // Compute B = dqddot/dqdot (perturb velocities, indices 5-9)
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        double y_pert[10];
        #pragma unroll
        for (int i = 0; i < 10; i++) y_pert[i] = y[i];
        y_pert[5 + j] += eps;

        double qdd_pert[5];
        compute_accelerations(
            y_pert[0], y_pert[1], y_pert[2], y_pert[3], y_pert[4],
            y_pert[5], y_pert[6], y_pert[7], y_pert[8], y_pert[9],
            u1, u2, phys, qdd_pert
        );

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            B[i][j] = (qdd_pert[i] - qdd0[i]) / eps;
        }
    }
}

// ============================================================================
// IMPLICIT MIDPOINT INTEGRATOR
// ============================================================================

__host__ __device__ inline void implicit_midpoint_step(
    double& x_foot, double& z_foot, double& phi_leg, double& phi_body, double& len_leg,
    double& ddt_x_foot, double& ddt_z_foot, double& ddt_phi_leg, double& ddt_phi_body, double& ddt_len_leg,
    double u1, double u2,
    const PhysicsParams& phys,
    double dt
) {
    // Pack current state
    double y[10] = {x_foot, z_foot, phi_leg, phi_body, len_leg,
                    ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg};
    double y_new[10];

    // Initial guess: explicit Euler step
    double f_curr[10];
    compute_state_derivative(y, u1, u2, phys, f_curr);

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        y_new[i] = y[i] + dt * f_curr[i];
    }

    // Newton iterations (fixed count for GPU uniformity)
    constexpr int NEWTON_ITERS = 4;

    for (int iter = 0; iter < NEWTON_ITERS; iter++) {
        // Compute midpoint
        double y_mid[10];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            y_mid[i] = 0.5 * (y[i] + y_new[i]);
        }

        // Evaluate dynamics at midpoint
        double f_mid[10];
        compute_state_derivative(y_mid, u1, u2, phys, f_mid);

        // Compute residual: G = y_new - y - dt * f(y_mid)
        double G[10];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            G[i] = y_new[i] - y[i] - dt * f_mid[i];
        }

        // Compute Jacobian blocks A and B at midpoint
        double A[5][5], B[5][5];
        compute_jacobian_blocks(y_mid, u1, u2, phys, A, B);

        // Solve for correction using block structure
        double G_q[5] = {G[0], G[1], G[2], G[3], G[4]};
        double G_qdot[5] = {G[5], G[6], G[7], G[8], G[9]};
        double dq[5], dqdot[5];

        solve_10x10_block(A, B, G_q, G_qdot, dt, dq, dqdot);

        // Update: y_new = y_new - delta
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            y_new[i] -= dq[i];
            y_new[5 + i] -= dqdot[i];
        }
    }

    // Unpack result
    x_foot = y_new[0];
    z_foot = y_new[1];
    phi_leg = y_new[2];
    phi_body = y_new[3];
    len_leg = y_new[4];
    ddt_x_foot = y_new[5];
    ddt_z_foot = y_new[6];
    ddt_phi_leg = y_new[7];
    ddt_phi_body = y_new[8];
    ddt_len_leg = y_new[9];
}

// ============================================================================
// SEMI-IMPLICIT EULER INTEGRATOR (fallback)
// ============================================================================

__host__ __device__ inline void semi_implicit_euler_step(
    double& x_foot, double& z_foot, double& phi_leg, double& phi_body, double& len_leg,
    double& ddt_x_foot, double& ddt_z_foot, double& ddt_phi_leg, double& ddt_phi_body, double& ddt_len_leg,
    double u1, double u2,
    const PhysicsParams& phys,
    double dt
) {
    // 1. Compute accelerations at current state
    double qdd[5];
    compute_accelerations(
        x_foot, z_foot, phi_leg, phi_body, len_leg,
        ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
        u1, u2, phys, qdd
    );

    // 2. Update velocities (explicit in acceleration)
    ddt_x_foot   += dt * qdd[0];
    ddt_z_foot   += dt * qdd[1];
    ddt_phi_leg  += dt * qdd[2];
    ddt_phi_body += dt * qdd[3];
    ddt_len_leg  += dt * qdd[4];

    // 3. Update positions (using NEW velocities)
    x_foot   += dt * ddt_x_foot;
    z_foot   += dt * ddt_z_foot;
    phi_leg  += dt * ddt_phi_leg;
    phi_body += dt * ddt_phi_body;
    len_leg  += dt * ddt_len_leg;
}

// ============================================================================
// INTEGRATOR SELECTION
// ============================================================================

template<int IntegratorType>
__host__ __device__ inline void integrator_step(
    double& x_foot, double& z_foot, double& phi_leg, double& phi_body, double& len_leg,
    double& ddt_x_foot, double& ddt_z_foot, double& ddt_phi_leg, double& ddt_phi_body, double& ddt_len_leg,
    double u1, double u2,
    const PhysicsParams& phys,
    double dt
);

template<>
__host__ __device__ inline void integrator_step<INTEGRATOR_IMPLICIT_MIDPOINT>(
    double& x_foot, double& z_foot, double& phi_leg, double& phi_body, double& len_leg,
    double& ddt_x_foot, double& ddt_z_foot, double& ddt_phi_leg, double& ddt_phi_body, double& ddt_len_leg,
    double u1, double u2,
    const PhysicsParams& phys,
    double dt
) {
    implicit_midpoint_step(
        x_foot, z_foot, phi_leg, phi_body, len_leg,
        ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
        u1, u2, phys, dt
    );
}

template<>
__host__ __device__ inline void integrator_step<INTEGRATOR_SEMI_IMPLICIT_EULER>(
    double& x_foot, double& z_foot, double& phi_leg, double& phi_body, double& len_leg,
    double& ddt_x_foot, double& ddt_z_foot, double& ddt_phi_leg, double& ddt_phi_body, double& ddt_len_leg,
    double u1, double u2,
    const PhysicsParams& phys,
    double dt
) {
    semi_implicit_euler_step(
        x_foot, z_foot, phi_leg, phi_body, len_leg,
        ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
        u1, u2, phys, dt
    );
}

// ============================================================================
// SINGLE HOPPER SIMULATION STEP (combines control + integration + FSM)
// ============================================================================

template<int IntegratorType>
__host__ __device__ inline void hopper_step(
    HopperState& state,
    double t,
    const ControlParams& ctrl,
    const PhysicsParams& phys,
    double dt
) {
    // 1. Compute control
    ControlOutput control = compute_control(t, state, ctrl, phys);

    // 2. Integrate
    integrator_step<IntegratorType>(
        state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
        state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
        control.u1, control.u2, phys, dt
    );

    // 3. Check FSM transition
    int new_fsm = check_fsm_transition(
        state.z_foot, state.len_leg, state.ddt_len_leg,
        state.fsm_state, phys.r_s0
    );

    // 4. Update FSM timing if transition occurred
    if (new_fsm != state.fsm_state) {
        if (state.fsm_state == FSM_COMPRESSION && new_fsm == FSM_THRUST) {
            state.T_compression = t - state.t_thrust_on;  // approximate
            state.t_thrust_on = t;
        }
        else if (state.fsm_state == FSM_THRUST && new_fsm == FSM_FLIGHT) {
            // Update stance time estimate
            // T_s = time_in_stance = T_compression + T_thrust
        }
        state.fsm_state = new_fsm;
    }
}

#endif // INTEGRATOR_CUDA_CUH
