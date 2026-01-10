// jacobian_analytical.cuh - Analytical Jacobian for hopper dynamics
//
// Computes A = d(qddot)/dq and B = d(qddot)/dqdot analytically,
// avoiding the 10 extra dynamics evaluations per Newton iteration
// required by finite-difference Jacobian.
//
// Derivation:
//   qddot = M(q)^{-1} * eta(q, qdot, u)
//
//   A = d(qddot)/dq = M^{-1} * d(eta)/dq - M^{-1} * (dM/dq * qddot)
//   B = d(qddot)/dqdot = M^{-1} * d(eta)/dqdot
//
// Note: We compute qddot first, then use it in A calculation.

#ifndef JACOBIAN_ANALYTICAL_CUH
#define JACOBIAN_ANALYTICAL_CUH

#include "hopper_types.cuh"
#include "hopper_cuda.cuh"

// ============================================================================
// PARTIAL DERIVATIVES OF MASS MATRIX M
// ============================================================================
// M depends on: phi_leg (q[2]), phi_body (q[3]), len_leg (q[4])
// We need: dM/d(phi_leg), dM/d(phi_body), dM/d(len_leg)
// These are 5x5 matrices.

// dM/d(phi_leg) - derivative of mass matrix w.r.t. phi_leg
__host__ __device__ inline void compute_dM_dphi_leg(
    Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    const PhysicsParams& p,
    Scalar dM[5][5]
) {
    Scalar R = len_leg - p.l_1;
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);

    // dc1/d(phi_leg) = -s1, ds1/d(phi_leg) = c1

    // Row 0
    dM[0][0] = 0;
    dM[0][1] = 0;
    dM[0][2] = (p.J_l - p.m_l * R * p.l_1) * (-s1);  // d(..* c1)/d(phi_leg)
    dM[0][3] = 0;
    dM[0][4] = 0;

    // Row 1
    dM[1][0] = 0;
    dM[1][1] = 0;
    dM[1][2] = (p.J_l - p.m_l * R * p.l_1) * c1;  // d(..* s1)/d(phi_leg)
    dM[1][3] = 0;
    dM[1][4] = 0;

    // Row 2
    dM[2][0] = 0;
    dM[2][1] = 0;
    dM[2][2] = (p.J_l + p.m * R * len_leg) * (-s1);  // d(..* c1)/d(phi_leg)
    dM[2][3] = 0;
    dM[2][4] = p.m * R * c1;  // d(m*R*s1)/d(phi_leg)

    // Row 3
    dM[3][0] = 0;
    dM[3][1] = 0;
    dM[3][2] = (p.J_l + p.m * R * len_leg) * c1;  // d(..* s1)/d(phi_leg)
    dM[3][3] = 0;
    dM[3][4] = p.m * R * s1;  // d(-m*R*c1)/d(phi_leg)

    // Row 4
    dM[4][0] = 0;
    dM[4][1] = 0;
    dM[4][2] = p.J_l * p.l_2 * (-sin(phi_leg - phi_body));  // d(cos(phi_leg - phi_body))/d(phi_leg)
    dM[4][3] = 0;
    dM[4][4] = 0;
}

// dM/d(phi_body) - derivative of mass matrix w.r.t. phi_body
__host__ __device__ inline void compute_dM_dphi_body(
    Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    const PhysicsParams& p,
    Scalar dM[5][5]
) {
    Scalar R = len_leg - p.l_1;
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);

    // dc2/d(phi_body) = -s2, ds2/d(phi_body) = c2

    // Most elements are zero
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            dM[i][j] = 0.0;
        }
    }

    // Only non-zero elements:
    dM[2][3] = p.m * R * p.l_2 * (-s2);  // d(m*R*l_2*c2)/d(phi_body)
    dM[3][3] = p.m * R * p.l_2 * c2;      // d(m*R*l_2*s2)/d(phi_body)
    dM[4][2] = p.J_l * p.l_2 * sin(phi_leg - phi_body);  // d(cos(phi_leg - phi_body))/d(phi_body)
}

// dM/d(len_leg) - derivative of mass matrix w.r.t. len_leg
__host__ __device__ inline void compute_dM_dlen_leg(
    Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    const PhysicsParams& p,
    Scalar dM[5][5]
) {
    // R = len_leg - l_1, so dR/d(len_leg) = 1
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);

    // Row 0: M[0][0] = -m_l * R
    dM[0][0] = -p.m_l;  // d(-m_l * R)/d(len_leg) = -m_l
    dM[0][1] = 0;
    dM[0][2] = -p.m_l * p.l_1 * c1;  // d((J_l - m_l*R*l_1)*c1)/d(len_leg) = -m_l*l_1*c1
    dM[0][3] = 0;
    dM[0][4] = 0;

    // Row 1: M[1][1] = m_l * R
    dM[1][0] = 0;
    dM[1][1] = p.m_l;  // d(m_l * R)/d(len_leg) = m_l
    dM[1][2] = -p.m_l * p.l_1 * s1;  // d((J_l - m_l*R*l_1)*s1)/d(len_leg)
    dM[1][3] = 0;
    dM[1][4] = 0;

    // Row 2: M[2][0] = m * R, M[2][2] = (J_l + m*R*len_leg)*c1
    dM[2][0] = p.m;  // d(m * R)/d(len_leg)
    dM[2][1] = 0;
    // d((J_l + m*R*len_leg)*c1)/d(len_leg) = m*(R + len_leg)*c1 = m*(2*len_leg - l_1)*c1
    // Wait, let's be more careful:
    // d(J_l + m*R*len_leg)/d(len_leg) = m*(dR/d(len_leg)*len_leg + R) = m*(len_leg + R) = m*(2*len_leg - l_1)
    dM[2][2] = p.m * (2*len_leg - p.l_1) * c1;
    dM[2][3] = p.m * p.l_2 * c2;  // d(m*R*l_2*c2)/d(len_leg) = m*l_2*c2
    dM[2][4] = p.m * s1;  // d(m*R*s1)/d(len_leg) = m*s1

    // Row 3
    dM[3][0] = 0;
    dM[3][1] = -p.m;  // d(-m*R)/d(len_leg)
    dM[3][2] = p.m * (2*len_leg - p.l_1) * s1;  // same pattern as [2][2]
    dM[3][3] = p.m * p.l_2 * s2;  // d(m*R*l_2*s2)/d(len_leg)
    dM[3][4] = -p.m * c1;  // d(-m*R*c1)/d(len_leg)

    // Row 4
    dM[4][0] = 0;
    dM[4][1] = 0;
    dM[4][2] = 0;  // J_l * l_2 * cos(...) doesn't depend on len_leg
    dM[4][3] = -p.J;  // d(-J*R)/d(len_leg) = -J
    dM[4][4] = 0;
}

// ============================================================================
// PARTIAL DERIVATIVES OF FORCE TERMS
// ============================================================================

// Ground force derivatives
__host__ __device__ inline void compute_ground_force_derivatives(
    Scalar z_foot, Scalar ddt_x_foot, Scalar ddt_z_foot,
    const PhysicsParams& p,
    Scalar& dFx_dz, Scalar& dFx_ddx,
    Scalar& dFz_dz, Scalar& dFz_ddz
) {
    if (z_foot < 0) {
        // F_x = -b_g * ddt_x_foot
        dFx_dz = 0;  // F_x doesn't depend on z directly (only on velocity)
        dFx_ddx = -p.b_g;

        // F_z = k_g * (-z_foot) + max(-b_g * ddt_z_foot, 0)
        dFz_dz = -p.k_g;  // d(k_g * (-z_foot))/d(z_foot)

        // The damping term: max(-b_g * ddt_z_foot, 0)
        // d(max(x, 0))/dx = 1 if x > 0, 0 otherwise
        if (-p.b_g * ddt_z_foot > 0) {
            dFz_ddz = -p.b_g;
        } else {
            dFz_ddz = 0;
        }
    } else {
        dFx_dz = 0;
        dFx_ddx = 0;
        dFz_dz = 0;
        dFz_ddz = 0;
    }
}

// Spring force derivatives
__host__ __device__ inline void compute_spring_force_derivatives(
    Scalar len_leg, Scalar ddt_len_leg,
    const PhysicsParams& p,
    Scalar& dFk_dlen, Scalar& dFk_ddlen
) {
    Scalar r_sd = p.r_s0 - len_leg;

    if (r_sd > 0) {
        // F_k = k_l * r_sd + u1
        // d(k_l * (r_s0 - len_leg))/d(len_leg) = -k_l
        dFk_dlen = -p.k_l;
        dFk_ddlen = 0;
    } else {
        // F_k = k_stop * r_sd + u1 - b_stop * ddt_len_leg
        dFk_dlen = -p.k_stop;
        dFk_ddlen = -p.b_stop;
    }
}

// ============================================================================
// COMPUTE d(eta)/dq - Partial derivatives of RHS vector w.r.t. positions
// ============================================================================
// eta depends on: x_foot (q[0]), z_foot (q[1]), phi_leg (q[2]), phi_body (q[3]), len_leg (q[4])
// Returns: deta[i][j] = d(eta[i])/d(q[j])

__host__ __device__ inline void compute_deta_dq(
    Scalar x_foot, Scalar z_foot, Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    Scalar ddt_x_foot, Scalar ddt_z_foot, Scalar ddt_phi_leg, Scalar ddt_phi_body, Scalar ddt_len_leg,
    Scalar u1, Scalar u2,
    const PhysicsParams& p,
    Scalar deta[5][5]
) {
    Scalar R = len_leg - p.l_1;
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);
    Scalar r_sd = p.r_s0 - len_leg;

    Scalar phi_leg_dot_sq = ddt_phi_leg * ddt_phi_leg;
    Scalar phi_body_dot_sq = ddt_phi_body * ddt_phi_body;

    // Current forces
    Scalar F_k, F_x, F_z;
    if (r_sd > 0) {
        F_k = p.k_l * r_sd + u1;
    } else {
        F_k = p.k_stop * r_sd + u1 - p.b_stop * ddt_len_leg;
    }
    if (z_foot < 0) {
        F_x = -p.b_g * ddt_x_foot;
        F_z = p.k_g * (-z_foot);
        Scalar damping = -p.b_g * ddt_z_foot;
        if (damping > 0) F_z += damping;
    } else {
        F_x = 0;
        F_z = 0;
    }

    Scalar a_torque = p.l_1 * F_z * s1 - p.l_1 * F_x * c1 - u2;

    // Force derivatives
    Scalar dFx_dz, dFx_ddx, dFz_dz, dFz_ddz;
    compute_ground_force_derivatives(z_foot, ddt_x_foot, ddt_z_foot, p, dFx_dz, dFx_ddx, dFz_dz, dFz_ddz);

    Scalar dFk_dlen, dFk_ddlen;
    compute_spring_force_derivatives(len_leg, ddt_len_leg, p, dFk_dlen, dFk_ddlen);

    // Initialize to zero
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            deta[i][j] = 0.0;
        }
    }

    // d(eta)/d(x_foot) - column 0
    // x_foot only affects forces indirectly, but F_x, F_z don't depend on x_foot
    // So all derivatives are 0

    // d(eta)/d(z_foot) - column 1
    // z_foot affects F_z and thus a_torque
    Scalar da_torque_dz = p.l_1 * dFz_dz * s1;  // - p.l_1 * dFx_dz * c1 = 0 since dFx_dz = 0

    deta[0][1] = da_torque_dz * c1 - R * (-dFz_dz * 0);  // F_x doesn't depend on z
    // Actually let me reconsider eta[0]:
    // eta[0] = a_torque * c1 - R * (F_x - F_k * s1 - m_l * l_1 * phi_leg_dot^2 * s1)
    // d(eta[0])/d(z_foot) = d(a_torque)/d(z) * c1 - R * (dF_x/dz - dF_k/dz * s1)
    //                     = da_torque_dz * c1 - R * (0 - 0) = da_torque_dz * c1
    deta[0][1] = da_torque_dz * c1;

    // eta[1] = a_torque * s1 + R * (m_l * l_1 * phi_leg_dot^2 * c1 + F_z - F_k * c1 - m_l * g)
    deta[1][1] = da_torque_dz * s1 + R * dFz_dz;

    // eta[2] = a_torque * c1 + R * F_k * s1 + m * R * (...)
    deta[2][1] = da_torque_dz * c1;

    // eta[3] = a_torque * s1 - R * (F_k * c1 - m * g) - m * R * (...)
    deta[3][1] = da_torque_dz * s1;

    // eta[4] = a_torque * l_2 * cos(phi_leg - phi_body) - R * (l_2 * F_k * sin(phi_body - phi_leg) + u2)
    deta[4][1] = da_torque_dz * p.l_2 * cos(phi_leg - phi_body);

    // d(eta)/d(phi_leg) - column 2
    Scalar da_torque_dphi = p.l_1 * F_z * c1 + p.l_1 * F_x * s1;  // d(l_1*F_z*s1 - l_1*F_x*c1)/d(phi_leg)

    // eta[0] = a_torque * c1 - R * (F_x - F_k * s1 - m_l * l_1 * phi_leg_dot^2 * s1)
    // d/d(phi_leg): da_torque_dphi * c1 + a_torque * (-s1) - R * (- F_k * c1 - m_l * l_1 * phi_leg_dot^2 * c1)
    deta[0][2] = da_torque_dphi * c1 - a_torque * s1 + R * (F_k * c1 + p.m_l * p.l_1 * phi_leg_dot_sq * c1);

    // eta[1] = a_torque * s1 + R * (m_l * l_1 * phi_leg_dot^2 * c1 + F_z - F_k * c1 - m_l * g)
    // d/d(phi_leg): da_torque_dphi * s1 + a_torque * c1 + R * (- m_l * l_1 * phi_leg_dot^2 * s1 + F_k * s1)
    deta[1][2] = da_torque_dphi * s1 + a_torque * c1 + R * (-p.m_l * p.l_1 * phi_leg_dot_sq * s1 + F_k * s1);

    // eta[2] = a_torque * c1 + R * F_k * s1 + m * R * (len_leg * phi_leg_dot^2 * s1 + l_2 * phi_body_dot^2 * s2 - 2 * ddt_len * ddt_phi_leg * c1)
    deta[2][2] = da_torque_dphi * c1 - a_torque * s1 + R * F_k * c1
               + p.m * R * (len_leg * phi_leg_dot_sq * c1 + 2.0 * ddt_len_leg * ddt_phi_leg * s1);

    // eta[3] = a_torque * s1 - R * (F_k * c1 - m * g) - m * R * (2 * ddt_len * ddt_phi_leg * s1 + len_leg * phi_leg_dot^2 * c1 + l_2 * phi_body_dot^2 * c2)
    deta[3][2] = da_torque_dphi * s1 + a_torque * c1 - R * (-F_k * s1)
               - p.m * R * (2.0 * ddt_len_leg * ddt_phi_leg * c1 - len_leg * phi_leg_dot_sq * s1);

    // eta[4] = a_torque * l_2 * cos(phi_leg - phi_body) - R * (l_2 * F_k * sin(phi_body - phi_leg) + u2)
    deta[4][2] = da_torque_dphi * p.l_2 * cos(phi_leg - phi_body)
               - a_torque * p.l_2 * sin(phi_leg - phi_body)
               - R * (p.l_2 * F_k * (-cos(phi_body - phi_leg)));  // d(sin(phi_body - phi_leg))/d(phi_leg) = -cos(...)

    // d(eta)/d(phi_body) - column 3
    // phi_body appears in s2, c2 and in eta[4]
    deta[0][3] = 0;  // a_torque doesn't depend on phi_body
    deta[1][3] = 0;
    deta[2][3] = p.m * R * p.l_2 * phi_body_dot_sq * c2;  // d(l_2 * phi_body_dot^2 * s2)/d(phi_body)
    deta[3][3] = -p.m * R * p.l_2 * phi_body_dot_sq * (-s2);  // d(l_2 * phi_body_dot^2 * c2)/d(phi_body)
    deta[4][3] = a_torque * p.l_2 * sin(phi_leg - phi_body)  // d(cos(phi_leg - phi_body))/d(phi_body) = sin(...)
               - R * p.l_2 * F_k * cos(phi_body - phi_leg);  // d(sin(phi_body - phi_leg))/d(phi_body)

    // d(eta)/d(len_leg) - column 4
    // len_leg affects R, F_k, and terms in eta directly
    Scalar dR_dlen = 1.0;

    // a_torque doesn't depend on len_leg directly (only through F_k if we had u1 dep, but u1 is input)
    // Wait, F_k depends on len_leg via r_sd
    Scalar da_torque_dlen = 0;  // F_z, F_x don't depend on len_leg

    // eta[0] = a_torque * c1 - R * (F_x - F_k * s1 - m_l * l_1 * phi_leg_dot^2 * s1)
    // d/d(len_leg): - dR_dlen * (...) - R * (-dFk_dlen * s1) = -(F_x - F_k*s1 - ...) + R * dFk_dlen * s1
    deta[0][4] = -(F_x - F_k * s1 - p.m_l * p.l_1 * phi_leg_dot_sq * s1) + R * dFk_dlen * s1;

    // eta[1] = a_torque * s1 + R * (m_l * l_1 * phi_leg_dot^2 * c1 + F_z - F_k * c1 - m_l * g)
    deta[1][4] = (p.m_l * p.l_1 * phi_leg_dot_sq * c1 + F_z - F_k * c1 - p.m_l * p.g) - R * dFk_dlen * c1;

    // eta[2]: has R*F_k*s1 and m*R*(len_leg*...)
    // Complicated - let me just compute the full derivative
    // eta[2] = a * c1 + R * F_k * s1 + m * R * (len_leg * phi_leg_dot^2 * s1 + l_2 * phi_body_dot^2 * s2 - 2 * ddt_len * ddt_phi_leg * c1)
    Scalar term2 = F_k * s1 + R * dFk_dlen * s1
                 + p.m * (len_leg * phi_leg_dot_sq * s1 + p.l_2 * phi_body_dot_sq * s2 - 2.0 * ddt_len_leg * ddt_phi_leg * c1)
                 + p.m * R * phi_leg_dot_sq * s1;  // d(len_leg*phi_leg_dot^2*s1)/d(len_leg)
    deta[2][4] = term2;

    // eta[3] = a * s1 - R * (F_k * c1 - m * g) - m * R * (2 * ddt_len * ddt_phi_leg * s1 + len_leg * phi_leg_dot^2 * c1 + l_2 * phi_body_dot^2 * c2)
    Scalar term3 = -(F_k * c1 - p.m * p.g) - R * dFk_dlen * c1
                 - p.m * (2.0 * ddt_len_leg * ddt_phi_leg * s1 + len_leg * phi_leg_dot_sq * c1 + p.l_2 * phi_body_dot_sq * c2)
                 - p.m * R * phi_leg_dot_sq * c1;
    deta[3][4] = term3;

    // eta[4] = a * l_2 * cos(phi_leg - phi_body) - R * (l_2 * F_k * sin(phi_body - phi_leg) + u2)
    Scalar term4 = -(p.l_2 * F_k * sin(phi_body - phi_leg) + u2) - R * p.l_2 * dFk_dlen * sin(phi_body - phi_leg);
    deta[4][4] = term4;
}

// ============================================================================
// COMPUTE d(eta)/dqdot - Partial derivatives of RHS vector w.r.t. velocities
// ============================================================================

__host__ __device__ inline void compute_deta_dqdot(
    Scalar x_foot, Scalar z_foot, Scalar phi_leg, Scalar phi_body, Scalar len_leg,
    Scalar ddt_x_foot, Scalar ddt_z_foot, Scalar ddt_phi_leg, Scalar ddt_phi_body, Scalar ddt_len_leg,
    Scalar u1, Scalar u2,
    const PhysicsParams& p,
    Scalar deta[5][5]
) {
    Scalar R = len_leg - p.l_1;
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);
    Scalar r_sd = p.r_s0 - len_leg;

    // Force derivatives w.r.t. velocities
    Scalar dFx_ddx = 0, dFz_ddz = 0;
    if (z_foot < 0) {
        dFx_ddx = -p.b_g;
        if (-p.b_g * ddt_z_foot > 0) {
            dFz_ddz = -p.b_g;
        }
    }

    Scalar dFk_ddlen = 0;
    if (r_sd <= 0) {
        dFk_ddlen = -p.b_stop;
    }

    // Initialize
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            deta[i][j] = 0.0;
        }
    }

    // Current forces for a_torque
    Scalar F_x = (z_foot < 0) ? -p.b_g * ddt_x_foot : 0;
    Scalar F_z = 0;
    if (z_foot < 0) {
        F_z = p.k_g * (-z_foot);
        if (-p.b_g * ddt_z_foot > 0) F_z += -p.b_g * ddt_z_foot;
    }

    // d(eta)/d(ddt_x_foot) - column 0
    // a_torque depends on F_x which depends on ddt_x_foot
    Scalar da_torque_ddx = -p.l_1 * dFx_ddx * c1;

    deta[0][0] = da_torque_ddx * c1 - R * dFx_ddx;
    deta[1][0] = da_torque_ddx * s1;
    deta[2][0] = da_torque_ddx * c1;
    deta[3][0] = da_torque_ddx * s1;
    deta[4][0] = da_torque_ddx * p.l_2 * cos(phi_leg - phi_body);

    // d(eta)/d(ddt_z_foot) - column 1
    Scalar da_torque_ddz = p.l_1 * dFz_ddz * s1;

    deta[0][1] = da_torque_ddz * c1;
    deta[1][1] = da_torque_ddz * s1 + R * dFz_ddz;
    deta[2][1] = da_torque_ddz * c1;
    deta[3][1] = da_torque_ddz * s1;
    deta[4][1] = da_torque_ddz * p.l_2 * cos(phi_leg - phi_body);

    // d(eta)/d(ddt_phi_leg) - column 2
    // Appears as phi_leg_dot^2 terms
    Scalar d_phi_leg_dot_sq = 2.0 * ddt_phi_leg;

    deta[0][2] = -R * (-p.m_l * p.l_1 * d_phi_leg_dot_sq * s1);
    deta[1][2] = R * p.m_l * p.l_1 * d_phi_leg_dot_sq * c1;
    deta[2][2] = p.m * R * (len_leg * d_phi_leg_dot_sq * s1 - 2.0 * ddt_len_leg * c1);
    deta[3][2] = -p.m * R * (2.0 * ddt_len_leg * s1 + len_leg * d_phi_leg_dot_sq * c1);
    deta[4][2] = 0;

    // d(eta)/d(ddt_phi_body) - column 3
    Scalar d_phi_body_dot_sq = 2.0 * ddt_phi_body;

    deta[0][3] = 0;
    deta[1][3] = 0;
    deta[2][3] = p.m * R * p.l_2 * d_phi_body_dot_sq * s2;
    deta[3][3] = -p.m * R * p.l_2 * d_phi_body_dot_sq * c2;
    deta[4][3] = 0;

    // d(eta)/d(ddt_len_leg) - column 4
    // Appears in F_k (if spring extended) and in Coriolis terms
    Scalar F_k;
    if (r_sd > 0) {
        F_k = p.k_l * r_sd + u1;
    } else {
        F_k = p.k_stop * r_sd + u1 - p.b_stop * ddt_len_leg;
    }

    deta[0][4] = R * dFk_ddlen * s1;
    deta[1][4] = -R * dFk_ddlen * c1;
    deta[2][4] = R * dFk_ddlen * s1 - p.m * R * 2.0 * ddt_phi_leg * c1;
    deta[3][4] = -R * dFk_ddlen * c1 - p.m * R * 2.0 * ddt_phi_leg * s1;
    deta[4][4] = -R * p.l_2 * dFk_ddlen * sin(phi_body - phi_leg);
}

// ============================================================================
// MAIN FUNCTION: Compute analytical Jacobian blocks A and B
// ============================================================================

__host__ __device__ inline void compute_jacobian_blocks_analytical(
    const Scalar y[10],
    Scalar u1, Scalar u2,
    const PhysicsParams& phys,
    Scalar A[5][5],    // d(qddot)/dq
    Scalar B[5][5]     // d(qddot)/dqdot
) {
    // Unpack state
    Scalar x_foot = y[0];
    Scalar z_foot = y[1];
    Scalar phi_leg = y[2];
    Scalar phi_body = y[3];
    Scalar len_leg = y[4];
    Scalar ddt_x_foot = y[5];
    Scalar ddt_z_foot = y[6];
    Scalar ddt_phi_leg = y[7];
    Scalar ddt_phi_body = y[8];
    Scalar ddt_len_leg = y[9];

    // First compute qddot (we need it for the A matrix)
    Scalar qdd[5];
    compute_accelerations(x_foot, z_foot, phi_leg, phi_body, len_leg,
                          ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                          u1, u2, phys, qdd);

    // Compute current mass matrix M
    Scalar R = len_leg - phys.l_1;
    Scalar s1 = sin(phi_leg);
    Scalar c1 = cos(phi_leg);
    Scalar s2 = sin(phi_body);
    Scalar c2 = cos(phi_body);

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

    // Compute deta/dq and deta/dqdot
    Scalar deta_dq[5][5], deta_dqdot[5][5];
    compute_deta_dq(x_foot, z_foot, phi_leg, phi_body, len_leg,
                    ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                    u1, u2, phys, deta_dq);
    compute_deta_dqdot(x_foot, z_foot, phi_leg, phi_body, len_leg,
                       ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                       u1, u2, phys, deta_dqdot);

    // Compute dM/dq for q[2], q[3], q[4] (only these affect M)
    Scalar dM_dphi_leg[5][5], dM_dphi_body[5][5], dM_dlen_leg[5][5];
    compute_dM_dphi_leg(phi_leg, phi_body, len_leg, phys, dM_dphi_leg);
    compute_dM_dphi_body(phi_leg, phi_body, len_leg, phys, dM_dphi_body);
    compute_dM_dlen_leg(phi_leg, phi_body, len_leg, phys, dM_dlen_leg);

    // For B: B = M^{-1} * deta_dqdot
    // We solve M * B[:,j] = deta_dqdot[:,j] for each column
    for (int j = 0; j < 5; j++) {
        Scalar rhs[5];
        for (int i = 0; i < 5; i++) {
            rhs[i] = deta_dqdot[i][j];
        }
        Scalar col[5];
        solve_5x5(M, rhs, col);
        for (int i = 0; i < 5; i++) {
            B[i][j] = col[i];
        }
    }

    // For A: A = M^{-1} * (deta_dq - dM_dq * qdd)
    // We need to handle columns 0, 1 (no M dependence) and 2, 3, 4 (have dM terms)
    for (int j = 0; j < 5; j++) {
        Scalar rhs[5];

        // Start with deta_dq[:,j]
        for (int i = 0; i < 5; i++) {
            rhs[i] = deta_dq[i][j];
        }

        // Subtract dM[:,j] * qdd if this column has M dependence
        if (j == 2) {  // phi_leg
            for (int i = 0; i < 5; i++) {
                Scalar dM_qdd = 0;
                for (int k = 0; k < 5; k++) {
                    dM_qdd += dM_dphi_leg[i][k] * qdd[k];
                }
                rhs[i] -= dM_qdd;
            }
        } else if (j == 3) {  // phi_body
            for (int i = 0; i < 5; i++) {
                Scalar dM_qdd = 0;
                for (int k = 0; k < 5; k++) {
                    dM_qdd += dM_dphi_body[i][k] * qdd[k];
                }
                rhs[i] -= dM_qdd;
            }
        } else if (j == 4) {  // len_leg
            for (int i = 0; i < 5; i++) {
                Scalar dM_qdd = 0;
                for (int k = 0; k < 5; k++) {
                    dM_qdd += dM_dlen_leg[i][k] * qdd[k];
                }
                rhs[i] -= dM_qdd;
            }
        }

        // Solve M * A[:,j] = rhs
        Scalar col[5];
        solve_5x5(M, rhs, col);
        for (int i = 0; i < 5; i++) {
            A[i][j] = col[i];
        }
    }
}

#endif // JACOBIAN_ANALYTICAL_CUH
