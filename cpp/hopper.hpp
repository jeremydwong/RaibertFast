// hopper.hpp - Raibert Hopper Simulation (C++ Port)
// Educational implementation demonstrating static typing and performance
//
// Unity build style: include this header in build.cpp
// Depends on: Eigen (for matrix solve)

#ifndef HOPPER_HPP
#define HOPPER_HPP

#include <cmath>
#include <cstdio>
#include <array>
#include <Eigen/Dense>

// ============================================================================
// DATA STRUCTURES
// ============================================================================

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

    static constexpr int SIZE = 10;

    // Array access for ODE solver compatibility
    double& operator[](int i) {
        switch (i) {
            case 0: return x_foot;
            case 1: return z_foot;
            case 2: return phi_leg;
            case 3: return phi_body;
            case 4: return len_leg;
            case 5: return ddt_x_foot;
            case 6: return ddt_z_foot;
            case 7: return ddt_phi_leg;
            case 8: return ddt_phi_body;
            case 9: return ddt_len_leg;
            default: return x_foot; // should not happen
        }
    }

    const double& operator[](int i) const {
        return const_cast<State*>(this)->operator[](i);
    }

    // Initialize to zero
    State() : x_foot(0), z_foot(0), phi_leg(0), phi_body(0), len_leg(0),
              ddt_x_foot(0), ddt_z_foot(0), ddt_phi_leg(0), ddt_phi_body(0), ddt_len_leg(0) {}
};

struct StateDot {
    // Velocities (first 5 elements of state derivative)
    double ddt_x_foot;
    double ddt_z_foot;
    double ddt_phi_leg;
    double ddt_phi_body;
    double ddt_len_leg;

    // Accelerations (second 5 elements)
    double dddt_x_foot;
    double dddt_z_foot;
    double dddt_phi_leg;
    double dddt_phi_body;
    double dddt_len_leg;

    static constexpr int SIZE = 10;

    double& operator[](int i) {
        switch (i) {
            case 0: return ddt_x_foot;
            case 1: return ddt_z_foot;
            case 2: return ddt_phi_leg;
            case 3: return ddt_phi_body;
            case 4: return ddt_len_leg;
            case 5: return dddt_x_foot;
            case 6: return dddt_z_foot;
            case 7: return dddt_phi_leg;
            case 8: return dddt_phi_body;
            case 9: return dddt_len_leg;
            default: return ddt_x_foot;
        }
    }

    const double& operator[](int i) const {
        return const_cast<StateDot*>(this)->operator[](i);
    }

    StateDot() : ddt_x_foot(0), ddt_z_foot(0), ddt_phi_leg(0), ddt_phi_body(0), ddt_len_leg(0),
                 dddt_x_foot(0), dddt_z_foot(0), dddt_phi_leg(0), dddt_phi_body(0), dddt_len_leg(0) {}
};

struct Parameters {
    // Physical constants
    double m      = 10.0;    // body mass [kg]
    double m_l    = 1.0;     // leg mass [kg]
    double J      = 10.0;    // body moment of inertia [kg*m^2]
    double J_l    = 1.0;     // leg moment of inertia [kg*m^2]
    double g      = 9.8;     // gravity [m/s^2]
    double k_l    = 1e3;     // leg spring constant [N/m]
    double k_stop = 1e5;     // leg stop spring constant [N/m]
    double b_stop = 1e3;     // leg stop damping [N*s/m]
    double k_g    = 1e4;     // ground spring constant [N/m]
    double b_g    = 300.0;   // ground damping [N*s/m]
    double r_s0   = 1.0;     // leg spring rest length [m]
    double l_1    = 0.5;     // foot to leg COM distance [m]
    double l_2    = 0.4;     // hip to body COM distance [m]

    // FSM state constants
    static constexpr int FSM_COMPRESSION = 0;
    static constexpr int FSM_THRUST      = 1;
    static constexpr int FSM_FLIGHT      = 2;
    static constexpr int FSM_NUM_STATES  = 3;

    // Mutable FSM state (changes during simulation)
    int    fsm_state        = FSM_FLIGHT;
    double t_state_switch   = 0.0;
    double x_dot_des        = 0.0;
    double T_s              = 0.425;
    double T_compression    = 0.0;
    double t_thrust_on      = 0.0;
    double T_MAX_THRUST_DUR = 0.425 * 0.35;
};

struct ControlOutput {
    double u1;      // Leg spring actuator force [N]
    double u2;      // Hip torque [N*m]
    double a_des;   // Desired leg angle (for logging) [rad]

    ControlOutput() : u1(0), u2(0), a_des(0) {}
};

struct DynamicsOutput {
    StateDot state_dot;
    ControlOutput control;
    double r_sd;        // Spring deflection [m]
    int fsm_state;      // Current FSM state
};

// ============================================================================
// MATRIX SOLVERS
// ============================================================================

// Hand-written 5x5 Gaussian elimination with partial pivoting
// Educational: shows students exactly what's happening
static void solve_5x5_gaussian(const double M_in[5][5], const double b_in[5], double x[5]) {
    // Copy inputs (we modify during elimination)
    double M[5][5];
    double b[5];
    for (int i = 0; i < 5; i++) {
        b[i] = b_in[i];
        for (int j = 0; j < 5; j++) {
            M[i][j] = M_in[i][j];
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < 5; col++) {
        // Find pivot (largest absolute value in column)
        int pivot_row = col;
        double pivot_val = std::abs(M[col][col]);
        for (int row = col + 1; row < 5; row++) {
            if (std::abs(M[row][col]) > pivot_val) {
                pivot_val = std::abs(M[row][col]);
                pivot_row = row;
            }
        }

        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 5; j++) {
                std::swap(M[col][j], M[pivot_row][j]);
            }
            std::swap(b[col], b[pivot_row]);
        }

        // Check for singular matrix
        if (std::abs(M[col][col]) < 1e-14) {
            fprintf(stderr, "WARNING: Near-singular matrix in solve_5x5_gaussian\n");
        }

        // Eliminate column below pivot
        for (int row = col + 1; row < 5; row++) {
            double factor = M[row][col] / M[col][col];
            for (int j = col; j < 5; j++) {
                M[row][j] -= factor * M[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for (int i = 4; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < 5; j++) {
            x[i] -= M[i][j] * x[j];
        }
        x[i] /= M[i][i];
    }
}

// Eigen-based 5x5 solver (default, faster for production)
static void solve_5x5_eigen(const double M_in[5][5], const double b_in[5], double x[5]) {
    Eigen::Matrix<double, 5, 5> M;
    Eigen::Matrix<double, 5, 1> b;

    for (int i = 0; i < 5; i++) {
        b(i) = b_in[i];
        for (int j = 0; j < 5; j++) {
            M(i, j) = M_in[i][j];
        }
    }

    Eigen::Matrix<double, 5, 1> result = M.partialPivLu().solve(b);

    for (int i = 0; i < 5; i++) {
        x[i] = result(i);
    }
}

// Function pointer for selecting solver (default: Eigen)
using MatrixSolver = void(*)(const double[5][5], const double[5], double[5]);
static MatrixSolver g_matrix_solver = solve_5x5_eigen;

// ============================================================================
// CONTROL (from hopperStateControl.m / hopper.py)
// ============================================================================

static ControlOutput hopper_control(double t, const State& q, const Parameters& p) {
    ControlOutput ctrl;

    // Control gains
    constexpr double k_fp  = 150.0;   // foot placement
    constexpr double b_fp  = 15.0;
    constexpr double k_att = 150.0;   // attitude
    constexpr double b_att = 15.0;
    constexpr double k_xdot = 0.02;   // forward speed
    const double thrust = 0.035 * p.k_l;
    constexpr double thr_z_low = 0.01;
    const double u_retract = -0.1 * p.k_l;

    double T_s = (p.T_s == 0) ? 0.425 : p.T_s;

    // Extract state variables for readability
    double d_xfoot_dt = q.ddt_x_foot;
    double y_foot = q.z_foot;
    double a = q.phi_leg;
    double dadt = q.ddt_phi_leg;
    double b = q.phi_body;
    double dbdt = q.ddt_phi_body;
    double l = q.len_leg;
    double dldt = q.ddt_len_leg;
    double stance_ang_des = a / 2.0;

    if (p.fsm_state == Parameters::FSM_THRUST) {
        ctrl.u1 = thrust;
        ctrl.u2 = -k_att * (b - stance_ang_des) - b_att * dbdt;
    }
    else if (p.fsm_state == Parameters::FSM_COMPRESSION) {
        ctrl.u1 = 0.0;
        ctrl.u2 = -k_att * (b - stance_ang_des) - b_att * dbdt;
    }
    else if (p.fsm_state == Parameters::FSM_FLIGHT) {
        // Compute body velocity
        double d_xbody_dt = d_xfoot_dt + dldt * std::sin(a) + l * std::cos(a) * dadt
                         + p.l_2 * std::cos(b) * dbdt;

        // Desired leg angle for foot placement
        double arg = (d_xbody_dt * T_s / 2.0 + k_xdot * (d_xbody_dt - p.x_dot_des)) / l;
        // Clamp argument to valid range for asin
        if (arg > 1.0) arg = 1.0;
        if (arg < -1.0) arg = -1.0;
        ctrl.a_des = -std::asin(arg);

        if (std::isnan(ctrl.a_des)) {
            fprintf(stderr, "WARNING: NaN in a_des at t=%.6f\n", t);
            ctrl.a_des = 0.0;
        }

        // Apply foot placement control if above ground
        if (y_foot > thr_z_low) {
            ctrl.u2 = k_fp * (a - ctrl.a_des) + b_fp * dadt;
        }
        ctrl.u1 = 0.0;
    }

    // Thrust duration limiting
    if (p.fsm_state == Parameters::FSM_THRUST) {
        if (t - p.t_thrust_on > p.T_MAX_THRUST_DUR) {
            ctrl.u1 = 0.0;
        }
    }

    return ctrl;
}

// ============================================================================
// DYNAMICS (from hopperDynamicsFwd.m / hopper.py)
// ============================================================================

static DynamicsOutput hopper_dynamics_fwd(double t, const State& q, const Parameters& p) {
    DynamicsOutput out;
    out.fsm_state = p.fsm_state;

    // Get control inputs
    ControlOutput ctrl = hopper_control(t, q, p);
    out.control = ctrl;
    double u1 = ctrl.u1;
    double u2 = ctrl.u2;

    // Geometric quantities
    double R = q.len_leg - p.l_1;
    double s1 = std::sin(q.phi_leg);
    double c1 = std::cos(q.phi_leg);
    double s2 = std::sin(q.phi_body);
    double c2 = std::cos(q.phi_body);

    // Spring deflection (positive when compressed)
    double r_sd = p.r_s0 - q.len_leg;
    out.r_sd = r_sd;

    // Leg spring force
    double F_k;
    if (r_sd > 0) {
        // Spring compressed: linear spring + actuator
        F_k = p.k_l * r_sd + u1;
    } else {
        // Spring extended past rest: hard stop with damping
        F_k = p.k_stop * r_sd + u1 - p.b_stop * q.ddt_len_leg;
    }

    // Ground reaction forces
    double F_x, F_z;
    if (q.z_foot < 0) {
        // Foot below ground: spring-damper contact
        F_x = -p.b_g * q.ddt_x_foot;
        F_z = p.k_g * (-q.z_foot);
        F_z = F_z + std::max(-p.b_g * q.ddt_z_foot, 0.0);
    } else {
        F_x = 0.0;
        F_z = 0.0;
    }

    // Torque about foot due to ground reaction and hip
    double a_torque = p.l_1 * F_z * s1 - p.l_1 * F_x * c1 - u2;

    // Mass matrix M (5x5)
    double M[5][5];
    M[0][0] = -p.m_l * R;
    M[0][1] = 0;
    M[0][2] = (p.J_l - p.m_l * R * p.l_1) * c1;
    M[0][3] = 0;
    M[0][4] = 0;

    M[1][0] = 0;
    M[1][1] = p.m_l * R;
    M[1][2] = (p.J_l - p.m_l * R * p.l_1) * s1;
    M[1][3] = 0;
    M[1][4] = 0;

    M[2][0] = p.m * R;
    M[2][1] = 0;
    M[2][2] = (p.J_l + p.m * R * q.len_leg) * c1;
    M[2][3] = p.m * R * p.l_2 * c2;
    M[2][4] = p.m * R * s1;

    M[3][0] = 0;
    M[3][1] = -p.m * R;
    M[3][2] = (p.J_l + p.m * R * q.len_leg) * s1;
    M[3][3] = p.m * R * p.l_2 * s2;
    M[3][4] = -p.m * R * c1;

    M[4][0] = 0;
    M[4][1] = 0;
    M[4][2] = p.J_l * p.l_2 * std::cos(q.phi_leg - q.phi_body);
    M[4][3] = -p.J * R;
    M[4][4] = 0;

    // RHS vector eta (5x1)
    double eta[5];
    double phi_leg_dot_sq = q.ddt_phi_leg * q.ddt_phi_leg;
    double phi_body_dot_sq = q.ddt_phi_body * q.ddt_phi_body;

    eta[0] = a_torque * c1 - R * (F_x - F_k * s1 - p.m_l * p.l_1 * phi_leg_dot_sq * s1);

    eta[1] = a_torque * s1 + R * (p.m_l * p.l_1 * phi_leg_dot_sq * c1 + F_z - F_k * c1 - p.m_l * p.g);

    eta[2] = a_torque * c1 + R * F_k * s1
           + p.m * R * (q.len_leg * phi_leg_dot_sq * s1 + p.l_2 * phi_body_dot_sq * s2
                      - 2.0 * q.ddt_len_leg * q.ddt_phi_leg * c1);

    eta[3] = a_torque * s1 - R * (F_k * c1 - p.m * p.g)
           - p.m * R * (2.0 * q.ddt_len_leg * q.ddt_phi_leg * s1
                      + q.len_leg * phi_leg_dot_sq * c1 + p.l_2 * phi_body_dot_sq * c2);

    eta[4] = a_torque * p.l_2 * std::cos(q.phi_leg - q.phi_body)
           - R * (p.l_2 * F_k * std::sin(q.phi_body - q.phi_leg) + u2);

    // Solve for accelerations: M * qdd = eta
    double qdd[5];
    g_matrix_solver(M, eta, qdd);

    // Assemble state derivative
    // First 5 elements: velocities
    out.state_dot.ddt_x_foot   = q.ddt_x_foot;
    out.state_dot.ddt_z_foot   = q.ddt_z_foot;
    out.state_dot.ddt_phi_leg  = q.ddt_phi_leg;
    out.state_dot.ddt_phi_body = q.ddt_phi_body;
    out.state_dot.ddt_len_leg  = q.ddt_len_leg;

    // Last 5 elements: accelerations
    out.state_dot.dddt_x_foot   = qdd[0];
    out.state_dot.dddt_z_foot   = qdd[1];
    out.state_dot.dddt_phi_leg  = qdd[2];
    out.state_dot.dddt_phi_body = qdd[3];
    out.state_dot.dddt_len_leg  = qdd[4];

    return out;
}

// ODE-compatible wrapper (returns just state derivative)
static StateDot hopper_dynamics(double t, const State& q, const Parameters& p) {
    return hopper_dynamics_fwd(t, q, p).state_dot;
}

// ============================================================================
// EVENT DETECTION (from eventsHopperControl.m / hopper.py)
// ============================================================================

// Returns event function value. Zero-crossing (positive direction) triggers state transition.
static double hopper_event(double t, const State& q, const Parameters& p) {
    constexpr double thresh_leg_extended = 0.0001;

    if (p.fsm_state == Parameters::FSM_COMPRESSION) {
        // Transition when leg stops compressing (ddt_len_leg crosses zero upward)
        return q.ddt_len_leg;
    }
    else if (p.fsm_state == Parameters::FSM_THRUST) {
        // Transition when leg fully extended
        return -(p.r_s0 - q.len_leg) - thresh_leg_extended;
    }
    else if (p.fsm_state == Parameters::FSM_FLIGHT) {
        // Touchdown: when foot hits ground
        return -q.z_foot;
    }

    // Should not reach here
    return 1.0;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static void print_state(const State& q, const char* label = "") {
    printf("%s State: x=%.4f z=%.4f phi_leg=%.4f phi_body=%.4f len=%.4f\n",
           label, q.x_foot, q.z_foot, q.phi_leg, q.phi_body, q.len_leg);
    printf("        dx=%.4f dz=%.4f dphi_leg=%.4f dphi_body=%.4f dlen=%.4f\n",
           q.ddt_x_foot, q.ddt_z_foot, q.ddt_phi_leg, q.ddt_phi_body, q.ddt_len_leg);
}

static void print_state_dot(const StateDot& qd, const char* label = "") {
    printf("%s StateDot: dx=%.4f dz=%.4f dphi_leg=%.4f dphi_body=%.4f dlen=%.4f\n",
           label, qd.ddt_x_foot, qd.ddt_z_foot, qd.ddt_phi_leg, qd.ddt_phi_body, qd.ddt_len_leg);
    printf("           ddx=%.4f ddz=%.4f ddphi_leg=%.4f ddphi_body=%.4f ddlen=%.4f\n",
           qd.dddt_x_foot, qd.dddt_z_foot, qd.dddt_phi_leg, qd.dddt_phi_body, qd.dddt_len_leg);
}

static const char* fsm_state_name(int state) {
    switch (state) {
        case Parameters::FSM_COMPRESSION: return "COMPRESSION";
        case Parameters::FSM_THRUST:      return "THRUST";
        case Parameters::FSM_FLIGHT:      return "FLIGHT";
        default: return "UNKNOWN";
    }
}

#endif // HOPPER_HPP
