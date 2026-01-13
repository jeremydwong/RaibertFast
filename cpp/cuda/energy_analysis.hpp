// energy_analysis.hpp - Energy accounting for hopper trajectories
//
// Port of hopperEnergy() from Python.
// Analyzes trajectory data to compute energy flows:
// - Kinetic energy (body and leg)
// - Gravitational potential energy
// - Spring potential energy (leg and ground)
// - Damping losses (leg stop, ground friction)
// - Work done by actuators (u1, u2)
//
// Used for debugging: E_delta should equal E_net_flow if conservation is correct.

#ifndef ENERGY_ANALYSIS_HPP
#define ENERGY_ANALYSIS_HPP

#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

#include "hopper_types.cuh"  // For Scalar typedef

struct PhysicsParamsEnergy {
    Scalar m = 10.0;      // body mass
    Scalar m_l = 1.0;     // leg mass
    Scalar J = 10.0;      // body moment of inertia
    Scalar J_l = 1.0;     // leg moment of inertia
    Scalar g = 9.8;       // gravity
    Scalar k_l = 1e3;     // leg spring constant
    Scalar k_stop = 1e5;  // leg stop spring constant
    Scalar b_stop = 1e3;  // leg stop damping
    Scalar k_g = 1e4;     // ground spring constant
    Scalar b_g = 300.0;   // ground damping
    Scalar r_s0 = 1.0;    // leg spring rest length
    Scalar l_1 = 0.5;     // foot to leg COM
    Scalar l_2 = 0.4;     // hip to body COM
};

struct TrajectoryPoint {
    Scalar t;
    Scalar x_foot, z_foot, phi_leg, phi_body, len_leg;
    Scalar ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg;
    int fsm_state;
    Scalar u1, u2;
};

struct EnergyResult {
    std::vector<Scalar> t;

    // Mechanical energies (instantaneous)
    std::vector<Scalar> E_kin_body;
    std::vector<Scalar> E_kin_leg;
    std::vector<Scalar> E_g_body;
    std::vector<Scalar> E_g_leg;
    std::vector<Scalar> E_leg_spring;
    std::vector<Scalar> E_foot_spring_z;

    // Cumulative work/losses
    std::vector<Scalar> work_u1;
    std::vector<Scalar> work_u2;
    std::vector<Scalar> E_leg_damp;
    std::vector<Scalar> E_foot_damp_x;
    std::vector<Scalar> E_foot_damp_z;

    // Summary quantities
    std::vector<Scalar> E_mechanical;    // Total mechanical energy
    std::vector<Scalar> E_loss;          // Total damping losses
    std::vector<Scalar> E_gain;          // Total work by actuators
    std::vector<Scalar> E_net_flow;      // E_gain + E_loss (loss is negative)
    std::vector<Scalar> E_delta;         // E_mechanical - E_mechanical[0]

    // Error metric: should be close to zero for energy conservation
    std::vector<Scalar> E_error;         // E_delta - E_net_flow
};

// Trapezoidal integration
inline Scalar trapz_step(Scalar f_prev, Scalar f_curr, Scalar dt) {
    return 0.5 * (f_prev + f_curr) * dt;
}

// Compute energy accounting from trajectory
inline EnergyResult compute_energy(const std::vector<TrajectoryPoint>& traj,
                                   const PhysicsParamsEnergy& P) {
    EnergyResult result;
    size_t n = traj.size();

    // Resize all vectors
    result.t.resize(n);
    result.E_kin_body.resize(n);
    result.E_kin_leg.resize(n);
    result.E_g_body.resize(n);
    result.E_g_leg.resize(n);
    result.E_leg_spring.resize(n);
    result.E_foot_spring_z.resize(n);
    result.work_u1.resize(n);
    result.work_u2.resize(n);
    result.E_leg_damp.resize(n);
    result.E_foot_damp_x.resize(n);
    result.E_foot_damp_z.resize(n);
    result.E_mechanical.resize(n);
    result.E_loss.resize(n);
    result.E_gain.resize(n);
    result.E_net_flow.resize(n);
    result.E_delta.resize(n);
    result.E_error.resize(n);

    // Initialize cumulative quantities
    result.work_u1[0] = 0;
    result.work_u2[0] = 0;
    result.E_leg_damp[0] = 0;
    result.E_foot_damp_x[0] = 0;
    result.E_foot_damp_z[0] = 0;

    for (size_t i = 0; i < n; i++) {
        const TrajectoryPoint& pt = traj[i];
        result.t[i] = pt.t;

        // Derived positions
        Scalar x_leg = pt.x_foot + P.l_1 * sin(pt.phi_leg);
        Scalar z_leg = pt.z_foot + P.l_1 * cos(pt.phi_leg);
        Scalar z_body = pt.z_foot + pt.len_leg * cos(pt.phi_leg) + P.l_2 * cos(pt.phi_body);

        // Leg COM velocity
        Scalar ddt_comx_leg = pt.ddt_x_foot + P.l_1 * cos(pt.phi_leg) * pt.ddt_phi_leg;
        Scalar ddt_comz_leg = pt.ddt_z_foot - P.l_1 * sin(pt.phi_leg) * pt.ddt_phi_leg;

        // Body COM velocity
        Scalar ddt_comx_body = pt.ddt_x_foot
            + pt.ddt_len_leg * sin(pt.phi_leg)
            + pt.len_leg * cos(pt.phi_leg) * pt.ddt_phi_leg
            + P.l_2 * cos(pt.phi_body) * pt.ddt_phi_body;
        Scalar ddt_comz_body = pt.ddt_z_foot
            + pt.ddt_len_leg * cos(pt.phi_leg)
            - pt.len_leg * sin(pt.phi_leg) * pt.ddt_phi_leg
            - P.l_2 * sin(pt.phi_body) * pt.ddt_phi_body;

        // Kinetic energies
        result.E_kin_body[i] = 0.5 * P.m * (ddt_comx_body*ddt_comx_body + ddt_comz_body*ddt_comz_body)
                             + 0.5 * P.J * pt.ddt_phi_body * pt.ddt_phi_body;
        result.E_kin_leg[i] = 0.5 * P.m_l * (ddt_comx_leg*ddt_comx_leg + ddt_comz_leg*ddt_comz_leg)
                            + 0.5 * P.J_l * pt.ddt_phi_leg * pt.ddt_phi_leg;

        // Gravitational potential energies
        result.E_g_body[i] = P.m * P.g * z_body;
        result.E_g_leg[i] = P.m_l * P.g * z_leg;

        // Spring deflection
        Scalar rs_d = P.r_s0 - pt.len_leg;

        // Leg spring energy
        if (rs_d > 0) {
            result.E_leg_spring[i] = 0.5 * P.k_l * rs_d * rs_d;
        } else {
            result.E_leg_spring[i] = 0.5 * P.k_stop * rs_d * rs_d;
        }

        // Ground spring energy
        if (pt.z_foot < 0) {
            result.E_foot_spring_z[i] = 0.5 * P.k_g * pt.z_foot * pt.z_foot;
        } else {
            result.E_foot_spring_z[i] = 0;
        }

        // Power calculations for cumulative integration
        if (i > 0) {
            Scalar dt = pt.t - traj[i-1].t;
            const TrajectoryPoint& prev = traj[i-1];

            // Power from u1 (leg actuator): P = u1 * d(len_leg)/dt
            // Note: u1 acts on leg length change
            Scalar power_u1_prev = prev.u1 * prev.ddt_len_leg;
            Scalar power_u1_curr = pt.u1 * pt.ddt_len_leg;
            result.work_u1[i] = result.work_u1[i-1] + trapz_step(power_u1_prev, power_u1_curr, dt);

            // Power from u2 (hip torque): P = u2 * (d(phi_body)/dt - d(phi_leg)/dt)
            Scalar omega_rel_prev = prev.ddt_phi_body - prev.ddt_phi_leg;
            Scalar omega_rel_curr = pt.ddt_phi_body - pt.ddt_phi_leg;
            Scalar power_u2_prev = prev.u2 * omega_rel_prev;
            Scalar power_u2_curr = pt.u2 * omega_rel_curr;
            result.work_u2[i] = result.work_u2[i-1] + trapz_step(power_u2_prev, power_u2_curr, dt);

            // Leg damper loss (only when extended past rest)
            Scalar rs_d_prev = P.r_s0 - prev.len_leg;
            Scalar power_leg_damp_prev = (rs_d_prev <= 0) ? -P.b_stop * prev.ddt_len_leg * prev.ddt_len_leg : 0;
            Scalar power_leg_damp_curr = (rs_d <= 0) ? -P.b_stop * pt.ddt_len_leg * pt.ddt_len_leg : 0;
            result.E_leg_damp[i] = result.E_leg_damp[i-1] + trapz_step(power_leg_damp_prev, power_leg_damp_curr, dt);

            // Ground friction loss in x
            Scalar power_foot_x_prev = (prev.z_foot < 0) ? -P.b_g * prev.ddt_x_foot * prev.ddt_x_foot : 0;
            Scalar power_foot_x_curr = (pt.z_foot < 0) ? -P.b_g * pt.ddt_x_foot * pt.ddt_x_foot : 0;
            result.E_foot_damp_x[i] = result.E_foot_damp_x[i-1] + trapz_step(power_foot_x_prev, power_foot_x_curr, dt);

            // Ground damping loss in z (only absorbs energy, not adds)
            Scalar F_damp_z_prev = (prev.z_foot < 0) ? std::max(-P.b_g * prev.ddt_z_foot, Scalar(0)) : Scalar(0);
            Scalar F_damp_z_curr = (pt.z_foot < 0) ? std::max(-P.b_g * pt.ddt_z_foot, Scalar(0)) : Scalar(0);
            Scalar power_foot_z_prev = F_damp_z_prev * prev.ddt_z_foot;
            Scalar power_foot_z_curr = F_damp_z_curr * pt.ddt_z_foot;
            result.E_foot_damp_z[i] = result.E_foot_damp_z[i-1] + trapz_step(power_foot_z_prev, power_foot_z_curr, dt);
        }

        // Total mechanical energy
        result.E_mechanical[i] = result.E_kin_body[i] + result.E_kin_leg[i]
                               + result.E_g_body[i] + result.E_g_leg[i]
                               + result.E_leg_spring[i] + result.E_foot_spring_z[i];

        // Summary quantities
        result.E_loss[i] = result.E_leg_damp[i] + result.E_foot_damp_x[i] + result.E_foot_damp_z[i];
        result.E_gain[i] = result.work_u1[i] + result.work_u2[i];
        result.E_net_flow[i] = result.E_gain[i] + result.E_loss[i];
        result.E_delta[i] = result.E_mechanical[i] - result.E_mechanical[0];
        result.E_error[i] = result.E_delta[i] - result.E_net_flow[i];
    }

    return result;
}

// Load trajectory from CSV file
inline std::vector<TrajectoryPoint> load_trajectory_csv(const std::string& filename) {
    std::vector<TrajectoryPoint> traj;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return traj;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        TrajectoryPoint pt;
        char comma;

        iss >> pt.t >> comma
            >> pt.x_foot >> comma
            >> pt.z_foot >> comma
            >> pt.phi_leg >> comma
            >> pt.phi_body >> comma
            >> pt.len_leg >> comma
            >> pt.ddt_x_foot >> comma
            >> pt.ddt_z_foot >> comma
            >> pt.ddt_phi_leg >> comma
            >> pt.ddt_phi_body >> comma
            >> pt.ddt_len_leg >> comma
            >> pt.fsm_state >> comma
            >> pt.u1 >> comma
            >> pt.u2;

        traj.push_back(pt);
    }

    return traj;
}

// Print energy summary
inline void print_energy_summary(const EnergyResult& result) {
    size_t n = result.t.size();
    if (n == 0) return;

    printf("\nEnergy Analysis Summary\n");
    printf("=======================\n");
    printf("Time span: %.3f to %.3f s (%zu samples)\n", result.t[0], result.t[n-1], n);
    printf("\n");

    printf("Initial mechanical energy: %.3f J\n", result.E_mechanical[0]);
    printf("Final mechanical energy:   %.3f J\n", result.E_mechanical[n-1]);
    printf("Change in mech. energy:    %.3f J\n", result.E_delta[n-1]);
    printf("\n");

    printf("Work by u1 (leg actuator): %.3f J\n", result.work_u1[n-1]);
    printf("Work by u2 (hip torque):   %.3f J\n", result.work_u2[n-1]);
    printf("Total work by actuators:   %.3f J\n", result.E_gain[n-1]);
    printf("\n");

    printf("Leg damper loss:           %.3f J\n", result.E_leg_damp[n-1]);
    printf("Ground friction (x) loss:  %.3f J\n", result.E_foot_damp_x[n-1]);
    printf("Ground damping (z) loss:   %.3f J\n", result.E_foot_damp_z[n-1]);
    printf("Total damping loss:        %.3f J\n", result.E_loss[n-1]);
    printf("\n");

    printf("Net energy flow (gain+loss): %.3f J\n", result.E_net_flow[n-1]);
    printf("Energy conservation error:   %.6f J\n", result.E_error[n-1]);

    // Find max error
    Scalar max_error = 0;
    for (size_t i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(result.E_error[i]));
    }
    printf("Max conservation error:      %.6f J\n", max_error);
}

// Export energy results to CSV
inline void export_energy_csv(const EnergyResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create " << filename << std::endl;
        return;
    }

    file << "t,E_kin_body,E_kin_leg,E_g_body,E_g_leg,E_leg_spring,E_foot_spring_z,"
         << "work_u1,work_u2,E_leg_damp,E_foot_damp_x,E_foot_damp_z,"
         << "E_mechanical,E_loss,E_gain,E_net_flow,E_delta,E_error\n";

    for (size_t i = 0; i < result.t.size(); i++) {
        file << result.t[i] << ","
             << result.E_kin_body[i] << ","
             << result.E_kin_leg[i] << ","
             << result.E_g_body[i] << ","
             << result.E_g_leg[i] << ","
             << result.E_leg_spring[i] << ","
             << result.E_foot_spring_z[i] << ","
             << result.work_u1[i] << ","
             << result.work_u2[i] << ","
             << result.E_leg_damp[i] << ","
             << result.E_foot_damp_x[i] << ","
             << result.E_foot_damp_z[i] << ","
             << result.E_mechanical[i] << ","
             << result.E_loss[i] << ","
             << result.E_gain[i] << ","
             << result.E_net_flow[i] << ","
             << result.E_delta[i] << ","
             << result.E_error[i] << "\n";
    }

    std::cout << "Energy results exported to: " << filename << std::endl;
}

#endif // ENERGY_ANALYSIS_HPP
