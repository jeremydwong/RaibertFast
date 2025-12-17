// build.cpp - Unity Build Entry Point for Raibert Hopper Simulation
//
// Compile with: ./build.sh
// Or directly:  clang++ -std=c++17 -O2 -I/path/to/eigen build.cpp -o hopper
//
// This is the Casey Muratori / Jonathan Blow style: single compilation unit.
// All code is included here, compiled together, fast builds.

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>
#include <chrono>

// External dependency: Eigen (header-only, for matrix solve)
#include <Eigen/Dense>

// Our code (all in headers for unity build)
#include "hopper.hpp"
#include "ode.hpp"

// ============================================================================
// SIMULATION RUNNER
// ============================================================================

struct SimulationResult {
    std::vector<double> t;
    std::vector<State> y;
    std::vector<int> fsm_state;
    std::vector<double> u1;
    std::vector<double> u2;
};

SimulationResult run_hopper_simulation(
    double t_start, double t_final,
    State y0, Parameters& p,
    double sample_rate = 1000.0
) {
    SimulationResult result;

    double t = t_start;
    State y = y0;

    RK45Config config;
    config.rtol = 1e-8;
    config.atol = 1e-8;
    config.max_step = 0.05;

    int segment = 0;
    while (t_final - t > 1e-6) {
        segment++;

        // Create dynamics and event functions bound to current parameters
        // (We copy p to avoid issues with changing fsm_state during integration)
        Parameters p_snapshot = p;

        auto dynamics = [p_snapshot](double time, const State& state) -> StateDot {
            return hopper_dynamics(time, state, p_snapshot);
        };

        auto event = [p_snapshot](double time, const State& state) -> double {
            return hopper_event(time, state, p_snapshot);
        };

        // Integrate until event or t_final
        auto seg_result = integrate_with_events<State, StateDot>(
            dynamics, event, t, t_final, y, config
        );

        // Store results (subsample to desired rate)
        double dt_sample = 1.0 / sample_rate;
        for (size_t i = 0; i < seg_result.t.size(); i++) {
            // Include first, last, and sampled points
            bool include = (i == 0) || (i == seg_result.t.size() - 1);
            if (!include && i > 0) {
                // Check if enough time has passed since last stored point
                if (result.t.empty() || seg_result.t[i] - result.t.back() >= dt_sample * 0.99) {
                    include = true;
                }
            }

            if (include) {
                result.t.push_back(seg_result.t[i]);
                result.y.push_back(seg_result.y[i]);
                result.fsm_state.push_back(p.fsm_state);

                // Compute control for logging
                auto ctrl = hopper_control(seg_result.t[i], seg_result.y[i], p);
                result.u1.push_back(ctrl.u1);
                result.u2.push_back(ctrl.u2);
            }
        }

        // Update for next segment
        if (seg_result.event_occurred) {
            // Update FSM timing
            if (p.fsm_state == Parameters::FSM_THRUST) {
                p.T_s = (seg_result.t_event - t) + p.T_compression;
            } else if (p.fsm_state == Parameters::FSM_COMPRESSION) {
                p.T_compression = seg_result.t_event - t;
                p.t_thrust_on = seg_result.t_event;
            }

            // Transition FSM state
            p.fsm_state = (p.fsm_state + 1) % Parameters::FSM_NUM_STATES;

            t = seg_result.t_event;
            y = seg_result.y_event;

            printf("  Segment %d: t=%.4f -> %.4f, event at t=%.4f, FSM -> %s\n",
                   segment, result.t.front(), t, t, fsm_state_name(p.fsm_state));
        } else {
            t = seg_result.t.back();
            y = seg_result.y.back();
            printf("  Segment %d: t=%.4f (no event, reached t_final)\n", segment, t);
        }
    }

    return result;
}

// ============================================================================
// EXPORT TRAJECTORY TO CSV
// ============================================================================

void export_trajectory(const char* filename, const SimulationResult& result) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s for writing\n", filename);
        return;
    }

    // Header
    fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
               "ddt_x_foot,ddt_z_foot,ddt_phi_leg,ddt_phi_body,ddt_len_leg,"
               "fsm_state,u1,u2\n");

    // Data
    for (size_t i = 0; i < result.t.size(); i++) {
        const State& y = result.y[i];
        fprintf(f, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                   "%.10e,%.10e,%.10e,%.10e,%.10e,"
                   "%d,%.10e,%.10e\n",
                result.t[i],
                y.x_foot, y.z_foot, y.phi_leg, y.phi_body, y.len_leg,
                y.ddt_x_foot, y.ddt_z_foot, y.ddt_phi_leg, y.ddt_phi_body, y.ddt_len_leg,
                result.fsm_state[i], result.u1[i], result.u2[i]);
    }

    fclose(f);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Raibert Hopper Simulation (C++)\n");
    printf("========================================\n\n");

    // Parse command line arguments
    double t_final = 5.0;
    double x_dot_des = 3.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            t_final = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            x_dot_des = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-t final_time] [-v desired_velocity]\n", argv[0]);
            printf("  -t  Simulation duration in seconds (default: 5.0)\n");
            printf("  -v  Desired forward velocity in m/s (default: 3.0)\n");
            return 0;
        }
    }

    // Set up parameters
    Parameters p;
    p.x_dot_des = x_dot_des;
    p.fsm_state = Parameters::FSM_FLIGHT;

    // Initial state: dropped from height with slight leg angle
    State y0;
    y0.x_foot = 0.0;
    y0.z_foot = 0.4;
    y0.phi_leg = 0.01;
    y0.phi_body = 0.0;
    y0.len_leg = 1.0;
    // Velocities default to 0

    printf("Configuration:\n");
    printf("  t_final = %.2f s\n", t_final);
    printf("  x_dot_des = %.2f m/s\n", x_dot_des);
    printf("  Initial state: z_foot=%.2f, phi_leg=%.3f\n\n", y0.z_foot, y0.phi_leg);

    // Run simulation
    printf("Running simulation...\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    SimulationResult result = run_hopper_simulation(0.0, t_final, y0, p);

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Summary
    printf("\n========================================\n");
    printf("Simulation Complete\n");
    printf("========================================\n");
    printf("  Simulated time:  %.4f s\n", result.t.back());
    printf("  Wall clock time: %.4f s\n", elapsed);
    printf("  Speedup:         %.1fx realtime\n", result.t.back() / elapsed);
    printf("  Data points:     %zu\n", result.t.size());

    if (!result.y.empty()) {
        const State& final_state = result.y.back();
        printf("\nFinal State:\n");
        printf("  Position: x=%.4f m, z=%.4f m\n", final_state.x_foot, final_state.z_foot);
        printf("  Angles:   phi_leg=%.4f rad, phi_body=%.4f rad\n",
               final_state.phi_leg, final_state.phi_body);
        printf("  Leg length: %.4f m\n", final_state.len_leg);

        // Compute body velocity
        double d_xbody_dt = final_state.ddt_x_foot
                         + final_state.ddt_len_leg * sin(final_state.phi_leg)
                         + final_state.len_leg * cos(final_state.phi_leg) * final_state.ddt_phi_leg
                         + p.l_2 * cos(final_state.phi_body) * final_state.ddt_phi_body;
        printf("  Body velocity: %.4f m/s (target: %.2f m/s)\n", d_xbody_dt, x_dot_des);
    }

    // Export trajectory
    const char* output_file = "trajectory.csv";
    export_trajectory(output_file, result);
    printf("\nExported trajectory to: %s\n", output_file);
    printf("Visualize with: python src/visualize_cpp_trajectory.py\n");

    return 0;
}
