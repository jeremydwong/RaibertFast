// test_implicit_cpu.cpp - CPU test of implicit midpoint integrator
//
// Verifies that the CUDA headers work correctly when compiled with a standard C++ compiler.
// This enables using the same physics code for both CPU verification and GPU parallel execution.
//
// Build: cl /std:c++17 /O2 test_implicit_cpu.cpp /Fe:test_implicit_cpu.exe
// Or:    clang++ -std=c++17 -O2 test_implicit_cpu.cpp -o test_implicit_cpu

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

// Include CUDA headers - these work on CPU due to compatibility layer
#include "cuda/hopper_types.cuh"
#include "cuda/hopper_cuda.cuh"
#include "cuda/integrator_cuda.cuh"

// Tolerance for floating-point comparisons (adapt to precision)
#ifdef HOPPER_USE_FLOAT32
constexpr Scalar TEST_TOL = 1e-5;
#else
constexpr Scalar TEST_TOL = TEST_TOL;
#endif

// ============================================================================
// SIMULATION RUNNER (CPU version using implicit midpoint)
// ============================================================================

struct SimResult {
    std::vector<Scalar> t;
    std::vector<HopperState> y;
    std::vector<int> fsm_state;
    std::vector<Scalar> u1;
    std::vector<Scalar> u2;
};

SimResult run_hopper_implicit(Scalar t_final, Scalar dt, HopperState y0,
                               const ControlParams& ctrl, const PhysicsParams& phys,
                               Scalar sample_rate = 1000.0) {
    SimResult result;

    Scalar t = 0.0;
    HopperState state = y0;
    Scalar last_sample_t = -1.0;
    Scalar dt_sample = 1.0 / sample_rate;

    int num_steps = static_cast<int>(t_final / dt) + 1;

    for (int step = 0; step < num_steps && t <= t_final; step++) {
        // Sample output
        bool should_sample = (t - last_sample_t >= dt_sample * 0.99) ||
                            (step == 0) ||
                            (t + dt > t_final);

        if (should_sample) {
            result.t.push_back(t);
            result.y.push_back(state);
            result.fsm_state.push_back(state.fsm_state);

            // Compute control for logging
            ControlOutput ctrl_out = compute_control(t, state, ctrl, phys);
            result.u1.push_back(ctrl_out.u1);
            result.u2.push_back(ctrl_out.u2);

            last_sample_t = t;
        }

        // Take one step using selected integrator
        hopper_step<HOPPER_INTEGRATOR>(state, t, ctrl, phys, dt);
        t += dt;
    }

    return result;
}

// ============================================================================
// EXPORT TRAJECTORY TO CSV
// ============================================================================

void export_trajectory(const char* filename, const SimResult& result) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s for writing\n", filename);
        return;
    }

    fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
               "ddt_x_foot,ddt_z_foot,ddt_phi_leg,ddt_phi_body,ddt_len_leg,"
               "fsm_state,u1,u2\n");

    for (size_t i = 0; i < result.t.size(); i++) {
        const HopperState& y = result.y[i];
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
// TESTS
// ============================================================================

int test_physics_params() {
    printf("Test: PhysicsParams initialization... ");
    PhysicsParams phys;

    if (fabs(phys.m - 10.0) > TEST_TOL ||
        fabs(phys.g - 9.8) > TEST_TOL ||
        fabs(phys.r_s0 - 1.0) > TEST_TOL) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_control_params() {
    printf("Test: ControlParams initialization... ");
    ControlParams ctrl;

    if (fabs(ctrl.k_fp - 150.0) > TEST_TOL ||
        fabs(ctrl.k_att - 150.0) > TEST_TOL) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_hopper_state() {
    printf("Test: HopperState default values... ");
    HopperState state;

    if (fabs(state.len_leg - 1.0) > TEST_TOL ||
        state.fsm_state != FSM_FLIGHT) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_5x5_solve() {
    printf("Test: 5x5 linear solve... ");

    // Simple test: identity matrix
    Scalar M[5][5] = {
        {1, 0, 0, 0, 0},
        {0, 2, 0, 0, 0},
        {0, 0, 3, 0, 0},
        {0, 0, 0, 4, 0},
        {0, 0, 0, 0, 5}
    };
    Scalar b[5] = {1, 2, 3, 4, 5};
    Scalar x[5];

    solve_5x5(M, b, x);

    Scalar expected[5] = {1, 1, 1, 1, 1};
    for (int i = 0; i < 5; i++) {
        if (fabs(x[i] - expected[i]) > TEST_TOL) {
            printf("FAILED (x[%d] = %f, expected %f)\n", i, x[i], expected[i]);
            return 1;
        }
    }
    printf("PASSED\n");
    return 0;
}

int test_compute_control_flight() {
    printf("Test: Control in flight phase... ");

    HopperState state;
    state.z_foot = 0.5;  // airborne
    state.fsm_state = FSM_FLIGHT;

    ControlParams ctrl;
    ctrl.x_dot_des = 3.0;

    PhysicsParams phys;

    ControlOutput out = compute_control(0.0, state, ctrl, phys);

    // In flight, u1 should be 0
    if (fabs(out.u1) > TEST_TOL) {
        printf("FAILED (u1 = %f, expected 0)\n", out.u1);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_compute_accelerations() {
    printf("Test: Dynamics accelerations... ");

    PhysicsParams phys;
    Scalar qdd[5];

    // State at rest, in air
    compute_accelerations(
        0.0, 1.0, 0.0, 0.0, 1.0,  // positions: foot at (0,1), vertical leg
        0.0, 0.0, 0.0, 0.0, 0.0,  // velocities: all zero
        0.0, 0.0,                  // control: no force/torque
        phys, qdd
    );

    // z acceleration should be negative (gravity)
    if (qdd[1] >= 0) {
        printf("FAILED (z_accel = %f, expected < 0)\n", qdd[1]);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_integrator_single_step() {
    printf("Test: Integrator single step... ");

    HopperState state;
    state.z_foot = 1.0;  // start high
    state.len_leg = 1.0;
    state.fsm_state = FSM_FLIGHT;

    ControlParams ctrl;
    PhysicsParams phys;

    Scalar z_before = state.z_foot;

    // Take one step using selected integrator
    hopper_step<HOPPER_INTEGRATOR>(state, 0.0, ctrl, phys, 0.001);

    // Foot should have moved down (falling)
    if (state.z_foot >= z_before) {
        printf("FAILED (z didn't decrease)\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

int test_full_simulation() {
    printf("Test: Full 1-second simulation... ");

    HopperState y0;
    y0.z_foot = 0.4;
    y0.phi_leg = 0.01;
    y0.len_leg = 1.0;
    y0.fsm_state = FSM_FLIGHT;

    ControlParams ctrl;
    ctrl.x_dot_des = 3.0;

    PhysicsParams phys;

    Scalar dt = 1e-4;
    SimResult result = run_hopper_implicit(1.0, dt, y0, ctrl, phys);

    if (result.t.empty()) {
        printf("FAILED (no output)\n");
        return 1;
    }

    // Should have traveled forward
    const HopperState& final_state = result.y.back();
    if (final_state.x_foot <= 0) {
        printf("FAILED (didn't move forward: x = %f)\n", final_state.x_foot);
        return 1;
    }

    printf("PASSED (x = %.2f m)\n", final_state.x_foot);
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("CPU Hopper Simulation\n");
    printf("(Using CUDA headers with compatibility layer)\n");
    printf("========================================\n\n");

    // Parse arguments
    bool run_tests = false;
    bool run_sim = false;
    bool run_multi = false;
    int num_hoppers = 128;
    Scalar t_final = 5.0;
    const char* output_file = "trajectory_cpu.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            run_tests = true;
        } else if (strcmp(argv[i], "--sim") == 0) {
            run_sim = true;
        } else if (strcmp(argv[i], "--multi") == 0) {
            run_multi = true;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_hoppers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            t_final = atof(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--test] [--sim] [--multi] [-n num] [-t time] [-o file]\n", argv[0]);
            printf("  --test   Run unit tests\n");
            printf("  --sim    Run single-hopper simulation\n");
            printf("  --multi  Run N hoppers sequentially (serial CPU baseline)\n");
            printf("  -n       Number of hoppers for --multi (default: 128)\n");
            printf("  -t       Simulation time (default: 5.0)\n");
            printf("  -o       Output file prefix (default: trajectory_cpu.csv)\n");
            return 0;
        }
    }

    // Default: run tests if no args
    if (!run_tests && !run_sim && !run_multi) {
        run_tests = true;
    }

    int failures = 0;

    if (run_tests) {
        printf("Running tests...\n\n");

        failures += test_physics_params();
        failures += test_control_params();
        failures += test_hopper_state();
        failures += test_5x5_solve();
        failures += test_compute_control_flight();
        failures += test_compute_accelerations();
        failures += test_integrator_single_step();
        failures += test_full_simulation();

        printf("\n========================================\n");
        if (failures == 0) {
            printf("All tests PASSED!\n");
        } else {
            printf("%d test(s) FAILED\n", failures);
        }
        printf("========================================\n");
    }

    if (run_sim) {
        printf("\nRunning simulation for %.2f seconds...\n", t_final);

        HopperState y0;
        y0.z_foot = 0.4;
        y0.phi_leg = 0.01;
        y0.len_leg = 1.0;
        y0.fsm_state = FSM_FLIGHT;

        ControlParams ctrl;
        ctrl.x_dot_des = 3.0;

        PhysicsParams phys;

        Scalar dt = 1e-4;

        auto start = std::chrono::high_resolution_clock::now();
        SimResult result = run_hopper_implicit(t_final, dt, y0, ctrl, phys);
        auto end = std::chrono::high_resolution_clock::now();

        Scalar elapsed = std::chrono::duration<Scalar>(end - start).count();

        printf("  Simulated time: %.4f s\n", result.t.back());
        printf("  Wall clock:     %.4f s\n", elapsed);
        printf("  Speedup:        %.1fx realtime\n", result.t.back() / elapsed);
        printf("  Data points:    %zu\n", result.t.size());

        const HopperState& final_state = result.y.back();
        printf("\nFinal state:\n");
        printf("  Position: x=%.4f m, z=%.4f m\n", final_state.x_foot, final_state.z_foot);
        printf("  Angles:   phi_leg=%.4f, phi_body=%.4f\n", final_state.phi_leg, final_state.phi_body);

        export_trajectory(output_file, result);
        printf("\nExported trajectory to: %s\n", output_file);
        printf("Visualize with: python src/visualize_cpp_trajectory.py %s\n", output_file);
    }

    if (run_multi) {
        printf("\nRunning %d hoppers sequentially (serial CPU) for %.2f seconds...\n", num_hoppers, t_final);

        PhysicsParams phys;
        Scalar dt = 1e-4;
        int num_steps = static_cast<int>(t_final / dt) + 1;

        // Create array of hopper states with varied initial conditions
        std::vector<HopperState> states(num_hoppers);
        std::vector<ControlParams> ctrls(num_hoppers);

        for (int i = 0; i < num_hoppers; i++) {
            states[i].z_foot = 0.4;
            states[i].phi_leg = 0.01;
            states[i].len_leg = 1.0;
            states[i].fsm_state = FSM_FLIGHT;
            // Vary desired velocity across hoppers
            ctrls[i].x_dot_des = 1.0 + 4.0 * i / (num_hoppers - 1);  // 1.0 to 5.0 m/s
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Simulate all hoppers sequentially
        for (int step = 0; step < num_steps; step++) {
            Scalar t = step * dt;
            for (int i = 0; i < num_hoppers; i++) {
                hopper_step<HOPPER_INTEGRATOR>(states[i], t, ctrls[i], phys, dt);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        Scalar elapsed = std::chrono::duration<Scalar>(end - start).count();

        printf("\n========================================\n");
        printf("Serial CPU Results:\n");
        printf("  Hoppers:        %d\n", num_hoppers);
        printf("  Sim time:       %.2f s\n", t_final);
        printf("  Wall clock:     %.4f s\n", elapsed);
        printf("  Total hopper-s: %.2f\n", t_final * num_hoppers);
        printf("  Throughput:     %.1f hopper-seconds/wall-second\n", (t_final * num_hoppers) / elapsed);
        printf("========================================\n");

        // Print sample of final states
        printf("\nSample final states:\n");
        for (int i = 0; i < num_hoppers; i += num_hoppers / 4) {
            printf("  Hopper %3d: x=%.2f m, x_dot_des=%.1f m/s\n",
                   i, states[i].x_foot, ctrls[i].x_dot_des);
        }
    }

    return failures;
}
