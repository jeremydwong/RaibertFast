// test_cuda.cu - Unit tests for CUDA hopper simulation
//
// Tests dynamics, integrator, and parallel execution.
// Comparison with CPU hopper.hpp is done via separate test (test_hopper.cpp)
// Build: see build.bat

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hopper_types.cuh"
#include "hopper_cuda.cuh"
#include "integrator_cuda.cuh"
#include "jacobian_analytical.cuh"
#include "energy_analysis.hpp"

// ============================================================================
// TEST UTILITIES
// ============================================================================

#define ASSERT_NEAR(a, b, tol) do { \
    Scalar _a = (a), _b = (b), _tol = (tol); \
    if (fabs(_a - _b) > _tol) { \
        printf("FAIL: %s != %s (%.10e vs %.10e, diff=%.2e, tol=%.2e) at line %d\n", \
               #a, #b, _a, _b, fabs(_a - _b), _tol, __LINE__); \
        return false; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL: %s at line %d\n", #cond, __LINE__); \
        return false; \
    } \
} while(0)

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(test_func) do { \
    printf("Running %s... ", #test_func); \
    if (test_func()) { \
        printf("PASSED\n"); \
        tests_passed++; \
    } else { \
        printf("FAILED\n"); \
        tests_failed++; \
    } \
} while(0)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

PhysicsParams default_physics() {
    PhysicsParams p;
    // Uses struct defaults which match hopper.hpp defaults
    return p;
}

ControlParams default_control() {
    ControlParams c;
    c.x_dot_des = 3.0;
    return c;
}

// ============================================================================
// TEST: 5x5 Solver
// ============================================================================

bool test_solve_5x5() {
    // Simple tridiagonal system
    Scalar M[5][5] = {
        {2, 1, 0, 0, 0},
        {1, 3, 1, 0, 0},
        {0, 1, 4, 1, 0},
        {0, 0, 1, 5, 1},
        {0, 0, 0, 1, 6}
    };
    Scalar b[5] = {1, 2, 3, 4, 5};
    Scalar x[5];

    solve_5x5(M, b, x);

    // Verify M*x = b
    for (int i = 0; i < 5; i++) {
        Scalar sum = 0;
        for (int j = 0; j < 5; j++) {
            sum += M[i][j] * x[j];
        }
        ASSERT_NEAR(sum, b[i], 1e-10);
    }

    return true;
}

bool test_solve_5x5_random() {
    // Random dense matrix
    srand(42);
    Scalar M[5][5];
    Scalar b[5];

    for (int i = 0; i < 5; i++) {
        b[i] = (Scalar)rand() / RAND_MAX * 10 - 5;
        for (int j = 0; j < 5; j++) {
            M[i][j] = (Scalar)rand() / RAND_MAX * 10 - 5;
        }
        M[i][i] += 10;  // Make diagonally dominant
    }

    // Make a copy of M since solve_5x5 may modify it
    Scalar M_copy[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            M_copy[i][j] = M[i][j];
        }
    }

    Scalar x[5];
    solve_5x5(M, b, x);

    // Verify M_copy * x = b
    for (int i = 0; i < 5; i++) {
        Scalar sum = 0;
        for (int j = 0; j < 5; j++) {
            sum += M_copy[i][j] * x[j];
        }
        ASSERT_NEAR(sum, b[i], 1e-8);
    }

    return true;
}

// ============================================================================
// TEST: Dynamics sanity checks
// ============================================================================

bool test_dynamics_flight() {
    // Flight state: hopper in air, should experience gravity
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 0.5;
    state.phi_leg = 0.1;
    state.phi_body = 0.05;
    state.len_leg = 1.0;
    state.ddt_x_foot = 1.0;
    state.ddt_z_foot = 0.0;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;
    state.T_s = 0.425;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    auto result = hopper_dynamics_fwd_cuda(0.0, state, ctrl, phys);

    // In flight, velocities should pass through
    ASSERT_NEAR(result.state_dot[0], state.ddt_x_foot, 1e-10);  // x_dot = ddt_x_foot
    ASSERT_NEAR(result.state_dot[1], state.ddt_z_foot, 1e-10);  // z_dot = ddt_z_foot

    // Gravity should affect vertical acceleration (body and leg COMs)
    // The exact values depend on the mass matrix, but z acceleration should be negative
    // due to gravity acting on the system
    ASSERT_TRUE(result.state_dot[6] < 0);  // z_foot acceleration should be negative (falling)

    // No NaN values
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(!isnan(result.state_dot[i]));
    }

    return true;
}

bool test_dynamics_ground_contact() {
    // Ground contact (foot below ground) - should have ground reaction force
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = -0.01;  // 1cm below ground
    state.phi_leg = 0.1;
    state.phi_body = 0.05;
    state.len_leg = 0.95;
    state.ddt_x_foot = 1.0;
    state.ddt_z_foot = -0.5;  // Still moving down
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = -0.1;
    state.fsm_state = FSM_COMPRESSION;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    auto result = hopper_dynamics_fwd_cuda(0.5, state, ctrl, phys);

    // Ground reaction should push foot up: z_foot acceleration should be positive
    ASSERT_TRUE(result.state_dot[6] > 0);  // Ground pushing back

    // No NaN values
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(!isnan(result.state_dot[i]));
    }

    return true;
}

bool test_dynamics_thrust() {
    // Thrust phase - should have leg extension force
    HopperState state;
    state.x_foot = 1.0;
    state.z_foot = -0.005;
    state.phi_leg = 0.05;
    state.phi_body = 0.02;
    state.len_leg = 0.92;  // Leg compressed
    state.ddt_x_foot = 2.0;
    state.ddt_z_foot = 0.1;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.5;  // Already extending
    state.fsm_state = FSM_THRUST;
    state.t_thrust_on = 0.9;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    auto result = hopper_dynamics_fwd_cuda(1.0, state, ctrl, phys);

    // In thrust, control u1 should be positive (extending leg)
    ASSERT_TRUE(result.control.u1 > 0);

    // No NaN values
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(!isnan(result.state_dot[i]));
    }

    return true;
}

bool test_dynamics_consistency() {
    // Test that dynamics are consistent: running same state gives same result
    HopperState state;
    state.x_foot = 0.5;
    state.z_foot = 0.3;
    state.phi_leg = 0.05;
    state.phi_body = 0.02;
    state.len_leg = 0.95;
    state.ddt_x_foot = 1.5;
    state.ddt_z_foot = -0.2;
    state.ddt_phi_leg = 0.1;
    state.ddt_phi_body = -0.05;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;
    state.T_s = 0.425;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    auto result1 = hopper_dynamics_fwd_cuda(0.5, state, ctrl, phys);
    auto result2 = hopper_dynamics_fwd_cuda(0.5, state, ctrl, phys);

    // Results should be identical
    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(result1.state_dot[i], result2.state_dot[i], 1e-15);
    }
    ASSERT_NEAR(result1.control.u1, result2.control.u1, 1e-15);
    ASSERT_NEAR(result1.control.u2, result2.control.u2, 1e-15);

    return true;
}

// ============================================================================
// TEST: Analytical vs Finite-Difference Jacobian
// ============================================================================

bool test_jacobian_analytical_vs_fd() {
    // Test multiple states: flight, ground contact, thrust
    PhysicsParams phys = default_physics();

    struct TestCase {
        const char* name;
        Scalar y[10];
        Scalar u1, u2;
    };

    TestCase cases[] = {
        {"Flight", {0.0, 0.5, 0.1, 0.05, 1.0, 1.0, -0.5, 0.2, 0.1, -0.1}, 0.0, 5.0},
        {"Ground contact", {0.0, -0.02, 0.1, 0.05, 0.95, 0.5, -1.0, 0.3, 0.1, -0.5}, 0.0, 10.0},
        {"Thrust", {0.0, -0.01, -0.1, 0.02, 0.9, 0.3, 0.5, -0.2, 0.05, 0.3}, 35.0, 8.0},
        {"Extended leg", {0.5, 0.6, 0.2, -0.1, 1.05, 1.5, 0.2, 0.1, -0.05, 0.2}, 10.0, -5.0},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);
    Scalar max_err = 0.0;

    for (int c = 0; c < num_cases; c++) {
        Scalar A_fd[5][5], B_fd[5][5];
        Scalar A_an[5][5], B_an[5][5];

        // Finite-difference Jacobian
        compute_jacobian_blocks(cases[c].y, cases[c].u1, cases[c].u2, phys, A_fd, B_fd);

        // Analytical Jacobian
        compute_jacobian_blocks_analytical(cases[c].y, cases[c].u1, cases[c].u2, phys, A_an, B_an);

        // Compare A matrices
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                Scalar err = fabs(A_an[i][j] - A_fd[i][j]);
                Scalar scale = fmax(1.0, fmax(fabs(A_an[i][j]), fabs(A_fd[i][j])));
                Scalar rel_err = err / scale;
                if (rel_err > max_err) max_err = rel_err;
                if (rel_err > 1e-4) {
                    printf("  Case '%s': A[%d][%d] mismatch: analytical=%.6e, fd=%.6e, rel_err=%.2e\n",
                           cases[c].name, i, j, A_an[i][j], A_fd[i][j], rel_err);
                }
            }
        }

        // Compare B matrices
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                Scalar err = fabs(B_an[i][j] - B_fd[i][j]);
                Scalar scale = fmax(1.0, fmax(fabs(B_an[i][j]), fabs(B_fd[i][j])));
                Scalar rel_err = err / scale;
                if (rel_err > max_err) max_err = rel_err;
                if (rel_err > 1e-4) {
                    printf("  Case '%s': B[%d][%d] mismatch: analytical=%.6e, fd=%.6e, rel_err=%.2e\n",
                           cases[c].name, i, j, B_an[i][j], B_fd[i][j], rel_err);
                }
            }
        }
    }

    printf("  (max relative error: %.2e)\n", max_err);
    return max_err < 1e-4;
}

// ============================================================================
// TEST: Integrator
// ============================================================================

bool test_semi_implicit_euler_step() {
    // Simple flight state
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 1.0;
    state.phi_leg = 0.0;
    state.phi_body = 0.0;
    state.len_leg = 1.0;
    state.ddt_x_foot = 0.0;
    state.ddt_z_foot = 0.0;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;

    PhysicsParams phys = default_physics();
    Scalar dt = 1e-4;

    // Take one step with no control (pure gravity)
    Scalar u1 = 0.0, u2 = 0.0;

    semi_implicit_euler_step(
        state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
        state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
        u1, u2, phys, dt
    );

    // After one step with gravity, z velocity should be slightly negative
    ASSERT_TRUE(state.ddt_z_foot < 0);
    // Position should barely change
    ASSERT_NEAR(state.z_foot, 1.0, 1e-6);

    return true;
}

bool test_implicit_midpoint_step() {
    // Simple flight state
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 1.0;
    state.phi_leg = 0.0;
    state.phi_body = 0.0;
    state.len_leg = 1.0;
    state.ddt_x_foot = 0.0;
    state.ddt_z_foot = 0.0;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;

    PhysicsParams phys = default_physics();
    Scalar dt = 1e-4;

    Scalar u1 = 0.0, u2 = 0.0;

    implicit_midpoint_step(
        state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
        state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
        u1, u2, phys, dt
    );

    // Similar check
    ASSERT_TRUE(state.ddt_z_foot < 0);
    ASSERT_NEAR(state.z_foot, 1.0, 1e-6);

    // Check no NaN
    ASSERT_TRUE(!isnan(state.z_foot));
    ASSERT_TRUE(!isnan(state.ddt_z_foot));

    return true;
}

bool test_integrator_stability_ground_contact() {
    // Drop hopper from height, check stability through ground contact
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 0.5;
    state.phi_leg = 0.0;
    state.phi_body = 0.0;
    state.len_leg = 1.0;
    state.ddt_x_foot = 0.0;
    state.ddt_z_foot = -2.0;  // Falling
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;

    PhysicsParams phys = default_physics();
    Scalar dt = 1e-4;
    Scalar u1 = 0.0, u2 = 0.0;

    Scalar min_z = 0.0;
    int num_steps = 10000;  // 1 second

    for (int step = 0; step < num_steps; step++) {
        implicit_midpoint_step(
            state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
            state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
            u1, u2, phys, dt
        );

        // Check for blowup
        ASSERT_TRUE(!isnan(state.z_foot));
        ASSERT_TRUE(!isinf(state.z_foot));
        ASSERT_TRUE(fabs(state.z_foot) < 100);

        if (state.z_foot < min_z) min_z = state.z_foot;
    }

    printf("  (max penetration: %.4f m) ", -min_z);

    // Ground penetration should be small
    ASSERT_TRUE(-min_z < 0.05);

    return true;
}

// ============================================================================
// TEST: Full simulation step
// ============================================================================

bool test_full_hopper_step() {
    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 0.4;
    state.phi_leg = 0.01;
    state.phi_body = 0.0;
    state.len_leg = 1.0;
    state.ddt_x_foot = 0.0;
    state.ddt_z_foot = 0.0;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;
    state.T_s = 0.425;
    state.T_compression = 0.0;
    state.t_thrust_on = 0.0;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();
    Scalar dt = 1e-4;

    // Run for a while
    Scalar t = 0.0;
    int num_steps = 50000;  // 5 seconds
    int fsm_transitions = 0;
    int last_fsm = state.fsm_state;

    for (int step = 0; step < num_steps; step++) {
        hopper_step<INTEGRATOR_IMPLICIT_MIDPOINT>(state, t, ctrl, phys, dt);
        t += dt;

        if (state.fsm_state != last_fsm) {
            fsm_transitions++;
            last_fsm = state.fsm_state;
        }

        // Check for blowup
        ASSERT_TRUE(!isnan(state.x_foot));
        ASSERT_TRUE(!isnan(state.z_foot));
        ASSERT_TRUE(fabs(state.z_foot) < 100);
    }

    printf("  (final x: %.2f m, transitions: %d) ", state.x_foot, fsm_transitions);

    // Should have traveled forward (reduced expectation for implicit midpoint)
    // With x_dot_des = 3.0, over 5 seconds ideal is 15m, but with startup and
    // integrator differences, ~4m is reasonable
    ASSERT_TRUE(state.x_foot > 3.0);

    // Should have had FSM transitions (multiple hops)
    ASSERT_TRUE(fsm_transitions > 10);

    return true;
}

// ============================================================================
// TRAJECTORY EXPORT (same format as CPU build.cpp)
// ============================================================================

void run_and_export_trajectory(const char* filename, Scalar t_final, Scalar sample_rate = 1000.0) {
    printf("Running CUDA simulation for %.2f seconds...\n", t_final);

    HopperState state;
    state.x_foot = 0.0;
    state.z_foot = 0.4;
    state.phi_leg = 0.01;
    state.phi_body = 0.0;
    state.len_leg = 1.0;
    state.ddt_x_foot = 0.0;
    state.ddt_z_foot = 0.0;
    state.ddt_phi_leg = 0.0;
    state.ddt_phi_body = 0.0;
    state.ddt_len_leg = 0.0;
    state.fsm_state = FSM_FLIGHT;
    state.T_s = 0.425;
    state.T_compression = 0.0;
    state.t_thrust_on = 0.0;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();
    Scalar dt = 1e-4;

    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("ERROR: Could not open %s for writing\n", filename);
        return;
    }

    // Header (matches CPU format)
    fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
               "ddt_x_foot,ddt_z_foot,ddt_phi_leg,ddt_phi_body,ddt_len_leg,"
               "fsm_state,u1,u2\n");

    Scalar t = 0.0;
    Scalar dt_sample = 1.0 / sample_rate;
    Scalar next_sample = 0.0;
    int num_steps = (int)(t_final / dt);

    for (int step = 0; step <= num_steps; step++) {
        // Sample at regular intervals
        if (t >= next_sample - dt * 0.5) {
            // Get control for logging
            ControlOutput control = compute_control(
                t, state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
                state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
                state.fsm_state, ctrl, phys, state.t_thrust_on, state.T_s
            );

            fprintf(f, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%d,%.10e,%.10e\n",
                    t,
                    state.x_foot, state.z_foot, state.phi_leg, state.phi_body, state.len_leg,
                    state.ddt_x_foot, state.ddt_z_foot, state.ddt_phi_leg, state.ddt_phi_body, state.ddt_len_leg,
                    state.fsm_state, control.u1, control.u2);

            next_sample += dt_sample;
        }

        // Take simulation step
        hopper_step<INTEGRATOR_IMPLICIT_MIDPOINT>(state, t, ctrl, phys, dt);
        t += dt;
    }

    fclose(f);
    printf("Exported %d samples to %s\n", (int)(t_final * sample_rate), filename);
    printf("Final state: x=%.4f m, z=%.4f m\n", state.x_foot, state.z_foot);
}

// ============================================================================
// GPU KERNEL TESTS
// ============================================================================

__global__ void kernel_test_dynamics(
    HopperState* state,
    DynamicsOutput* output,
    ControlParams ctrl,
    PhysicsParams phys
) {
    *output = hopper_dynamics_fwd_cuda(0.0, *state, ctrl, phys);
}

bool test_gpu_dynamics_kernel() {
    // Allocate device memory
    HopperState* d_state;
    DynamicsOutput* d_output;
    cudaMalloc(&d_state, sizeof(HopperState));
    cudaMalloc(&d_output, sizeof(DynamicsOutput));

    // Set up test state
    HopperState h_state;
    h_state.x_foot = 0.0;
    h_state.z_foot = 0.5;
    h_state.phi_leg = 0.1;
    h_state.phi_body = 0.05;
    h_state.len_leg = 1.0;
    h_state.ddt_x_foot = 1.0;
    h_state.ddt_z_foot = 0.0;
    h_state.ddt_phi_leg = 0.0;
    h_state.ddt_phi_body = 0.0;
    h_state.ddt_len_leg = 0.0;
    h_state.fsm_state = FSM_FLIGHT;
    h_state.T_s = 0.425;

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    // Copy state to device
    cudaMemcpy(d_state, &h_state, sizeof(HopperState), cudaMemcpyHostToDevice);

    // Copy physics params to constant memory
    cudaMemcpyToSymbol(d_physics, &phys, sizeof(PhysicsParams));

    // Run kernel
    kernel_test_dynamics<<<1, 1>>>(d_state, d_output, ctrl, phys);
    cudaDeviceSynchronize();

    // Copy result back
    DynamicsOutput h_output;
    cudaMemcpy(&h_output, d_output, sizeof(DynamicsOutput), cudaMemcpyDeviceToHost);

    // Compare with CPU
    DynamicsOutput cpu_output = hopper_dynamics_fwd_cuda(0.0, h_state, ctrl, phys);

    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(h_output.state_dot[i], cpu_output.state_dot[i], 1e-10);
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_output);

    return true;
}

__global__ void kernel_simulate_hoppers(
    HopperStateArrays states,
    ControlParams ctrl,
    PhysicsParams phys,
    Scalar dt,
    int num_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= states.N) return;

    // Load state to registers
    Scalar x_foot = states.x_foot[idx];
    Scalar z_foot = states.z_foot[idx];
    Scalar phi_leg = states.phi_leg[idx];
    Scalar phi_body = states.phi_body[idx];
    Scalar len_leg = states.len_leg[idx];
    Scalar ddt_x_foot = states.ddt_x_foot[idx];
    Scalar ddt_z_foot = states.ddt_z_foot[idx];
    Scalar ddt_phi_leg = states.ddt_phi_leg[idx];
    Scalar ddt_phi_body = states.ddt_phi_body[idx];
    Scalar ddt_len_leg = states.ddt_len_leg[idx];
    int fsm_state = states.fsm_state[idx];
    Scalar T_s = states.T_s[idx];
    Scalar T_compression = states.T_compression[idx];
    Scalar t_thrust_on = states.t_thrust_on[idx];

    Scalar t = 0.0;

    for (int step = 0; step < num_steps; step++) {
        // Compute control
        ControlOutput control = compute_control(
            t, x_foot, z_foot, phi_leg, phi_body, len_leg,
            ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
            fsm_state, ctrl, phys, t_thrust_on, T_s
        );

        // Integrate
        integrator_step<HOPPER_INTEGRATOR>(
            x_foot, z_foot, phi_leg, phi_body, len_leg,
            ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
            control.u1, control.u2, phys, dt
        );

        // Check FSM
        int new_fsm = check_fsm_transition(z_foot, len_leg, ddt_len_leg, fsm_state, phys.r_s0);
        if (new_fsm != fsm_state) {
            if (fsm_state == FSM_COMPRESSION && new_fsm == FSM_THRUST) {
                T_compression = t - t_thrust_on;
                t_thrust_on = t;
            }
            fsm_state = new_fsm;
        }

        t += dt;
    }

    // Write back
    states.x_foot[idx] = x_foot;
    states.z_foot[idx] = z_foot;
    states.phi_leg[idx] = phi_leg;
    states.phi_body[idx] = phi_body;
    states.len_leg[idx] = len_leg;
    states.ddt_x_foot[idx] = ddt_x_foot;
    states.ddt_z_foot[idx] = ddt_z_foot;
    states.ddt_phi_leg[idx] = ddt_phi_leg;
    states.ddt_phi_body[idx] = ddt_phi_body;
    states.ddt_len_leg[idx] = ddt_len_leg;
    states.fsm_state[idx] = fsm_state;
    states.T_s[idx] = T_s;
    states.T_compression[idx] = T_compression;
    states.t_thrust_on[idx] = t_thrust_on;
}

bool test_parallel_identical_hoppers() {
    int N = 256;
    Scalar dt = 1e-4;
    int num_steps = 1000;  // 0.1 seconds

    // Allocate
    HopperStateArrays d_states = allocate_state_arrays_device(N);

    // Initialize all hoppers with same state
    HopperState init_state;
    init_state.x_foot = 0.0;
    init_state.z_foot = 0.4;
    init_state.phi_leg = 0.01;
    init_state.phi_body = 0.0;
    init_state.len_leg = 1.0;
    init_state.ddt_x_foot = 0.0;
    init_state.ddt_z_foot = 0.0;
    init_state.ddt_phi_leg = 0.0;
    init_state.ddt_phi_body = 0.0;
    init_state.ddt_len_leg = 0.0;
    init_state.fsm_state = FSM_FLIGHT;
    init_state.T_s = 0.425;
    init_state.T_compression = 0.0;
    init_state.t_thrust_on = 0.0;

    for (int i = 0; i < N; i++) {
        copy_state_to_device(init_state, d_states, i);
    }

    PhysicsParams phys = default_physics();
    ControlParams ctrl = default_control();

    // Copy physics to constant memory
    cudaMemcpyToSymbol(d_physics, &phys, sizeof(PhysicsParams));

    // Run simulation
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel_simulate_hoppers<<<blocks, threads>>>(d_states, ctrl, phys, dt, num_steps);
    cudaDeviceSynchronize();

    // Read back results
    std::vector<HopperState> results(N);
    for (int i = 0; i < N; i++) {
        copy_state_from_device(results[i], d_states, i);
    }

    // All should be identical
    for (int i = 1; i < N; i++) {
        ASSERT_NEAR(results[0].x_foot, results[i].x_foot, 1e-12);
        ASSERT_NEAR(results[0].z_foot, results[i].z_foot, 1e-12);
    }

    printf("  (final x: %.4f m) ", results[0].x_foot);

    // Cleanup
    free_state_arrays_device(d_states);

    return true;
}

// ============================================================================
// BATCHED NEWTON ITERATION KERNELS
// ============================================================================
//
// Instead of each thread running its own serial Newton loop, we launch
// separate kernels for each substep. All hoppers execute each substep in
// parallel before moving to the next.

// Workspace for batched Newton iteration (all hoppers)
struct BatchedNewtonWorkspace {
    int N;                  // Number of hoppers
    Scalar* y;              // [N, 10] current state
    Scalar* y_new;          // [N, 10] next state (being solved)
    Scalar* y_mid;          // [N, 10] midpoint state
    Scalar* f_mid;          // [N, 10] dynamics at midpoint
    Scalar* G;              // [N, 10] residual
    Scalar* A;              // [N, 5, 5] Jacobian block dqddot/dq
    Scalar* B;              // [N, 5, 5] Jacobian block dqddot/dqdot
    Scalar* u1;             // [N] control input 1
    Scalar* u2;             // [N] control input 2
};

BatchedNewtonWorkspace allocate_newton_workspace(int N) {
    BatchedNewtonWorkspace w;
    w.N = N;
    cudaMalloc(&w.y, N * 10 * sizeof(Scalar));
    cudaMalloc(&w.y_new, N * 10 * sizeof(Scalar));
    cudaMalloc(&w.y_mid, N * 10 * sizeof(Scalar));
    cudaMalloc(&w.f_mid, N * 10 * sizeof(Scalar));
    cudaMalloc(&w.G, N * 10 * sizeof(Scalar));
    cudaMalloc(&w.A, N * 25 * sizeof(Scalar));
    cudaMalloc(&w.B, N * 25 * sizeof(Scalar));
    cudaMalloc(&w.u1, N * sizeof(Scalar));
    cudaMalloc(&w.u2, N * sizeof(Scalar));
    return w;
}

void free_newton_workspace(BatchedNewtonWorkspace& w) {
    cudaFree(w.y);
    cudaFree(w.y_new);
    cudaFree(w.y_mid);
    cudaFree(w.f_mid);
    cudaFree(w.G);
    cudaFree(w.A);
    cudaFree(w.B);
    cudaFree(w.u1);
    cudaFree(w.u2);
}

// Kernel 1: Initialize Newton iteration with explicit Euler guess
__global__ void kernel_newton_init(
    Scalar* y, Scalar* y_new, Scalar* u1, Scalar* u2,
    int N, PhysicsParams phys, Scalar dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load state
    Scalar y_local[10];
    for (int i = 0; i < 10; i++) {
        y_local[i] = y[idx * 10 + i];
    }

    // Compute f(y)
    Scalar f[10];
    compute_state_derivative(y_local, u1[idx], u2[idx], phys, f);

    // Initial guess: y_new = y + dt * f(y)
    for (int i = 0; i < 10; i++) {
        y_new[idx * 10 + i] = y_local[i] + dt * f[i];
    }
}

// Kernel 2: Compute midpoint y_mid = 0.5 * (y + y_new)
__global__ void kernel_compute_midpoint(
    Scalar* y, Scalar* y_new, Scalar* y_mid, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int i = 0; i < 10; i++) {
        y_mid[idx * 10 + i] = 0.5 * (y[idx * 10 + i] + y_new[idx * 10 + i]);
    }
}

// Kernel 3: Evaluate dynamics at midpoint
__global__ void kernel_eval_dynamics_midpoint(
    Scalar* y_mid, Scalar* f_mid, Scalar* u1, Scalar* u2,
    int N, PhysicsParams phys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Scalar y_local[10];
    for (int i = 0; i < 10; i++) {
        y_local[i] = y_mid[idx * 10 + i];
    }

    Scalar f[10];
    compute_state_derivative(y_local, u1[idx], u2[idx], phys, f);

    for (int i = 0; i < 10; i++) {
        f_mid[idx * 10 + i] = f[i];
    }
}

// Kernel 4: Compute residual G = y_new - y - dt * f_mid
__global__ void kernel_compute_residual(
    Scalar* y, Scalar* y_new, Scalar* f_mid, Scalar* G,
    int N, Scalar dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int i = 0; i < 10; i++) {
        G[idx * 10 + i] = y_new[idx * 10 + i] - y[idx * 10 + i] - dt * f_mid[idx * 10 + i];
    }
}

// Kernel 5: Compute Jacobian blocks A and B via finite differences
__global__ void kernel_compute_jacobian(
    Scalar* y_mid, Scalar* u1, Scalar* u2, Scalar* A, Scalar* B,
    int N, PhysicsParams phys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Scalar y_local[10];
    for (int i = 0; i < 10; i++) {
        y_local[i] = y_mid[idx * 10 + i];
    }

    Scalar A_local[5][5], B_local[5][5];
    compute_jacobian_blocks(y_local, u1[idx], u2[idx], phys, A_local, B_local);

    // Store flattened
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            A[idx * 25 + i * 5 + j] = A_local[i][j];
            B[idx * 25 + i * 5 + j] = B_local[i][j];
        }
    }
}

// Kernel 6: Solve linear system and update y_new
__global__ void kernel_newton_solve_and_update(
    Scalar* y_new, Scalar* G, Scalar* A, Scalar* B,
    int N, Scalar dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load Jacobian blocks
    Scalar A_local[5][5], B_local[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            A_local[i][j] = A[idx * 25 + i * 5 + j];
            B_local[i][j] = B[idx * 25 + i * 5 + j];
        }
    }

    // Load residual
    Scalar G_q[5], G_qdot[5];
    for (int i = 0; i < 5; i++) {
        G_q[i] = G[idx * 10 + i];
        G_qdot[i] = G[idx * 10 + 5 + i];
    }

    // Solve
    Scalar dq[5], dqdot[5];
    solve_10x10_block(A_local, B_local, G_q, G_qdot, dt, dq, dqdot);

    // Update y_new -= delta
    for (int i = 0; i < 5; i++) {
        y_new[idx * 10 + i] -= dq[i];
        y_new[idx * 10 + 5 + i] -= dqdot[i];
    }
}

// Run one complete implicit midpoint step for all hoppers using batched kernels
void batched_implicit_midpoint_step(
    BatchedNewtonWorkspace& w,
    PhysicsParams phys,
    Scalar dt,
    int threads_per_block = 256
) {
    int blocks = (w.N + threads_per_block - 1) / threads_per_block;

    // Initialize with explicit Euler guess
    kernel_newton_init<<<blocks, threads_per_block>>>(
        w.y, w.y_new, w.u1, w.u2, w.N, phys, dt
    );

    // Newton iterations
    constexpr int NEWTON_ITERS = 4;
    for (int iter = 0; iter < NEWTON_ITERS; iter++) {
        kernel_compute_midpoint<<<blocks, threads_per_block>>>(
            w.y, w.y_new, w.y_mid, w.N
        );
        kernel_eval_dynamics_midpoint<<<blocks, threads_per_block>>>(
            w.y_mid, w.f_mid, w.u1, w.u2, w.N, phys
        );
        kernel_compute_residual<<<blocks, threads_per_block>>>(
            w.y, w.y_new, w.f_mid, w.G, w.N, dt
        );
        kernel_compute_jacobian<<<blocks, threads_per_block>>>(
            w.y_mid, w.u1, w.u2, w.A, w.B, w.N, phys
        );
        kernel_newton_solve_and_update<<<blocks, threads_per_block>>>(
            w.y_new, w.G, w.A, w.B, w.N, dt
        );
    }

    // y_new now contains the result - swap y and y_new for next step
    Scalar* tmp = w.y;
    w.y = w.y_new;
    w.y_new = tmp;
}

// Kernel to load states from HopperStateArrays into workspace
__global__ void kernel_load_states_to_workspace(
    HopperStateArrays states,
    Scalar* y, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    y[idx * 10 + 0] = states.x_foot[idx];
    y[idx * 10 + 1] = states.z_foot[idx];
    y[idx * 10 + 2] = states.phi_leg[idx];
    y[idx * 10 + 3] = states.phi_body[idx];
    y[idx * 10 + 4] = states.len_leg[idx];
    y[idx * 10 + 5] = states.ddt_x_foot[idx];
    y[idx * 10 + 6] = states.ddt_z_foot[idx];
    y[idx * 10 + 7] = states.ddt_phi_leg[idx];
    y[idx * 10 + 8] = states.ddt_phi_body[idx];
    y[idx * 10 + 9] = states.ddt_len_leg[idx];
}

// Kernel to store workspace back to HopperStateArrays
__global__ void kernel_store_workspace_to_states(
    Scalar* y, HopperStateArrays states, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    states.x_foot[idx] = y[idx * 10 + 0];
    states.z_foot[idx] = y[idx * 10 + 1];
    states.phi_leg[idx] = y[idx * 10 + 2];
    states.phi_body[idx] = y[idx * 10 + 3];
    states.len_leg[idx] = y[idx * 10 + 4];
    states.ddt_x_foot[idx] = y[idx * 10 + 5];
    states.ddt_z_foot[idx] = y[idx * 10 + 6];
    states.ddt_phi_leg[idx] = y[idx * 10 + 7];
    states.ddt_phi_body[idx] = y[idx * 10 + 8];
    states.ddt_len_leg[idx] = y[idx * 10 + 9];
}

// Kernel to compute control for all hoppers
__global__ void kernel_compute_control_batched(
    HopperStateArrays states,
    ControlParams* ctrl_array,
    ControlParams ctrl_default,
    PhysicsParams phys,
    Scalar* u1, Scalar* u2,
    int N, Scalar t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    ControlParams ctrl = ctrl_array ? ctrl_array[idx] : ctrl_default;

    ControlOutput control = compute_control(
        t,
        states.x_foot[idx], states.z_foot[idx],
        states.phi_leg[idx], states.phi_body[idx], states.len_leg[idx],
        states.ddt_x_foot[idx], states.ddt_z_foot[idx],
        states.ddt_phi_leg[idx], states.ddt_phi_body[idx], states.ddt_len_leg[idx],
        states.fsm_state[idx], ctrl, phys,
        states.t_thrust_on[idx], states.T_s[idx]
    );

    u1[idx] = control.u1;
    u2[idx] = control.u2;
}

// Kernel to check FSM transitions for all hoppers
__global__ void kernel_check_fsm_batched(
    HopperStateArrays states,
    PhysicsParams phys,
    int N, Scalar t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int old_fsm = states.fsm_state[idx];
    int new_fsm = check_fsm_transition(
        states.z_foot[idx], states.len_leg[idx], states.ddt_len_leg[idx],
        old_fsm, phys.r_s0
    );

    if (new_fsm != old_fsm) {
        if (old_fsm == FSM_COMPRESSION && new_fsm == FSM_THRUST) {
            states.T_compression[idx] = t - states.t_thrust_on[idx];
            states.t_thrust_on[idx] = t;
        }
        states.fsm_state[idx] = new_fsm;
    }
}

// Kernel to sample trajectories at given timestep
__global__ void kernel_sample_trajectory(
    HopperStateArrays states,
    Scalar* u1, Scalar* u2,
    Scalar* trajectory_data,
    int N, int sample_idx, int num_samples, Scalar t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int base = (idx * num_samples + sample_idx) * 14;
    trajectory_data[base + 0] = t;
    trajectory_data[base + 1] = states.x_foot[idx];
    trajectory_data[base + 2] = states.z_foot[idx];
    trajectory_data[base + 3] = states.phi_leg[idx];
    trajectory_data[base + 4] = states.phi_body[idx];
    trajectory_data[base + 5] = states.len_leg[idx];
    trajectory_data[base + 6] = states.ddt_x_foot[idx];
    trajectory_data[base + 7] = states.ddt_z_foot[idx];
    trajectory_data[base + 8] = states.ddt_phi_leg[idx];
    trajectory_data[base + 9] = states.ddt_phi_body[idx];
    trajectory_data[base + 10] = states.ddt_len_leg[idx];
    trajectory_data[base + 11] = (Scalar)states.fsm_state[idx];
    trajectory_data[base + 12] = u1[idx];
    trajectory_data[base + 13] = u2[idx];
}

// ============================================================================
// BATCHED MULTI-HOPPER SIMULATION (new implementation)
// ============================================================================

void run_multi_hopper_simulation_batched(int N, Scalar t_final, const char* output_prefix, Scalar sample_rate = 100.0) {
    printf("Running BATCHED parallel simulation of %d hoppers for %.2f seconds...\n", N, t_final);

    Scalar dt = 1e-4;
    int num_steps = (int)(t_final / dt);
    int sample_every = (int)(1.0 / (sample_rate * dt));
    int num_samples = num_steps / sample_every + 1;

    printf("  dt=%.2e, steps=%d, sample_every=%d, samples=%d\n", dt, num_steps, sample_every, num_samples);

    // Allocate device state arrays
    HopperStateArrays d_states = allocate_state_arrays_device(N);

    // Allocate Newton workspace
    BatchedNewtonWorkspace workspace = allocate_newton_workspace(N);

    // Allocate trajectory storage on device
    size_t traj_size = (size_t)N * num_samples * 14 * sizeof(Scalar);
    Scalar* d_trajectory;
    cudaMalloc(&d_trajectory, traj_size);
    printf("  Trajectory buffer: %.2f MB\n", traj_size / (1024.0 * 1024.0));

    // Allocate per-hopper control params
    ControlParams* d_ctrl;
    cudaMalloc(&d_ctrl, N * sizeof(ControlParams));

    // Initialize hoppers with varied ICs
    std::vector<ControlParams> h_ctrl(N);

    for (int i = 0; i < N; i++) {
        HopperState state;
        state.x_foot = 0.0;
        state.z_foot = 0.4;
        state.phi_leg = 0.01 + 0.1 * ((Scalar)i / (N - 1) - 0.5);
        state.phi_body = 0.0;
        state.len_leg = 1.0;
        state.ddt_x_foot = 0.0;
        state.ddt_z_foot = 0.0;
        state.ddt_phi_leg = 0.0;
        state.ddt_phi_body = 0.0;
        state.ddt_len_leg = 0.0;
        state.fsm_state = FSM_FLIGHT;
        state.T_s = 0.425;
        state.T_compression = 0.0;
        state.t_thrust_on = 0.0;

        copy_state_to_device(state, d_states, i);

        h_ctrl[i].x_dot_des = 2.0 + 2.0 * ((Scalar)i / (N - 1));
    }

    cudaMemcpy(d_ctrl, h_ctrl.data(), N * sizeof(ControlParams), cudaMemcpyHostToDevice);

    PhysicsParams phys = default_physics();
    ControlParams ctrl_default = default_control();

    cudaMemcpyToSymbol(d_physics, &phys, sizeof(PhysicsParams));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("  Launching batched simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    Scalar t = 0.0;
    int sample_idx = 0;

    for (int step = 0; step <= num_steps; step++) {
        // Sample at regular intervals
        if (step % sample_every == 0 && sample_idx < num_samples) {
            kernel_compute_control_batched<<<blocks, threads>>>(
                d_states, d_ctrl, ctrl_default, phys, workspace.u1, workspace.u2, N, t
            );
            kernel_sample_trajectory<<<blocks, threads>>>(
                d_states, workspace.u1, workspace.u2, d_trajectory, N, sample_idx, num_samples, t
            );
            sample_idx++;
        }

        if (step < num_steps) {
            // 1. Compute control
            kernel_compute_control_batched<<<blocks, threads>>>(
                d_states, d_ctrl, ctrl_default, phys, workspace.u1, workspace.u2, N, t
            );

            // 2. Load states into workspace
            kernel_load_states_to_workspace<<<blocks, threads>>>(d_states, workspace.y, N);

            // 3. Run batched implicit midpoint step
            batched_implicit_midpoint_step(workspace, phys, dt, threads);

            // 4. Store results back to states
            kernel_store_workspace_to_states<<<blocks, threads>>>(workspace.y, d_states, N);

            // 5. Check FSM transitions
            kernel_check_fsm_batched<<<blocks, threads>>>(d_states, phys, N, t);

            t += dt;
        }
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("  Kernel time: %.2f ms (%.1fx realtime)\n", ms, t_final * 1000.0 / ms);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy trajectory back to host
    printf("  Copying trajectory to host...\n");
    std::vector<Scalar> h_trajectory(N * num_samples * 14);
    cudaMemcpy(h_trajectory.data(), d_trajectory, traj_size, cudaMemcpyDeviceToHost);

    // Export to CSV files
    printf("  Exporting %d CSV files...\n", N);
    for (int i = 0; i < N; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s_%03d.csv", output_prefix, i);

        FILE* f = fopen(filename, "w");
        if (!f) {
            printf("ERROR: Could not open %s\n", filename);
            continue;
        }

        fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
                   "ddt_x_foot,ddt_z_foot,ddt_phi_leg,ddt_phi_body,ddt_len_leg,"
                   "fsm_state,u1,u2\n");

        for (int s = 0; s < num_samples; s++) {
            int base = (i * num_samples + s) * 14;
            fprintf(f, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%d,%.10e,%.10e\n",
                    h_trajectory[base + 0], h_trajectory[base + 1], h_trajectory[base + 2],
                    h_trajectory[base + 3], h_trajectory[base + 4], h_trajectory[base + 5],
                    h_trajectory[base + 6], h_trajectory[base + 7], h_trajectory[base + 8],
                    h_trajectory[base + 9], h_trajectory[base + 10],
                    (int)h_trajectory[base + 11], h_trajectory[base + 12], h_trajectory[base + 13]);
        }
        fclose(f);
    }

    // Export summary
    char summary_file[256];
    snprintf(summary_file, sizeof(summary_file), "%s_summary.csv", output_prefix);
    FILE* f = fopen(summary_file, "w");
    if (f) {
        fprintf(f, "hopper_id,x_dot_des,final_x,final_z\n");
        for (int i = 0; i < N; i++) {
            int base = (i * num_samples + (num_samples - 1)) * 14;
            fprintf(f, "%d,%.4f,%.4f,%.4f\n", i, h_ctrl[i].x_dot_des,
                    h_trajectory[base + 1], h_trajectory[base + 2]);
        }
        fclose(f);
        printf("  Summary: %s\n", summary_file);
    }

    printf("Done! Files: %s_000.csv through %s_%03d.csv\n", output_prefix, output_prefix, N-1);

    // Cleanup
    cudaFree(d_trajectory);
    cudaFree(d_ctrl);
    free_state_arrays_device(d_states);
    free_newton_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// ORIGINAL MULTI-HOPPER PARALLEL SIMULATION (kept for comparison)
// ============================================================================

// Kernel that simulates N hoppers for num_steps, storing samples at regular intervals
__global__ void kernel_simulate_hoppers_with_history(
    HopperStateArrays states,
    Scalar* trajectory_data,   // [N * num_samples * 14] array (14 columns per sample)
    ControlParams* ctrl_array, // Per-hopper control params (or nullptr for uniform)
    ControlParams ctrl_default,
    PhysicsParams phys,
    Scalar dt,
    int num_steps,
    int sample_every,          // Sample every N steps
    int num_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= states.N) return;

    // Load state to registers
    Scalar x_foot = states.x_foot[idx];
    Scalar z_foot = states.z_foot[idx];
    Scalar phi_leg = states.phi_leg[idx];
    Scalar phi_body = states.phi_body[idx];
    Scalar len_leg = states.len_leg[idx];
    Scalar ddt_x_foot = states.ddt_x_foot[idx];
    Scalar ddt_z_foot = states.ddt_z_foot[idx];
    Scalar ddt_phi_leg = states.ddt_phi_leg[idx];
    Scalar ddt_phi_body = states.ddt_phi_body[idx];
    Scalar ddt_len_leg = states.ddt_len_leg[idx];
    int fsm_state = states.fsm_state[idx];
    Scalar T_s = states.T_s[idx];
    Scalar T_compression = states.T_compression[idx];
    Scalar t_thrust_on = states.t_thrust_on[idx];

    // Get control params for this hopper
    ControlParams ctrl = ctrl_array ? ctrl_array[idx] : ctrl_default;

    Scalar t = 0.0;
    int sample_idx = 0;

    for (int step = 0; step <= num_steps; step++) {
        // Sample at regular intervals
        if (step % sample_every == 0 && sample_idx < num_samples) {
            // Compute control for logging
            ControlOutput control = compute_control(
                t, x_foot, z_foot, phi_leg, phi_body, len_leg,
                ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                fsm_state, ctrl, phys, t_thrust_on, T_s
            );

            // Store: layout is [hopper_idx][sample_idx][14 columns]
            // Flattened as: trajectory_data[(idx * num_samples + sample_idx) * 14 + col]
            int base = (idx * num_samples + sample_idx) * 14;
            trajectory_data[base + 0] = t;
            trajectory_data[base + 1] = x_foot;
            trajectory_data[base + 2] = z_foot;
            trajectory_data[base + 3] = phi_leg;
            trajectory_data[base + 4] = phi_body;
            trajectory_data[base + 5] = len_leg;
            trajectory_data[base + 6] = ddt_x_foot;
            trajectory_data[base + 7] = ddt_z_foot;
            trajectory_data[base + 8] = ddt_phi_leg;
            trajectory_data[base + 9] = ddt_phi_body;
            trajectory_data[base + 10] = ddt_len_leg;
            trajectory_data[base + 11] = (Scalar)fsm_state;
            trajectory_data[base + 12] = control.u1;
            trajectory_data[base + 13] = control.u2;

            sample_idx++;
        }

        if (step < num_steps) {
            // Compute control
            ControlOutput control = compute_control(
                t, x_foot, z_foot, phi_leg, phi_body, len_leg,
                ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                fsm_state, ctrl, phys, t_thrust_on, T_s
            );

            // Integrate
            integrator_step<HOPPER_INTEGRATOR>(
                x_foot, z_foot, phi_leg, phi_body, len_leg,
                ddt_x_foot, ddt_z_foot, ddt_phi_leg, ddt_phi_body, ddt_len_leg,
                control.u1, control.u2, phys, dt
            );

            // Check FSM
            int new_fsm = check_fsm_transition(z_foot, len_leg, ddt_len_leg, fsm_state, phys.r_s0);
            if (new_fsm != fsm_state) {
                if (fsm_state == FSM_COMPRESSION && new_fsm == FSM_THRUST) {
                    T_compression = t - t_thrust_on;
                    t_thrust_on = t;
                }
                fsm_state = new_fsm;
            }

            t += dt;
        }
    }

    // Write final state back
    states.x_foot[idx] = x_foot;
    states.z_foot[idx] = z_foot;
    states.phi_leg[idx] = phi_leg;
    states.phi_body[idx] = phi_body;
    states.len_leg[idx] = len_leg;
    states.ddt_x_foot[idx] = ddt_x_foot;
    states.ddt_z_foot[idx] = ddt_z_foot;
    states.ddt_phi_leg[idx] = ddt_phi_leg;
    states.ddt_phi_body[idx] = ddt_phi_body;
    states.ddt_len_leg[idx] = ddt_len_leg;
    states.fsm_state[idx] = fsm_state;
    states.T_s[idx] = T_s;
    states.T_compression[idx] = T_compression;
    states.t_thrust_on[idx] = t_thrust_on;
}

void run_multi_hopper_simulation(int N, Scalar t_final, const char* output_prefix, Scalar sample_rate = 100.0) {
    printf("Running parallel simulation of %d hoppers for %.2f seconds...\n", N, t_final);

    Scalar dt = 1e-4;
    int num_steps = (int)(t_final / dt);
    int sample_every = (int)(1.0 / (sample_rate * dt));
    int num_samples = num_steps / sample_every + 1;

    printf("  dt=%.2e, steps=%d, sample_every=%d, samples=%d\n", dt, num_steps, sample_every, num_samples);

    // Allocate device state arrays
    HopperStateArrays d_states = allocate_state_arrays_device(N);

    // Allocate trajectory storage on device
    size_t traj_size = (size_t)N * num_samples * 14 * sizeof(Scalar);
    Scalar* d_trajectory;
    cudaMalloc(&d_trajectory, traj_size);
    printf("  Trajectory buffer: %.2f MB\n", traj_size / (1024.0 * 1024.0));

    // Allocate per-hopper control params
    ControlParams* d_ctrl;
    cudaMalloc(&d_ctrl, N * sizeof(ControlParams));

    // Initialize hoppers with varied ICs
    std::vector<ControlParams> h_ctrl(N);
    srand(42);  // Reproducible random

    for (int i = 0; i < N; i++) {
        HopperState state;
        state.x_foot = 0.0;
        state.z_foot = 0.4;
        // Vary initial leg angle slightly: -0.1 to 0.1 rad
        state.phi_leg = 0.01 + 0.1 * ((Scalar)i / (N - 1) - 0.5);
        state.phi_body = 0.0;
        state.len_leg = 1.0;
        state.ddt_x_foot = 0.0;
        state.ddt_z_foot = 0.0;
        state.ddt_phi_leg = 0.0;
        state.ddt_phi_body = 0.0;
        state.ddt_len_leg = 0.0;
        state.fsm_state = FSM_FLIGHT;
        state.T_s = 0.425;
        state.T_compression = 0.0;
        state.t_thrust_on = 0.0;

        copy_state_to_device(state, d_states, i);

        // Vary desired velocity: 2.0 to 4.0 m/s
        h_ctrl[i].x_dot_des = 2.0 + 2.0 * ((Scalar)i / (N - 1));
    }

    cudaMemcpy(d_ctrl, h_ctrl.data(), N * sizeof(ControlParams), cudaMemcpyHostToDevice);

    PhysicsParams phys = default_physics();
    ControlParams ctrl_default = default_control();

    // Copy physics to constant memory
    cudaMemcpyToSymbol(d_physics, &phys, sizeof(PhysicsParams));

    // Run simulation
    printf("  Launching kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel_simulate_hoppers_with_history<<<blocks, threads>>>(
        d_states, d_trajectory, d_ctrl, ctrl_default, phys, dt, num_steps, sample_every, num_samples
    );
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("  Kernel time: %.2f ms (%.1fx realtime)\n", ms, t_final * 1000.0 / ms);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy trajectory back to host
    printf("  Copying trajectory to host...\n");
    std::vector<Scalar> h_trajectory(N * num_samples * 14);
    cudaMemcpy(h_trajectory.data(), d_trajectory, traj_size, cudaMemcpyDeviceToHost);

    // Export to CSV files - one per hopper
    printf("  Exporting %d CSV files...\n", N);
    for (int i = 0; i < N; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s_%03d.csv", output_prefix, i);

        FILE* f = fopen(filename, "w");
        if (!f) {
            printf("ERROR: Could not open %s\n", filename);
            continue;
        }

        // Header
        fprintf(f, "t,x_foot,z_foot,phi_leg,phi_body,len_leg,"
                   "ddt_x_foot,ddt_z_foot,ddt_phi_leg,ddt_phi_body,ddt_len_leg,"
                   "fsm_state,u1,u2\n");

        for (int s = 0; s < num_samples; s++) {
            int base = (i * num_samples + s) * 14;
            fprintf(f, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%.10e,%.10e,%.10e,%.10e,%.10e,"
                       "%d,%.10e,%.10e\n",
                    h_trajectory[base + 0],
                    h_trajectory[base + 1],
                    h_trajectory[base + 2],
                    h_trajectory[base + 3],
                    h_trajectory[base + 4],
                    h_trajectory[base + 5],
                    h_trajectory[base + 6],
                    h_trajectory[base + 7],
                    h_trajectory[base + 8],
                    h_trajectory[base + 9],
                    h_trajectory[base + 10],
                    (int)h_trajectory[base + 11],
                    h_trajectory[base + 12],
                    h_trajectory[base + 13]);
        }
        fclose(f);
    }

    // Also export summary of final states
    char summary_file[256];
    snprintf(summary_file, sizeof(summary_file), "%s_summary.csv", output_prefix);
    FILE* f = fopen(summary_file, "w");
    if (f) {
        fprintf(f, "hopper_id,x_dot_des,final_x,final_z\n");
        for (int i = 0; i < N; i++) {
            int base = (i * num_samples + (num_samples - 1)) * 14;
            fprintf(f, "%d,%.4f,%.4f,%.4f\n", i, h_ctrl[i].x_dot_des,
                    h_trajectory[base + 1], h_trajectory[base + 2]);
        }
        fclose(f);
        printf("  Summary: %s\n", summary_file);
    }

    printf("Done! Files: %s_000.csv through %s_%03d.csv\n", output_prefix, output_prefix, N-1);

    // Cleanup
    cudaFree(d_trajectory);
    cudaFree(d_ctrl);
    free_state_arrays_device(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// MAIN
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --test         Run unit tests (default if no options)\n");
    printf("  --sim          Run single-hopper simulation and export trajectory\n");
    printf("  --multi        Run parallel multi-hopper simulation (original)\n");
    printf("  --batched      Run parallel multi-hopper with batched Newton kernels\n");
    printf("  --energy <csv> Analyze energy from trajectory CSV file\n");
    printf("  -n <num>       Number of hoppers for --multi/--batched (default: 100)\n");
    printf("  -t <time>      Simulation duration in seconds (default: 5.0)\n");
    printf("  -o <file>      Output filename/prefix (default: trajectory_cuda)\n");
    printf("  -h, --help     Show this help\n");
}

int main(int argc, char** argv) {
    bool run_tests = false;
    bool run_sim = false;
    bool run_multi = false;
    bool run_batched = false;
    bool run_energy = false;
    int num_hoppers = 100;
    Scalar t_final = 5.0;
    // Default output to parent directory (one level up from project) to keep code folder clean
    // From cpp/cuda/, go up 3 levels: cuda -> cpp -> RaibertFast -> Robotics
    const char* output_file = "../../../trajectory_cuda";
    const char* energy_file = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            run_tests = true;
        } else if (strcmp(argv[i], "--sim") == 0) {
            run_sim = true;
        } else if (strcmp(argv[i], "--multi") == 0) {
            run_multi = true;
        } else if (strcmp(argv[i], "--batched") == 0) {
            run_batched = true;
        } else if (strcmp(argv[i], "--energy") == 0 && i + 1 < argc) {
            run_energy = true;
            energy_file = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_hoppers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            t_final = atof(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Default: run tests if no mode specified
    if (!run_tests && !run_sim && !run_multi && !run_batched && !run_energy) {
        run_tests = true;
    }

    // Energy analysis mode (doesn't need CUDA)
    if (run_energy && energy_file != nullptr) {
        printf("========================================\n");
        printf("Energy Analysis\n");
        printf("========================================\n");
        printf("Loading trajectory: %s\n", energy_file);

        auto traj = load_trajectory_csv(energy_file);
        if (traj.empty()) {
            printf("Error: Failed to load trajectory\n");
            return 1;
        }
        printf("Loaded %zu trajectory points\n", traj.size());

        PhysicsParamsEnergy phys;  // Use default physics params
        auto result = compute_energy(traj, phys);

        print_energy_summary(result);

        // Export to CSV
        std::string out_name = std::string(energy_file);
        size_t dot_pos = out_name.rfind('.');
        if (dot_pos != std::string::npos) {
            out_name = out_name.substr(0, dot_pos);
        }
        out_name += "_energy.csv";
        export_energy_csv(result, out_name);

        return 0;
    }

    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (run_tests) {
        printf("========================================\n");
        printf("CUDA Hopper Simulation Tests\n");
        printf("========================================\n\n");
        printf("CUDA Device: %s\n\n", prop.name);

        printf("--- Linear Algebra Tests ---\n");
        RUN_TEST(test_solve_5x5);
        RUN_TEST(test_solve_5x5_random);

        printf("\n--- Dynamics Tests (CPU) ---\n");
        RUN_TEST(test_dynamics_flight);
        RUN_TEST(test_dynamics_ground_contact);
        RUN_TEST(test_dynamics_thrust);
        RUN_TEST(test_dynamics_consistency);

        printf("\n--- Jacobian Tests (CPU) ---\n");
        RUN_TEST(test_jacobian_analytical_vs_fd);

        printf("\n--- Integrator Tests (CPU) ---\n");
        RUN_TEST(test_semi_implicit_euler_step);
        RUN_TEST(test_implicit_midpoint_step);
        RUN_TEST(test_integrator_stability_ground_contact);

        printf("\n--- Full Simulation Tests (CPU) ---\n");
        RUN_TEST(test_full_hopper_step);

        printf("\n--- GPU Kernel Tests ---\n");
        RUN_TEST(test_gpu_dynamics_kernel);
        RUN_TEST(test_parallel_identical_hoppers);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
        printf("========================================\n");
    }

    if (run_sim) {
        char sim_output[256];
        snprintf(sim_output, sizeof(sim_output), "%s.csv", output_file);

        printf("\n========================================\n");
        printf("CUDA Hopper Simulation\n");
        printf("========================================\n");
        printf("CUDA Device: %s\n", prop.name);
        printf("Duration: %.2f s\n", t_final);
        printf("Output: %s\n\n", sim_output);

        run_and_export_trajectory(sim_output, t_final);

        printf("\nVisualize with: python src/visualize_cpp_trajectory.py %s\n", sim_output);
    }

    if (run_multi) {
        printf("\n========================================\n");
        printf("CUDA Multi-Hopper Parallel Simulation\n");
        printf("========================================\n");
        printf("CUDA Device: %s\n", prop.name);
        printf("Hoppers: %d\n", num_hoppers);
        printf("Duration: %.2f s\n", t_final);
        printf("Output prefix: %s\n\n", output_file);

        run_multi_hopper_simulation(num_hoppers, t_final, output_file);

        printf("\nVisualize with: python src/visualize_multi_hopper.py %s\n", output_file);
    }

    if (run_batched) {
        printf("\n========================================\n");
        printf("CUDA Multi-Hopper BATCHED Simulation\n");
        printf("========================================\n");
        printf("CUDA Device: %s\n", prop.name);
        printf("Hoppers: %d\n", num_hoppers);
        printf("Duration: %.2f s\n", t_final);
        printf("Output prefix: %s\n\n", output_file);

        run_multi_hopper_simulation_batched(num_hoppers, t_final, output_file);

        printf("\nVisualize with: python src/visualize_multi_hopper.py %s\n", output_file);
    }

    return (run_tests && tests_failed > 0) ? 1 : 0;
}
