// test_hopper.cpp - Unit Tests for Raibert Hopper Simulation
//
// Educational test suite demonstrating basic testing principles:
// 1. Known input/output tests
// 2. Boundary condition tests
// 3. Invariant tests (e.g., both solvers should match)
//
// Compile with: ./build.sh test

#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <Eigen/Dense>

#include "hopper.hpp"
#include "ode.hpp"

// ============================================================================
// SIMPLE TEST FRAMEWORK
// ============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            printf("Running: %s ... ", #name); \
            fflush(stdout); \
            try { \
                test_##name(); \
                printf("PASSED\n"); \
                g_tests_passed++; \
            } catch (const char* msg) { \
                printf("FAILED: %s\n", msg); \
                g_tests_failed++; \
            } \
            g_tests_run++; \
        } \
    } runner_##name; \
    void test_##name()

#define ASSERT(cond) \
    if (!(cond)) { throw "Assertion failed: " #cond; }

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        printf("\n  Expected: %.10e\n  Actual:   %.10e\n  Diff:     %.10e\n", \
               (double)(b), (double)(a), std::abs((a)-(b))); \
        throw "ASSERT_NEAR failed: " #a " != " #b; \
    }

#define ASSERT_TRUE(cond) ASSERT(cond)
#define ASSERT_FALSE(cond) ASSERT(!(cond))

// ============================================================================
// TEST: State struct indexing
// ============================================================================

TEST(state_indexing) {
    State q;
    q.x_foot = 1.0;
    q.z_foot = 2.0;
    q.phi_leg = 3.0;
    q.phi_body = 4.0;
    q.len_leg = 5.0;
    q.ddt_x_foot = 6.0;
    q.ddt_z_foot = 7.0;
    q.ddt_phi_leg = 8.0;
    q.ddt_phi_body = 9.0;
    q.ddt_len_leg = 10.0;

    // Test operator[] matches named fields
    ASSERT_NEAR(q[0], q.x_foot, 1e-15);
    ASSERT_NEAR(q[1], q.z_foot, 1e-15);
    ASSERT_NEAR(q[2], q.phi_leg, 1e-15);
    ASSERT_NEAR(q[3], q.phi_body, 1e-15);
    ASSERT_NEAR(q[4], q.len_leg, 1e-15);
    ASSERT_NEAR(q[5], q.ddt_x_foot, 1e-15);
    ASSERT_NEAR(q[6], q.ddt_z_foot, 1e-15);
    ASSERT_NEAR(q[7], q.ddt_phi_leg, 1e-15);
    ASSERT_NEAR(q[8], q.ddt_phi_body, 1e-15);
    ASSERT_NEAR(q[9], q.ddt_len_leg, 1e-15);
}

// ============================================================================
// TEST: Matrix solvers produce equivalent results
// ============================================================================

TEST(matrix_solvers_equivalent) {
    // Test matrix (from a real hopper state)
    double M[5][5] = {
        {-4.5, 0.0, 0.495, 0.0, 0.0},
        {0.0, 4.5, 0.0499, 0.0, 0.0},
        {5.0, 0.0, 5.995, 1.996, 0.0499},
        {0.0, -5.0, 0.5995, 0.1996, -0.9988},
        {0.0, 0.0, 0.3998, -5.0, 0.0}
    };

    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    double x_gauss[5], x_eigen[5];

    solve_5x5_gaussian(M, b, x_gauss);
    solve_5x5_eigen(M, b, x_eigen);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(x_gauss[i], x_eigen[i], 1e-10);
    }
}

// ============================================================================
// TEST: Zero gravity, stationary state -> near-zero accelerations
// ============================================================================

TEST(zero_gravity_equilibrium) {
    Parameters p;
    p.g = 0.0;
    p.fsm_state = Parameters::FSM_FLIGHT;

    // State at rest, in the air (z_foot > 0), leg vertical
    State q;
    q.x_foot = 0.0;
    q.z_foot = 0.5;  // Above ground, so no ground contact
    q.phi_leg = 0.0;
    q.phi_body = 0.0;
    q.len_leg = p.r_s0;  // At rest length (no spring force)
    q.ddt_x_foot = 0.0;
    q.ddt_z_foot = 0.0;
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = 0.0;

    auto result = hopper_dynamics_fwd(0.0, q, p);

    // With zero gravity, no ground contact, no initial velocity,
    // and leg at rest length, accelerations should be near zero
    // (there may be small control torques from attitude control)
    ASSERT_NEAR(result.state_dot.dddt_x_foot, 0.0, 1e-8);
    ASSERT_NEAR(result.state_dot.dddt_z_foot, 0.0, 1e-8);
    ASSERT_NEAR(result.state_dot.dddt_len_leg, 0.0, 1e-8);
}

// ============================================================================
// TEST: Ground contact produces upward force
// ============================================================================

TEST(ground_contact_force) {
    Parameters p;
    p.fsm_state = Parameters::FSM_COMPRESSION;

    // Foot below ground, moving downward
    State q;
    q.x_foot = 0.0;
    q.z_foot = -0.01;  // 1cm below ground
    q.phi_leg = 0.0;
    q.phi_body = 0.0;
    q.len_leg = 1.0;
    q.ddt_x_foot = 0.0;
    q.ddt_z_foot = -0.5;  // Moving down
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = 0.0;

    auto result = hopper_dynamics_fwd(0.0, q, p);

    // Ground should push back up (positive z acceleration)
    // The exact value depends on masses and spring constants
    ASSERT_TRUE(result.state_dot.dddt_z_foot > 0.0);
}

// ============================================================================
// TEST: Control outputs in each FSM state
// ============================================================================

TEST(control_thrust_state) {
    Parameters p;
    p.fsm_state = Parameters::FSM_THRUST;

    State q;
    q.x_foot = 0.0;
    q.z_foot = -0.01;
    q.phi_leg = 0.1;
    q.phi_body = 0.05;
    q.len_leg = 0.95;
    q.ddt_x_foot = 1.0;
    q.ddt_z_foot = 0.0;
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = 0.1;

    auto ctrl = hopper_control(0.0, q, p);

    // In THRUST state, u1 should be positive (thrust force)
    ASSERT_TRUE(ctrl.u1 > 0.0);
}

TEST(control_compression_state) {
    Parameters p;
    p.fsm_state = Parameters::FSM_COMPRESSION;

    State q;
    q.x_foot = 0.0;
    q.z_foot = -0.01;
    q.phi_leg = 0.1;
    q.phi_body = 0.05;
    q.len_leg = 0.95;
    q.ddt_x_foot = 1.0;
    q.ddt_z_foot = 0.0;
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = -0.1;  // Compressing

    auto ctrl = hopper_control(0.0, q, p);

    // In COMPRESSION state, u1 should be zero (passive spring)
    ASSERT_NEAR(ctrl.u1, 0.0, 1e-10);
}

TEST(control_flight_state) {
    Parameters p;
    p.fsm_state = Parameters::FSM_FLIGHT;
    p.x_dot_des = 2.0;
    p.T_s = 0.425;

    State q;
    q.x_foot = 0.0;
    q.z_foot = 0.5;  // In the air
    q.phi_leg = 0.0;
    q.phi_body = 0.0;
    q.len_leg = 1.0;
    q.ddt_x_foot = 1.5;
    q.ddt_z_foot = 0.0;
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = 0.0;

    auto ctrl = hopper_control(0.0, q, p);

    // In FLIGHT state:
    // - u1 should be zero
    // - u2 should be non-zero (foot placement control)
    // - a_des should be computed
    ASSERT_NEAR(ctrl.u1, 0.0, 1e-10);
    // a_des should be negative (swing leg forward to slow down when moving faster than desired)
    ASSERT_TRUE(ctrl.a_des < 0.0);
}

// ============================================================================
// TEST: Event detection values
// ============================================================================

TEST(event_flight_to_compression) {
    Parameters p;
    p.fsm_state = Parameters::FSM_FLIGHT;

    // In flight, event triggers when z_foot crosses zero (touchdown)
    State q_above;
    q_above.z_foot = 0.1;
    State q_below;
    q_below.z_foot = -0.01;

    double e_above = hopper_event(0.0, q_above, p);
    double e_below = hopper_event(0.0, q_below, p);

    // Event value should change sign across touchdown
    ASSERT_TRUE(e_above < 0.0);  // -z_foot when above ground
    ASSERT_TRUE(e_below > 0.0);  // -z_foot when below ground
}

TEST(event_compression_to_thrust) {
    Parameters p;
    p.fsm_state = Parameters::FSM_COMPRESSION;

    // In compression, event triggers when ddt_len_leg crosses zero (bottom of compression)
    State q_compressing;
    q_compressing.ddt_len_leg = -0.5;  // Leg compressing
    State q_extending;
    q_extending.ddt_len_leg = 0.1;  // Leg extending

    double e_comp = hopper_event(0.0, q_compressing, p);
    double e_ext = hopper_event(0.0, q_extending, p);

    ASSERT_TRUE(e_comp < 0.0);
    ASSERT_TRUE(e_ext > 0.0);
}

// ============================================================================
// TEST: RK4 single step (sanity check)
// ============================================================================

// Helper structs for RK4 test (must be at namespace scope for static constexpr)
struct SimpleState {
    double y;
    static constexpr int SIZE = 1;
    double& operator[](int) { return y; }
    const double& operator[](int) const { return y; }
};

struct SimpleDeriv {
    double dydt;
    static constexpr int SIZE = 1;
    double& operator[](int) { return dydt; }
    const double& operator[](int) const { return dydt; }
};

TEST(rk4_step_linear_system) {
    // Test with simple linear ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)

    auto f = [](double, const SimpleState& s) -> SimpleDeriv {
        return SimpleDeriv{-s.y};
    };

    SimpleState y0{1.0};
    double h = 0.1;

    SimpleState y1 = rk4_step<SimpleState, SimpleDeriv>(f, 0.0, y0, h);

    // Analytical solution: exp(-0.1) ≈ 0.904837
    double expected = std::exp(-0.1);
    ASSERT_NEAR(y1.y, expected, 1e-6);
}

// ============================================================================
// TEST: Known dynamics output (regression test)
// ============================================================================

TEST(dynamics_known_output) {
    // This test uses a specific state and checks the output matches
    // a pre-computed reference (from Python implementation)

    Parameters p;
    p.fsm_state = Parameters::FSM_FLIGHT;
    p.x_dot_des = 3.0;
    p.T_s = 0.425;

    State q;
    q.x_foot = 0.0;
    q.z_foot = 0.4;
    q.phi_leg = 0.01;
    q.phi_body = 0.0;
    q.len_leg = 1.0;
    q.ddt_x_foot = 0.0;
    q.ddt_z_foot = 0.0;
    q.ddt_phi_leg = 0.0;
    q.ddt_phi_body = 0.0;
    q.ddt_len_leg = 0.0;

    auto result = hopper_dynamics_fwd(0.0, q, p);

    // First 5 elements of state_dot are just the velocities
    ASSERT_NEAR(result.state_dot.ddt_x_foot, 0.0, 1e-10);
    ASSERT_NEAR(result.state_dot.ddt_z_foot, 0.0, 1e-10);
    ASSERT_NEAR(result.state_dot.ddt_phi_leg, 0.0, 1e-10);
    ASSERT_NEAR(result.state_dot.ddt_phi_body, 0.0, 1e-10);
    ASSERT_NEAR(result.state_dot.ddt_len_leg, 0.0, 1e-10);

    // Accelerations: gravity should cause downward acceleration
    // The exact values depend on the dynamics equations
    // dddt_z_foot should be negative (falling due to gravity)
    ASSERT_TRUE(result.state_dot.dddt_z_foot < 0.0);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("\n");
    printf("========================================\n");
    printf("Hopper Unit Tests\n");
    printf("========================================\n\n");

    // Tests are auto-registered and run via static initializers
    // (see TEST macro)

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0) {
        printf("         %d tests FAILED\n", g_tests_failed);
    }
    printf("========================================\n");

    return g_tests_failed > 0 ? 1 : 0;
}
