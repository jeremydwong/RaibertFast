// ode.hpp - Custom ODE Integrator with Event Detection
// Educational implementation: RK45 (Dormand-Prince) with adaptive stepping
//
// No external dependencies beyond standard library.
// Students can read and understand every line.

#ifndef ODE_HPP
#define ODE_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

// ============================================================================
// CONFIGURATION
// ============================================================================

struct RK45Config {
    double rtol = 1e-6;       // Relative tolerance
    double atol = 1e-8;       // Absolute tolerance
    double max_step = 0.05;   // Maximum step size
    double min_step = 1e-12;  // Minimum step size
    double safety = 0.9;      // Safety factor for step size adjustment
    int max_steps = 100000;   // Maximum number of steps
};

// ============================================================================
// INTEGRATION RESULT
// ============================================================================

template<typename StateType>
struct IntegrateResult {
    std::vector<double> t;
    std::vector<StateType> y;
    bool event_occurred = false;
    double t_event = 0.0;
    StateType y_event;
    bool success = true;
    int num_steps = 0;
};

// ============================================================================
// DORMAND-PRINCE RK45 COEFFICIENTS
// ============================================================================
// These are the standard coefficients for the 4th/5th order adaptive method

namespace DP45 {
    // Butcher tableau for Dormand-Prince
    constexpr double c2 = 1.0/5.0;
    constexpr double c3 = 3.0/10.0;
    constexpr double c4 = 4.0/5.0;
    constexpr double c5 = 8.0/9.0;
    constexpr double c6 = 1.0;
    constexpr double c7 = 1.0;

    constexpr double a21 = 1.0/5.0;
    constexpr double a31 = 3.0/40.0;
    constexpr double a32 = 9.0/40.0;
    constexpr double a41 = 44.0/45.0;
    constexpr double a42 = -56.0/15.0;
    constexpr double a43 = 32.0/9.0;
    constexpr double a51 = 19372.0/6561.0;
    constexpr double a52 = -25360.0/2187.0;
    constexpr double a53 = 64448.0/6561.0;
    constexpr double a54 = -212.0/729.0;
    constexpr double a61 = 9017.0/3168.0;
    constexpr double a62 = -355.0/33.0;
    constexpr double a63 = 46732.0/5247.0;
    constexpr double a64 = 49.0/176.0;
    constexpr double a65 = -5103.0/18656.0;
    constexpr double a71 = 35.0/384.0;
    constexpr double a72 = 0.0;
    constexpr double a73 = 500.0/1113.0;
    constexpr double a74 = 125.0/192.0;
    constexpr double a75 = -2187.0/6784.0;
    constexpr double a76 = 11.0/84.0;

    // 5th order weights (for y_new)
    constexpr double b1 = 35.0/384.0;
    constexpr double b2 = 0.0;
    constexpr double b3 = 500.0/1113.0;
    constexpr double b4 = 125.0/192.0;
    constexpr double b5 = -2187.0/6784.0;
    constexpr double b6 = 11.0/84.0;
    constexpr double b7 = 0.0;

    // 4th order weights (for error estimate)
    constexpr double e1 = 35.0/384.0 - 5179.0/57600.0;
    constexpr double e2 = 0.0;
    constexpr double e3 = 500.0/1113.0 - 7571.0/16695.0;
    constexpr double e4 = 125.0/192.0 - 393.0/640.0;
    constexpr double e5 = -2187.0/6784.0 + 92097.0/339200.0;
    constexpr double e6 = 11.0/84.0 - 187.0/2100.0;
    constexpr double e7 = -1.0/40.0;
}

// ============================================================================
// RK45 SINGLE STEP
// ============================================================================

template<typename StateType, typename DerivType>
struct RK45StepResult {
    StateType y_new;      // New state (5th order)
    double error;         // Error estimate
    double h_new;         // Suggested next step size
    bool accepted;        // Whether step was accepted
};

template<typename StateType, typename DerivType>
RK45StepResult<StateType, DerivType> rk45_step(
    std::function<DerivType(double, const StateType&)> f,
    double t, const StateType& y, double h,
    const RK45Config& config = RK45Config()
) {
    constexpr int N = StateType::SIZE;

    // Compute k1 through k7
    DerivType k1 = f(t, y);

    StateType y2;
    for (int i = 0; i < N; i++) {
        y2[i] = y[i] + h * DP45::a21 * k1[i];
    }
    DerivType k2 = f(t + DP45::c2 * h, y2);

    StateType y3;
    for (int i = 0; i < N; i++) {
        y3[i] = y[i] + h * (DP45::a31 * k1[i] + DP45::a32 * k2[i]);
    }
    DerivType k3 = f(t + DP45::c3 * h, y3);

    StateType y4;
    for (int i = 0; i < N; i++) {
        y4[i] = y[i] + h * (DP45::a41 * k1[i] + DP45::a42 * k2[i] + DP45::a43 * k3[i]);
    }
    DerivType k4 = f(t + DP45::c4 * h, y4);

    StateType y5;
    for (int i = 0; i < N; i++) {
        y5[i] = y[i] + h * (DP45::a51 * k1[i] + DP45::a52 * k2[i] + DP45::a53 * k3[i] + DP45::a54 * k4[i]);
    }
    DerivType k5 = f(t + DP45::c5 * h, y5);

    StateType y6;
    for (int i = 0; i < N; i++) {
        y6[i] = y[i] + h * (DP45::a61 * k1[i] + DP45::a62 * k2[i] + DP45::a63 * k3[i]
                         + DP45::a64 * k4[i] + DP45::a65 * k5[i]);
    }
    DerivType k6 = f(t + DP45::c6 * h, y6);

    // 5th order solution
    StateType y_new;
    for (int i = 0; i < N; i++) {
        y_new[i] = y[i] + h * (DP45::b1 * k1[i] + DP45::b3 * k3[i] + DP45::b4 * k4[i]
                             + DP45::b5 * k5[i] + DP45::b6 * k6[i]);
    }

    DerivType k7 = f(t + h, y_new);

    // Error estimate (difference between 4th and 5th order)
    double err_max = 0.0;
    for (int i = 0; i < N; i++) {
        double err_i = std::abs(h * (DP45::e1 * k1[i] + DP45::e3 * k3[i] + DP45::e4 * k4[i]
                                   + DP45::e5 * k5[i] + DP45::e6 * k6[i] + DP45::e7 * k7[i]));
        double scale = config.atol + config.rtol * std::max(std::abs(y[i]), std::abs(y_new[i]));
        err_max = std::max(err_max, err_i / scale);
    }

    // Compute new step size
    RK45StepResult<StateType, DerivType> result;
    result.y_new = y_new;
    result.error = err_max;

    if (err_max <= 1.0) {
        // Step accepted
        result.accepted = true;
        if (err_max < 1e-10) {
            result.h_new = h * 2.0;  // Can safely increase
        } else {
            result.h_new = h * config.safety * std::pow(err_max, -0.2);
        }
    } else {
        // Step rejected - decrease step size
        result.accepted = false;
        result.h_new = h * config.safety * std::pow(err_max, -0.25);
    }

    // Clamp step size
    result.h_new = std::max(config.min_step, std::min(config.max_step, result.h_new));

    return result;
}

// ============================================================================
// EVENT DETECTION VIA BISECTION
// ============================================================================

template<typename StateType, typename DerivType, typename ParamType>
double find_event_time(
    std::function<DerivType(double, const StateType&, const ParamType&)> dynamics,
    std::function<double(double, const StateType&, const ParamType&)> event,
    const ParamType& params,
    double t0_in, const StateType& y0_in, double e0_in,
    double t1_in, const StateType& y1_in, double e1_in,
    double tol = 1e-10,
    int max_iter = 50
) {
    // Bisection to find zero crossing
    // Precondition: e0 and e1 have opposite signs

    // Make mutable copies for bisection
    double t0 = t0_in, t1 = t1_in;
    StateType y0 = y0_in, y1 = y1_in;
    double e0 = e0_in, e1 = e1_in;

    for (int iter = 0; iter < max_iter; iter++) {
        double t_mid = 0.5 * (t0 + t1);

        if (t1 - t0 < tol) {
            return t_mid;
        }

        // Interpolate state at midpoint (linear for simplicity)
        StateType y_mid;
        double alpha = (t_mid - t0) / (t1 - t0);
        for (int i = 0; i < StateType::SIZE; i++) {
            y_mid[i] = y0[i] + alpha * (y1[i] - y0[i]);
        }

        double e_mid = event(t_mid, y_mid, params);

        if (std::abs(e_mid) < tol) {
            return t_mid;
        }

        // Determine which half contains the root
        if ((e0 > 0) == (e_mid > 0)) {
            // Root is in [t_mid, t1]
            t0 = t_mid;
            y0 = y_mid;
            e0 = e_mid;
        } else {
            // Root is in [t0, t_mid]
            t1 = t_mid;
            y1 = y_mid;
            e1 = e_mid;
        }
    }

    return 0.5 * (t0 + t1);
}

// ============================================================================
// MAIN INTEGRATION LOOP WITH EVENT DETECTION
// ============================================================================

template<typename StateType, typename DerivType, typename ParamType>
IntegrateResult<StateType> integrate_with_events(
    std::function<DerivType(double, const StateType&, const ParamType&)> dynamics,
    std::function<double(double, const StateType&, const ParamType&)> event,
    double t0, double tf,
    const StateType& y0,
    const ParamType& params,
    RK45Config config = RK45Config()
) {
    IntegrateResult<StateType> result;
    result.t.push_back(t0);
    result.y.push_back(y0);

    double t = t0;
    StateType y = y0;
    double h = std::min(config.max_step, (tf - t0) / 10.0);

    double e_prev = event(t, y, params);

    int step_count = 0;
    while (t < tf && step_count < config.max_steps) {
        step_count++;

        // Don't overshoot final time
        if (t + h > tf) {
            h = tf - t;
        }

        // Take a step (wrap dynamics to match expected signature)
        auto dynamics_wrapper = [&dynamics, &params](double time, const StateType& state) -> DerivType {
            return dynamics(time, state, params);
        };
        auto step = rk45_step<StateType, DerivType>(dynamics_wrapper, t, y, h, config);

        if (!step.accepted) {
            // Step rejected, try again with smaller step
            h = step.h_new;
            continue;
        }

        // Check for event
        double e_new = event(t + h, step.y_new, params);

        // Event detected if sign changed (going positive direction)
        bool event_detected = (e_prev <= 0 && e_new > 0);

        if (event_detected) {
            // Find precise event time via bisection
            double t_event = find_event_time<StateType, DerivType, ParamType>(
                dynamics, event, params,
                t, y, e_prev,
                t + h, step.y_new, e_new
            );

            // Interpolate state at event time
            double alpha = (t_event - t) / h;
            StateType y_event;
            for (int i = 0; i < StateType::SIZE; i++) {
                y_event[i] = y[i] + alpha * (step.y_new[i] - y[i]);
            }

            result.t.push_back(t_event);
            result.y.push_back(y_event);
            result.event_occurred = true;
            result.t_event = t_event;
            result.y_event = y_event;
            result.num_steps = step_count;
            return result;
        }

        // Accept step
        t += h;
        y = step.y_new;
        e_prev = e_new;
        h = step.h_new;

        result.t.push_back(t);
        result.y.push_back(y);
    }

    result.num_steps = step_count;
    result.success = (step_count < config.max_steps);
    return result;
}

// ============================================================================
// SIMPLE INTEGRATION (NO EVENTS)
// ============================================================================

template<typename StateType, typename DerivType, typename ParamType>
IntegrateResult<StateType> integrate(
    std::function<DerivType(double, const StateType&, const ParamType&)> dynamics,
    double t0, double tf,
    const StateType& y0,
    const ParamType& params,
    RK45Config config = RK45Config()
) {
    // Dummy event that never triggers
    auto no_event = [](double, const StateType&, const ParamType&) { return -1.0; };
    return integrate_with_events<StateType, DerivType, ParamType>(dynamics, no_event, t0, tf, y0, params, config);
}

// ============================================================================
// FIXED-STEP RK4 (SIMPLER, FOR TESTING)
// ============================================================================

template<typename StateType, typename DerivType>
StateType rk4_step(
    std::function<DerivType(double, const StateType&)> f,
    double t, const StateType& y, double h
) {
    constexpr int N = StateType::SIZE;

    DerivType k1 = f(t, y);

    StateType y2;
    for (int i = 0; i < N; i++) y2[i] = y[i] + 0.5 * h * k1[i];
    DerivType k2 = f(t + 0.5 * h, y2);

    StateType y3;
    for (int i = 0; i < N; i++) y3[i] = y[i] + 0.5 * h * k2[i];
    DerivType k3 = f(t + 0.5 * h, y3);

    StateType y4;
    for (int i = 0; i < N; i++) y4[i] = y[i] + h * k3[i];
    DerivType k4 = f(t + h, y4);

    StateType y_new;
    for (int i = 0; i < N; i++) {
        y_new[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    return y_new;
}

#endif // ODE_HPP
