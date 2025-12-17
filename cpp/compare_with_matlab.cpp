// compare_with_matlab.cpp - Compare C++ dynamics against MATLAB/Python reference
//
// Loads test cases from CSV (exported from MATLAB .mat file) and compares
// the C++ dynamics output to the reference values.
//
// Compile with: ./build.sh compare

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>

#include "hopper.hpp"

// ============================================================================
// TEST CASE STRUCTURE
// ============================================================================

struct TestCase {
    double t;
    State state;
    StateDot expected_state_dot;

    // FSM parameters that vary during simulation
    int fsm_state;
    double t_state_switch;
    double x_dot_des;
    double T_s;
    double T_compression;
    double t_thrust_on;
};

// ============================================================================
// CSV PARSING
// ============================================================================

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

std::vector<TestCase> load_test_cases(const char* filename) {
    std::vector<TestCase> cases;

    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Could not open %s\n", filename);
        return cases;
    }

    std::string line;

    // Skip header
    if (!std::getline(file, line)) {
        fprintf(stderr, "ERROR: Empty file %s\n", filename);
        return cases;
    }

    // Parse data lines
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto fields = split_csv_line(line);

        // Expected format (27 fields):
        // t, state[0:10], state_dot[0:10], fsm_state, t_state_switch, x_dot_des, T_s, T_compression, t_thrust_on
        if (fields.size() < 27) {
            fprintf(stderr, "WARNING: Skipping line with %zu fields (expected 27)\n", fields.size());
            continue;
        }

        TestCase tc;
        int idx = 0;

        tc.t = std::stod(fields[idx++]);

        // State (10 elements)
        tc.state.x_foot = std::stod(fields[idx++]);
        tc.state.z_foot = std::stod(fields[idx++]);
        tc.state.phi_leg = std::stod(fields[idx++]);
        tc.state.phi_body = std::stod(fields[idx++]);
        tc.state.len_leg = std::stod(fields[idx++]);
        tc.state.ddt_x_foot = std::stod(fields[idx++]);
        tc.state.ddt_z_foot = std::stod(fields[idx++]);
        tc.state.ddt_phi_leg = std::stod(fields[idx++]);
        tc.state.ddt_phi_body = std::stod(fields[idx++]);
        tc.state.ddt_len_leg = std::stod(fields[idx++]);

        // Expected state derivative (10 elements)
        tc.expected_state_dot.ddt_x_foot = std::stod(fields[idx++]);
        tc.expected_state_dot.ddt_z_foot = std::stod(fields[idx++]);
        tc.expected_state_dot.ddt_phi_leg = std::stod(fields[idx++]);
        tc.expected_state_dot.ddt_phi_body = std::stod(fields[idx++]);
        tc.expected_state_dot.ddt_len_leg = std::stod(fields[idx++]);
        tc.expected_state_dot.dddt_x_foot = std::stod(fields[idx++]);
        tc.expected_state_dot.dddt_z_foot = std::stod(fields[idx++]);
        tc.expected_state_dot.dddt_phi_leg = std::stod(fields[idx++]);
        tc.expected_state_dot.dddt_phi_body = std::stod(fields[idx++]);
        tc.expected_state_dot.dddt_len_leg = std::stod(fields[idx++]);

        // FSM parameters
        tc.fsm_state = std::stoi(fields[idx++]);
        tc.t_state_switch = std::stod(fields[idx++]);
        tc.x_dot_des = std::stod(fields[idx++]);
        tc.T_s = std::stod(fields[idx++]);
        tc.T_compression = std::stod(fields[idx++]);
        tc.t_thrust_on = std::stod(fields[idx++]);

        cases.push_back(tc);
    }

    return cases;
}

// ============================================================================
// COMPARISON
// ============================================================================

double max_abs_diff(const StateDot& a, const StateDot& b) {
    double max_err = 0.0;
    for (int i = 0; i < StateDot::SIZE; i++) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("C++ vs MATLAB/Python Comparison\n");
    printf("========================================\n\n");

    const char* test_file = "test_data/reference_cases.csv";
    if (argc > 1) {
        test_file = argv[1];
    }

    printf("Loading test cases from: %s\n", test_file);
    auto test_cases = load_test_cases(test_file);

    if (test_cases.empty()) {
        printf("\nNo test cases loaded.\n");
        printf("Generate test cases with: python src/export_test_cases.py\n");
        return 1;
    }

    printf("Loaded %zu test cases.\n\n", test_cases.size());

    // Run comparison
    double max_error_overall = 0.0;
    int worst_case = -1;
    int num_large_errors = 0;

    std::vector<double> all_errors;
    std::vector<double> errors_by_state[3];  // FSM states 0, 1, 2

    for (size_t i = 0; i < test_cases.size(); i++) {
        const auto& tc = test_cases[i];

        // Set up parameters to match test case
        Parameters p;
        p.fsm_state = tc.fsm_state;
        p.t_state_switch = tc.t_state_switch;
        p.x_dot_des = tc.x_dot_des;
        p.T_s = tc.T_s;
        p.T_compression = tc.T_compression;
        p.t_thrust_on = tc.t_thrust_on;

        // Compute dynamics
        auto result = hopper_dynamics_fwd(tc.t, tc.state, p);

        // Compare
        double error = max_abs_diff(result.state_dot, tc.expected_state_dot);
        all_errors.push_back(error);

        if (tc.fsm_state >= 0 && tc.fsm_state < 3) {
            errors_by_state[tc.fsm_state].push_back(error);
        }

        if (error > max_error_overall) {
            max_error_overall = error;
            worst_case = (int)i;
        }

        if (error > 1e-6) {
            num_large_errors++;
        }

        // Progress update
        if ((i + 1) % 500 == 0) {
            printf("  Processed %zu/%zu cases...\n", i + 1, test_cases.size());
        }
    }

    // Summary statistics
    printf("\n========================================\n");
    printf("COMPARISON SUMMARY\n");
    printf("========================================\n");
    printf("Total test cases:        %zu\n", test_cases.size());
    printf("Max error (overall):     %.6e\n", max_error_overall);
    printf("Cases with error > 1e-6: %d\n", num_large_errors);
    printf("Worst case index:        %d\n", worst_case);

    // Compute mean/median
    if (!all_errors.empty()) {
        double sum = 0.0;
        for (double e : all_errors) sum += e;
        double mean = sum / all_errors.size();

        std::vector<double> sorted_errors = all_errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());
        double median = sorted_errors[sorted_errors.size() / 2];

        printf("Mean error:              %.6e\n", mean);
        printf("Median error:            %.6e\n", median);
    }

    // Error by FSM state
    printf("\n========================================\n");
    printf("ERROR STATISTICS BY FSM STATE\n");
    printf("========================================\n");
    const char* state_names[] = {"COMPRESSION", "THRUST", "FLIGHT"};
    for (int s = 0; s < 3; s++) {
        if (errors_by_state[s].empty()) continue;

        double sum = 0.0;
        double max_e = 0.0;
        for (double e : errors_by_state[s]) {
            sum += e;
            max_e = std::max(max_e, e);
        }
        double mean = sum / errors_by_state[s].size();

        printf("FSM State %d (%s):\n", s, state_names[s]);
        printf("  Count:      %zu\n", errors_by_state[s].size());
        printf("  Max error:  %.6e\n", max_e);
        printf("  Mean error: %.6e\n", mean);
    }

    // Show worst case details
    if (worst_case >= 0) {
        const auto& tc = test_cases[worst_case];
        Parameters p;
        p.fsm_state = tc.fsm_state;
        p.t_state_switch = tc.t_state_switch;
        p.x_dot_des = tc.x_dot_des;
        p.T_s = tc.T_s;
        p.T_compression = tc.T_compression;
        p.t_thrust_on = tc.t_thrust_on;

        auto result = hopper_dynamics_fwd(tc.t, tc.state, p);

        printf("\n========================================\n");
        printf("WORST CASE DETAILS (case %d)\n", worst_case);
        printf("========================================\n");
        printf("Time:      %.6f\n", tc.t);
        printf("FSM State: %d (%s)\n", tc.fsm_state, state_names[tc.fsm_state]);
        printf("\nState derivative comparison:\n");
        printf("Index | Expected       | C++            | Error\n");
        printf("------|----------------|----------------|----------------\n");
        for (int i = 0; i < StateDot::SIZE; i++) {
            printf("  %2d  | %14.6e | %14.6e | %14.6e\n",
                   i, tc.expected_state_dot[i], result.state_dot[i],
                   std::abs(tc.expected_state_dot[i] - result.state_dot[i]));
        }
    }

    // Final result
    printf("\n========================================\n");
    if (max_error_overall < 1e-8) {
        printf("RESULT: PASS (max error < 1e-8)\n");
    } else if (max_error_overall < 1e-6) {
        printf("RESULT: PASS (max error < 1e-6)\n");
    } else {
        printf("RESULT: FAIL (max error = %.6e)\n", max_error_overall);
    }
    printf("========================================\n");

    return (max_error_overall < 1e-6) ? 0 : 1;
}
