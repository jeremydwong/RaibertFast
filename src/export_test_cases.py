#!/usr/bin/env python3
"""
Export test cases from MATLAB .mat file to CSV for C++ comparison.

This script reads the MATLAB hopper_replication_data.mat file and exports
it to a CSV format that the C++ compare_with_matlab tool can read.

Usage:
    python src/export_test_cases.py
"""

import numpy as np
import scipy.io
import os
import sys

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    matlab_file = os.path.join(root_dir, 'MATLAB', 'hopper_replication_data.mat')
    output_dir = os.path.join(root_dir, 'cpp', 'test_data')
    output_file = os.path.join(output_dir, 'reference_cases.csv')

    # Check input file exists
    if not os.path.exists(matlab_file):
        print(f"ERROR: Cannot find {matlab_file}")
        print("\nTo generate this file, run the MATLAB simulation with data logging enabled.")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load MATLAB data
    print(f"Loading: {matlab_file}")
    mat_data = scipy.io.loadmat(matlab_file, squeeze_me=True)
    replication_data = mat_data['replication_data']

    num_cases = len(replication_data)
    print(f"Found {num_cases} test cases")

    # Write CSV
    print(f"Writing: {output_file}")
    with open(output_file, 'w') as f:
        # Header
        header = [
            't',
            'x_foot', 'z_foot', 'phi_leg', 'phi_body', 'len_leg',
            'ddt_x_foot', 'ddt_z_foot', 'ddt_phi_leg', 'ddt_phi_body', 'ddt_len_leg',
            'stated_0', 'stated_1', 'stated_2', 'stated_3', 'stated_4',
            'stated_5', 'stated_6', 'stated_7', 'stated_8', 'stated_9',
            'fsm_state', 't_state_switch', 'x_dot_des', 'T_s', 'T_compression', 't_thrust_on'
        ]
        f.write(','.join(header) + '\n')

        # Data
        for i in range(num_cases):
            t = float(replication_data[i]['t'].item())
            state = replication_data[i]['state'].item()
            state_d = replication_data[i]['state_d'].item()
            p = replication_data[i]['p'].item()

            # Extract FSM parameters
            fsm_state = int(p['fsm_state'])
            t_state_switch = float(p['t_state_switch'])
            x_dot_des = float(p['x_dot_des'])
            T_s = float(p['T_s'])
            T_compression = float(p['T_compression'])
            t_thrust_on = float(p['t_thrust_on'])

            # Write row
            row = [f'{t:.15e}']

            # State (10 elements)
            for j in range(10):
                row.append(f'{state[j]:.15e}')

            # State derivative (10 elements)
            for j in range(10):
                row.append(f'{state_d[j]:.15e}')

            # FSM parameters
            row.append(str(fsm_state))
            row.append(f'{t_state_switch:.15e}')
            row.append(f'{x_dot_des:.15e}')
            row.append(f'{T_s:.15e}')
            row.append(f'{T_compression:.15e}')
            row.append(f'{t_thrust_on:.15e}')

            f.write(','.join(row) + '\n')

            if (i + 1) % 500 == 0:
                print(f"  Wrote {i + 1}/{num_cases} cases...")

    print(f"\nDone! Exported {num_cases} test cases to:")
    print(f"  {output_file}")
    print("\nRun C++ comparison with:")
    print("  cd cpp && ./build.sh compare")

    return 0

if __name__ == '__main__':
    sys.exit(main())
