#!/usr/bin/env python3
"""
Compare Python hopperDynamicsFwd implementation against MATLAB reference data.

This script loads the MATLAB replication data and runs the Python implementation
on the same inputs to identify any discrepancies.
"""

import numpy as np
import sys
import os

# Import the hopper module
from hopper import hopperDynamicsFwd, hopperParameters

def parse_p(p):
    
    p['r_s0'].item()

def compare_single_case(t, state, p_python, case_num):
    """
    Compare a single test case.

    Args:
        t: time value
        state: state vector (10 elements)
        p_matlab: parameters from MATLAB
        case_num: test case number for reporting

    Returns:
        dict with comparison results
    """
    # Get Python parameters
    
    # Copy over the state-dependent parameters from MATLAB
    # These are the ones that change during simulation
    result_python = hopperDynamicsFwd(t, state, p_python)

    return result_python

def main():
    # Load MATLAB reference data
    root_dir = os.path.dirname(os.path.dirname(__file__))
    matlab_file = 'MATLAB/hopper_replication_data.mat'
    pathfile = os.path.join(root_dir,matlab_file)
    if not os.path.exists(pathfile):
        print(f"ERROR: Cannot find {matlab_file}")
        sys.exit(1)

    print("Loading MATLAB reference data...")
    import scipy.io
    mat_data = scipy.io.loadmat(pathfile,squeeze_me =True)
    
    replication_data = mat_data['replication_data']
    num_cases = len(replication_data)
    print(f"\nProcessing {num_cases} test cases...")

    all_errors = []
    max_error_overall = 0
    worst_case = None
    p = hopperParameters()
    for i in range(num_cases):
        t = replication_data[i]['t'].item()
        p_matlab = replication_data[i]['p'].item()
        state = replication_data[i]['state'].item()
        stated_matlab = replication_data[i]['state_d'].item()
        replication_data[0]['state'].item()
        # claude did a pretty good job of checking which states are actually changed during a run. 
        p.fsm_state = int(p_matlab['fsm_state'])
        p.t_state_switch = float(p_matlab['t_state_switch'])
        p.x_dot_des = float(p_matlab['x_dot_des'])
        p.T_s = float(p_matlab['T_s'])
        p.T_compression = float(p_matlab['T_compression'])
        p.t_thrust_on = float(p_matlab['t_thrust_on'])
    
        # Run Python version
        result_python = hopperDynamicsFwd(t,state,p)
        stated_python = result_python['stated']

        # Compare
        error = np.abs(stated_python - stated_matlab)
        max_error = np.max(error)
        rms_error = np.sqrt(np.mean(error**2))

        all_errors.append({
            'case': i,
            't': t,
            'max_error': max_error,
            'rms_error': rms_error,
            'error_vector': error,
            'stated_matlab': stated_matlab,
            'stated_python': stated_python,
            'fsm_state': p.fsm_state
        })

        if max_error > max_error_overall:
            max_error_overall = max_error
            worst_case = i

        # Print progress every 100 cases
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_cases} cases...")

    # Summary statistics
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    max_errors = [e['max_error'] for e in all_errors]
    rms_errors = [e['rms_error'] for e in all_errors]

    print(f"Total test cases:        {len(all_errors)}")
    print(f"Max error (overall):     {np.max(max_errors):.6e}")
    print(f"Mean max error:          {np.mean(max_errors):.6e}")
    print(f"Median max error:        {np.median(max_errors):.6e}")
    print(f"Mean RMS error:          {np.mean(rms_errors):.6e}")
    print(f"Worst case index:        {worst_case}")

    # Show worst case details
    if worst_case is not None:
        worst = all_errors[worst_case]
        print("\n" + "="*70)
        print(f"WORST CASE DETAILS (case {worst_case})")
        print("="*70)
        print(f"Time:                    {worst['t']:.6f}")
        print(f"FSM State:               {worst['fsm_state']}")
        print(f"Max error:               {worst['max_error']:.6e}")
        print(f"RMS error:               {worst['rms_error']:.6e}")
        print("\nState derivative comparison:")
        print("Index | MATLAB         | Python         | Error")
        print("------|----------------|----------------|----------------")
        for i in range(len(worst['stated_matlab'])):
            print(f"  {i:2d}  | {worst['stated_matlab'][i]:14.6e} | {worst['stated_python'][i]:14.6e} | {worst['error_vector'][i]:14.6e}")

    # Show error distribution by FSM state
    print("\n" + "="*70)
    print("ERROR STATISTICS BY FSM STATE")
    print("="*70)

    by_state = {}
    for e in all_errors:
        state = e['fsm_state']
        if state not in by_state:
            by_state[state] = []
        by_state[state].append(e['max_error'])

    for state in sorted(by_state.keys()):
        errors = by_state[state]
        print(f"FSM State {state}:")
        print(f"  Count:       {len(errors)}")
        print(f"  Max error:   {np.max(errors):.6e}")
        print(f"  Mean error:  {np.mean(errors):.6e}")
        print(f"  Median error: {np.median(errors):.6e}")

    # Find top 10 worst cases
    print("\n" + "="*70)
    print("TOP 10 WORST CASES")
    print("="*70)
    print("Rank | Case | Time     | FSM | Max Error")
    print("-----|------|----------|-----|----------------")

    sorted_errors = sorted(all_errors, key=lambda x: x['max_error'], reverse=True)
    for rank, e in enumerate(sorted_errors[:10], 1):
        print(f" {rank:2d}  | {e['case']:4d} | {e['t']:8.4f} | {e['fsm_state']:3d} | {e['max_error']:14.6e}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
