# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains MATLAB, Python, C++ and CUDA implementations of a Raibert hopper - a one-legged robotic hopper that uses a finite state machine to control hopping motion. The simulation includes dynamics, control, visualization, and energy analysis.

## Running the Simulation

### Python Version

**Environment Setup (Windows dev machine):**

Anaconda Python location: `C:\Users\wongj\anaconda3\python.exe`

**Option 1: Virtual environment (venv)**
```bash
# Create venv (run from project root)
"C:\Users\wongj\anaconda3\python.exe" -m venv .venv

# Activate in Git Bash
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Conda environment**
```bash
# Initialize conda for Git Bash
eval "$("C:\Users\wongj\anaconda3\Scripts\conda.exe" shell.bash hook)"

# Create and activate environment
conda env create -f environment.yaml
conda activate hopper
```

**Running the python simulation:**

```bash
cd src
python hopper.py
```

Or run the Jupyter notebook:

```bash
jupyter notebook hopper_demo.ipynb
```

### MATLAB Version

Open MATLAB, navigate to project directory, run `call_hopper`

## Code Architecture

### State Variables (10-dimensional state vector q)
1. q(1) - x position of foot
2. q(2) - y position of foot  
3. q(3) - absolute angle of leg (from vertical)
4. q(4) - absolute angle of body (from vertical)
5. q(5) - leg length
6. q(6:10) - time derivatives of q(1:5)

### Control Variables
- u(1) - position of leg spring actuator (axial force)
- u(2) - torque at the hip

### Finite State Machine (4 states)
- FSM_COMPRESSION (0): Leg compressing after ground contact
- FSM_THRUST (1): Active thrust phase to add energy
- FSM_FLIGHT (2): Airborne phase with foot placement control

### Core Files Structure

#### MATLAB Files

**Main Simulation Loop**: `call_hopper.m`
- ODE integration with event detection
- State machine transitions
- Data collection and visualization

**Dynamics**:
- `raibertHopperDynamics.m` - wrapper calling forward dynamics
- `raibertHopperDynamicsFwd.m` - full dynamics computation with control input

**Control**: `raibertStateControl.m`
- State-dependent control laws
- Foot placement for velocity control during flight
- Attitude control during stance

**Events**: `eventsHopperControl.m`
- State transition conditions
- Ground contact detection
- Leg compression/extension events

**Visualization**: `draw.m`
- Real-time animated display of hopper
- Persistent figure management with automatic camera tracking

**Configuration**: `defaultRaibertParameters.m`
- Physical parameters (masses, inertias, spring constants)
- FSM state definitions
- Control gains and timing parameters

**Energy Analysis**: `RaibertEnergy.m` (incomplete)
- Kinetic and potential energy calculations

#### Python Files

**Main Module**: `src/hopper.py` (complete standalone implementation)
- All-in-one Python module containing all functions for the hopper simulation
- Can be run directly: `python hopper.py`

**Key Functions in hopper.py**:
- `hopperParameters()` - Returns Parameters dataclass with physical constants and FSM configuration
- `hopperStateControl(t, q, param)` - Implements state-dependent control (foot placement, thrust, attitude)
- `hopperDynamicsFwd(t, q, p_obj)` - Computes forward dynamics with control inputs
- `hopperDynamics(t, q, p_obj)` - Wrapper for dynamics (used by ODE solver)
- `eventsHopperControl(t, q, param)` - Event detection for FSM state transitions
- `hopperEnergy(t, State, P)` - Complete energy accounting (kinetic, potential, spring, damping losses, control work)
- `draw(p_obj, t, q)` - Real-time visualization with matplotlib
- `call_hopper(...)` - Main simulation loop with event-driven integration using scipy.solve_ivp

**Test Files**:
- `src/test_hopper.py` - Unit tests for dynamics functions
- `src/compare_matlab_python.py` - Comparison utility between MATLAB and Python implementations

**Jupyter Notebook**: `hopper_demo.ipynb`
- Interactive demonstration of the hopper simulation
- Includes documentation, visualization, and animation
- Colab-compatible with installation instructions
- Uses HTML5 video for animation display in notebooks

**Python Implementation Details**:
- Uses `scipy.integrate.solve_ivp` with Radau method (stiff ODE solver) for robust event detection
- Event detection with terminal events for FSM transitions
- Deep copying of parameters to avoid closure issues in lambda functions
- Animation via `matplotlib.animation.FuncAnimation`
- Energy analysis with `scipy.integrate.cumulative_trapezoid`

## Key Control Strategies

1. **Velocity Control**: Achieved through foot placement during flight phase
2. **Attitude Control**: Hip torque to maintain body orientation
3. **Energy Injection**: Thrust during stance phase to maintain hopping
4. **Adaptive Timing**: Contact time estimation for improved foot placement

## Common Development Tasks

### MATLAB

To run the simulation: Open MATLAB, navigate to project directory, run `call_hopper`

To modify parameters: Edit values in `defaultRaibertParameters.m`

To adjust control gains: Modify parameters in `raibertStateControl.m` (k_fp, b_fp, k_att, b_att, k_xdot, thrust)

To change visualization: Modify `draw.m` or adjust video parameters in `call_hopper.m` (sr_video, ds)

### Python

To run the simulation:
```bash
eval "$(/opt/miniconda3/bin/conda shell.bash hook)" && conda activate hopper
cd src
python hopper.py
```

To modify parameters: Edit the `Parameters` dataclass in `src/hopper.py` or modify values in `hopperParameters()` function

To adjust control gains: Modify constants in `hopperStateControl()` function in `src/hopper.py` (k_fp, b_fp, k_att, b_att, k_xdot, thrust)

To change simulation parameters: Modify arguments to `call_hopper()`:
- `tfinal` - simulation duration (default: 5s)
- `x_dot_des` - desired forward velocity (default: 3.0 m/s)
- `y0` - initial state vector
- `sr_sim` - simulation sample rate (default: 1000 Hz)
- `save_figures` - whether to save plots and animation (default: True)

To run in Jupyter: `jupyter notebook hopper_demo.ipynb`

### C++ and CUDA

The C++ implementation has two different implementations for the ODE Solver:
ODE45-based and Implicit

**Build Environment (Windows with Git Bash):**

Since Claude Code runs in Git Bash, use full paths to the compilers:

- **MSVC (cl.exe)**: `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"`
- **CUDA (nvcc)**: `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"`

**Building C++ files:**
```bash
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" /std:c++17 /O2 /EHsc cpp/test_implicit_cpu.cpp /Fe:cpp/test_implicit_cpu.exe
```

**Building CUDA files:**
nvcc needs cl.exe in PATH or use `-ccbin` to specify it:
```bash
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -O2 cpp/cuda/test_cuda.cu -o cpp/cuda/test_cuda.exe
```

**Note**: Paths with spaces must be quoted. The `-ccbin` flag tells nvcc where to find the host compiler (cl.exe).