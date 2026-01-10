# RaibertFast - One-Legged Hopping Robot Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/RaibertFast/blob/master/hopper_demo.ipynb)

A multi-language implementation of the classic Raibert hopping robot controller. This simulation demonstrates a one-legged hopping robot that uses three key control strategies:

1. **Forward Speed Control via Foot Placement** - The foot is placed ahead or behind the body to control forward velocity
2. **Hopping Height Control via Thrust Timing** - Thrust force is applied during stance to maintain hopping height
3. **Body Attitude Control** - Body angle is controlled during stance for stability

## Implementations

| Language | Integrator | Purpose | Location |
|----------|------------|---------|----------|
| **Python** | RK45 (adaptive) | Reference, visualization | `src/hopper.py` |
| **MATLAB** | ode45 (adaptive) | Reference | `*.m` files |
| **C++** | RK45 (adaptive) | Fast CPU simulation | `cpp/` |
| **CUDA** | Implicit Midpoint (fixed-step) | Massively parallel GPU simulation | `cpp/cuda/` |

## Quick Start

Click the "Open in Colab" badge above to run the Python simulation in your browser without any installation!

## Local Installation

### Python

**Option 1: Virtual Environment (recommended)**
```bash
git clone https://github.com/jeremydwong/RaibertFast.git
cd RaibertFast

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Conda**
```bash
git clone https://github.com/jeremydwong/RaibertFast.git
cd RaibertFast
conda env create -f environment.yaml
conda activate hopper
```

### C++ (requires Eigen)
```bash
cd cpp
# Linux/Mac
./build.sh

# Windows (Visual Studio)
cl /std:c++17 /O2 /I"C:\libs\eigen-3.4.1" build.cpp /Fe:hopper.exe
```

### CUDA (requires CUDA Toolkit + Eigen)
```bash
cd cpp/cuda
# Edit build.bat to set your CUDA/Eigen paths, then:
./build.bat
```

## Running the Simulation

### Python
```python
from src.hopper import call_hopper

# Run simulation with desired velocity of 2 m/s
tout, yout, State, p, anim = call_hopper(tfinal=5, x_dot_des=2.0)
```

Or run directly:
```bash
cd src && python hopper.py
```

### C++
```bash
cd cpp
./hopper -t 5.0 -v 3.0    # 5 seconds, target 3 m/s
```
Exports trajectory to `trajectory.csv`.

### CUDA
```bash
cd cpp/cuda

# Run tests
./test_cuda.exe --test

# Run simulation and export trajectory
./test_cuda.exe --sim -t 5.0 -o trajectory_cuda.csv
```

### Jupyter Notebook
Open `hopper_demo.ipynb` in Jupyter or Colab for interactive simulation with visualization.

## Architecture

### State Variables (10D)
- Positions (5): foot x, foot z, leg angle, body angle, leg length
- Velocities (5): time derivatives of above

### Finite State Machine
- **COMPRESSION**: Leg compressing after touchdown
- **THRUST**: Active thrust applied to leg spring
- **FLIGHT**: Robot is airborne, foot placement control active

### CUDA Design

The CUDA implementation uses:
- **Implicit Midpoint Integrator**: Second-order, symplectic, unconditionally stable
- **Fixed timestep** (dt=1e-4): Avoids warp divergence from adaptive stepping
- **Structure of Arrays (SoA)**: Coalesced GPU memory access
- **Schur complement**: Reduces 10x10 Newton solve to 5x5

See `cpp/DesignCuda.md` for detailed design documentation.

## Visualization

Python scripts can visualize trajectories from any implementation:
```bash
python src/visualize_cpp_trajectory.py cpp/trajectory.csv
python src/visualize_cpp_trajectory.py cpp/cuda/trajectory_cuda.csv
```

## Features

- Full dynamics simulation using Lagrangian mechanics
- Finite state machine for stance/flight phase transitions
- Real-time animation of hopping behavior
- Energy analysis and accounting
- Configurable parameters (velocity, initial conditions, etc.)
- CSV trajectory export for cross-implementation comparison

## References

- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
- Tedrake, R. (2024). *Underactuated Robotics*. MIT Press.

## License

MIT License
