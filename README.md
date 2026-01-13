# RaibertFast - Massively Parallel Hopping Robot Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/RaibertFast/blob/master/hopper_demo.ipynb)

A GPU-accelerated implementation of the classic Raibert hopping robot controller. Simulates thousands of one-legged hoppers in parallel using CUDA.

![64 Hoppers Animation](docs/multi_hopper_demo.gif)

## Performance

| Configuration | 4096 hoppers × 5s | Wall Clock |
|--------------|-------------------|------------|
| CPU (serial) | - | 15.0 s |
| GPU (FP32) | 16× faster | 0.77 s |

Kernel time: 133ms for 4096 hoppers (37× realtime)

## Quick Start

**Run in browser:** Click the Colab badge above - no installation needed!

**Local GPU simulation:**
```bash
# Build (Windows with CUDA + MSVC)
nvcc -O2 -DHOPPER_USE_FLOAT32 cpp/cuda/test_cuda.cu -o test_cuda.exe

# Run 64 hoppers, export trajectories
./test_cuda.exe --multi -n 64 -t 5.0 -o demo

# Benchmark mode (no file I/O)
./test_cuda.exe --multi -n 4096 -t 5.0 --no-export

# Visualize
python src/visualize_multi_hopper_matplotlib.py . demo 64
```

## Architecture

### GPU Design: Thread-per-Hopper

Each CUDA thread simulates one complete hopper trajectory:

```
┌─────────────────────────────────────────────────────────┐
│  GPU Kernel (single launch)                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐   │
│  │Thread 0 │ │Thread 1 │ │Thread 2 │ ... │Thread N │   │
│  │Hopper 0 │ │Hopper 1 │ │Hopper 2 │     │Hopper N │   │
│  │         │ │         │ │         │     │         │   │
│  │ 50,000  │ │ 50,000  │ │ 50,000  │     │ 50,000  │   │
│  │ steps   │ │ steps   │ │ steps   │     │ steps   │   │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘   │
└─────────────────────────────────────────────────────────┘
```

- **State in registers**: 10 state variables + FSM state per thread
- **No inter-thread communication**: Hoppers are independent
- **Single kernel launch**: Avoids launch overhead (vs batched multi-kernel)
- **FP32 mode**: 6-9× faster than FP64 on consumer GPUs

### Integrators

| Integrator | Order | Compile Flag |
|------------|-------|--------------|
| Semi-Implicit Euler | 1st | `-DHOPPER_INTEGRATOR=2` |
| Implicit Midpoint | 2nd | `-DHOPPER_INTEGRATOR=1` |
| Implicit Midpoint (analytical Jacobian) | 2nd | `-DHOPPER_INTEGRATOR=4` |

### Precision

```cpp
// In hopper_types.cuh - single source of truth
#ifdef HOPPER_USE_FLOAT32
    using Scalar = float;   // Fast on consumer GPUs
#else
    using Scalar = double;  // Higher precision
#endif
```

## The Raibert Controller

Three key control strategies for stable hopping:

1. **Forward Speed Control** - Foot placement during flight phase
2. **Hopping Height Control** - Thrust timing during stance
3. **Body Attitude Control** - Hip torque for balance

### Finite State Machine

```
       ┌──────────┐
       │  FLIGHT  │◄────────────────┐
       └────┬─────┘                 │
            │ touchdown             │ liftoff
            ▼                       │
       ┌──────────┐            ┌────┴─────┐
       │COMPRESS- │───────────►│  THRUST  │
       │   ION    │ leg fully  └──────────┘
       └──────────┘ compressed
```

### State Variables (10D)

| Variable | Description |
|----------|-------------|
| x_foot, z_foot | Foot position |
| phi_leg | Leg angle from vertical |
| phi_body | Body angle from vertical |
| len_leg | Leg spring length |
| ddt_* | Time derivatives |

## Project Structure

```
RaibertFast/
├── src/
│   ├── hopper.py                    # Python reference implementation
│   └── visualize_*.py               # Visualization scripts
├── cpp/
│   ├── cuda/
│   │   ├── test_cuda.cu             # Main GPU simulation
│   │   ├── hopper_cuda.cuh          # Dynamics & control
│   │   ├── integrator_cuda.cuh      # Integrator implementations
│   │   └── hopper_types.cuh         # Scalar typedef, structs
│   └── test_implicit_cpu.cpp        # CPU version (same headers)
├── MATLAB/                          # Original MATLAB implementation
└── README.md
```

## Building

### Requirements

- CUDA Toolkit 11.0+
- MSVC (Windows) or GCC (Linux)
- Python 3.8+ with numpy, matplotlib (for visualization)

### Windows (MSVC + CUDA)

```bash
# Set up environment
export CCBIN="/path/to/MSVC/bin/Hostx64/x64"

# GPU build
nvcc -O2 -DHOPPER_USE_FLOAT32 -DHOPPER_INTEGRATOR=2 \
    cpp/cuda/test_cuda.cu -o test_cuda.exe

# CPU build (uses same headers)
nvcc -O2 -DHOPPER_USE_FLOAT32 -x cu \
    cpp/test_implicit_cpu.cpp -o test_cpu.exe
```

### Linux

```bash
nvcc -O2 -DHOPPER_USE_FLOAT32 cpp/cuda/test_cuda.cu -o test_cuda
```

## Command Line Options

```
./test_cuda.exe [options]
  --test         Run unit tests
  --sim          Single hopper, export trajectory
  --multi        Parallel multi-hopper simulation
  --no-export    Skip CSV export (benchmark mode)
  -n <num>       Number of hoppers (default: 100)
  -t <time>      Simulation duration (default: 5.0)
  -o <prefix>    Output file prefix
```

## References

- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
- Tedrake, R. (2024). *Underactuated Robotics*. MIT Press.

## License

MIT License
