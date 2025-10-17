# RaibertFast - One-Legged Hopping Robot Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/RaibertFast/blob/master/hopper_demo.ipynb)

A Python implementation of the classic Raibert hopping robot controller. This simulation demonstrates a one-legged hopping robot that uses three key control strategies:

1. **Forward Speed Control via Foot Placement** - The foot is placed ahead or behind the body to control forward velocity
2. **Hopping Height Control via Thrust Timing** - Thrust force is applied during stance to maintain hopping height
3. **Body Attitude Control** - Body angle is controlled during stance for stability

## Quick Start

Click the "Open in Colab" badge above to run the simulation in your browser without any installation!

## Local Installation

```bash
# Clone the repository
git clone https://github.com/jeremydwong/RaibertFast.git
cd RaibertFast

# Install dependencies
pip install numpy matplotlib scipy
```

## Running the Simulation

### Python Script
```python
from src.hopper import call_hopper

# Run simulation with desired velocity of 2 m/s
tout, yout, State, p, anim = call_hopper(tfinal=5, x_dot_des=2.0)
```

### Jupyter Notebook
Open `hopper_demo.ipynb` in Jupyter or Colab to run an interactive simulation with visualization.

## Features

- Full dynamics simulation using Lagrangian mechanics
- Finite state machine for stance/flight phase transitions
- Real-time animation of hopping behavior
- Energy analysis and accounting
- Configurable parameters (velocity, initial conditions, etc.)

## Controller States

The controller uses a finite state machine with three states:
- **COMPRESSION**: Leg is compressing after touchdown
- **THRUST**: Active thrust is applied to the leg
- **FLIGHT**: Robot is in the air

## References

- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
- Tedrake, R. (2024). *Underactuated Robotics*. MIT Press.

## License

MIT License
