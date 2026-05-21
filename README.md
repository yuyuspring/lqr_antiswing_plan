# Coaxial Multirotor Simulation

## Overview

This project implements a truth-state closed-loop simulation for the coaxial 8-rotor aircraft described in the markdown specifications.

Implemented modules:
- 6-DOF rigid-body dynamics
- 8-motor first-order actuator dynamics
- Cascaded position, velocity, attitude, and body-rate controller
- Motor allocation for M1-M8
- Simple phase-based planner for takeoff, hover, translation, hold, and landing
- Plot generation for trajectory, attitude, rates, motor commands, and servo channels

## Assumptions

- World frame uses z-up.
- Body frame uses x-forward, y-right, z-up.
- Motor PWM linearly maps to commanded RPM, while thrust and reaction torque scale with squared RPM ratio.
- Lower coaxial rotors use a fixed 0.67 thrust scale.
- Truth state is fed directly to the controller.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run simulation:

```bash
python3 simulate.py
```

Output plot:
- outputs/closed_loop_summary.png

## Main files

- coaxial_multirotor/config.py
- coaxial_multirotor/dynamics.py
- coaxial_multirotor/controllers.py
- coaxial_multirotor/planner.py
- coaxial_multirotor/simulation.py
- simulate.py
