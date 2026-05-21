# Coaxial Multirotor Simulation

## Overview

This project implements a truth-state closed-loop simulation for the coaxial 8-rotor aircraft described in the markdown specifications.

Implemented modules:
- 6-DOF rigid-body dynamics
- 3D suspended-load model with rigid massless rope and payload back-reaction on UAV translation
- 8-motor first-order actuator dynamics
- Cascaded position, velocity, attitude, and body-rate controller
- World/body velocity-acceleration tracking modes for trajectory-following inputs
- Motor allocation for M1-M8
- Simple phase-based planner for takeoff, hover, translation, hold, and landing
- LQR replay pipeline compatible with the xp_16_7 batch-processing flow
- Plot generation for trajectory, attitude, rates, motor commands, and servo channels

## Assumptions

- World frame uses z-up.
- Body frame uses x-forward, y-right, z-up.
- Swing log sign follows xp_16_7 semantics: `gyro_angle_y` front-positive and `gyro_angle_x` left-positive.
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

Run LQR dataset replay:

```bash
python3 replay_lqr_dataset.py
```

Output plot:
- outputs/closed_loop_summary.png
- outputs/lqr_replay_summary.png

## Main files

- coaxial_multirotor/config.py
- coaxial_multirotor/dynamics.py
- coaxial_multirotor/controllers.py
- coaxial_multirotor/suspended_load.py
- coaxial_multirotor/lqr/standalone_lqr.py
- coaxial_multirotor/lqr/lqr_swing_controller.py
- coaxial_multirotor/replay.py
- coaxial_multirotor/planner.py
- coaxial_multirotor/simulation.py
- simulate.py
- replay_lqr_dataset.py
