from pathlib import Path
from typing import Optional

import numpy as np

from .config import SystemConfig, build_default_config
from .controllers import CoaxialController
from .dynamics import CoaxialMultirotorDynamics, VehicleState
from .planner import WaypointPlanner
from .plotting import plot_summary


def run_closed_loop_simulation(system: Optional[SystemConfig] = None) -> dict:
    system = system or build_default_config()
    state = VehicleState.from_config(system)
    controller = CoaxialController.build(system)
    planner = WaypointPlanner(system.scenario)
    dynamics = CoaxialMultirotorDynamics(system)

    sample_count = int(system.simulation.duration_s / system.simulation.dt_s) + 1
    log = {
        "time_s": np.zeros(sample_count),
        "position_m": np.zeros((sample_count, 3)),
        "velocity_mps": np.zeros((sample_count, 3)),
        "euler_deg": np.zeros((sample_count, 3)),
        "body_rates_degps": np.zeros((sample_count, 3)),
        "accel_world_mps2": np.zeros((sample_count, 3)),
        "motor_pwm": np.zeros((sample_count, 8)),
        "motor_rpm": np.zeros((sample_count, 8)),
        "position_cmd_m": np.zeros((sample_count, 3)),
        "attitude_cmd_deg": np.zeros((sample_count, 2)),
        "yaw_cmd_deg": np.zeros(sample_count),
        "servo": np.zeros((sample_count, 4)),
    }

    for index in range(sample_count):
        time_s = index * system.simulation.dt_s
        command = planner.sample(time_s)
        truth = state.as_truth()
        control = controller.step(command, truth, system.simulation.dt_s, system.vehicle.gravity_mps2)
        state = dynamics.step(state, control["motor_pwm"], system.simulation.dt_s)
        truth = state.as_truth()

        log["time_s"][index] = time_s
        log["position_m"][index] = truth["position_m"]
        log["velocity_mps"][index] = truth["velocity_mps"]
        log["euler_deg"][index] = truth["euler_deg"]
        log["body_rates_degps"][index] = truth["body_rates_degps"]
        log["accel_world_mps2"][index] = truth["accel_world_mps2"]
        log["motor_pwm"][index] = control["motor_pwm"]
        log["motor_rpm"][index] = truth["motor_rpm"]
        log["position_cmd_m"][index] = np.array([command["x_m"], command["y_m"], command["z_m"]])
        log["attitude_cmd_deg"][index] = np.array([control["roll_cmd_deg"], control["pitch_cmd_deg"]])
        log["yaw_cmd_deg"][index] = command["yaw_deg"]
        log["servo"][index] = np.array([control["servo_thro"], control["servo_roll"], control["servo_pitch"], control["servo_yaw"]])

    Path(system.logging.output_dir).mkdir(parents=True, exist_ok=True)
    plot_summary(log, system.logging.save_plot_path)
    return log
