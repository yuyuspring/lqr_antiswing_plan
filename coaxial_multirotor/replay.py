from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .config import SystemConfig, build_default_config
from .controllers import CoaxialController
from .dynamics import CoaxialMultirotorDynamics, VehicleState
from .lqr import LqrSwingController
from .lqr.lqr_swing_controller import Config as LqrControllerConfig


def load_lqr_batches(csv_path: str) -> List[List[dict]]:
    grouped: OrderedDict[str, List[dict]] = OrderedDict()
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            grouped.setdefault(row["timestamp_us"], []).append(row)
    batches = []
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["point_idx"]))
        batches.append(rows)
    return batches


def plot_replay_summary(log: dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    time_s = log["time_s"]
    figure, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(time_s, log["position_m"][:, 0], label="north/x")
    axes[0, 0].plot(time_s, log["position_m"][:, 1], label="east/y")
    axes[0, 0].set_title("UAV Position")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(time_s, log["body_velocity_ref_mps"][:, 0], "--", label="vx_orig")
    axes[0, 1].plot(time_s, log["body_velocity_lqr_mps"][:, 0], label="vx_lqr")
    axes[0, 1].plot(time_s, log["body_velocity_mps"][:, 0], label="vx_resp")
    axes[0, 1].plot(time_s, log["body_velocity_ref_mps"][:, 1], "--", label="vy_orig")
    axes[0, 1].plot(time_s, log["body_velocity_lqr_mps"][:, 1], label="vy_lqr")
    axes[0, 1].plot(time_s, log["body_velocity_mps"][:, 1], label="vy_resp")
    axes[0, 1].set_title("Body Velocity: Orig vs LQR vs Response")
    axes[0, 1].grid(True)
    axes[0, 1].legend(ncol=2)

    axes[1, 0].plot(time_s, np.rad2deg(log["gyro_angle_x_rad"]), label="gyro_angle_x")
    axes[1, 0].plot(time_s, np.rad2deg(log["gyro_angle_y_rad"]), label="gyro_angle_y")
    axes[1, 0].set_title("Swing Angles")
    axes[1, 0].set_ylabel("deg")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(time_s, np.rad2deg(log["gyro_rate_x_radps"]), label="gyro_rate_x")
    axes[1, 1].plot(time_s, np.rad2deg(log["gyro_rate_y_radps"]), label="gyro_rate_y")
    axes[1, 1].set_title("Swing Rates")
    axes[1, 1].set_ylabel("deg/s")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[2, 0].plot(time_s, log["body_accel_ref_mps2"][:, 0], "--", label="ax_orig")
    axes[2, 0].plot(time_s, log["body_accel_cmd_mps2"][:, 0], label="ax_lqr")
    axes[2, 0].plot(time_s, log["body_accel_mps2"][:, 0], label="ax_resp")
    axes[2, 0].plot(time_s, log["body_accel_ref_mps2"][:, 1], "--", label="ay_orig")
    axes[2, 0].plot(time_s, log["body_accel_cmd_mps2"][:, 1], label="ay_lqr")
    axes[2, 0].plot(time_s, log["body_accel_mps2"][:, 1], label="ay_resp")
    axes[2, 0].set_title("Body Accel: Orig vs LQR vs Response")
    axes[2, 0].grid(True)
    axes[2, 0].legend(ncol=2)

    axes[2, 1].plot(time_s, log["motor_pwm"])
    axes[2, 1].set_title("Motor PWM")
    axes[2, 1].grid(True)

    figure.tight_layout()
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def run_lqr_replay_simulation(system: SystemConfig | None = None) -> dict:
    system = system or build_default_config()
    state = VehicleState.from_config(system)
    controller = CoaxialController.build(system)
    dynamics = CoaxialMultirotorDynamics(system)
    lqr_controller = LqrSwingController(
        LqrControllerConfig(
            ropeLength=system.suspended_load.rope_length_m,
            payloadMass=system.suspended_load.payload_mass_kg,
            droneMass=system.vehicle.mass_kg,
            dt=system.lqr.dt_s,
            lqrAxMax=system.lqr.accel_limit_mps2,
            lqrJerkMax=system.lqr.jerk_limit_mps3,
        )
    )
    batches = load_lqr_batches(system.replay.dataset_csv_path)
    step_dt_s = system.replay.batch_dt_s
    execute_steps = system.replay.execute_steps_per_batch
    total_steps = sum(min(len(batch), execute_steps) for batch in batches)

    log = {
        "time_s": np.zeros(total_steps),
        "position_m": np.zeros((total_steps, 3)),
        "body_velocity_mps": np.zeros((total_steps, 3)),
        "body_velocity_ref_mps": np.zeros((total_steps, 2)),
        "body_velocity_lqr_mps": np.zeros((total_steps, 2)),
        "body_accel_ref_mps2": np.zeros((total_steps, 2)),
        "body_accel_cmd_mps2": np.zeros((total_steps, 2)),
        "body_accel_mps2": np.zeros((total_steps, 3)),
        "gyro_angle_x_rad": np.zeros(total_steps),
        "gyro_angle_y_rad": np.zeros(total_steps),
        "gyro_rate_x_radps": np.zeros(total_steps),
        "gyro_rate_y_radps": np.zeros(total_steps),
        "motor_pwm": np.zeros((total_steps, 8)),
    }

    step_index = 0
    for batch in batches:
        truth = state.as_truth()
        body_velocity = truth["body_velocity_mps"]
        theta_body_x = truth["gyro_angle_y_rad"]
        omega_body_x = truth["gyro_rate_y_radps"]
        theta_body_y = -truth["gyro_angle_x_rad"]
        omega_body_y = -truth["gyro_rate_x_radps"]
        lqr_controller.update_params(system.suspended_load.rope_length_m, system.suspended_load.payload_mass_kg, system.vehicle.mass_kg)
        vx_ref_body = [float(row["orig_body_vel_x"]) for row in batch]
        vy_ref_body = [float(row["orig_body_vel_y"]) for row in batch]
        ax_ref_body = [float(row["orig_body_acc_x"]) for row in batch]
        ay_ref_body = [float(row["orig_body_acc_y"]) for row in batch]
        batch_output = lqr_controller.process_batch(
            vx_ref_body=vx_ref_body,
            vy_ref_body=vy_ref_body,
            ax_ref_body=ax_ref_body,
            ay_ref_body=ay_ref_body,
            px0_body=0.0,
            vx0_body=body_velocity[0],
            theta_body_x0=theta_body_x,
            omega_body_x0=omega_body_x,
            py0_body=0.0,
            vy0_body=body_velocity[1],
            theta_body_y0=theta_body_y,
            omega_body_y0=omega_body_y,
        )
        for local_index in range(min(len(batch), execute_steps)):
            truth = state.as_truth()
            command = {
                "horizontal_mode": "body_velocity_accel",
                "vx_body_mps": batch_output["vx"][local_index],
                "vy_body_mps": batch_output["vy"][local_index],
                "ax_body_mps2": batch_output["ax"][local_index],
                "ay_body_mps2": batch_output["ay"][local_index],
                "z_m": 0.0,
                "yaw_deg": 0.0,
            }
            control = controller.step(command, truth, step_dt_s, system.vehicle.gravity_mps2)
            state = dynamics.step(state, control["motor_pwm"], step_dt_s)
            truth = state.as_truth()
            log["time_s"][step_index] = step_index * step_dt_s
            log["position_m"][step_index] = truth["position_m"]
            log["body_velocity_mps"][step_index] = truth["body_velocity_mps"]
            log["body_velocity_ref_mps"][step_index] = np.array([vx_ref_body[local_index], vy_ref_body[local_index]])
            log["body_velocity_lqr_mps"][step_index] = np.array([batch_output["vx"][local_index], batch_output["vy"][local_index]])
            log["body_accel_ref_mps2"][step_index] = np.array([ax_ref_body[local_index], ay_ref_body[local_index]])
            log["body_accel_cmd_mps2"][step_index] = np.array([batch_output["ax"][local_index], batch_output["ay"][local_index]])
            log["body_accel_mps2"][step_index] = truth["body_accel_mps2"]
            log["gyro_angle_x_rad"][step_index] = truth["gyro_angle_x_rad"]
            log["gyro_angle_y_rad"][step_index] = truth["gyro_angle_y_rad"]
            log["gyro_rate_x_radps"][step_index] = truth["gyro_rate_x_radps"]
            log["gyro_rate_y_radps"][step_index] = truth["gyro_rate_y_radps"]
            log["motor_pwm"][step_index] = control["motor_pwm"]
            step_index += 1

    plot_path = str(Path(system.logging.output_dir) / "lqr_replay_summary.png")
    plot_replay_summary(log, plot_path)
    log["plot_path"] = plot_path
    return log
