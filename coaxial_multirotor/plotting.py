from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_summary(log: dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    time_s = log["time_s"]

    figure, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(time_s, log["position_m"][:, 0], label="x")
    axes[0, 0].plot(time_s, log["position_m"][:, 1], label="y")
    axes[0, 0].plot(time_s, log["position_m"][:, 2], label="z")
    axes[0, 0].plot(time_s, log["position_cmd_m"][:, 0], "--", label="x_cmd")
    axes[0, 0].plot(time_s, log["position_cmd_m"][:, 1], "--", label="y_cmd")
    axes[0, 0].plot(time_s, log["position_cmd_m"][:, 2], "--", label="z_cmd")
    axes[0, 0].set_title("Position")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].grid(True)
    axes[0, 0].legend(ncol=3)

    axes[0, 1].plot(time_s, log["velocity_mps"][:, 0], label="vx")
    axes[0, 1].plot(time_s, log["velocity_mps"][:, 1], label="vy")
    axes[0, 1].plot(time_s, log["velocity_mps"][:, 2], label="vz")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].set_ylabel("m/s")
    axes[0, 1].grid(True)
    axes[0, 1].legend(ncol=3)

    axes[1, 0].plot(time_s, log["euler_deg"][:, 0], label="roll")
    axes[1, 0].plot(time_s, log["euler_deg"][:, 1], label="pitch")
    axes[1, 0].plot(time_s, log["euler_deg"][:, 2], label="yaw")
    axes[1, 0].plot(time_s, log["attitude_cmd_deg"][:, 0], "--", label="roll_cmd")
    axes[1, 0].plot(time_s, log["attitude_cmd_deg"][:, 1], "--", label="pitch_cmd")
    axes[1, 0].plot(time_s, np.full_like(time_s, log["yaw_cmd_deg"][0]), "--", label="yaw_cmd")
    axes[1, 0].set_title("Attitude")
    axes[1, 0].set_ylabel("deg")
    axes[1, 0].grid(True)
    axes[1, 0].legend(ncol=3)

    axes[1, 1].plot(time_s, log["body_rates_degps"][:, 0], label="p")
    axes[1, 1].plot(time_s, log["body_rates_degps"][:, 1], label="q")
    axes[1, 1].plot(time_s, log["body_rates_degps"][:, 2], label="r")
    axes[1, 1].set_title("Body Rates")
    axes[1, 1].set_ylabel("deg/s")
    axes[1, 1].grid(True)
    axes[1, 1].legend(ncol=3)

    axes[2, 0].plot(time_s, log["motor_pwm"])
    axes[2, 0].set_title("Motor PWM")
    axes[2, 0].set_xlabel("s")
    axes[2, 0].set_ylabel("pwm")
    axes[2, 0].grid(True)

    axes[2, 1].plot(time_s, log["servo"][:, 0], label="thro")
    axes[2, 1].plot(time_s, log["servo"][:, 1], label="roll")
    axes[2, 1].plot(time_s, log["servo"][:, 2], label="pitch")
    axes[2, 1].plot(time_s, log["servo"][:, 3], label="yaw")
    axes[2, 1].set_title("Servo Channels")
    axes[2, 1].set_xlabel("s")
    axes[2, 1].grid(True)
    axes[2, 1].legend(ncol=2)

    figure.tight_layout()
    figure.savefig(save_path, dpi=160)
    plt.close(figure)
