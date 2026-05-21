from pathlib import Path

import numpy as np

from coaxial_multirotor import build_default_config
from coaxial_multirotor.replay import run_lqr_replay_comparison


def main() -> None:
    system = build_default_config()
    result = run_lqr_replay_comparison(system)
    print("LQR replay comparison complete")
    for swing_model, log in result["logs"].items():
        final_position = log["position_m"][-1]
        final_swing = np.array([log["gyro_angle_x_rad"][-1], log["gyro_angle_y_rad"][-1]])
        print(f"[{swing_model}] Final position [m]: {np.array2string(final_position, precision=3)}")
        print(f"[{swing_model}] Final swing angles [deg]: {np.array2string(np.rad2deg(final_swing), precision=3)}")
        print(f"[{swing_model}] Summary plot: {Path(log['plot_path']).resolve()}")
    print(f"Comparison plot: {Path(result['plot_path']).resolve()}")


if __name__ == "__main__":
    main()
