from pathlib import Path

import numpy as np

from coaxial_multirotor import build_default_config, run_closed_loop_simulation


def main() -> None:
    system = build_default_config()
    log = run_closed_loop_simulation(system)
    final_position = log["position_m"][-1]
    final_attitude = log["euler_deg"][-1]
    print("Simulation complete")
    print(f"Final position [m]: {np.array2string(final_position, precision=3)}")
    print(f"Final attitude [deg]: {np.array2string(final_attitude, precision=3)}")
    print(f"Plot saved to: {Path(system.logging.save_plot_path).resolve()}")


if __name__ == "__main__":
    main()
