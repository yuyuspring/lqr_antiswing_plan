from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .suspended_load import SuspendedLoadConfig


@dataclass
class VehicleConfig:
    mass_kg: float = 120.0
    gravity_mps2: float = 9.81
    inertia_kgm2: np.ndarray = field(
        default_factory=lambda: np.diag([67.0, 67.0, 75.0])
    )
    arm_length_m: float = 1.0
    upper_lower_thrust_ratio: float = 0.67
    max_thrust_kgf: float = 100.0
    max_reaction_torque_nm: float = 80.0
    max_rpm: float = 1800.0
    max_pwm: float = 1000.0
    motor_bandwidth_hz: float = 2.0

    @property
    def max_thrust_n(self) -> float:
        return self.max_thrust_kgf * self.gravity_mps2

    @property
    def motor_time_constant_s(self) -> float:
        return 1.0 / (2.0 * np.pi * self.motor_bandwidth_hz)


@dataclass
class ActuatorLayout:
    positions_m: np.ndarray
    yaw_signs: np.ndarray
    layer_scale: np.ndarray


@dataclass
class SimulationConfig:
    dt_s: float = 0.01
    duration_s: float = 55.0
    initial_position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_euler_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_body_rates_dps: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ScenarioConfig:
    altitude_step_m: float = 10.0
    horizontal_target_m: np.ndarray = field(default_factory=lambda: np.array([20.0, 10.0]))
    yaw_target_deg: float = 0.0
    phase_times_s: Dict[str, float] = field(
        default_factory=lambda: {
            "idle": 2.0,
            "climb_end": 8.0,
            "hover_end": 20.0,
            "translate_end": 35.0,
            "hold_end": 45.0,
            "land_end": 55.0,
        }
    )


@dataclass
class HorizontalMappingConfig:
    pitch_from_ax_sign: float = 1.0
    roll_from_ay_sign: float = -1.0
    max_tilt_deg: float = 35.0


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"
    save_plot_path: str = "outputs/closed_loop_summary.png"


@dataclass
class LqrConfig:
    dt_s: float = 0.02
    lookahead_steps: int = 0
    executed_steps: int = 5
    accel_limit_mps2: float = 5.0
    jerk_limit_mps3: float = 10.0
    gain_table_path: str = ""


@dataclass
class ReplayConfig:
    dataset_csv_path: str = "/home/hcy/work_space/xp_16_7/src/app/planner/test/lqr_analysis/data_15/work/lqr_comparison.csv"
    lqr_dt_s: float = 0.02
    dynamics_dt_s: float = 0.001
    execute_steps_per_batch: int = 5
    swing_model: str = "3d_rope"
    compare_swing_models: List[str] = field(default_factory=lambda: ["3d_rope", "planar_2d"])


@dataclass
class SystemConfig:
    vehicle: VehicleConfig
    layout: ActuatorLayout
    suspended_load: SuspendedLoadConfig
    simulation: SimulationConfig
    scenario: ScenarioConfig
    horizontal_mapping: HorizontalMappingConfig
    lqr: LqrConfig
    replay: ReplayConfig
    logging: LoggingConfig


MOTOR_NAMES: List[str] = [f"M{index}" for index in range(1, 9)]


def build_default_layout(arm_length_m: float, upper_lower_ratio: float) -> ActuatorLayout:
    positions = np.array(
        [
            [arm_length_m, -arm_length_m, 0.0],
            [arm_length_m, arm_length_m, 0.0],
            [-arm_length_m, arm_length_m, 0.0],
            [-arm_length_m, -arm_length_m, 0.0],
            [arm_length_m, -arm_length_m, 0.0],
            [arm_length_m, arm_length_m, 0.0],
            [-arm_length_m, arm_length_m, 0.0],
            [-arm_length_m, -arm_length_m, 0.0],
        ],
        dtype=float,
    )
    yaw_signs = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=float)
    layer_scale = np.array([1.0, 1.0, 1.0, 1.0, upper_lower_ratio, upper_lower_ratio, upper_lower_ratio, upper_lower_ratio], dtype=float)
    return ActuatorLayout(positions_m=positions, yaw_signs=yaw_signs, layer_scale=layer_scale)


def build_default_config() -> SystemConfig:
    vehicle = VehicleConfig()
    layout = build_default_layout(vehicle.arm_length_m, vehicle.upper_lower_thrust_ratio)
    return SystemConfig(
        vehicle=vehicle,
        layout=layout,
        suspended_load=SuspendedLoadConfig(),
        simulation=SimulationConfig(),
        scenario=ScenarioConfig(),
        horizontal_mapping=HorizontalMappingConfig(),
        lqr=LqrConfig(),
        replay=ReplayConfig(),
        logging=LoggingConfig(),
    )
