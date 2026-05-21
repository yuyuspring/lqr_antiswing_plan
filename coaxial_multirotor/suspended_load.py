from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .math_utils import clamp


@dataclass
class SuspendedLoadConfig:
    rope_length_m: float = 15.0
    payload_mass_kg: float = 150.0
    attach_point_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tangential_damping: float = 0.08


@dataclass
class SuspendedLoadState:
    direction_world: np.ndarray
    direction_rate_world: np.ndarray


@dataclass
class PlanarSwingState:
    theta_x_rad: float = 0.0
    omega_x_radps: float = 0.0
    theta_y_rad: float = 0.0
    omega_y_radps: float = 0.0


def build_default_suspended_load_state() -> SuspendedLoadState:
    return SuspendedLoadState(
        direction_world=np.array([0.0, 0.0, -1.0], dtype=float),
        direction_rate_world=np.zeros(3, dtype=float),
    )


def build_default_planar_swing_state() -> PlanarSwingState:
    return PlanarSwingState()


def normalize_direction(direction_world: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(direction_world)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return direction_world / norm


def project_to_tangent(vector: np.ndarray, direction_world: np.ndarray) -> np.ndarray:
    return vector - direction_world * float(np.dot(direction_world, vector))


def compute_load_tension(
    payload_mass_kg: float,
    rope_length_m: float,
    direction_world: np.ndarray,
    direction_rate_world: np.ndarray,
    attach_accel_world_mps2: np.ndarray,
    gravity_world_mps2: np.ndarray,
) -> float:
    centripetal_term = rope_length_m * float(np.dot(direction_rate_world, direction_rate_world))
    gravity_term = float(np.dot(direction_world, gravity_world_mps2))
    attach_term = float(np.dot(direction_world, attach_accel_world_mps2))
    return payload_mass_kg * max(centripetal_term + gravity_term - attach_term, 0.0)


def step_suspended_load_state(
    state: SuspendedLoadState,
    config: SuspendedLoadConfig,
    attach_accel_world_mps2: np.ndarray,
    gravity_world_mps2: np.ndarray,
    dt_s: float,
) -> SuspendedLoadState:
    direction_world = normalize_direction(state.direction_world)
    direction_rate_world = project_to_tangent(state.direction_rate_world, direction_world)
    tangent_drive = project_to_tangent(gravity_world_mps2 - attach_accel_world_mps2, direction_world)
    direction_accel_world = (
        tangent_drive / max(config.rope_length_m, 1e-6)
        - np.dot(direction_rate_world, direction_rate_world) * direction_world
        - config.tangential_damping * direction_rate_world
    )
    direction_rate_world = direction_rate_world + direction_accel_world * dt_s
    direction_rate_world = project_to_tangent(direction_rate_world, direction_world)
    direction_world = normalize_direction(direction_world + direction_rate_world * dt_s)
    direction_rate_world = project_to_tangent(direction_rate_world, direction_world)
    return SuspendedLoadState(
        direction_world=direction_world,
        direction_rate_world=direction_rate_world,
    )


def _compute_planar_derivative(
    theta_rad: float,
    omega_radps: float,
    accel_mps2: float,
    config: SuspendedLoadConfig,
    gravity_mps2: float,
    drone_mass_kg: float,
) -> Tuple[float, float]:
    payload_mass_kg = max(config.payload_mass_kg, 0.0)
    if payload_mass_kg < 1e-6:
        return 0.0, 0.0
    pendulum_gain = payload_mass_kg / max(payload_mass_kg + drone_mass_kg, 1e-6)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    denom = max(1.0 - pendulum_gain * cos_theta * cos_theta, 1e-3)
    rope_length_m = max(config.rope_length_m, 1e-6)
    drone_accel_mps2 = (
        (1.0 - pendulum_gain) * accel_mps2
        + pendulum_gain * gravity_mps2 * sin_theta * cos_theta
        + pendulum_gain * rope_length_m * omega_radps * omega_radps * sin_theta
    ) / denom
    dtheta_radps = omega_radps
    domega_radps2 = -(
        gravity_mps2 * sin_theta + drone_accel_mps2 * cos_theta
    ) / rope_length_m - 0.15 * omega_radps
    return dtheta_radps, domega_radps2


def _step_planar_channel(
    theta_rad: float,
    omega_radps: float,
    accel_mps2: float,
    config: SuspendedLoadConfig,
    gravity_mps2: float,
    drone_mass_kg: float,
    dt_s: float,
) -> Tuple[float, float]:
    k1_theta, k1_omega = _compute_planar_derivative(theta_rad, omega_radps, accel_mps2, config, gravity_mps2, drone_mass_kg)
    k2_theta, k2_omega = _compute_planar_derivative(theta_rad + 0.5 * dt_s * k1_theta, omega_radps + 0.5 * dt_s * k1_omega, accel_mps2, config, gravity_mps2, drone_mass_kg)
    k3_theta, k3_omega = _compute_planar_derivative(theta_rad + 0.5 * dt_s * k2_theta, omega_radps + 0.5 * dt_s * k2_omega, accel_mps2, config, gravity_mps2, drone_mass_kg)
    k4_theta, k4_omega = _compute_planar_derivative(theta_rad + dt_s * k3_theta, omega_radps + dt_s * k3_omega, accel_mps2, config, gravity_mps2, drone_mass_kg)
    theta_next = theta_rad + dt_s * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) / 6.0
    omega_next = omega_radps + dt_s * (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) / 6.0
    return theta_next, omega_next


def step_planar_swing_state(
    state: PlanarSwingState,
    config: SuspendedLoadConfig,
    accel_body_x_mps2: float,
    accel_body_y_mps2: float,
    gravity_mps2: float,
    drone_mass_kg: float,
    dt_s: float,
) -> PlanarSwingState:
    theta_y_rad, omega_y_radps = _step_planar_channel(
        state.theta_y_rad,
        state.omega_y_radps,
        accel_body_x_mps2,
        config,
        gravity_mps2,
        drone_mass_kg,
        dt_s,
    )
    theta_x_rad, omega_x_radps = _step_planar_channel(
        state.theta_x_rad,
        state.omega_x_radps,
        accel_body_y_mps2,
        config,
        gravity_mps2,
        drone_mass_kg,
        dt_s,
    )
    return PlanarSwingState(
        theta_x_rad=theta_x_rad,
        omega_x_radps=omega_x_radps,
        theta_y_rad=theta_y_rad,
        omega_y_radps=omega_y_radps,
    )


def compute_planar_tension_force_body(
    state: PlanarSwingState,
    config: SuspendedLoadConfig,
    gravity_mps2: float,
) -> np.ndarray:
    payload_mass_kg = max(config.payload_mass_kg, 0.0)
    if payload_mass_kg < 1e-6:
        return np.zeros(3, dtype=float)
    rope_length_m = max(config.rope_length_m, 1e-6)
    theta_mag = np.sqrt(state.theta_x_rad * state.theta_x_rad + state.theta_y_rad * state.theta_y_rad)
    tension_x_n = payload_mass_kg * (gravity_mps2 * np.cos(state.theta_y_rad) + rope_length_m * state.omega_y_radps * state.omega_y_radps)
    tension_y_n = payload_mass_kg * (gravity_mps2 * np.cos(state.theta_x_rad) + rope_length_m * state.omega_x_radps * state.omega_x_radps)
    tension_n = max(0.5 * (tension_x_n + tension_y_n), 0.0)
    return np.array(
        [
            tension_n * np.sin(state.theta_y_rad),
            -tension_n * np.sin(state.theta_x_rad),
            -tension_n * np.cos(theta_mag),
        ],
        dtype=float,
    )


def extract_planar_swing_state(state: PlanarSwingState) -> dict:
    return {
        "gyro_angle_x_rad": state.theta_x_rad,
        "gyro_angle_y_rad": state.theta_y_rad,
        "gyro_rate_x_radps": state.omega_x_radps,
        "gyro_rate_y_radps": state.omega_y_radps,
    }


def extract_xp_swing_state(
    direction_body: np.ndarray,
    direction_rate_body: np.ndarray,
) -> dict:
    dir_body = normalize_direction(direction_body)
    rate_body = direction_rate_body

    theta_y = float(np.arctan2(dir_body[0], -dir_body[2]))
    theta_x = float(np.arctan2(-dir_body[1], -dir_body[2]))

    denom_y = max(dir_body[0] * dir_body[0] + dir_body[2] * dir_body[2], 1e-9)
    gyro_rate_y = float(((-dir_body[2]) * rate_body[0] + dir_body[0] * rate_body[2]) / denom_y)

    u_x = -dir_body[1]
    v_x = -dir_body[2]
    u_x_dot = -rate_body[1]
    v_x_dot = -rate_body[2]
    denom_x = max(u_x * u_x + v_x * v_x, 1e-9)
    gyro_rate_x = float((v_x * u_x_dot - u_x * v_x_dot) / denom_x)

    return {
        "gyro_angle_x_rad": theta_x,
        "gyro_angle_y_rad": theta_y,
        "gyro_rate_x_radps": gyro_rate_x,
        "gyro_rate_y_radps": gyro_rate_y,
    }
