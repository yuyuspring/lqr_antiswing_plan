from dataclasses import dataclass, field

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


def build_default_suspended_load_state() -> SuspendedLoadState:
    return SuspendedLoadState(
        direction_world=np.array([0.0, 0.0, -1.0], dtype=float),
        direction_rate_world=np.zeros(3, dtype=float),
    )


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
