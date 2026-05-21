from dataclasses import dataclass

import numpy as np

from .config import SystemConfig
from .math_utils import (
    DEG2RAD,
    RAD2DEG,
    body_rates_to_quaternion_derivative,
    clamp,
    euler321_to_quaternion,
    normalize_quaternion,
    quaternion_to_euler321,
    quaternion_to_rotation_matrix,
    skew,
)
from .suspended_load import (
    SuspendedLoadState,
    build_default_suspended_load_state,
    extract_xp_swing_state,
    step_suspended_load_state,
)


@dataclass
class VehicleState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion: np.ndarray
    body_rates_radps: np.ndarray
    motor_rpm: np.ndarray
    accel_world_mps2: np.ndarray
    load_state: SuspendedLoadState

    @classmethod
    def from_config(cls, system: SystemConfig) -> "VehicleState":
        return cls(
            position_m=system.simulation.initial_position_m.astype(float).copy(),
            velocity_mps=system.simulation.initial_velocity_mps.astype(float).copy(),
            quaternion=euler321_to_quaternion(system.simulation.initial_euler_deg * DEG2RAD),
            body_rates_radps=system.simulation.initial_body_rates_dps * DEG2RAD,
            motor_rpm=np.zeros(8, dtype=float),
            accel_world_mps2=np.zeros(3, dtype=float),
            load_state=build_default_suspended_load_state(),
        )

    def as_truth(self) -> dict:
        rotation_body_to_world = quaternion_to_rotation_matrix(self.quaternion)
        rotation_world_to_body = rotation_body_to_world.T
        euler_rad = quaternion_to_euler321(self.quaternion)
        roll_rad, pitch_rad, yaw_rad = euler_rad
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        rotation_heading_to_world = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation_world_to_heading = rotation_heading_to_world.T
        direction_heading = rotation_world_to_heading @ self.load_state.direction_world
        cos_pitch = max(np.cos(pitch_rad), 1e-6)
        yaw_rate_radps = (
            self.body_rates_radps[1] * np.sin(roll_rad)
            + self.body_rates_radps[2] * np.cos(roll_rad)
        ) / cos_pitch
        heading_omega_heading = np.array([0.0, 0.0, yaw_rate_radps])
        direction_rate_heading = (
            rotation_world_to_heading @ self.load_state.direction_rate_world
            - np.cross(heading_omega_heading, direction_heading)
        )
        swing_state = extract_xp_swing_state(direction_heading, direction_rate_heading)
        return {
            "position_m": self.position_m.copy(),
            "velocity_mps": self.velocity_mps.copy(),
            "body_velocity_mps": rotation_world_to_body @ self.velocity_mps,
            "body_accel_mps2": rotation_world_to_body @ self.accel_world_mps2,
            "euler_deg": euler_rad * RAD2DEG,
            "body_rates_degps": self.body_rates_radps * RAD2DEG,
            "accel_world_mps2": self.accel_world_mps2.copy(),
            "motor_rpm": self.motor_rpm.copy(),
            "direction_world": self.load_state.direction_world.copy(),
            "direction_rate_world": self.load_state.direction_rate_world.copy(),
            "gyro_angle_x_rad": swing_state["gyro_angle_x_rad"],
            "gyro_angle_y_rad": swing_state["gyro_angle_y_rad"],
            "gyro_rate_x_radps": swing_state["gyro_rate_x_radps"],
            "gyro_rate_y_radps": swing_state["gyro_rate_y_radps"],
        }


class CoaxialMultirotorDynamics:
    def __init__(self, system: SystemConfig) -> None:
        self.system = system
        self.inertia = system.vehicle.inertia_kgm2
        self.inertia_inv = np.linalg.inv(self.inertia)

    def motor_command_to_rpm(self, motor_pwm: np.ndarray) -> np.ndarray:
        return motor_pwm / self.system.vehicle.max_pwm * self.system.vehicle.max_rpm

    def step(self, state: VehicleState, motor_pwm_cmd: np.ndarray, dt_s: float) -> VehicleState:
        rpm_cmd = self.motor_command_to_rpm(clamp(motor_pwm_cmd, 0.0, self.system.vehicle.max_pwm))
        tau_motor = self.system.vehicle.motor_time_constant_s
        state.motor_rpm = state.motor_rpm + dt_s * (rpm_cmd - state.motor_rpm) / tau_motor
        rpm_ratio = clamp(state.motor_rpm / self.system.vehicle.max_rpm, 0.0, 1.0)
        rpm_ratio_sq = rpm_ratio * rpm_ratio

        thrust_per_motor_n = rpm_ratio_sq * self.system.vehicle.max_thrust_n * self.system.layout.layer_scale
        reaction_torque_nm = rpm_ratio_sq * self.system.vehicle.max_reaction_torque_nm * self.system.layout.yaw_signs

        total_force_body_n = np.array([0.0, 0.0, thrust_per_motor_n.sum()])
        moment_from_arms = np.sum(np.cross(self.system.layout.positions_m, np.column_stack([np.zeros(8), np.zeros(8), thrust_per_motor_n])), axis=0)
        total_moment_body_nm = moment_from_arms + np.array([0.0, 0.0, reaction_torque_nm.sum()])

        rotation_body_to_world = quaternion_to_rotation_matrix(state.quaternion)
        gravity_world = np.array([0.0, 0.0, -self.system.vehicle.gravity_mps2])

        body_rates = state.body_rates_radps
        body_rates_dot = self.inertia_inv @ (total_moment_body_nm - skew(body_rates) @ (self.inertia @ body_rates))
        quaternion_dot = body_rates_to_quaternion_derivative(state.quaternion, body_rates)

        attach_offset_body = self.system.suspended_load.attach_point_body_m
        attach_accel_world = rotation_body_to_world @ (
            np.cross(body_rates_dot, attach_offset_body)
            + np.cross(body_rates, np.cross(body_rates, attach_offset_body))
        )
        direction_world = state.load_state.direction_world
        direction_rate_world = state.load_state.direction_rate_world
        base_accel_world = rotation_body_to_world @ total_force_body_n / self.system.vehicle.mass_kg + gravity_world
        payload_mass_kg = self.system.suspended_load.payload_mass_kg
        rope_length_m = self.system.suspended_load.rope_length_m
        coupling_term = (
            rope_length_m * float(np.dot(direction_rate_world, direction_rate_world))
            + float(np.dot(direction_world, gravity_world))
            - float(np.dot(direction_world, attach_accel_world))
            - float(np.dot(direction_world, base_accel_world))
        )
        effective_mass = payload_mass_kg * self.system.vehicle.mass_kg / (payload_mass_kg + self.system.vehicle.mass_kg)
        load_tension_n = max(effective_mass * coupling_term, 0.0)
        tension_force_world = load_tension_n * direction_world
        accel_world = base_accel_world + tension_force_world / self.system.vehicle.mass_kg

        state.velocity_mps = state.velocity_mps + accel_world * dt_s
        state.position_m = state.position_m + state.velocity_mps * dt_s
        state.body_rates_radps = state.body_rates_radps + body_rates_dot * dt_s
        state.quaternion = normalize_quaternion(state.quaternion + quaternion_dot * dt_s)
        state.accel_world_mps2 = accel_world
        state.load_state = step_suspended_load_state(
            state=state.load_state,
            config=self.system.suspended_load,
            attach_accel_world_mps2=accel_world + attach_accel_world,
            gravity_world_mps2=gravity_world,
            dt_s=dt_s,
        )
        return state
