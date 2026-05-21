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


@dataclass
class VehicleState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion: np.ndarray
    body_rates_radps: np.ndarray
    motor_rpm: np.ndarray
    accel_world_mps2: np.ndarray

    @classmethod
    def from_config(cls, system: SystemConfig) -> "VehicleState":
        return cls(
            position_m=system.simulation.initial_position_m.astype(float).copy(),
            velocity_mps=system.simulation.initial_velocity_mps.astype(float).copy(),
            quaternion=euler321_to_quaternion(system.simulation.initial_euler_deg * DEG2RAD),
            body_rates_radps=system.simulation.initial_body_rates_dps * DEG2RAD,
            motor_rpm=np.zeros(8, dtype=float),
            accel_world_mps2=np.zeros(3, dtype=float),
        )

    def as_truth(self) -> dict:
        return {
            "position_m": self.position_m.copy(),
            "velocity_mps": self.velocity_mps.copy(),
            "euler_deg": quaternion_to_euler321(self.quaternion) * RAD2DEG,
            "body_rates_degps": self.body_rates_radps * RAD2DEG,
            "accel_world_mps2": self.accel_world_mps2.copy(),
            "motor_rpm": self.motor_rpm.copy(),
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
        accel_world = rotation_body_to_world @ total_force_body_n / self.system.vehicle.mass_kg + gravity_world

        body_rates = state.body_rates_radps
        body_rates_dot = self.inertia_inv @ (total_moment_body_nm - skew(body_rates) @ (self.inertia @ body_rates))
        quaternion_dot = body_rates_to_quaternion_derivative(state.quaternion, body_rates)

        state.velocity_mps = state.velocity_mps + accel_world * dt_s
        state.position_m = state.position_m + state.velocity_mps * dt_s
        state.body_rates_radps = state.body_rates_radps + body_rates_dot * dt_s
        state.quaternion = normalize_quaternion(state.quaternion + quaternion_dot * dt_s)
        state.accel_world_mps2 = accel_world
        return state
