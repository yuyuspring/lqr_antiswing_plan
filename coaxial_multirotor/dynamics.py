from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SystemConfig
from .math_utils import (
    DEG2RAD,
    RAD2DEG,
    body_rates_to_quaternion_derivative,
    clamp,
    euler321_to_quaternion,
    internal_body_rates_to_user,
    internal_euler_to_user,
    normalize_quaternion,
    quaternion_to_euler321,
    quaternion_to_rotation_matrix,
    skew,
    user_body_rates_to_internal,
    user_euler_to_internal,
)
from .suspended_load import (
    PlanarSwingState,
    SuspendedLoadState,
    build_default_planar_swing_state,
    build_default_suspended_load_state,
    compute_planar_tension_force_body,
    extract_planar_swing_state,
    extract_xp_swing_state,
    step_planar_swing_state,
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
    planar_swing_state: PlanarSwingState

    @classmethod
    def from_config(cls, system: SystemConfig) -> "VehicleState":
        hover_pwm = (
            np.sqrt(
                (system.vehicle.mass_kg + system.suspended_load.payload_mass_kg) * system.vehicle.gravity_mps2
                / (system.vehicle.max_thrust_n * np.sum(system.layout.layer_scale))
            )
            * system.vehicle.max_pwm
        )
        hover_rpm = hover_pwm / system.vehicle.max_pwm * system.vehicle.max_rpm
        return cls(
            position_m=system.simulation.initial_position_m.astype(float).copy(),
            velocity_mps=system.simulation.initial_velocity_mps.astype(float).copy(),
            quaternion=euler321_to_quaternion(user_euler_to_internal(system.simulation.initial_euler_deg * DEG2RAD)),
            body_rates_radps=user_body_rates_to_internal(system.simulation.initial_body_rates_dps * DEG2RAD),
            motor_rpm=np.full(8, hover_rpm, dtype=float),
            accel_world_mps2=np.zeros(3, dtype=float),
            load_state=build_default_suspended_load_state(),
            planar_swing_state=build_default_planar_swing_state(),
        )

    def as_truth(self, swing_model: str = "3d_rope") -> dict:
        rotation_body_to_world = quaternion_to_rotation_matrix(self.quaternion)
        rotation_world_to_body = rotation_body_to_world.T
        euler_rad_internal = quaternion_to_euler321(self.quaternion)
        euler_rad = internal_euler_to_user(euler_rad_internal)
        if swing_model == "planar_2d":
            swing_state = extract_planar_swing_state(self.planar_swing_state)
        else:
            roll_rad, pitch_rad, yaw_rad = euler_rad_internal
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
            "body_rates_degps": internal_body_rates_to_user(self.body_rates_radps) * RAD2DEG,
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

    def _rigid_body_derivative(
        self,
        quaternion: np.ndarray,
        velocity_mps: np.ndarray,
        body_rates_radps: np.ndarray,
        total_force_body_n: np.ndarray,
        total_moment_body_nm: np.ndarray,
        gravity_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rotation_body_to_world = quaternion_to_rotation_matrix(normalize_quaternion(quaternion))
        accel_world = rotation_body_to_world @ total_force_body_n / self.system.vehicle.mass_kg + gravity_world
        body_rates_dot = self.inertia_inv @ (total_moment_body_nm - skew(body_rates_radps) @ (self.inertia @ body_rates_radps))
        quaternion_dot = body_rates_to_quaternion_derivative(normalize_quaternion(quaternion), body_rates_radps)
        return velocity_mps, accel_world, quaternion_dot, body_rates_dot

    def _integrate_rigid_body_rk4(
        self,
        state: VehicleState,
        total_force_body_n: np.ndarray,
        total_moment_body_nm: np.ndarray,
        gravity_world: np.ndarray,
        dt_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos0 = state.position_m.copy()
        vel0 = state.velocity_mps.copy()
        quat0 = state.quaternion.copy()
        rates0 = state.body_rates_radps.copy()

        k1_pos, k1_vel, k1_quat, k1_rates = self._rigid_body_derivative(
            quat0,
            vel0,
            rates0,
            total_force_body_n,
            total_moment_body_nm,
            gravity_world,
        )
        k2_pos, k2_vel, k2_quat, k2_rates = self._rigid_body_derivative(
            quat0 + 0.5 * dt_s * k1_quat,
            vel0 + 0.5 * dt_s * k1_vel,
            rates0 + 0.5 * dt_s * k1_rates,
            total_force_body_n,
            total_moment_body_nm,
            gravity_world,
        )
        k3_pos, k3_vel, k3_quat, k3_rates = self._rigid_body_derivative(
            quat0 + 0.5 * dt_s * k2_quat,
            vel0 + 0.5 * dt_s * k2_vel,
            rates0 + 0.5 * dt_s * k2_rates,
            total_force_body_n,
            total_moment_body_nm,
            gravity_world,
        )
        k4_pos, k4_vel, k4_quat, k4_rates = self._rigid_body_derivative(
            quat0 + dt_s * k3_quat,
            vel0 + dt_s * k3_vel,
            rates0 + dt_s * k3_rates,
            total_force_body_n,
            total_moment_body_nm,
            gravity_world,
        )

        position_next = pos0 + dt_s * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos) / 6.0
        velocity_next = vel0 + dt_s * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel) / 6.0
        quaternion_next = normalize_quaternion(
            quat0 + dt_s * (k1_quat + 2.0 * k2_quat + 2.0 * k3_quat + k4_quat) / 6.0
        )
        body_rates_next = rates0 + dt_s * (k1_rates + 2.0 * k2_rates + 2.0 * k3_rates + k4_rates) / 6.0
        return position_next, velocity_next, quaternion_next, body_rates_next

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

        attach_offset_body = self.system.suspended_load.attach_point_body_m
        attach_accel_world = rotation_body_to_world @ (
            np.cross(body_rates_dot, attach_offset_body)
            + np.cross(body_rates, np.cross(body_rates, attach_offset_body))
        )
        direction_world = state.load_state.direction_world
        direction_rate_world = state.load_state.direction_rate_world
        base_accel_world = rotation_body_to_world @ total_force_body_n / self.system.vehicle.mass_kg + gravity_world
        if self.system.replay.swing_model == "planar_2d":
            planar_tension_force_body = compute_planar_tension_force_body(
                state.planar_swing_state,
                self.system.suspended_load,
                self.system.vehicle.gravity_mps2,
            )
            tension_force_world = rotation_body_to_world @ planar_tension_force_body
            accel_world = base_accel_world + tension_force_world / self.system.vehicle.mass_kg
        else:
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

        total_force_world_n = self.system.vehicle.mass_kg * (accel_world - gravity_world)
        total_force_body_with_tension_n = rotation_body_to_world.T @ total_force_world_n
        position_next, velocity_next, quaternion_next, body_rates_next = self._integrate_rigid_body_rk4(
            state,
            total_force_body_with_tension_n,
            total_moment_body_nm,
            gravity_world,
            dt_s,
        )
        state.position_m = position_next
        state.velocity_mps = velocity_next
        state.body_rates_radps = body_rates_next
        state.quaternion = quaternion_next
        state.accel_world_mps2 = accel_world
        if self.system.replay.swing_model == "planar_2d":
            accel_body = total_force_body_with_tension_n / self.system.vehicle.mass_kg + rotation_body_to_world.T @ gravity_world
            state.planar_swing_state = step_planar_swing_state(
                state=state.planar_swing_state,
                config=self.system.suspended_load,
                accel_body_x_mps2=accel_body[0],
                accel_body_y_mps2=accel_body[1],
                gravity_mps2=self.system.vehicle.gravity_mps2,
                drone_mass_kg=self.system.vehicle.mass_kg,
                dt_s=dt_s,
            )
        else:
            state.load_state = step_suspended_load_state(
                state=state.load_state,
                config=self.system.suspended_load,
                attach_accel_world_mps2=accel_world + attach_accel_world,
                gravity_world_mps2=gravity_world,
                dt_s=dt_s,
            )
        return state
