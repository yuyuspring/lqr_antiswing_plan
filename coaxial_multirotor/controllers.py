from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from .config import HorizontalMappingConfig, SystemConfig
from .math_utils import DEG2RAD, RAD2DEG, clamp


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    integral_limit: Optional[float] = None
    output_limit: Optional[float] = None
    integral: float = 0.0
    previous_error: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def step(self, error: float, dt_s: float) -> float:
        self.integral += error * dt_s
        if self.integral_limit is not None:
            self.integral = float(clamp(self.integral, -self.integral_limit, self.integral_limit))
        derivative = (error - self.previous_error) / dt_s if dt_s > 0.0 else 0.0
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limit is not None:
            output = float(clamp(output, -self.output_limit, self.output_limit))
        return output


@dataclass
class DiscreteTransferFunction:
    numerator: np.ndarray
    denominator: np.ndarray
    x_state: np.ndarray = field(init=False)
    y_state: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x_state = np.zeros(max(len(self.numerator) - 1, 0))
        self.y_state = np.zeros(max(len(self.denominator) - 1, 0))

    @classmethod
    def from_continuous(cls, num, den, dt_s: float) -> "DiscreteTransferFunction":
        discrete_num, discrete_den, _ = signal.cont2discrete((num, den), dt_s, method="bilinear")[:3]
        discrete_num = np.squeeze(discrete_num)
        discrete_den = np.squeeze(discrete_den)
        discrete_num = discrete_num / discrete_den[0]
        discrete_den = discrete_den / discrete_den[0]
        return cls(numerator=discrete_num, denominator=discrete_den)

    def reset(self) -> None:
        self.x_state.fill(0.0)
        self.y_state.fill(0.0)

    def step(self, input_value: float) -> float:
        y_val = self.numerator[0] * input_value
        for index in range(1, len(self.numerator)):
            y_val += self.numerator[index] * self.x_state[index - 1]
        for index in range(1, len(self.denominator)):
            y_val -= self.denominator[index] * self.y_state[index - 1]
        if self.x_state.size > 0:
            self.x_state[1:] = self.x_state[:-1]
            self.x_state[0] = input_value
        if self.y_state.size > 0:
            self.y_state[1:] = self.y_state[:-1]
            self.y_state[0] = y_val
        return float(y_val)


@dataclass
class CascadeAxisController:
    angle_pid: PIDController
    rate_pid: PIDController
    lead_lag: Optional[DiscreteTransferFunction] = None

    def reset(self) -> None:
        self.angle_pid.reset()
        self.rate_pid.reset()
        if self.lead_lag is not None:
            self.lead_lag.reset()

    def step(self, angle_cmd_deg: float, angle_deg: float, rate_degps: float, dt_s: float) -> float:
        rate_cmd_degps = self.angle_pid.step(angle_cmd_deg - angle_deg, dt_s)
        servo = self.rate_pid.step(rate_cmd_degps - rate_degps, dt_s)
        if self.lead_lag is not None:
            servo = self.lead_lag.step(servo)
        return servo


@dataclass
class AltitudeController:
    position_pid: PIDController
    velocity_pid: PIDController
    acceleration_pid: PIDController
    hover_pwm: float

    def reset(self) -> None:
        self.position_pid.reset()
        self.velocity_pid.reset()
        self.acceleration_pid.reset()

    def step(self, altitude_cmd_m: float, altitude_m: float, vel_z_mps: float, accel_z_mps2: float, roll_rad: float, pitch_rad: float, dt_s: float) -> float:
        vel_cmd = self.position_pid.step(altitude_cmd_m - altitude_m, dt_s)
        accel_cmd = self.velocity_pid.step(vel_cmd - vel_z_mps, dt_s)
        servo = self.acceleration_pid.step(accel_cmd - accel_z_mps2, dt_s)
        tilt_comp = max(np.cos(roll_rad) * np.cos(pitch_rad), 0.2)
        return self.hover_pwm + servo / tilt_comp


@dataclass
class HorizontalController:
    position_pid_x: PIDController
    position_pid_y: PIDController
    velocity_pid_x: PIDController
    velocity_pid_y: PIDController
    mapping: HorizontalMappingConfig

    def reset(self) -> None:
        self.position_pid_x.reset()
        self.position_pid_y.reset()
        self.velocity_pid_x.reset()
        self.velocity_pid_y.reset()

    def step(self, pos_cmd_m: np.ndarray, pos_m: np.ndarray, vel_mps: np.ndarray, yaw_rad: float, gravity_mps2: float, dt_s: float) -> Tuple[float, float]:
        vel_cmd_x = self.position_pid_x.step(pos_cmd_m[0] - pos_m[0], dt_s)
        vel_cmd_y = self.position_pid_y.step(pos_cmd_m[1] - pos_m[1], dt_s)
        accel_cmd_world_x = self.velocity_pid_x.step(vel_cmd_x - vel_mps[0], dt_s)
        accel_cmd_world_y = self.velocity_pid_y.step(vel_cmd_y - vel_mps[1], dt_s)

        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        accel_body_x = cos_yaw * accel_cmd_world_x + sin_yaw * accel_cmd_world_y
        accel_body_y = -sin_yaw * accel_cmd_world_x + cos_yaw * accel_cmd_world_y

        pitch_cmd_rad = self.mapping.pitch_from_ax_sign * accel_body_x / gravity_mps2
        roll_cmd_rad = self.mapping.roll_from_ay_sign * accel_body_y / gravity_mps2
        max_tilt_rad = self.mapping.max_tilt_deg * DEG2RAD
        return (
            float(clamp(roll_cmd_rad * RAD2DEG, -self.mapping.max_tilt_deg, self.mapping.max_tilt_deg)),
            float(clamp(pitch_cmd_rad * RAD2DEG, -self.mapping.max_tilt_deg, self.mapping.max_tilt_deg)),
        )


@dataclass
class CoaxialController:
    roll_axis: CascadeAxisController
    pitch_axis: CascadeAxisController
    yaw_axis: CascadeAxisController
    altitude_axis: AltitudeController
    horizontal_axis: HorizontalController
    max_pwm: float

    @classmethod
    def build(cls, system: SystemConfig) -> "CoaxialController":
        dt_s = system.simulation.dt_s
        hover_pwm = (
            np.sqrt(
                system.vehicle.mass_kg * system.vehicle.gravity_mps2
                / (system.vehicle.max_thrust_n * np.sum(system.layout.layer_scale))
            )
            * system.vehicle.max_pwm
        )
        roll_filter = DiscreteTransferFunction.from_continuous(
            [1.0 / (2.0 * np.pi * 2.0), 1.0],
            [1.0 / (2.0 * np.pi * 20.0), 1.0],
            dt_s,
        )
        pitch_filter = DiscreteTransferFunction.from_continuous(
            [1.0 / (2.0 * np.pi * 2.0), 1.0],
            [1.0 / (2.0 * np.pi * 20.0), 1.0],
            dt_s,
        )
        return cls(
            roll_axis=CascadeAxisController(
                angle_pid=PIDController(5.0, 0.0, 0.0, output_limit=60.0),
                rate_pid=PIDController(100.0 / 7.0 * 0.1, 100.0 / 7.0 * 0.05, 0.0, integral_limit=20.0, output_limit=300.0),
                lead_lag=roll_filter,
            ),
            pitch_axis=CascadeAxisController(
                angle_pid=PIDController(5.0, 0.0, 0.0, output_limit=60.0),
                rate_pid=PIDController(100.0 / 7.0 * 0.1, 100.0 / 7.0 * 0.05, 0.0, integral_limit=20.0, output_limit=300.0),
                lead_lag=pitch_filter,
            ),
            yaw_axis=CascadeAxisController(
                angle_pid=PIDController(2.0, 0.0, 0.0, output_limit=45.0),
                rate_pid=PIDController(100.0 / 10.0 * 0.3, 100.0 / 10.0 * 0.1, 0.0, integral_limit=30.0, output_limit=250.0),
                lead_lag=None,
            ),
            altitude_axis=AltitudeController(
                position_pid=PIDController(0.7, 0.0, 0.0, output_limit=5.0),
                velocity_pid=PIDController(2.0, 0.0, 0.0, output_limit=5.0),
                acceleration_pid=PIDController(10.0, 200.0, 0.0, integral_limit=0.3, output_limit=700.0),
                hover_pwm=hover_pwm,
            ),
            horizontal_axis=HorizontalController(
                position_pid_x=PIDController(1.0, 0.0, 0.0, output_limit=5.0),
                position_pid_y=PIDController(1.0, 0.0, 0.0, output_limit=5.0),
                velocity_pid_x=PIDController(1.0, 0.02, 0.0, integral_limit=5.0, output_limit=4.0),
                velocity_pid_y=PIDController(1.0, 0.02, 0.0, integral_limit=5.0, output_limit=4.0),
                mapping=system.horizontal_mapping,
            ),
            max_pwm=system.vehicle.max_pwm,
        )

    def allocate_motors(self, servo_roll: float, servo_pitch: float, servo_yaw: float, servo_thro: float) -> np.ndarray:
        motor_pwm = np.array(
            [
                servo_thro - servo_roll + servo_pitch - servo_yaw,
                servo_thro + servo_roll + servo_pitch + servo_yaw,
                servo_thro + servo_roll - servo_pitch - servo_yaw,
                servo_thro - servo_roll - servo_pitch + servo_yaw,
                servo_thro - servo_roll + servo_pitch + servo_yaw,
                servo_thro + servo_roll + servo_pitch - servo_yaw,
                servo_thro + servo_roll - servo_pitch + servo_yaw,
                servo_thro - servo_roll - servo_pitch - servo_yaw,
            ],
            dtype=float,
        )
        return clamp(motor_pwm, 0.0, self.max_pwm)

    def step(self, command: dict, truth: dict, dt_s: float, gravity_mps2: float) -> dict:
        euler_deg = truth["euler_deg"]
        body_rates_degps = truth["body_rates_degps"]
        position = truth["position_m"]
        velocity = truth["velocity_mps"]
        accel_world = truth["accel_world_mps2"]

        roll_cmd_deg, pitch_cmd_deg = self.horizontal_axis.step(
            pos_cmd_m=np.array([command["x_m"], command["y_m"]]),
            pos_m=position[:2],
            vel_mps=velocity[:2],
            yaw_rad=euler_deg[2] * DEG2RAD,
            gravity_mps2=gravity_mps2,
            dt_s=dt_s,
        )
        servo_roll = self.roll_axis.step(roll_cmd_deg, euler_deg[0], body_rates_degps[0], dt_s)
        servo_pitch = -self.pitch_axis.step(pitch_cmd_deg, euler_deg[1], body_rates_degps[1], dt_s)
        servo_yaw = -self.yaw_axis.step(command["yaw_deg"], euler_deg[2], body_rates_degps[2], dt_s)
        servo_thro = self.altitude_axis.step(
            altitude_cmd_m=command["z_m"],
            altitude_m=position[2],
            vel_z_mps=velocity[2],
            accel_z_mps2=accel_world[2],
            roll_rad=euler_deg[0] * DEG2RAD,
            pitch_rad=euler_deg[1] * DEG2RAD,
            dt_s=dt_s,
        )
        motor_pwm = self.allocate_motors(servo_roll, servo_pitch, servo_yaw, servo_thro)
        return {
            "roll_cmd_deg": roll_cmd_deg,
            "pitch_cmd_deg": pitch_cmd_deg,
            "servo_roll": servo_roll,
            "servo_pitch": servo_pitch,
            "servo_yaw": servo_yaw,
            "servo_thro": servo_thro,
            "motor_pwm": motor_pwm,
        }
