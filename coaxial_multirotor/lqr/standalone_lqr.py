from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
from typing import List, Optional, Sequence, Tuple


K_GRAVITY = 9.81
K_PI = 3.14159265358979323846


def saturate(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def clamp(value: float, min_val: float, max_val: float) -> float:
    if value > max_val:
        return max_val
    if value < min_val:
        return min_val
    return value


@dataclass
class State:
    time: float = 0.0
    p: float = 0.0
    v: float = 0.0
    a: float = 0.0
    theta: float = 0.0
    omega: float = 0.0


@dataclass
class StepMetrics:
    currentVel: float = 0.0
    vRef: float = 0.0
    err: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    axRef: float = 0.0
    feedforwardAcc: float = 0.0
    swingUavAcc: float = 0.0
    integralBefore: float = 0.0
    integralPreview: float = 0.0
    integralAfter: float = 0.0
    antiWindupGain: float = 0.0
    antiWindupDelta: float = 0.0
    antiWindupCorrection: float = 0.0
    antiWindupPreviewAcc: float = 0.0
    antiWindupPreviewSatAcc: float = 0.0
    targetAcc: float = 0.0
    cmdAcc: float = 0.0


@dataclass
class Config:
    dt: float = 0.02
    ropeLength: float = 15.0
    lqrAxMax: float = 5.0
    lqrJerkMax: float = 10.0
    payloadMass: float = 180.0
    droneMass: float = 120.0
    linearDampingCoeff: float = 0.15
    dragCoeff: float = 1.0
    dragArea: float = 0.5
    airDensity: float = 1.225
    initialTheta: float = 0.0
    initialOmega: float = 0.0
    initialP: float = 0.0
    initialV: float = 0.0


class StandaloneLqrSimulator:
    K_MU_TABLE: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    K_L_TABLE: Tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
    K_DEFAULT_GAIN_TABLE: Tuple[Tuple[Tuple[float, float, float, float, float], ...], ...] = (
        ((0.0000, 4.2467, -38.9056, -12.2858, 0.9324), (0.0000, 4.2724, -37.8918, -8.1534, 0.9468), (0.0000, 4.2720, -37.1976, -5.6508, 0.9514), (0.0000, 4.2673, -36.5576, -3.3143, 0.9543), (0.0000, 4.2573, -35.7052, -0.0513, 0.9571), (0.0000, 4.2388, -34.5180, 4.9252, 0.9598), (0.0000, 4.2211, -33.5375, 9.5043, 0.9613), (0.0000, 4.2047, -32.6974, 13.8028, 0.9623), (0.0000, 4.1754, -31.2914, 21.7963, 0.9634)),
        ((0.1000, 4.2499, -37.9249, -12.1477, 0.9327), (0.1000, 4.2747, -36.8723, -7.9667, 0.9470), (0.1000, 4.2738, -36.1590, -5.4425, 0.9516), (0.1000, 4.2687, -35.5040, -3.0902, 0.9545), (0.1000, 4.2582, -34.6348, 0.1891, 0.9573), (0.1000, 4.2392, -33.4292, 5.1808, 0.9599), (0.1000, 4.2212, -32.4374, 9.7670, 0.9614), (0.1000, 4.2045, -31.5899, 14.0683, 0.9623), (0.1000, 4.1750, -30.1757, 22.0610, 0.9634)),
        ((0.2000, 4.2547, -36.7345, -11.9759, 0.9329), (0.2000, 4.2783, -35.6363, -7.7328, 0.9473), (0.2000, 4.2769, -34.9006, -5.1805, 0.9518), (0.2000, 4.2714, -34.2280, -2.8071, 0.9547), (0.2000, 4.2603, -33.3390, 0.4950, 0.9575), (0.2000, 4.2408, -32.1117, 5.5103, 0.9601), (0.2000, 4.2223, -31.1064, 10.1102, 0.9615), (0.2000, 4.2054, -30.2502, 14.4200, 0.9624), (0.2000, 4.1756, -28.8259, 22.4221, 0.9635)),
        ((0.3000, 4.2621, -35.2612, -11.7567, 0.9333), (0.3000, 4.2844, -34.1095, -7.4318, 0.9475), (0.3000, 4.2823, -33.3472, -4.8414, 0.9521), (0.3000, 4.2762, -32.6538, -2.4385, 0.9549), (0.3000, 4.2646, -31.7414, 0.8968, 0.9577), (0.3000, 4.2443, -30.4884, 5.9497, 0.9602), (0.3000, 4.2254, -29.4671, 10.5753, 0.9617), (0.3000, 4.2082, -28.6001, 14.9044, 0.9626), (0.3000, 4.1780, -27.1630, 22.9354, 0.9636)),
        ((0.4000, 4.2740, -33.3950, -11.4674, 0.9337), (0.4000, 4.2949, -32.1805, -7.0301, 0.9479), (0.4000, 4.2920, -31.3870, -4.3856, 0.9524), (0.4000, 4.2853, -30.6691, -1.9395, 0.9552), (0.4000, 4.2729, -29.7290, 1.4467, 0.9579), (0.4000, 4.2518, -28.4456, 6.5625, 0.9605), (0.4000, 4.2324, -27.4049, 11.2360, 0.9619), (0.4000, 4.2148, -26.5248, 15.6049, 0.9627), (0.4000, 4.1842, -25.0710, 23.7034, 0.9637)),
        ((0.5000, 4.2945, -30.9637, -11.0686, 0.9342), (0.5000, 4.3138, -29.6781, -6.4682, 0.9484), (0.5000, 4.3100, -28.8487, -3.7419, 0.9528), (0.5000, 4.3026, -28.1025, -1.2283, 0.9556), (0.5000, 4.2894, -27.1306, 2.2415, 0.9583), (0.5000, 4.2674, -25.8116, 7.4688, 0.9607), (0.5000, 4.2474, -24.7478, 12.2347, 0.9621), (0.5000, 4.2294, -23.8512, 16.6859, 0.9629), (0.5000, 4.1982, -22.3744, 24.9325, 0.9639)),
        ((0.6000, 4.3331, -27.6883, -10.4851, 0.9350), (0.6000, 4.3508, -26.3305, -5.6291, 0.9490), (0.6000, 4.3462, -25.4633, -2.7686, 0.9534), (0.6000, 4.3381, -24.6872, -0.1393, 0.9562), (0.6000, 4.3241, -23.6810, 3.4809, 0.9588), (0.6000, 4.3011, -22.3228, 8.9226, 0.9611), (0.6000, 4.2805, -21.2315, 13.8785, 0.9624), (0.6000, 4.2620, -20.3135, 18.5062, 0.9632), (0.6000, 4.2300, -18.8025, 27.0828, 0.9641)),
        ((0.7000, 4.4147, -23.0980, -9.5538, 0.9359), (0.7000, 4.4319, -21.6982, -4.2507, 0.9499), (0.7000, 4.4269, -20.8035, -1.1412, 0.9542), (0.7000, 4.4184, -20.0042, 1.7121, 0.9569), (0.7000, 4.4039, -18.9698, 5.6380, 0.9594), (0.7000, 4.3803, -17.5741, 11.5424, 0.9616), (0.7000, 4.3592, -16.4505, 16.9289, 0.9628), (0.7000, 4.3402, -15.5017, 21.9687, 0.9636), (0.7000, 4.3072, -13.9304, 31.3340, 0.9644)),
        ((0.8000, 4.6224, -16.3609, -7.8453, 0.9372), (0.8000, 4.6436, -15.0640, -1.6096, 0.9510), (0.8000, 4.6399, -14.1936, 2.0548, 0.9552), (0.8000, 4.6322, -13.4046, 5.4302, 0.9578), (0.8000, 4.6185, -12.3697, 10.1000, 0.9602), (0.8000, 4.5954, -10.9465, 17.1841, 0.9623), (0.8000, 4.5741, -9.7742, 23.7072, 0.9634), (0.8000, 4.5546, -8.7633, 29.8552, 0.9640), (0.8000, 4.5198, -7.0452, 41.3725, 0.9648)),
    )

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pendulumGain = config.payloadMass / (config.droneMass + config.payloadMass)
        self.kV = 0.0
        self.kTheta = 0.0
        self.kOmega = 0.0
        self.kIntegral = 0.0
        self.integral = 0.0
        self.axCmd = 0.0
        self.axRef = 0.0
        self.state = State()
        self.lastStepMetrics = StepMetrics()
        self.gainTable = self._load_gain_table()
        self.recompute_gains()
        self.reset_all(config.initialP, config.initialV, config.initialTheta, config.initialOmega)

    def _load_gain_table(self) -> List[Tuple[float, float, float, float, float]]:
        search_paths = []
        env_path = os.getenv("LQR_GAIN_TABLE_PATH")
        if env_path:
            search_paths.append(env_path)
        search_paths.extend([
            "/opt/apollo/share/planner/lqr_gain_table.bin",
            "lqr_gain_table.bin",
        ])
        for path in search_paths:
            loaded = self._load_gain_table_from_file(path)
            if loaded is not None:
                return loaded
        flattened = []
        for row in self.K_DEFAULT_GAIN_TABLE:
            flattened.extend(row)
        return flattened

    def _load_gain_table_from_file(self, path: str) -> Optional[List[Tuple[float, float, float, float, float]]]:
        if not os.path.exists(path):
            return None
        header_fmt = "<4I5d2I"
        entry_fmt = "<5d"
        with open(path, "rb") as file_obj:
            header_size = struct.calcsize(header_fmt)
            raw_header = file_obj.read(header_size)
            if len(raw_header) != header_size:
                return None
            magic, version, mu_count, l_count, _qv, _qt, _qo, _qi, _r, _r0, _r1 = struct.unpack(header_fmt, raw_header)
            if magic != 0x4C515254 or version != 1:
                return None
            if mu_count != len(self.K_MU_TABLE) or l_count != len(self.K_L_TABLE):
                return None
            entry_size = struct.calcsize(entry_fmt)
            entries = []
            for _ in range(mu_count * l_count):
                raw_entry = file_obj.read(entry_size)
                if len(raw_entry) != entry_size:
                    return None
                entries.append(struct.unpack(entry_fmt, raw_entry))
            return entries

    def _entry(self, mu_index: int, rope_index: int) -> Tuple[float, float, float, float, float]:
        return self.gainTable[mu_index * len(self.K_L_TABLE) + rope_index]

    def bilinear_interpolate_gain(self, mu: float, rope_length: float) -> Tuple[float, float, float, float]:
        mu_clamped = max(self.K_MU_TABLE[0], min(self.K_MU_TABLE[-1], mu))
        rope_clamped = max(self.K_L_TABLE[0], min(self.K_L_TABLE[-1], rope_length))
        mu_index = 0
        rope_index = 0
        for index in range(len(self.K_MU_TABLE) - 1):
            if self.K_MU_TABLE[index] <= mu_clamped <= self.K_MU_TABLE[index + 1]:
                mu_index = index
                break
        for index in range(len(self.K_L_TABLE) - 1):
            if self.K_L_TABLE[index] <= rope_clamped <= self.K_L_TABLE[index + 1]:
                rope_index = index
                break
        t_mu = (mu_clamped - self.K_MU_TABLE[mu_index]) / (self.K_MU_TABLE[mu_index + 1] - self.K_MU_TABLE[mu_index])
        t_rope = (rope_clamped - self.K_L_TABLE[rope_index]) / (self.K_L_TABLE[rope_index + 1] - self.K_L_TABLE[rope_index])

        def interp(component_index: int) -> float:
            f00 = self._entry(mu_index, rope_index)[component_index]
            f10 = self._entry(mu_index + 1, rope_index)[component_index]
            f01 = self._entry(mu_index, rope_index + 1)[component_index]
            f11 = self._entry(mu_index + 1, rope_index + 1)[component_index]
            return (
                (1.0 - t_mu) * (1.0 - t_rope) * f00
                + t_mu * (1.0 - t_rope) * f10
                + (1.0 - t_mu) * t_rope * f01
                + t_mu * t_rope * f11
            )

        return interp(1), interp(2), interp(3), interp(4)

    def recompute_gains(self) -> None:
        self.kV, self.kTheta, self.kOmega, self.kIntegral = self.bilinear_interpolate_gain(
            self.pendulumGain,
            self.config.ropeLength,
        )

    def set_rope_length(self, rope_length: float) -> None:
        if abs(rope_length - self.config.ropeLength) < 0.5:
            return
        self.config.ropeLength = rope_length
        self.recompute_gains()

    def set_mass(self, payload_mass: float, drone_mass: float) -> None:
        if payload_mass < 10.0:
            payload_mass = 0.0
        if abs(payload_mass - self.config.payloadMass) < 10.0 and abs(drone_mass - self.config.droneMass) < 10.0:
            return
        self.config.payloadMass = payload_mass
        self.config.droneMass = drone_mass
        self.pendulumGain = payload_mass / (payload_mass + drone_mass) if payload_mass + drone_mass > 1e-6 else 0.0
        self.recompute_gains()

    def set_physical_state(self, p: float, v: float, theta: float, omega: float) -> None:
        self.state.p = p
        self.state.v = v
        no_payload = self.pendulumGain < 1e-6
        self.state.theta = 0.0 if no_payload else theta
        self.state.omega = 0.0 if no_payload else omega

    def reset_all(self, p: float, v: float, theta: float, omega: float) -> None:
        self.state = State(p=p, v=v, theta=0.0 if self.pendulumGain < 1e-6 else theta, omega=0.0 if self.pendulumGain < 1e-6 else omega)
        self.integral = 0.0
        self.axCmd = 0.0
        self.axRef = 0.0

    def compute_derivative(self, state: State, a1: float) -> Tuple[float, float, float, float]:
        mu = self.pendulumGain
        if mu < 1e-6:
            return state.v, a1, 0.0, 0.0
        sin_theta = math.sin(state.theta)
        cos_theta = math.cos(state.theta)
        denom = 1.0 - mu * cos_theta * cos_theta
        if denom < 1e-3:
            denom = 1e-3
        dv = ((1.0 - mu) * a1 + mu * K_GRAVITY * sin_theta * cos_theta + mu * self.config.ropeLength * state.omega * state.omega * sin_theta) / denom
        vx_payload = state.v + self.config.ropeLength * state.omega * cos_theta
        vz_payload = self.config.ropeLength * state.omega * sin_theta
        vel_abs_sq = vx_payload * vx_payload + vz_payload * vz_payload
        vel_abs = math.sqrt(vel_abs_sq)
        quadratic_drag = 0.0
        if vel_abs > 1e-6 and self.config.payloadMass > 1e-6:
            drag_force_mag = 0.5 * self.config.airDensity * self.config.dragCoeff * self.config.dragArea * vel_abs_sq
            drag_x = -drag_force_mag * vx_payload / vel_abs
            drag_z = -drag_force_mag * vz_payload / vel_abs
            tangential_drag = drag_x * cos_theta + drag_z * sin_theta
            quadratic_drag = tangential_drag / (self.config.payloadMass * self.config.ropeLength)
        d_omega = -(K_GRAVITY * sin_theta + dv * cos_theta) / self.config.ropeLength - self.config.linearDampingCoeff * state.omega + quadratic_drag
        return state.v, dv, state.omega, d_omega

    def step_rk4(self, state: State, a1: float, dt: float) -> None:
        k1 = self.compute_derivative(state, a1)
        s2 = State(time=state.time, p=state.p + k1[0] * dt * 0.5, v=state.v + k1[1] * dt * 0.5, theta=state.theta + k1[2] * dt * 0.5, omega=state.omega + k1[3] * dt * 0.5)
        k2 = self.compute_derivative(s2, a1)
        s3 = State(time=state.time, p=state.p + k2[0] * dt * 0.5, v=state.v + k2[1] * dt * 0.5, theta=state.theta + k2[2] * dt * 0.5, omega=state.omega + k2[3] * dt * 0.5)
        k3 = self.compute_derivative(s3, a1)
        s4 = State(time=state.time, p=state.p + k3[0] * dt, v=state.v + k3[1] * dt, theta=state.theta + k3[2] * dt, omega=state.omega + k3[3] * dt)
        k4 = self.compute_derivative(s4, a1)
        state.p += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * dt / 6.0
        state.v += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * dt / 6.0
        state.theta += (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) * dt / 6.0
        state.omega += (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) * dt / 6.0
        state.time += dt
        _, dv, _, _ = self.compute_derivative(state, a1)
        state.a = dv

    def compute_feedforward(self, guide_acc: float, theta: float, omega: float) -> float:
        if self.pendulumGain < 1e-6:
            return guide_acc
        mu = self.pendulumGain
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        denom = 1.0 - mu * cos_theta * cos_theta
        if denom < 1e-3:
            return guide_acc
        one_minus_mu = 1.0 - mu
        if one_minus_mu < 1e-3:
            return guide_acc
        numerator = guide_acc * denom - mu * K_GRAVITY * sin_theta * cos_theta - mu * self.config.ropeLength * omega * omega * sin_theta
        return numerator / one_minus_mu

    def compute_control(self, velocity: float, theta: float, omega: float, velocity_ref: float) -> float:
        err = velocity - velocity_ref
        integral_before = self.integral
        no_payload = self.pendulumGain < 1e-6
        ctrl_theta = 0.0 if no_payload else theta
        ctrl_omega = 0.0 if no_payload else omega
        feedforward = self.compute_feedforward(self.axRef, ctrl_theta, ctrl_omega)
        swing_uav_acc = 0.0
        if not no_payload:
            sin_theta = math.sin(ctrl_theta)
            cos_theta = math.cos(ctrl_theta)
            denom = 1.0 - self.pendulumGain * cos_theta * cos_theta
            if denom < 1e-3:
                denom = 1e-3
            swing_uav_acc = (self.pendulumGain * K_GRAVITY * sin_theta * cos_theta + self.pendulumGain * self.config.ropeLength * ctrl_omega * ctrl_omega * sin_theta) / denom
        int_preview = self.integral + err * self.config.dt
        feedback_preview = -(self.kV * err + self.kTheta * ctrl_theta + self.kOmega * ctrl_omega + self.kIntegral * int_preview)
        preview_acc = feedforward + feedback_preview
        preview_sat = saturate(preview_acc, self.config.lqrAxMax)
        delta = preview_acc - preview_sat
        anti_windup_gain = self.kIntegral / self.kV if self.kV > 1e-6 else 0.0
        anti_windup_correction = anti_windup_gain * delta * self.config.dt
        self.integral = clamp(int_preview + anti_windup_correction, -3.0, 3.0)
        velocity_feedback = -self.kV * err
        pendulum_feedback = -(self.kTheta * ctrl_theta + self.kOmega * ctrl_omega)
        integral_feedback = -self.kIntegral * self.integral
        target_acc = saturate(feedforward + velocity_feedback + pendulum_feedback + integral_feedback, self.config.lqrAxMax)
        self.lastStepMetrics = StepMetrics(
            currentVel=velocity,
            vRef=velocity_ref,
            err=err,
            theta=ctrl_theta,
            omega=ctrl_omega,
            axRef=self.axRef,
            feedforwardAcc=feedforward,
            swingUavAcc=swing_uav_acc,
            integralBefore=integral_before,
            integralPreview=int_preview,
            integralAfter=self.integral,
            antiWindupGain=anti_windup_gain,
            antiWindupDelta=delta,
            antiWindupCorrection=anti_windup_correction,
            antiWindupPreviewAcc=preview_acc,
            antiWindupPreviewSatAcc=preview_sat,
            targetAcc=target_acc,
            cmdAcc=self.axCmd,
        )
        return target_acc

    def apply_jerk_limit(self, target: float, dt: float) -> float:
        jerk = (target - self.axCmd) / dt
        if jerk > self.config.lqrJerkMax:
            self.axCmd += self.config.lqrJerkMax * dt
        elif jerk < -self.config.lqrJerkMax:
            self.axCmd -= self.config.lqrJerkMax * dt
        else:
            self.axCmd = target
        self.lastStepMetrics.cmdAcc = self.axCmd
        return self.axCmd

    def step(self, velocity_ref: float) -> None:
        ax_target = self.compute_control(self.state.v, self.state.theta, self.state.omega, velocity_ref)
        ax_cmd = self.apply_jerk_limit(ax_target, self.config.dt)
        self.step_rk4(self.state, ax_cmd, self.config.dt)

    def set_integral(self, value: float) -> None:
        self.integral = value

    def set_ax_cmd(self, value: float) -> None:
        self.axCmd = value

    def set_ax_ref(self, value: float) -> None:
        self.axRef = value
