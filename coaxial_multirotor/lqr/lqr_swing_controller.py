from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .standalone_lqr import Config as StandaloneConfig
from .standalone_lqr import StandaloneLqrSimulator, StepMetrics


@dataclass
class Config:
    ropeLength: float = 10.0
    payloadMass: float = 180.0
    droneMass: float = 120.0
    dt: float = 0.02
    lqrAxMax: float = 5.0
    lqrJerkMax: float = 10.0


@dataclass
class ExecutedStepLog:
    pointIdx: int
    x: StepMetrics
    y: StepMetrics


class BatchTuning:
    K_LOOKAHEAD_STEPS = 0
    K_EXECUTED_STEPS = 5
    K_STICK_RELEASE_VREF_THRESH = 0.05
    K_STICK_RELEASE_V_THRESH = 0.5


class LqrSwingController:
    def __init__(self, config: Config) -> None:
        lqr_config_x = StandaloneConfig(
            dt=config.dt,
            ropeLength=config.ropeLength,
            payloadMass=config.payloadMass,
            droneMass=config.droneMass,
            lqrAxMax=config.lqrAxMax,
            lqrJerkMax=config.lqrJerkMax,
        )
        lqr_config_y = StandaloneConfig(
            dt=config.dt,
            ropeLength=config.ropeLength,
            payloadMass=config.payloadMass,
            droneMass=config.droneMass,
            lqrAxMax=config.lqrAxMax,
            lqrJerkMax=config.lqrJerkMax,
        )
        self.lqrX = StandaloneLqrSimulator(lqr_config_x)
        self.lqrY = StandaloneLqrSimulator(lqr_config_y)
        self.executedStepLogs: List[ExecutedStepLog] = []

    def update_params(self, rope_length: float, payload_mass: float, drone_mass: float) -> None:
        self.lqrX.set_rope_length(rope_length)
        self.lqrX.set_mass(payload_mass, drone_mass)
        self.lqrY.set_rope_length(rope_length)
        self.lqrY.set_mass(payload_mass, drone_mass)

    def process_batch(
        self,
        vx_ref_body: List[float],
        vy_ref_body: List[float],
        ax_ref_body: List[float],
        ay_ref_body: List[float],
        px0_body: float,
        vx0_body: float,
        theta_body_x0: float,
        omega_body_x0: float,
        py0_body: float,
        vy0_body: float,
        theta_body_y0: float,
        omega_body_y0: float,
    ) -> dict:
        count = len(vx_ref_body)
        if count == 0 or len(vy_ref_body) != count:
            return {"vx": [], "vy": [], "ax": [], "ay": []}
        if ax_ref_body and len(ax_ref_body) != count:
            return {"vx": [], "vy": [], "ax": [], "ay": []}
        if ay_ref_body and len(ay_ref_body) != count:
            return {"vx": [], "vy": [], "ax": [], "ay": []}

        vx_shifted = list(vx_ref_body)
        vy_shifted = list(vy_ref_body)
        ax_shifted = list(ax_ref_body)
        ay_shifted = list(ay_ref_body)
        lookahead_steps = BatchTuning.K_LOOKAHEAD_STEPS
        executed_steps = BatchTuning.K_EXECUTED_STEPS
        if count > lookahead_steps and lookahead_steps > 0:
            for index in range(count - lookahead_steps):
                vx_shifted[index] = vx_shifted[index + lookahead_steps]
                vy_shifted[index] = vy_shifted[index + lookahead_steps]
            for index in range(count - lookahead_steps, count):
                vx_shifted[index] = vx_shifted[count - lookahead_steps - 1]
                vy_shifted[index] = vy_shifted[count - lookahead_steps - 1]
            if ax_shifted:
                for index in range(count - lookahead_steps):
                    ax_shifted[index] = ax_shifted[index + lookahead_steps]
                    ay_shifted[index] = ay_shifted[index + lookahead_steps]
                for index in range(count - lookahead_steps, count):
                    ax_shifted[index] = ax_shifted[count - lookahead_steps - 1]
                    ay_shifted[index] = ay_shifted[count - lookahead_steps - 1]

        self.executedStepLogs = []
        self.lqrX.set_physical_state(px0_body, vx0_body, theta_body_x0, omega_body_x0)
        self.lqrY.set_physical_state(py0_body, vy0_body, theta_body_y0, omega_body_y0)

        saved_ax_cmd_x = 0.0
        saved_integral_x = 0.0
        saved_ax_cmd_y = 0.0
        saved_integral_y = 0.0
        stick_released_x = bool(vx_ref_body) and abs(vx_ref_body[0]) < BatchTuning.K_STICK_RELEASE_VREF_THRESH and abs(vx0_body) < BatchTuning.K_STICK_RELEASE_V_THRESH
        stick_released_y = bool(vy_ref_body) and abs(vy_ref_body[0]) < BatchTuning.K_STICK_RELEASE_VREF_THRESH and abs(vy0_body) < BatchTuning.K_STICK_RELEASE_V_THRESH

        vx_out = [0.0] * count
        vy_out = [0.0] * count
        ax_out = [0.0] * count
        ay_out = [0.0] * count
        for index in range(count):
            if index == executed_steps:
                saved_ax_cmd_x = self.lqrX.axCmd
                saved_integral_x = self.lqrX.integral
                saved_ax_cmd_y = self.lqrY.axCmd
                saved_integral_y = self.lqrY.integral
            if index < len(ax_shifted):
                self.lqrX.set_ax_ref(0.0 if stick_released_x else ax_shifted[index])
            if index < len(ay_shifted):
                self.lqrY.set_ax_ref(0.0 if stick_released_y else ay_shifted[index])
            self.lqrX.step(vx_shifted[index])
            self.lqrY.step(vy_shifted[index])
            if index < executed_steps:
                self.executedStepLogs.append(ExecutedStepLog(index, self.lqrX.lastStepMetrics, self.lqrY.lastStepMetrics))
            vx_out[index] = self.lqrX.state.v
            vy_out[index] = self.lqrY.state.v
            ax_out[index] = self.lqrX.axCmd
            ay_out[index] = self.lqrY.axCmd

        if count <= executed_steps:
            saved_ax_cmd_x = self.lqrX.axCmd
            saved_integral_x = self.lqrX.integral
            saved_ax_cmd_y = self.lqrY.axCmd
            saved_integral_y = self.lqrY.integral

        self.lqrX.set_ax_cmd(saved_ax_cmd_x)
        self.lqrX.set_integral(saved_integral_x)
        self.lqrY.set_ax_cmd(saved_ax_cmd_y)
        self.lqrY.set_integral(saved_integral_y)
        return {"vx": vx_out, "vy": vy_out, "ax": ax_out, "ay": ay_out}
