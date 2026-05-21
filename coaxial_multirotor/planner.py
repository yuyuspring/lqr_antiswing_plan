from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig


@dataclass
class WaypointPlanner:
    scenario: ScenarioConfig

    def sample(self, time_s: float) -> dict:
        phases = self.scenario.phase_times_s
        target_xy = self.scenario.horizontal_target_m
        altitude = self.scenario.altitude_step_m
        if time_s < phases["idle"]:
            x_cmd, y_cmd, z_cmd = 0.0, 0.0, 0.0
        elif time_s < phases["climb_end"]:
            ratio = (time_s - phases["idle"]) / max(phases["climb_end"] - phases["idle"], 1e-6)
            x_cmd, y_cmd, z_cmd = 0.0, 0.0, altitude * ratio
        elif time_s < phases["hover_end"]:
            x_cmd, y_cmd, z_cmd = 0.0, 0.0, altitude
        elif time_s < phases["translate_end"]:
            ratio = (time_s - phases["hover_end"]) / max(phases["translate_end"] - phases["hover_end"], 1e-6)
            x_cmd = target_xy[0] * ratio
            y_cmd = target_xy[1] * ratio
            z_cmd = altitude
        elif time_s < phases["hold_end"]:
            x_cmd, y_cmd, z_cmd = target_xy[0], target_xy[1], altitude
        else:
            ratio = (time_s - phases["hold_end"]) / max(phases["land_end"] - phases["hold_end"], 1e-6)
            x_cmd, y_cmd = target_xy[0], target_xy[1]
            z_cmd = altitude * (1.0 - np.clip(ratio, 0.0, 1.0))
        return {"x_m": x_cmd, "y_m": y_cmd, "z_m": z_cmd, "yaw_deg": self.scenario.yaw_target_deg}
