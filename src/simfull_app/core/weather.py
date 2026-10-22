"""Weather engine: ambient conditions, diurnal cycle, and solar position/irradiance."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class WeatherCondition:
    time_seconds: float
    ambient_temp_c: float
    ground_temp_c: float
    solar_direct_w_m2: float
    solar_diffuse_w_m2: float
    solar_zenith_deg: float
    solar_azimuth_deg: float

    @property
    def solar_ghi_w_m2(self) -> float:
        """Global Horizontal Irradiance GHI = DNI * cos(zenith) + DHI."""
        cos_z = max(0.0, math.cos(math.radians(self.solar_zenith_deg)))
        return float(self.solar_direct_w_m2 * cos_z + self.solar_diffuse_w_m2)


class WeatherDriver:
    """Calculates diurnal ambient temperature and solar radiation curves."""

    PRESETS = {
        "winter_design_day": {
            "name": "Winter Peak Design Day",
            "t_min": -4.0, "t_max": 2.0, "t_ground": 8.0,
            "max_solar_direct": 180.0, "max_solar_diffuse": 80.0,
            "daylight_hours": 8.0,
        },
        "summer_design_day": {
            "name": "Summer Peak Cooling Day",
            "t_min": 17.0, "t_max": 31.0, "t_ground": 16.0,
            "max_solar_direct": 750.0, "max_solar_diffuse": 220.0,
            "daylight_hours": 15.0,
        },
        "london_shoulder": {
            "name": "London Shoulder Season (October/March)",
            "t_min": 5.0, "t_max": 14.0, "t_ground": 11.0,
            "max_solar_direct": 420.0, "max_solar_diffuse": 140.0,
            "daylight_hours": 11.0,
        },
        "continental_annual": {
            "name": "Frankfurt / Central European Average",
            "t_min": 1.0, "t_max": 9.0, "t_ground": 10.0,
            "max_solar_direct": 350.0, "max_solar_diffuse": 120.0,
            "daylight_hours": 10.0,
        },
    }

    def __init__(self, preset: str = "winter_design_day") -> None:
        self.preset_key = preset
        self.config = self.PRESETS.get(preset, self.PRESETS["winter_design_day"])

    def evaluate(self, t_seconds: float) -> WeatherCondition:
        """Evaluate ambient conditions at time t seconds from midnight."""
        hour = (t_seconds / 3600.0) % 24.0
        c = self.config

        # Diurnal temperature sinusoidal wave peaking at ~15:00, lowest at ~05:00
        t_mean = 0.5 * (c["t_min"] + c["t_max"])
        t_amp = 0.5 * (c["t_max"] - c["t_min"])
        t_amb = t_mean + t_amp * math.sin((hour - 9.0) * (2.0 * math.pi / 24.0))

        # Solar position (simplified diurnal arch)
        solar_noon = 12.0
        daylight = c["daylight_hours"]
        sunrise = solar_noon - daylight * 0.5
        sunset = solar_noon + daylight * 0.5

        if sunrise <= hour <= sunset:
            frac = (hour - sunrise) / daylight
            solar_elev = math.sin(frac * math.pi)
            zenith_deg = 90.0 - (solar_elev * 60.0) # max elevation 60°
            azimuth_deg = 90.0 + frac * 180.0      # 90° East -> 180° South -> 270° West
            i_dir = c["max_solar_direct"] * (solar_elev ** 1.2)
            i_diff = c["max_solar_diffuse"] * math.sin(frac * math.pi)
        else:
            zenith_deg = 90.0
            azimuth_deg = 0.0
            i_dir = 0.0
            i_diff = 0.0

        return WeatherCondition(
            time_seconds=t_seconds,
            ambient_temp_c=t_amb,
            ground_temp_c=c["t_ground"],
            solar_direct_w_m2=i_dir,
            solar_diffuse_w_m2=i_diff,
            solar_zenith_deg=zenith_deg,
            solar_azimuth_deg=azimuth_deg,
        )

    def calculate_incident_solar(
        self,
        cond: WeatherCondition,
        surface_azimuth_deg: float,
        surface_tilt_deg: float = 90.0,
    ) -> float:
        """Calculate solar radiation incident on a surface with given azimuth and tilt."""
        if cond.solar_direct_w_m2 == 0.0 and cond.solar_diffuse_w_m2 == 0.0:
            return 0.0

        zenith_rad = math.radians(cond.solar_zenith_deg)
        elev_rad = max(0.0, 0.5 * math.pi - zenith_rad)
        sun_az_rad = math.radians(cond.solar_azimuth_deg)
        surf_az_rad = math.radians(surface_azimuth_deg)
        tilt_rad = math.radians(surface_tilt_deg)

        # Cosine of angle of incidence
        cos_theta = (
            math.sin(elev_rad) * math.cos(tilt_rad)
            + math.cos(elev_rad) * math.sin(tilt_rad) * math.cos(sun_az_rad - surf_az_rad)
        )
        cos_theta = max(0.0, cos_theta)

        # Beam + Sky diffuse + Ground reflection (Perez / Liu-Jordan isotropic)
        i_beam = cond.solar_direct_w_m2 * cos_theta
        i_diffuse = cond.solar_diffuse_w_m2 * (0.5 * (1.0 + math.cos(tilt_rad)))
        i_ground = 0.2 * (cond.solar_direct_w_m2 * math.sin(elev_rad) + cond.solar_diffuse_w_m2) * (0.5 * (1.0 - math.cos(tilt_rad)))

        return max(0.0, i_beam + i_diffuse + i_ground)
