"""API request bodies and schemas for simfull simulation software."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class SimulateRequest(BaseModel):
    """Body of POST /api/simulate/network: runs full multi-zone simulation."""
    model_config = ConfigDict(extra="ignore")

    geometry: dict[str, Any] = Field(default_factory=dict, description="BuildingGeometry JSON")
    weather_preset: str = Field("winter_design_day", description="Weather location / design day preset")
    construction: str = Field("brick_cavity", description="Wall construction preset")
    hp_kw: float = Field(8.0, ge=1.0, le=100.0, description="Nominal heat pump capacity in kW")
    pv_m2: float = Field(20.0, ge=0.0, le=500.0, description="Roof solar PV area in m^2")
    duration_days: float = Field(1.0, ge=0.1, le=14.0, description="Simulation length in days")
    dt_seconds: float = Field(300.0, ge=30.0, le=3600.0, description="Time step in seconds")
    run_detailed_fields: bool = Field(True, description="Run Level 2 1D wall and 2D thermal bridge models")


class ThermalBridgeRequest(BaseModel):
    """Body of POST /api/simulate/thermal-bridge: 2D EN ISO 10211 solver."""
    model_config = ConfigDict(extra="ignore")

    junction_type: Literal["wall_corner", "wall_floor", "window_reveal"] = "wall_corner"
    t_inside: float = Field(20.0, description="Internal boundary temperature in °C")
    t_outside: float = Field(0.0, description="External boundary temperature in °C")
    r_si: float = Field(0.13, ge=0.01, le=1.0, description="Internal surface resistance m^2*K/W")
    r_se: float = Field(0.04, ge=0.01, le=1.0, description="External surface resistance m^2*K/W")
    wall_u_value: float = Field(0.22, ge=0.05, le=5.0, description="Flanking wall U-value W/(m^2*K)")
    grid_resolution: int = Field(30, ge=15, le=60, description="Grid nodes in each direction")


class Wall1DRequest(BaseModel):
    """Body of POST /api/simulate/wall-1d: 1D finite-difference multi-layer conduction."""
    model_config = ConfigDict(extra="ignore")

    construction: str = "brick_cavity"
    t_inside: float = 21.0
    t_outside: float = 0.0
    duration_hours: float = 24.0
    dt_seconds: float = 120.0


class STLImportRequest(BaseModel):
    """Body of POST /api/geometry/from-stl: parse uploaded 3D STL file."""
    model_config = ConfigDict(extra="ignore")

    content: str = Field(..., description="ASCII text or base64-encoded binary STL")
    is_base64: bool = False
    zone_name: str = "Imported 3D Building"
