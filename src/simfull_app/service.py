"""Service layer for simfull application logic, independent of web framework."""
from __future__ import annotations

import base64
from typing import Any

from .core.builder import CONSTRUCTIONS
from .core.components import MATERIALS
from .core.fields import MultiLayerWall1D, ThermalBridge2D
from .core.geometry import BuildingGeometry, Point2D, PolygonZone, STLParser, WallSegment
from .core.simulation import FullSimulationEngine
from .core.weather import WeatherDriver
from .models import (
    SimulateRequest,
    STLImportRequest,
    ThermalBridgeRequest,
    Wall1DRequest,
)


def default_two_room_building() -> BuildingGeometry:
    """Standard 2-room test building geometry (Living Room + Bedroom)."""
    # Living room: 6m x 5m (30 m^2)
    # Bedroom: 4m x 5m (20 m^2)
    p0 = Point2D(0.0, 0.0)
    p1 = Point2D(6.0, 0.0)
    p2 = Point2D(10.0, 0.0)
    p3 = Point2D(10.0, 5.0)
    p4 = Point2D(6.0, 5.0)
    p5 = Point2D(0.0, 5.0)

    z1 = PolygonZone(
        id="living_room",
        name="Living Room (South)",
        vertices=[p0, p1, p4, p5],
        height=2.6,
        setpoint_temp_c=21.0,
        internal_gain_w=180.0,
        ventilation_ach=0.5,
    )
    z2 = PolygonZone(
        id="bedroom",
        name="Bedroom (East)",
        vertices=[p1, p2, p3, p4],
        height=2.6,
        setpoint_temp_c=19.0,
        internal_gain_w=100.0,
        ventilation_ach=0.5,
    )

    walls = [
        # South exterior walls
        WallSegment(id="w_living_south", p1=p0, p2=p1, zone_a="living_room", window_area=3.5),
        WallSegment(id="w_bed_south", p1=p1, p2=p2, zone_a="bedroom", window_area=2.0),
        # East exterior wall
        WallSegment(id="w_bed_east", p1=p2, p2=p3, zone_a="bedroom", window_area=1.5),
        # North exterior walls
        WallSegment(id="w_bed_north", p1=p3, p2=p4, zone_a="bedroom", window_area=1.0),
        WallSegment(id="w_living_north", p1=p4, p2=p5, zone_a="living_room", window_area=1.5),
        # West exterior wall
        WallSegment(id="w_living_west", p1=p5, p2=p0, zone_a="living_room", window_area=2.0),
        # Interior partition between Living Room and Bedroom
        WallSegment(id="w_partition", p1=p1, p2=p4, zone_a="living_room", zone_b="bedroom", u_value=1.5),
    ]

    return BuildingGeometry(
        zones=[z1, z2],
        walls=walls,
        roof_u_value=0.15,
        floor_u_value=0.18,
        metadata={"preset": "two_room_cottage"},
    )


class SimulationService:
    """Manages building network simulations, spatial fields, and CAD processing."""

    @staticmethod
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": "simfull-app",
            "version": "0.1.0",
            "capabilities": [
                "network_model_level1",
                "finite_difference_wall1d_level2",
                "thermal_bridge_2d_en_iso_10211",
                "stl_mesh_3d_import",
                "multi_domain_expansion",
            ],
        }

    @staticmethod
    def weather_presets() -> list[dict[str, Any]]:
        return [
            {"id": k, "name": v["name"], "t_min": v["t_min"], "t_max": v["t_max"]}
            for k, v in WeatherDriver.PRESETS.items()
        ]

    @staticmethod
    def materials() -> dict[str, Any]:
        return MATERIALS

    @staticmethod
    def constructions() -> dict[str, Any]:
        return {
            k: [
                {"material": l.material, "thickness_mm": int(l.thickness * 1000), "name": l.name}
                for l in layers
            ]
            for k, layers in CONSTRUCTIONS.items()
        }

    def simulate_network(self, req: SimulateRequest) -> dict[str, Any]:
        """Execute Level 1 network simulation and optional Level 2 fields."""
        if req.geometry and req.geometry.get("zones"):
            geom = BuildingGeometry.from_dict(req.geometry)
        else:
            geom = default_two_room_building()

        engine = FullSimulationEngine(
            geometry=geom,
            weather_preset=req.weather_preset,
            construction=req.construction,
            hp_kw=req.hp_kw,
            pv_m2=req.pv_m2,
            run_detailed_fields=req.run_detailed_fields,
        )

        output = engine.run(
            duration_days=req.duration_days,
            dt_seconds=req.dt_seconds,
        )
        return output.to_dict()

    @staticmethod
    def simulate_thermal_bridge(req: ThermalBridgeRequest) -> dict[str, Any]:
        """Execute 2D EN ISO 10211 thermal bridge calculation."""
        tb = ThermalBridge2D(nx=req.grid_resolution, ny=req.grid_resolution)
        res = tb.solve_junction(
            junction_type=req.junction_type,
            t_inside=req.t_inside,
            t_outside=req.t_outside,
            r_si=req.r_si,
            r_se=req.r_se,
            wall_u_value=req.wall_u_value,
        )
        return res.to_dict()

    @staticmethod
    def simulate_wall_1d(req: Wall1DRequest) -> dict[str, Any]:
        """Execute 1D finite-difference multi-layer wall conduction."""
        layers = CONSTRUCTIONS.get(req.construction, CONSTRUCTIONS["brick_cavity"])
        solver = MultiLayerWall1D(layers=layers, nodes_per_layer=5)
        res = solver.simulate(
            t_inside=req.t_inside,
            t_outside=req.t_outside,
            duration_hours=req.duration_hours,
            dt_seconds=req.dt_seconds,
        )
        return res.to_dict()

    @staticmethod
    def parse_stl(req: STLImportRequest) -> dict[str, Any]:
        """Parse uploaded ASCII or Binary STL file and synthesize thermal building model."""
        if req.is_base64:
            content_bytes = base64.b64decode(req.content)
            geom = STLParser.to_building_geometry(content_bytes, zone_name=req.zone_name)
        else:
            geom = STLParser.to_building_geometry(req.content, zone_name=req.zone_name)
        return geom.to_dict()
