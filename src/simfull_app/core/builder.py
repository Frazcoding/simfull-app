"""Graph builder: converts building geometry and plant specifications into a simulation network."""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from .components import (
    HeatPumpPlant,
    MultiLayerWall,
    PhotovoltaicArray,
    StratifiedBufferTank,
    StructuralThermalStressElement,
    ThermalZone,
    WallLayer,
    WindowGlazing,
)
from .geometry import BuildingGeometry
from .network import Domain, NetworkLink, NetworkNode, SimulationNetwork


# Standard multi-layer wall constructions
CONSTRUCTIONS: dict[str, list[WallLayer]] = {
    "brick_cavity": [
        WallLayer("plasterboard", 0.015, "Internal Plasterboard"),
        WallLayer("concrete", 0.100, "Inner Concrete Block"),
        WallLayer("insulation_pir", 0.100, "PIR Cavity Insulation"),
        WallLayer("air_cavity", 0.050, "Residual Air Cavity"),
        WallLayer("brick", 0.102, "External Facing Brick"),
    ],
    "timber_frame": [
        WallLayer("plasterboard", 0.015, "Internal Plasterboard"),
        WallLayer("insulation_mineral_wool", 0.140, "Mineral Wool between Studs"),
        WallLayer("timber", 0.015, "OSB Sheathing Board"),
        WallLayer("insulation_pir", 0.050, "External Insulation"),
        WallLayer("timber", 0.020, "Timber Cladding"),
    ],
    "solid_brick_uninsulated": [
        WallLayer("plasterboard", 0.015, "Internal Plaster"),
        WallLayer("brick", 0.215, "Solid Brickwork"),
    ],
}


class BuildingNetworkBuilder:
    """Synthesizes a complete multi-zone, plant, and multi-domain network from geometry."""

    def __init__(
        self,
        geometry: BuildingGeometry,
        construction_preset: str = "brick_cavity",
        has_heat_pump: bool = True,
        has_buffer_tank: bool = True,
        has_solar_pv: bool = True,
        has_structural_monitor: bool = True,
        hp_nominal_kw: float = 8.0,
        pv_area_m2: float = 20.0,
    ) -> None:
        self.geometry = geometry
        self.layers = CONSTRUCTIONS.get(construction_preset, CONSTRUCTIONS["brick_cavity"])
        self.has_heat_pump = has_heat_pump
        self.has_buffer_tank = has_buffer_tank
        self.has_solar_pv = has_solar_pv
        self.has_structural_monitor = has_structural_monitor
        self.hp_capacity_kw = hp_nominal_kw
        self.pv_area_m2 = pv_area_m2

    def build(self) -> tuple[SimulationNetwork, dict[str, Any]]:
        network = SimulationNetwork()

        # 1. External boundary nodes
        network.add_node(NetworkNode(
            id="ambient", domain=Domain.THERMAL, name="Outdoor Ambient Air",
            capacity=0.0, value=5.0, initial_value=5.0, fixed=True
        ))
        network.add_node(NetworkNode(
            id="ground", domain=Domain.THERMAL, name="Ground Temperature",
            capacity=0.0, value=10.0, initial_value=10.0, fixed=True
        ))

        # 2. Zones (air nodes, ventilation, ground slab, roof)
        zone_components: dict[str, ThermalZone] = {}
        for z in self.geometry.zones:
            tz = ThermalZone(
                zone_id=z.id,
                floor_area=z.floor_area,
                height=z.height,
                internal_gain_w=z.internal_gain_w,
                air_changes_per_hour=z.ventilation_ach,
                outside_node_id="ambient",
            )
            tz.build_network(network)
            zone_components[z.id] = tz

            # Floor heat loss to ground: G = U_floor * Area
            g_floor = self.geometry.floor_u_value * z.floor_area
            network.add_link(NetworkLink(
                id=f"{z.id}.floor_ground", node_a=tz.air_node_id, node_b="ground",
                conductance=g_floor, kind="conduction", area=z.floor_area, equation="Ground Slab U*A"
            ))

            # Roof heat loss to ambient: G = U_roof * Area
            g_roof = self.geometry.roof_u_value * z.floor_area
            network.add_link(NetworkLink(
                id=f"{z.id}.roof_ambient", node_a=tz.air_node_id, node_b="ambient",
                conductance=g_roof, kind="conduction", area=z.floor_area, equation="Roof U*A"
            ))

        # 3. Walls and Windows
        for w in self.geometry.walls:
            if w.zone_b is None:
                # Exterior wall exposed to ambient
                wall_comp = MultiLayerWall(
                    wall_id=w.id,
                    area=w.net_wall_area,
                    layers=self.layers,
                    inside_node_id=f"{w.zone_a}.air",
                    outside_node_id="ambient",
                    azimuth=w.azimuth,
                )
                wall_comp.build_network(network)

                # Window glazing
                if w.window_area > 0:
                    win_comp = WindowGlazing(
                        window_id=f"{w.id}.window",
                        area=w.window_area,
                        u_value=1.3,
                        inside_node_id=f"{w.zone_a}.air",
                        outside_node_id="ambient",
                        azimuth=w.azimuth,
                    )
                    win_comp.build_network(network)
            else:
                # Interior partition between zone_a and zone_b
                # Internal partition U-value ~ 1.5 W/(m^2*K)
                g_part = 1.5 * w.gross_area
                network.add_link(NetworkLink(
                    id=f"{w.id}.partition",
                    node_a=f"{w.zone_a}.air",
                    node_b=f"{w.zone_b}.air",
                    conductance=g_part,
                    kind="conduction",
                    area=w.gross_area,
                    equation="Partition U*A"
                ))

        # 4. Plant: Stratified Buffer Tank and Heat Pump
        tank_top_node = "plant.tank.layer_0" if self.has_buffer_tank else None
        hp_plant = None

        if self.has_buffer_tank:
            tank = StratifiedBufferTank(
                tank_id="plant.tank", volume_litres=300.0, num_layers=4, ambient_node_id="ambient"
            )
            tank.build_network(network)

        if self.has_heat_pump:
            hp_plant = HeatPumpPlant(
                plant_id="plant.hp",
                nominal_capacity_kw=self.hp_capacity_kw,
                cop_nominal=3.8,
                source_node_id="ambient",
                supply_node_id=tank_top_node or f"{self.geometry.zones[0].id}.air",
            )
            hp_plant.build_network(network)

        # 5. Hydronic Emitters in each zone
        if tank_top_node is not None:
            for z in self.geometry.zones:
                # Emitter link between tank top layer and zone air (conductance modulated by thermostat)
                network.add_link(NetworkLink(
                    id=f"{z.id}.emitter",
                    node_a=tank_top_node,
                    node_b=f"{z.id}.air",
                    conductance=120.0, # nominal W/K design radiator
                    kind="emitter",
                    equation="Hydronic Emitter Q = G*(T_w - T_room)"
                ))

        # 6. Multi-domain extensions: Solar PV and Structural Stress
        pv_comp = None
        if self.has_solar_pv:
            pv_comp = PhotovoltaicArray(pv_id="solar.pv", area_m2=self.pv_area_m2)
            pv_comp.build_network(network)

        struct_comp = None
        if self.has_structural_monitor and self.geometry.walls:
            # Monitor thermal stress in the first exterior wall
            first_wall = self.geometry.walls[0]
            struct_comp = StructuralThermalStressElement(
                struct_id="struct.corner", thermal_node_id=f"{first_wall.id}.surf_out"
            )
            struct_comp.build_network(network)

        metadata = {
            "num_zones": len(self.geometry.zones),
            "num_walls": len(self.geometry.walls),
            "total_floor_area": self.geometry.total_floor_area(),
            "has_heat_pump": self.has_heat_pump,
            "has_buffer_tank": self.has_buffer_tank,
            "has_solar_pv": self.has_solar_pv,
            "has_structural_monitor": self.has_structural_monitor,
            "hp_plant": hp_plant,
            "pv_comp": pv_comp,
            "struct_comp": struct_comp,
        }

        return network, metadata
