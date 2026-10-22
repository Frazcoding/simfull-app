"""Extensible physics components across Thermal, Fluid, Electrical, and Structural domains.

Every component defines physical equations, registers state nodes and conductance links
into the simulation network, and updates dynamically with environmental drivers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from .network import Domain, NetworkLink, NetworkNode, SimulationNetwork


# Standard building material library: thermal conductivity k (W/m*K), density rho (kg/m^3), specific heat cp (J/kg*K)
MATERIALS: dict[str, dict[str, float]] = {
    "concrete": {"k": 1.75, "rho": 2400.0, "cp": 1000.0, "name": "Heavyweight Concrete"},
    "brick": {"k": 0.77, "rho": 1700.0, "cp": 840.0, "name": "Clay Brickwork"},
    "insulation_eps": {"k": 0.035, "rho": 25.0, "cp": 1400.0, "name": "Expanded Polystyrene (EPS)"},
    "insulation_mineral_wool": {"k": 0.038, "rho": 40.0, "cp": 1030.0, "name": "Mineral Wool"},
    "insulation_pir": {"k": 0.022, "rho": 32.0, "cp": 1400.0, "name": "Polyisocyanurate (PIR)"},
    "plasterboard": {"k": 0.25, "rho": 800.0, "cp": 1000.0, "name": "Gypsum Plasterboard"},
    "screed": {"k": 1.40, "rho": 2000.0, "cp": 1000.0, "name": "Floor Screed"},
    "timber": {"k": 0.13, "rho": 500.0, "cp": 1600.0, "name": "Structural Timber"},
    "glass": {"k": 1.05, "rho": 2500.0, "cp": 840.0, "name": "Window Glass"},
    "air_cavity": {"k": 0.18, "rho": 1.2, "cp": 1005.0, "name": "Unventilated Air Cavity"},
}


class BaseComponent(ABC):
    """Abstract base component across all physical domains."""
    name: str

    @abstractmethod
    def build_network(self, network: SimulationNetwork) -> None:
        """Register state nodes and conductance links into the network."""
        pass

    def step(self, t: float, dt: float, states: dict[str, float]) -> None:
        """Optional hook for component-specific state or control updates."""
        pass


@dataclass
class WallLayer:
    """A single homogenous material layer in a construction assembly."""
    material: str
    thickness: float               # meters
    name: str = ""

    @property
    def properties(self) -> dict[str, float]:
        return MATERIALS.get(self.material, {"k": 0.5, "rho": 1000.0, "cp": 1000.0})

    @property
    def thermal_resistance(self) -> float:
        k = self.properties["k"]
        return self.thickness / max(1e-5, k)

    @property
    def thermal_capacity_per_m2(self) -> float:
        props = self.properties
        return props["rho"] * props["cp"] * self.thickness


class MultiLayerWall(BaseComponent):
    """A building envelope wall with multi-layer construction.

    Models interior surface node, interior mass capacitance node, and exterior surface node.
    """
    def __init__(
        self,
        wall_id: str,
        area: float,
        layers: list[WallLayer],
        inside_node_id: str,
        outside_node_id: str,
        r_si: float = 0.13,        # ISO 6946 internal surface resistance (m^2*K/W)
        r_se: float = 0.04,        # ISO 6946 external surface resistance (m^2*K/W)
        azimuth: float = 180.0,    # South = 180, North = 0
        tilt: float = 90.0,        # Vertical wall = 90, Flat roof = 0
    ) -> None:
        self.wall_id = wall_id
        self.area = area
        self.layers = layers
        self.inside_node_id = inside_node_id
        self.outside_node_id = outside_node_id
        self.r_si = r_si
        self.r_se = r_se
        self.azimuth = azimuth
        self.tilt = tilt

    @property
    def r_layers(self) -> float:
        return sum(layer.thermal_resistance for layer in self.layers)

    @property
    def r_total(self) -> float:
        return self.r_si + self.r_layers + self.r_se

    @property
    def u_value(self) -> float:
        return 1.0 / max(1e-4, self.r_total)

    @property
    def total_capacity(self) -> float:
        return sum(layer.thermal_capacity_per_m2 for layer in self.layers) * self.area

    def build_network(self, network: SimulationNetwork) -> None:
        # Create interior surface node and mass node
        surf_in_id = f"{self.wall_id}.surf_in"
        mass_id = f"{self.wall_id}.mass"
        surf_out_id = f"{self.wall_id}.surf_out"

        # Convective link from room air to interior surface
        g_si = (1.0 / self.r_si) * self.area
        # Internal conduction through outer and inner half of construction
        g_half = (2.0 / max(1e-4, self.r_layers)) * self.area
        # Convective link from exterior surface to ambient
        g_se = (1.0 / self.r_se) * self.area

        network.add_node(NetworkNode(
            id=surf_in_id, domain=Domain.THERMAL, name=f"{self.wall_id} Inside Surface",
            capacity=0.05 * self.total_capacity, initial_value=20.0
        ))
        network.add_node(NetworkNode(
            id=mass_id, domain=Domain.THERMAL, name=f"{self.wall_id} Core Mass",
            capacity=0.90 * self.total_capacity, initial_value=18.0
        ))
        network.add_node(NetworkNode(
            id=surf_out_id, domain=Domain.THERMAL, name=f"{self.wall_id} Outside Surface",
            capacity=0.05 * self.total_capacity, initial_value=10.0
        ))

        # Links
        network.add_link(NetworkLink(
            id=f"{self.wall_id}.link_in", node_a=self.inside_node_id, node_b=surf_in_id,
            conductance=g_si, kind="convection", area=self.area, equation="ISO 6946 R_si"
        ))
        network.add_link(NetworkLink(
            id=f"{self.wall_id}.link_mass_in", node_a=surf_in_id, node_b=mass_id,
            conductance=g_half, kind="conduction", area=self.area, equation="Fourier Conduction k*A/d"
        ))
        network.add_link(NetworkLink(
            id=f"{self.wall_id}.link_mass_out", node_a=mass_id, node_b=surf_out_id,
            conductance=g_half, kind="conduction", area=self.area, equation="Fourier Conduction k*A/d"
        ))
        network.add_link(NetworkLink(
            id=f"{self.wall_id}.link_out", node_a=surf_out_id, node_b=self.outside_node_id,
            conductance=g_se, kind="convection", area=self.area, equation="ISO 6946 R_se"
        ))


class WindowGlazing(BaseComponent):
    """Window glazing element with heat transfer and solar aperture."""
    def __init__(
        self,
        window_id: str,
        area: float,
        u_value: float = 1.3,      # W/(m^2*K) double/triple glazed
        g_value: float = 0.55,     # SHGC solar factor
        inside_node_id: str = "",
        outside_node_id: str = "",
        azimuth: float = 180.0,
    ) -> None:
        self.window_id = window_id
        self.area = area
        self.u_value = u_value
        self.g_value = g_value
        self.inside_node_id = inside_node_id
        self.outside_node_id = outside_node_id
        self.azimuth = azimuth

    def build_network(self, network: SimulationNetwork) -> None:
        conductance = self.u_value * self.area
        network.add_link(NetworkLink(
            id=f"{self.window_id}.transmission",
            node_a=self.inside_node_id,
            node_b=self.outside_node_id,
            conductance=conductance,
            kind="conduction",
            area=self.area,
            equation="EN 673 Window U-Value"
        ))


class ThermalZone(BaseComponent):
    """A thermal room zone containing air mass and internal heat gains."""
    def __init__(
        self,
        zone_id: str,
        floor_area: float,
        height: float = 2.6,
        internal_gain_w: float = 150.0,   # occupants + appliances (W)
        air_changes_per_hour: float = 0.5, # infiltration + ventilation (ACH)
        outside_node_id: str = "ambient",
    ) -> None:
        self.zone_id = zone_id
        self.floor_area = floor_area
        self.height = height
        self.volume = floor_area * height
        self.internal_gain_w = internal_gain_w
        self.ach = air_changes_per_hour
        self.outside_node_id = outside_node_id

    @property
    def air_node_id(self) -> str:
        return f"{self.zone_id}.air"

    def build_network(self, network: SimulationNetwork) -> None:
        # Air thermal capacity C = rho * cp * V (air: 1.2 kg/m^3 * 1005 J/kg*K * V)
        # Plus lumped furniture capacity (approx 5x air capacity per ISO 13790)
        c_air = 1.2 * 1005.0 * self.volume
        c_internal_mass = 5.0 * c_air

        network.add_node(NetworkNode(
            id=self.air_node_id,
            domain=Domain.THERMAL,
            name=f"Zone {self.zone_id} Air",
            capacity=c_air + c_internal_mass,
            value=20.0,
            initial_value=20.0,
            source=self.internal_gain_w,
            zone=self.zone_id,
        ))

        # Infiltration & ventilation conductance: G_vent = rho * cp * (V * ACH / 3600)
        v_flow = (self.volume * self.ach) / 3600.0
        g_vent = 1.2 * 1005.0 * v_flow

        network.add_link(NetworkLink(
            id=f"{self.zone_id}.ventilation",
            node_a=self.air_node_id,
            node_b=self.outside_node_id,
            conductance=g_vent,
            kind="advection",
            equation="ISO 13790 Ventilation Q = rho*cp*V_dot*(T_i - T_e)"
        ))


class HeatPumpPlant(BaseComponent):
    """Heat pump plant delivering heat to heating circuits or buffer tanks."""
    def __init__(
        self,
        plant_id: str,
        nominal_capacity_kw: float = 8.0,
        cop_nominal: float = 3.8,
        source_node_id: str = "ambient",
        supply_node_id: str = "",
        carnot_efficiency: float = 0.48,
    ) -> None:
        self.plant_id = plant_id
        self.capacity_w = nominal_capacity_kw * 1000.0
        self.cop_nominal = cop_nominal
        self.source_node_id = source_node_id
        self.supply_node_id = supply_node_id
        self.carnot_efficiency = carnot_efficiency

    def calculate_cop(self, t_source_c: float, t_supply_c: float) -> float:
        """Thermodynamic COP based on Carnot cycle with empirical efficiency factor."""
        t_src_k = t_source_c + 273.15
        t_sup_k = t_supply_c + 273.15
        delta_t = max(5.0, t_sup_k - t_src_k)
        cop_carnot = t_sup_k / delta_t
        cop = self.carnot_efficiency * cop_carnot
        return float(np.clip(cop, 1.5, 6.5))

    def build_network(self, network: SimulationNetwork) -> None:
        # Electrical consumption node
        network.add_node(NetworkNode(
            id=f"{self.plant_id}.electrical",
            domain=Domain.ELECTRICAL,
            name=f"{self.plant_id} Electrical Power Demand",
            capacity=0.0,
            value=0.0,
            initial_value=0.0
        ))


class StratifiedBufferTank(BaseComponent):
    """Stratified thermal storage tank discretized into N vertical layers."""
    def __init__(
        self,
        tank_id: str,
        volume_litres: float = 300.0,
        num_layers: int = 4,
        heat_loss_coef_u: float = 0.8, # W/(m^2*K) insulated tank
        ambient_node_id: str = "ambient",
    ) -> None:
        self.tank_id = tank_id
        self.volume_m3 = volume_litres / 1000.0
        self.num_layers = num_layers
        self.u_loss = heat_loss_coef_u
        self.ambient_node_id = ambient_node_id

    def build_network(self, network: SimulationNetwork) -> None:
        v_layer = self.volume_m3 / self.num_layers
        m_layer = v_layer * 990.0  # kg water
        c_layer = m_layer * 4182.0 # J/K heat capacity per layer

        # Cylinder aspect ratio height ~ 3 * radius
        r = (self.volume_m3 / (3.0 * math.pi)) ** (1.0 / 3.0)
        h = 3.0 * r
        area_per_layer = (2.0 * math.pi * r * (h / self.num_layers))

        layer_node_ids = []
        for i in range(self.num_layers):
            nid = f"{self.tank_id}.layer_{i}"
            layer_node_ids.append(nid)
            # Top layer warmer (e.g. 48°C), bottom layer cooler (e.g. 40°C)
            t_init = 48.0 - (8.0 * i / max(1, self.num_layers - 1))
            network.add_node(NetworkNode(
                id=nid, domain=Domain.THERMAL, name=f"{self.tank_id} Layer {i} (top={i==0})",
                capacity=c_layer, value=t_init, initial_value=t_init
            ))

            # Heat loss to ambient / plant room
            network.add_link(NetworkLink(
                id=f"{nid}.loss", node_a=nid, node_b=self.ambient_node_id,
                conductance=self.u_loss * area_per_layer, kind="convection",
                equation="Cylinder Tank Heat Loss U*A"
            ))

        # Vertical conduction between adjacent water layers
        k_water = 0.63
        a_cross = math.pi * r * r
        dz = h / self.num_layers
        g_internal = (k_water * a_cross) / dz

        for i in range(self.num_layers - 1):
            network.add_link(NetworkLink(
                id=f"{self.tank_id}.conduction_{i}_{i+1}",
                node_a=layer_node_ids[i],
                node_b=layer_node_ids[i+1],
                conductance=g_internal,
                kind="conduction",
                equation="Stratified Water Conduction k*A/dz"
            ))


# -------------------------------------------------------------------------------------------------
# Multi-Domain Extensions: Electrical (PV + Battery) and Structural (Thermal Stress)
# -------------------------------------------------------------------------------------------------

class PhotovoltaicArray(BaseComponent):
    """Photovoltaic generation element feeding electrical nodes (Electrical Domain)."""
    def __init__(
        self,
        pv_id: str,
        area_m2: float = 24.0,
        efficiency_stc: float = 0.20,
        temp_coefficient: float = -0.004, # -0.4% / K above 25°C
        electrical_bus_node_id: str = "grid.bus",
    ) -> None:
        self.pv_id = pv_id
        self.area_m2 = area_m2
        self.efficiency_stc = efficiency_stc
        self.temp_coef = temp_coefficient
        self.bus_id = electrical_bus_node_id

    def build_network(self, network: SimulationNetwork) -> None:
        # PV generation node
        network.add_node(NetworkNode(
            id=f"{self.pv_id}.generation",
            domain=Domain.ELECTRICAL,
            name=f"{self.pv_id} Solar PV Power Output",
            capacity=0.0,
            value=0.0,
            initial_value=0.0
        ))

    def calculate_power(self, solar_irradiance_w_m2: float, ambient_c: float) -> float:
        """Calculate electrical output from incident solar radiation."""
        t_cell = ambient_c + 0.03 * solar_irradiance_w_m2
        eff = self.efficiency_stc * (1.0 + self.temp_coef * (t_cell - 25.0))
        power_w = self.area_m2 * solar_irradiance_w_m2 * max(0.0, eff)
        return float(power_w)


class StructuralThermalStressElement(BaseComponent):
    """Structural constraint calculating thermal expansion stress from temperature (Structural Domain)."""
    def __init__(
        self,
        struct_id: str,
        thermal_node_id: str,
        youngs_modulus_gpa: float = 30.0,      # GPa (concrete: 30 GPa, steel: 210 GPa)
        thermal_expansion_alpha: float = 1.0e-5,# 1/K (concrete: 1.0e-5, steel: 1.2e-5)
        reference_temperature_c: float = 15.0, # zero-stress manufacturing/assembly temperature
        cross_section_area_m2: float = 0.05,
    ) -> None:
        self.struct_id = struct_id
        self.thermal_node_id = thermal_node_id
        self.e_pa = youngs_modulus_gpa * 1e9
        self.alpha = thermal_expansion_alpha
        self.t_ref = reference_temperature_c
        self.area = cross_section_area_m2

    def build_network(self, network: SimulationNetwork) -> None:
        network.add_node(NetworkNode(
            id=f"{self.struct_id}.stress_mpa",
            domain=Domain.STRUCTURAL,
            name=f"{self.struct_id} Thermal Restraint Stress",
            capacity=0.0,
            value=0.0,
            initial_value=0.0
        ))

    def evaluate_stress(self, current_temp_c: float) -> float:
        """Calculate axial thermal stress in MPa under full restraint."""
        delta_t = current_temp_c - self.t_ref
        strain_th = self.alpha * delta_t
        stress_pa = self.e_pa * strain_th
        return float(stress_pa / 1e6) # MPa (positive = compression, negative = tension)
