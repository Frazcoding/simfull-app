"""Full simulation workflow orchestrating building network, plant controls, weather, and detailed spatial fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .builder import BuildingNetworkBuilder
from .components import MATERIALS, HeatPumpPlant, PhotovoltaicArray, StructuralThermalStressElement
from .fields import MultiLayerWall1D, ThermalBridge2D, ThermalBridgeResult, Wall1DResult
from .geometry import BuildingGeometry
from .network import SimulationNetwork, SimulationResult
from .weather import WeatherDriver


@dataclass
class FullSimulationOutput:
    """Comprehensive output encompassing multi-zone network, plant time-series, and detailed sub-fields."""
    network_result: SimulationResult
    wall_1d_result: Wall1DResult | None
    thermal_bridge_result: ThermalBridgeResult | None
    metrics: dict[str, Any]
    geometry_dict: dict[str, Any]

    def to_dict(self, sample_step: int = 1) -> dict[str, Any]:
        return {
            "network": self.network_result.to_dict(sample_step=sample_step),
            "wall_1d": self.wall_1d_result.to_dict(max_points=50) if self.wall_1d_result else None,
            "thermal_bridge": self.thermal_bridge_result.to_dict() if self.thermal_bridge_result else None,
            "metrics": self.metrics,
            "geometry": self.geometry_dict,
        }


class FullSimulationEngine:
    """High-level simulation orchestrator combining Level 1 network and Level 2 spatial fields."""

    def __init__(
        self,
        geometry: BuildingGeometry,
        weather_preset: str = "winter_design_day",
        construction: str = "brick_cavity",
        hp_kw: float = 8.0,
        pv_m2: float = 20.0,
        run_detailed_fields: bool = True,
    ) -> None:
        self.geometry = geometry
        self.weather = WeatherDriver(preset=weather_preset)
        self.builder = BuildingNetworkBuilder(
            geometry=geometry,
            construction_preset=construction,
            hp_nominal_kw=hp_kw,
            pv_area_m2=pv_m2,
        )
        self.run_detailed_fields = run_detailed_fields

    def run(
        self,
        duration_days: float = 1.0,
        dt_seconds: float = 300.0, # 5-minute time steps
    ) -> FullSimulationOutput:
        """Execute full integrated simulation."""
        network, meta = self.builder.build()
        hp_plant: HeatPumpPlant | None = meta.get("hp_plant")
        pv_comp: PhotovoltaicArray | None = meta.get("pv_comp")
        struct_comp: StructuralThermalStressElement | None = meta.get("struct_comp")

        duration_sec = duration_days * 86400.0
        num_steps = int(duration_sec / dt_seconds)

        # Tracking arrays for plant & metrics
        cop_history = np.zeros(num_steps + 1)
        hp_power_history = np.zeros(num_steps + 1)
        pv_power_history = np.zeros(num_steps + 1)
        stress_history = np.zeros(num_steps + 1)

        # Weather driver closure
        def weather_callback(t: float) -> dict[str, float]:
            cond = self.weather.evaluate(t)
            bc = {
                "ambient": cond.ambient_temp_c,
                "ground": cond.ground_temp_c,
            }
            # Inject solar gains through windows into zone air nodes
            for w in self.geometry.walls:
                if w.window_area > 0 and w.zone_b is None:
                    i_sol = self.weather.calculate_incident_solar(cond, w.azimuth, 90.0)
                    # Solar heat gain: Q = Area * g_value * I_sol (g_value ~ 0.55)
                    q_solar = w.window_area * 0.55 * i_sol
                    air_node = network.get_node(f"{w.zone_a}.air")
                    # Internal gain + solar gain
                    air_node.source = 120.0 + q_solar
            return bc

        # HVAC controller closure
        def hvac_controller(t: float, states: dict[str, float]) -> None:
            step_idx = min(num_steps, int(t / dt_seconds))
            cond = self.weather.evaluate(t)
            t_amb = states.get("ambient", cond.ambient_temp_c)

            # Zone thermostat control
            tank_top = states.get("plant.tank.layer_0", 45.0)
            total_heating_demand = 0.0

            for z in self.geometry.zones:
                t_zone = states.get(f"{z.id}.air", 20.0)
                emitter_link_id = f"{z.id}.emitter"
                if emitter_link_id in network.links:
                    emitter_link = network.links[emitter_link_id]
                    if t_zone < z.setpoint_temp_c:
                        # Modulate emitter flow conductance to satisfy demand
                        error = z.setpoint_temp_c - t_zone
                        emitter_link.conductance = float(np.clip(120.0 * (1.0 + error), 40.0, 300.0))
                        total_heating_demand += max(0.0, emitter_link.conductance * (tank_top - t_zone))
                    else:
                        # Thermostatic valve closed
                        emitter_link.conductance = 5.0

            # Heat pump operation: maintains buffer tank top layer >= 45°C
            hp_elec_node = network.nodes.get("plant.hp.electrical")
            tank_layer_top = network.nodes.get("plant.tank.layer_0")

            if hp_plant is not None:
                if tank_top < 45.0:
                    cop = hp_plant.calculate_cop(t_amb, 48.0)
                    cop_history[step_idx] = cop
                    hp_heat_w = min(hp_plant.capacity_w, max(1500.0, total_heating_demand * 1.5))
                    el_w = hp_heat_w / cop
                    hp_power_history[step_idx] = el_w
                    if hp_elec_node:
                        hp_elec_node.value = el_w
                    # Inject heat into tank supply top layer
                    if tank_layer_top:
                        tank_layer_top.source = hp_heat_w
                else:
                    cop_history[step_idx] = 4.0
                    hp_power_history[step_idx] = 0.0
                    if hp_elec_node:
                        hp_elec_node.value = 0.0
                    if tank_layer_top:
                        tank_layer_top.source = 0.0

            # Solar PV generation
            if pv_comp is not None:
                pv_gen = pv_comp.calculate_power(cond.solar_direct_w_m2 + cond.solar_diffuse_w_m2, t_amb)
                pv_power_history[step_idx] = pv_gen
                pv_node = network.nodes.get("solar.pv.generation")
                if pv_node:
                    pv_node.value = pv_gen

            # Structural thermal stress
            if struct_comp is not None:
                surf_node = states.get(struct_comp.thermal_node_id, t_amb)
                stress_mpa = struct_comp.evaluate_stress(surf_node)
                stress_history[step_idx] = stress_mpa
                stress_node = network.nodes.get("struct.corner.stress_mpa")
                if stress_node:
                    stress_node.value = stress_mpa

        # Run Level 1 Network
        sim_res = network.simulate(
            duration=duration_sec,
            dt=dt_seconds,
            weather_driver=weather_callback,
            hvac_controller=hvac_controller,
        )

        # Run Level 2 Local Spatial Fields
        wall_1d_res = None
        tb_res = None

        if self.run_detailed_fields:
            # 1D finite-difference multi-layer wall
            wall_solver = MultiLayerWall1D(layers=self.builder.layers, nodes_per_layer=5)
            # Use ambient temperature curve from simulation
            amb_curve = sim_res.temperatures.get("ambient", np.full(num_steps + 1, 5.0))
            wall_1d_res = wall_solver.simulate(
                t_inside=21.0,
                t_outside=amb_curve,
                duration_hours=duration_days * 24.0,
                dt_seconds=dt_seconds,
            )

            # 2D EN ISO 10211 thermal bridge solver
            tb_solver = ThermalBridge2D(nx=30, ny=30)
            t_amb_min = float(np.min(amb_curve))
            tb_res = tb_solver.solve_junction(
                junction_type="wall_corner",
                t_inside=21.0,
                t_outside=t_amb_min,
                wall_u_value=0.22,
            )

        # Compute summary metrics
        times_arr = sim_res.times
        total_hp_elec_kwh = float(np.trapezoid(hp_power_history, times_arr) / 3.6e6)
        total_pv_gen_kwh = float(np.trapezoid(pv_power_history, times_arr) / 3.6e6)
        total_heating_kwh = sum(
            val for k, val in sim_res.total_energy_kwh.items() if k.endswith(".emitter")
        )
        scop = (total_heating_kwh / max(0.01, total_hp_elec_kwh)) if total_hp_elec_kwh > 0 else 3.8

        metrics = {
            "duration_days": duration_days,
            "total_floor_area_m2": round(self.geometry.total_floor_area(), 1),
            "space_heating_demand_kwh": round(total_heating_kwh, 2),
            "heat_pump_electricity_kwh": round(total_hp_elec_kwh, 2),
            "seasonal_cop": round(scop, 2),
            "solar_pv_generation_kwh": round(total_pv_gen_kwh, 2),
            "net_grid_electricity_kwh": round(max(0.0, total_hp_elec_kwh - total_pv_gen_kwh), 2),
            "max_structural_stress_mpa": round(float(np.max(np.abs(stress_history))), 2),
            "thermal_bridge_psi": round(float(tb_res.linear_thermal_transmittance_psi) if tb_res else 0.05, 3),
            "condensation_risk": tb_res.condensation_risk if tb_res else False,
            "balance_error_max_w": round(float(sim_res.balance_error_max), 6),
        }

        # Store arrays into summary for dashboard graphing
        sim_res.summary = {
            "cop_history": [round(float(c), 2) for c in cop_history],
            "hp_power_w": [round(float(p), 1) for p in hp_power_history],
            "pv_power_w": [round(float(p), 1) for p in pv_power_history],
            "stress_mpa": [round(float(s), 2) for s in stress_history],
        }

        return FullSimulationOutput(
            network_result=sim_res,
            wall_1d_result=wall_1d_res,
            thermal_bridge_result=tb_res,
            metrics=metrics,
            geometry_dict=self.geometry.to_dict(),
        )
