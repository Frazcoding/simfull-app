"""Unit tests for building network synthesis, weather drivers, and integrated simulation engine."""
import numpy as np
import pytest

from simfull_app.core.builder import BuildingNetworkBuilder
from simfull_app.core.geometry import BuildingGeometry, Point2D, PolygonZone, WallSegment
from simfull_app.core.simulation import FullSimulationEngine
from simfull_app.core.weather import WeatherDriver
from simfull_app.service import default_two_room_building


def test_weather_driver_presets_and_solar():
    for preset in ["winter_design_day", "summer_design_day", "london_shoulder", "frankfurt_mild"]:
        wd = WeatherDriver(preset=preset)
        cond_noon = wd.evaluate(12.0 * 3600.0)
        assert -20.0 < cond_noon.ambient_temp_c < 45.0
        assert cond_noon.solar_ghi_w_m2 >= 0.0

        # South-facing vertical wall (azimuth 180, tilt 90) at noon
        i_south = wd.calculate_incident_solar(cond_noon, surface_azimuth_deg=180.0, surface_tilt_deg=90.0)
        assert i_south >= 0.0

        # North-facing vertical wall (azimuth 0, tilt 90)
        i_north = wd.calculate_incident_solar(cond_noon, surface_azimuth_deg=0.0, surface_tilt_deg=90.0)
        assert i_north >= 0.0
        # In northern hemisphere winter/shoulder, south-facing gets more solar radiation than north-facing
        if cond_noon.solar_ghi_w_m2 > 100.0:
            assert i_south >= i_north


def test_building_network_builder_synthesis():
    geom = default_two_room_building()
    builder = BuildingNetworkBuilder(
        geometry=geom,
        construction_preset="brick_cavity",
        hp_nominal_kw=8.0,
        pv_area_m2=20.0,
    )
    network, meta = builder.build()

    # Verify ambient and ground boundary nodes exist
    assert "ambient" in network.nodes
    assert network.nodes["ambient"].fixed is True
    assert "ground" in network.nodes
    assert network.nodes["ground"].fixed is True

    # Verify both zone air nodes exist
    assert "living_room.air" in network.nodes
    assert "bedroom.air" in network.nodes

    # Verify stratified tank layers exist
    for i in range(4):
        assert f"plant.tank.layer_{i}" in network.nodes

    # Verify multi-domain nodes exist: electrical (PV) and structural (thermal stress)
    assert "solar.pv.generation" in network.nodes
    assert network.nodes["solar.pv.generation"].domain.value == "electrical"
    assert "struct.corner.stress_mpa" in network.nodes
    assert network.nodes["struct.corner.stress_mpa"].domain.value == "structural"

    # Verify meta components
    assert "hp_plant" in meta
    assert "pv_comp" in meta
    assert "struct_comp" in meta


def test_full_simulation_engine_run():
    geom = default_two_room_building()
    engine = FullSimulationEngine(
        geometry=geom,
        weather_preset="winter_design_day",
        construction="brick_cavity",
        hp_kw=8.0,
        pv_m2=20.0,
        run_detailed_fields=True,
    )

    # Run for 1 day with 10-minute time steps (dt = 600s)
    output = engine.run(duration_days=1.0, dt_seconds=600.0)

    # 1. Check network simulation results
    net_res = output.network_result
    assert len(net_res.hours) == 145  # 24 * 6 + 1
    assert "living_room.air" in net_res.temperatures
    assert "bedroom.air" in net_res.temperatures

    # Living room should stay reasonably warm (e.g. >= 15°C) with HP heating
    assert np.all(net_res.temperatures["living_room.air"] > 15.0)

    # Energy balance residual must be strictly conserved
    assert net_res.balance_error_max < 1e-4

    # 2. Check metrics
    metrics = output.metrics
    assert "seasonal_cop" in metrics
    assert metrics["seasonal_cop"] > 1.5  # Realistic heat pump COP
    assert "space_heating_demand_kwh" in metrics
    assert metrics["space_heating_demand_kwh"] > 0.0
    assert "solar_pv_generation_kwh" in metrics
    assert metrics["solar_pv_generation_kwh"] >= 0.0
    assert "max_structural_stress_mpa" in metrics

    # 3. Check Level 2 detailed spatial fields
    assert output.wall_1d_result is not None
    assert output.wall_1d_result.u_value > 0.0
    assert output.thermal_bridge_result is not None
    assert output.thermal_bridge_result.f_rsi > 0.0

    # 4. Check JSON export
    d = output.to_dict()
    assert "network" in d
    assert "wall_1d" in d
    assert "thermal_bridge" in d
    assert "metrics" in d
    assert "geometry" in d
