"""Unit tests for physical components across thermal, fluid, electrical, and structural domains."""
import math
import pytest

from simfull_app.core.network import Domain, SimulationNetwork
from simfull_app.core.components import (
    MATERIALS,
    HeatPumpPlant,
    MultiLayerWall,
    PhotovoltaicArray,
    StratifiedBufferTank,
    StructuralThermalStressElement,
    ThermalZone,
    WallLayer,
    WindowGlazing,
)


def test_wall_layer_properties():
    layer = WallLayer(material="brick", thickness=0.10)
    # k = 0.77, rho = 1700, cp = 840
    assert layer.thermal_resistance == pytest.approx(0.10 / 0.77, rel=1e-4)
    expected_cap = 1700.0 * 840.0 * 0.10
    assert layer.thermal_capacity_per_m2 == pytest.approx(expected_cap, rel=1e-4)


def test_multi_layer_wall_u_value_and_network_nodes():
    layers = [
        WallLayer(material="brick", thickness=0.102),
        WallLayer(material="insulation_pir", thickness=0.080),  # k = 0.022
        WallLayer(material="concrete", thickness=0.100),       # k = 1.75
        WallLayer(material="plasterboard", thickness=0.0125),   # k = 0.25
    ]
    r_layers = (
        0.102 / 0.77 +
        0.080 / 0.022 +
        0.100 / 1.75 +
        0.0125 / 0.25
    )
    r_si = 0.13
    r_se = 0.04
    r_total = r_si + r_layers + r_se
    expected_u = 1.0 / r_total

    wall = MultiLayerWall(
        wall_id="w_ext_south",
        area=15.0,
        layers=layers,
        inside_node_id="room.air",
        outside_node_id="ambient",
        r_si=r_si,
        r_se=r_se,
        azimuth=180.0,
    )

    assert wall.r_total == pytest.approx(r_total, rel=1e-3)
    assert wall.u_value == pytest.approx(expected_u, rel=1e-3)
    assert wall.u_value < 0.30  # High performance insulated wall

    net = SimulationNetwork()
    net.add_node(wall_room_node := net.add_node(
        type("DummyNode", (), {"id": "room.air", "fixed": False, "value": 20.0, "initial_value": 20.0, "capacity": 1000.0, "source": 0.0, "domain": Domain.THERMAL, "zone": None, "metadata": {}, "reset": lambda: None})()
    ))
    net.add_node(type("DummyNode", (), {"id": "ambient", "fixed": True, "value": 0.0, "initial_value": 0.0, "capacity": 0.0, "source": 0.0, "domain": Domain.THERMAL, "zone": None, "metadata": {}, "reset": lambda: None})())

    wall.build_network(net)

    assert "w_ext_south.surf_in" in net.nodes
    assert "w_ext_south.mass" in net.nodes
    assert "w_ext_south.surf_out" in net.nodes
    assert "w_ext_south.link_in" in net.links
    assert "w_ext_south.link_mass_in" in net.links
    assert "w_ext_south.link_mass_out" in net.links
    assert "w_ext_south.link_out" in net.links


def test_window_glazing():
    window = WindowGlazing(
        window_id="win_1",
        area=3.0,
        u_value=1.2,
        g_value=0.60,
        inside_node_id="room.air",
        outside_node_id="ambient",
        azimuth=180.0,
    )
    net = SimulationNetwork()
    from simfull_app.core.network import NetworkNode
    net.add_node(NetworkNode(id="room.air"))
    net.add_node(NetworkNode(id="ambient", fixed=True))

    window.build_network(net)
    link = net.links["win_1.transmission"]
    assert link.conductance == pytest.approx(3.0 * 1.2)


def test_thermal_zone_capacitance_and_ventilation():
    zone = ThermalZone(
        zone_id="living_room",
        floor_area=30.0,
        height=2.5,
        internal_gain_w=200.0,
        air_changes_per_hour=0.6,
        outside_node_id="ambient",
    )
    net = SimulationNetwork()
    from simfull_app.core.network import NetworkNode
    net.add_node(NetworkNode(id="ambient", fixed=True))

    zone.build_network(net)
    air_node = net.get_node("living_room.air")
    # Volume = 75 m^3
    c_air = 1.2 * 1005.0 * 75.0
    c_furn = 5.0 * c_air
    assert air_node.capacity == pytest.approx(c_air + c_furn)
    assert air_node.source == 200.0

    vent_link = net.links["living_room.ventilation"]
    expected_v_dot = (75.0 * 0.6) / 3600.0
    expected_g_vent = 1.2 * 1005.0 * expected_v_dot
    assert vent_link.conductance == pytest.approx(expected_g_vent)


def test_heat_pump_cop_calculation():
    hp = HeatPumpPlant(
        plant_id="hp_main",
        nominal_capacity_kw=7.0,
        cop_nominal=4.0,
        carnot_efficiency=0.50,
    )

    # Ambient 7°C, Supply 35°C (standard EN 14511 floor heating test point)
    # T_sup_k = 35 + 273.15 = 308.15 K, Delta T = 28 K
    # COP_carnot = 308.15 / 28 = 11.005 -> COP = 0.50 * 11.005 = 5.50
    cop_floor = hp.calculate_cop(t_source_c=7.0, t_supply_c=35.0)
    assert 5.0 < cop_floor < 6.0

    # Ambient -5°C, Supply 55°C (cold winter high temp radiator test point)
    # Delta T = 60 K -> Carnot = 328.15 / 60 = 5.469 -> COP = 0.50 * 5.469 = 2.73
    cop_cold = hp.calculate_cop(t_source_c=-5.0, t_supply_c=55.0)
    assert 2.5 < cop_cold < 3.0

    # Bounded between 1.5 and 6.5
    assert hp.calculate_cop(-30.0, 75.0) >= 1.5
    assert hp.calculate_cop(25.0, 26.0) <= 6.5


def test_stratified_buffer_tank():
    tank = StratifiedBufferTank(
        tank_id="buffer",
        volume_litres=400.0,
        num_layers=4,
        heat_loss_coef_u=0.7,
        ambient_node_id="plant_room",
    )
    net = SimulationNetwork()
    from simfull_app.core.network import NetworkNode
    net.add_node(NetworkNode(id="plant_room", fixed=True, value=18.0))

    tank.build_network(net)

    # Check 4 layers created
    for i in range(4):
        assert f"buffer.layer_{i}" in net.nodes
        assert f"buffer.layer_{i}.loss" in net.links

    # Stratification: Top layer warmer than bottom layer initially
    t_top = net.nodes["buffer.layer_0"].value
    t_bottom = net.nodes["buffer.layer_3"].value
    assert t_top > t_bottom

    # Vertical conduction links between layer 0-1, 1-2, 2-3
    assert "buffer.conduction_0_1" in net.links
    assert "buffer.conduction_1_2" in net.links
    assert "buffer.conduction_2_3" in net.links


def test_photovoltaic_multi_domain_extension():
    pv = PhotovoltaicArray(
        pv_id="pv_roof",
        area_m2=20.0,
        efficiency_stc=0.21,
        temp_coefficient=-0.004,
    )
    net = SimulationNetwork()
    pv.build_network(net)
    assert "pv_roof.generation" in net.nodes
    assert net.nodes["pv_roof.generation"].domain == Domain.ELECTRICAL

    # Standard Test Conditions (STC): 1000 W/m^2, cell temp 25°C
    # In calculate_power: t_cell = ambient_c + 0.03 * G
    # For ambient 25 - 0.03*1000 = -5°C, t_cell = 25°C -> eff = 0.21
    p_stc = pv.calculate_power(solar_irradiance_w_m2=1000.0, ambient_c=25.0 - 30.0)
    assert p_stc == pytest.approx(20.0 * 1000.0 * 0.21)

    # On a hot summer afternoon: ambient 30°C, 800 W/m2 -> cell = 30 + 24 = 54°C
    # delta T = 29 K -> eff drop ~ 29 * -0.004 = -11.6%
    p_hot = pv.calculate_power(solar_irradiance_w_m2=800.0, ambient_c=30.0)
    assert p_hot > 0.0
    assert p_hot < (20.0 * 800.0 * 0.21)  # Efficiency is reduced due to heat


def test_structural_thermal_stress_multi_domain_extension():
    struct = StructuralThermalStressElement(
        struct_id="beam_slab",
        thermal_node_id="slab.mass",
        youngs_modulus_gpa=30.0,         # Concrete E = 30 GPa
        thermal_expansion_alpha=1.0e-5,  # 1e-5 / K
        reference_temperature_c=15.0,    # Zero stress at 15°C
    )
    net = SimulationNetwork()
    struct.build_network(net)
    assert "beam_slab.stress_mpa" in net.nodes
    assert net.nodes["beam_slab.stress_mpa"].domain == Domain.STRUCTURAL

    # At reference temperature, stress is zero
    assert struct.evaluate_stress(15.0) == pytest.approx(0.0)

    # At 35°C (+20 K temperature rise under full axial restraint):
    # sigma = E * alpha * Delta T = 30e9 * 1e-5 * 20 = 6,000,000 Pa = 6.0 MPa compression
    stress_warm = struct.evaluate_stress(35.0)
    assert stress_warm == pytest.approx(6.0, rel=1e-4)

    # At -5°C (-20 K temperature drop under full axial restraint):
    # sigma = -6.0 MPa (tension)
    stress_cold = struct.evaluate_stress(-5.0)
    assert stress_cold == pytest.approx(-6.0, rel=1e-4)
