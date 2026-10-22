"""Unit tests for the Level 1 graph-based simulation network."""
import math
import numpy as np
import pytest

from simfull_app.core.network import (
    Domain,
    NetworkLink,
    NetworkNode,
    SimulationNetwork,
    SimulationResult,
)


def test_network_node_domains():
    t_node = NetworkNode(id="n1", domain=Domain.THERMAL, capacity=1000.0, value=21.0)
    e_node = NetworkNode(id="n2", domain=Domain.ELECTRICAL, capacity=0.0, value=230.0)
    s_node = NetworkNode(id="n3", domain=Domain.STRUCTURAL, capacity=0.0, value=0.0)
    f_node = NetworkNode(id="n4", domain=Domain.FLUID, capacity=500.0, value=3.0)

    assert t_node.domain == Domain.THERMAL
    assert e_node.domain == Domain.ELECTRICAL
    assert s_node.domain == Domain.STRUCTURAL
    assert f_node.domain == Domain.FLUID

    t_node.value = 35.0
    t_node.reset()
    assert t_node.value == 20.0  # default initial_value is 20.0


def test_network_link_dynamic_conductance():
    def dynamic_g(t: float, states: dict[str, float]) -> float:
        # e.g., temperature-dependent conductance
        t_a = states.get("a", 20.0)
        return 10.0 + 0.1 * t_a

    link = NetworkLink(
        id="link_dyn",
        node_a="a",
        node_b="b",
        conductance=10.0,
        conductance_fn=dynamic_g,
    )

    states = {"a": 30.0, "b": 10.0}
    assert link.current_conductance(0.0, states) == pytest.approx(13.0)


def test_network_link_invalid_node_error():
    net = SimulationNetwork()
    net.add_node(NetworkNode(id="a", capacity=100.0))

    with pytest.raises(KeyError, match="unknown node_b"):
        net.add_link(NetworkLink(id="l1", node_a="a", node_b="nonexistent", conductance=5.0))

    with pytest.raises(KeyError, match="unknown node_a"):
        net.add_link(NetworkLink(id="l2", node_a="nonexistent", node_b="a", conductance=5.0))


def test_implicit_stepping_analytical_rc_relaxation():
    """Verify implicit Euler matches exact backward Euler discretization of RC decay."""
    net = SimulationNetwork()

    # Fixed ambient node at 0°C
    net.add_node(NetworkNode(id="ambient", fixed=True, value=0.0, initial_value=0.0))

    # Room air node at 20°C with thermal capacity C = 10,000 J/K
    C = 10000.0
    G = 100.0  # Conductance = 100 W/K -> tau = C/G = 100 seconds
    net.add_node(NetworkNode(id="room", capacity=C, value=20.0, initial_value=20.0))

    net.add_link(NetworkLink(id="loss", node_a="room", node_b="ambient", conductance=G))

    dt = 50.0  # seconds
    # Backward Euler step: T_new = (C/dt * T_old + G * T_amb) / (C/dt + G)
    # T_new = (200 * 20 + 100 * 0) / (200 + 100) = 4000 / 300 = 13.3333... °C
    expected_t1 = ( (C / dt) * 20.0 + G * 0.0 ) / ((C / dt) + G)

    new_vals, flows, res = net.step(0.0, dt)

    assert new_vals["room"] == pytest.approx(expected_t1, rel=1e-5)
    assert res < 1e-6  # Conservation residual

    # Step again
    expected_t2 = ( (C / dt) * expected_t1 + G * 0.0 ) / ((C / dt) + G)
    new_vals2, flows2, res2 = net.step(dt, dt)
    assert new_vals2["room"] == pytest.approx(expected_t2, rel=1e-5)


def test_network_conservation_with_internal_source():
    """Verify energy balance holds when internal heat sources exist."""
    net = SimulationNetwork()
    net.add_node(NetworkNode(id="ambient", fixed=True, value=5.0, initial_value=5.0))
    net.add_node(NetworkNode(id="room", capacity=50000.0, value=15.0, source=2000.0))  # 2 kW heater
    net.add_link(NetworkLink(id="wall", node_a="room", node_b="ambient", conductance=150.0))

    res = net.simulate(
        duration=3600.0,
        dt=300.0,
        weather_driver=lambda t: {"ambient": 5.0},
    )

    assert res.balance_error_max < 1e-5
    # Room must heat up since source 2000 W > heat loss (15-5)*150 = 1500 W initially
    assert res.temperatures["room"][-1] > 15.0

    # Test conversion to dict
    out_dict = res.to_dict()
    assert "temperatures" in out_dict
    assert "room" in out_dict["temperatures"]
    assert len(out_dict["hours"]) == len(res.hours)
