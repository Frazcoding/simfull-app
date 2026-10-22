"""Unit tests for Level 2 local spatial fields (1D wall conduction & 2D EN ISO 10211 thermal bridge)."""
import numpy as np
import pytest

from simfull_app.core.components import WallLayer
from simfull_app.core.fields import MultiLayerWall1D, ThermalBridge2D


def test_wall_1d_discretization_and_u_value():
    layers = [
        WallLayer(material="brick", thickness=0.10, name="Brick"),
        WallLayer(material="insulation_eps", thickness=0.10, name="EPS"),
        WallLayer(material="plasterboard", thickness=0.015, name="Plaster"),
    ]
    # k_brick = 0.77, k_eps = 0.035, k_plaster = 0.25
    r_layers = 0.10 / 0.77 + 0.10 / 0.035 + 0.015 / 0.25
    expected_u = 1.0 / (0.13 + r_layers + 0.04)

    wall_1d = MultiLayerWall1D(layers, nodes_per_layer=5)

    assert wall_1d.u_value == pytest.approx(expected_u, rel=1e-3)
    assert len(wall_1d.x) == 15
    assert len(wall_1d.layer_names) == 15


def test_wall_1d_steady_state_asymptote():
    """Verify that after sufficient time, 1D heat flux reaches U * (T_in - T_out)."""
    layers = [
        WallLayer(material="insulation_pir", thickness=0.10, name="PIR"),
    ]
    wall_1d = MultiLayerWall1D(layers, nodes_per_layer=8)

    t_in = 20.0
    t_out = 0.0
    expected_flux = wall_1d.u_value * (t_in - t_out)

    # Run for 48 hours to reach steady state (PIR has low thermal mass)
    res = wall_1d.simulate(
        t_inside=t_in,
        t_outside=t_out,
        duration_hours=48.0,
        dt_seconds=300.0,
        t_initial=10.0,
    )

    final_flux_in = res.heat_flux_in_w_m2[-1]
    final_flux_out = res.heat_flux_out_w_m2[-1]

    # At steady state, flux in ~ flux out ~ U * delta T
    assert final_flux_in == pytest.approx(expected_flux, rel=0.05)
    assert final_flux_out == pytest.approx(expected_flux, rel=0.05)

    # Temperatures monotonically decrease from inside to outside
    final_temps = res.temperatures[-1]
    assert np.all(np.diff(final_temps) < 0)

    # Check to_dict serialization
    res_dict = res.to_dict(max_points=20)
    assert "temperatures" in res_dict
    assert "x_positions_m" in res_dict
    assert res_dict["u_value"] > 0


def test_thermal_bridge_2d_corner_junction():
    tb = ThermalBridge2D(nx=25, ny=25, width=1.0, height=1.0)
    res = tb.solve_junction(
        junction_type="wall_corner",
        t_inside=20.0,
        t_outside=0.0,
        r_si=0.13,
        r_se=0.04,
        wall_u_value=0.25,
    )

    # Temperature grid bounds
    assert res.temperature_grid.shape == (25, 25)
    assert np.all(res.temperature_grid >= 0.0 - 1e-4)
    assert np.all(res.temperature_grid <= 20.0 + 1e-4)

    # Minimum internal surface temperature must be between outside and inside
    assert 0.0 < res.t_si_min < 20.0

    # Temperature factor f_Rsi = (T_si_min - T_out) / (T_in - T_out)
    assert 0.0 < res.f_rsi < 1.0

    # Linear thermal transmittance Psi (W/m*K)
    assert res.linear_thermal_transmittance_psi >= 0.0

    # Total heat flow rate per meter L^2D > 0
    assert res.heat_flow_rate_w_m > 0.0


def test_thermal_bridge_junction_types():
    tb = ThermalBridge2D(nx=20, ny=20)

    # Test all 3 standard junction types
    for j_type in ["wall_corner", "wall_floor", "window_reveal"]:
        res = tb.solve_junction(junction_type=j_type, t_inside=21.0, t_outside=-2.0)
        assert res.summary["junction_type"] == j_type
        assert res.f_rsi > 0.0
        assert res.f_rsi < 1.0
        assert res.to_dict()["condensation_risk"] in [True, False]
