"""API integration tests for simfull FastAPI application."""
import base64
import pytest
from fastapi.testclient import TestClient

from simfull_app.server import SimulationApp


@pytest.fixture(scope="module")
def client():
    sim_app = SimulationApp()
    return TestClient(sim_app.app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "capabilities" in data


def test_api_weather_presets(client):
    res = client.get("/api/weather/presets")
    assert res.status_code == 200
    presets = res.json()
    assert len(presets) >= 4
    preset_ids = [p["id"] for p in presets]
    assert "winter_design_day" in preset_ids
    assert "summer_design_day" in preset_ids


def test_api_materials_and_constructions(client):
    res_m = client.get("/api/materials")
    assert res_m.status_code == 200
    materials = res_m.json()
    assert "brick" in materials
    assert "insulation_pir" in materials

    res_c = client.get("/api/constructions")
    assert res_c.status_code == 200
    constructions = res_c.json()
    assert "brick_cavity" in constructions


def test_api_default_geometry(client):
    res = client.get("/api/geometry/default")
    assert res.status_code == 200
    geom = res.json()
    assert "zones" in geom
    assert "walls" in geom
    assert len(geom["zones"]) == 2


def test_api_simulate_network(client):
    geom_res = client.get("/api/geometry/default")
    geom = geom_res.json()

    payload = {
        "geometry": geom,
        "weather_preset": "winter_design_day",
        "construction": "brick_cavity",
        "heat_pump_nominal_kw": 8.0,
        "pv_area_m2": 20.0,
        "duration_days": 1.0,
        "dt_seconds": 600.0,
        "run_detailed_fields": True,
    }
    res = client.post("/api/simulate/network", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "network" in data
    assert "metrics" in data
    assert "geometry" in data
    assert data["metrics"]["seasonal_cop"] > 1.5
    assert data["metrics"]["balance_error_max_w"] < 1e-4


def test_api_simulate_thermal_bridge(client):
    payload = {
        "junction_type": "wall_corner",
        "t_inside": 21.0,
        "t_outside": -2.0,
        "grid_resolution": 25,
    }
    res = client.post("/api/simulate/thermal-bridge", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "temperature_grid" in data
    assert "t_si_min" in data
    assert "f_rsi" in data
    assert 0.0 < data["f_rsi"] < 1.0
    assert "linear_thermal_transmittance_psi" in data


def test_api_simulate_wall_1d(client):
    payload = {
        "construction": "brick_cavity",
        "t_inside": 21.0,
        "t_outside": 0.0,
        "duration_hours": 12.0,
        "dt_seconds": 300.0,
    }
    res = client.post("/api/simulate/wall-1d", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "times_hr" in data
    assert "x_positions_m" in data
    assert "temperatures" in data
    assert data["u_value"] > 0.0


def test_api_geometry_from_stl(client):
    sample_stl = """solid cube
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0 0 3
    vertex 6 0 3
    vertex 6 5 3
  endloop
endfacet
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0 0 3
    vertex 6 5 3
    vertex 0 5 3
  endloop
endfacet
facet normal 0.0 -1.0 0.0
  outer loop
    vertex 0 0 0
    vertex 6 0 0
    vertex 6 0 3
  endloop
endfacet
endsolid cube
"""
    b64_content = base64.b64encode(sample_stl.encode("utf-8")).decode("ascii")
    payload = {
        "content": b64_content,
        "is_base64": True,
        "zone_name": "Test Cube Zone",
    }
    res = client.post("/api/geometry/from-stl", json=payload)
    assert res.status_code == 200
    geom = res.json()
    assert len(geom["zones"]) >= 1
    assert len(geom["walls"]) >= 1
    assert geom["metadata"]["source"] == "stl_import"

    # Also test direct ASCII text upload
    payload_ascii = {
        "content": sample_stl,
        "is_base64": False,
    }
    res_ascii = client.post("/api/geometry/from-stl", json=payload_ascii)
    assert res_ascii.status_code == 200
    geom_ascii = res_ascii.json()
    assert len(geom_ascii["zones"]) >= 1


def test_api_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "<title>simfull" in res.text
    assert "canvas id=\"floorplan-canvas\"" in res.text
