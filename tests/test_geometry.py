"""Unit tests for 2D floorplan geometry, 3D STL mesh parsing, and automated thermal building conversion."""
import struct
import pytest

from simfull_app.core.geometry import (
    BuildingGeometry,
    Point2D,
    PolygonZone,
    STLFacet,
    STLParser,
    WallSegment,
)


def test_point_and_wall_segment():
    p1 = Point2D(0.0, 0.0)
    p2 = Point2D(4.0, 3.0)

    wall = WallSegment(
        id="w1",
        p1=p1,
        p2=p2,
        zone_a="z1",
        height=2.5,
        window_area=2.0,
    )

    # Length of 3-4-5 triangle = 5.0
    assert wall.length == pytest.approx(5.0)
    assert wall.gross_area == pytest.approx(12.5)
    assert wall.net_wall_area == pytest.approx(10.5)

    # Outward normal for (0,0)->(4,3) is (3, -4)
    # atan2(3, -4) in degrees: approx 143.1°
    assert 140.0 < wall.azimuth < 145.0

    d = wall.to_dict()
    assert d["length"] == 5.0
    assert d["area"] == 12.5


def test_polygon_zone_shoelace_area():
    # 5m x 4m rectangle = 20 m^2
    vertices = [
        Point2D(0.0, 0.0),
        Point2D(5.0, 0.0),
        Point2D(5.0, 4.0),
        Point2D(0.0, 4.0),
    ]
    zone = PolygonZone(id="living", name="Living Room", vertices=vertices, height=2.8)

    assert zone.floor_area == pytest.approx(20.0)
    assert zone.volume == pytest.approx(56.0)

    # Triangle: (0,0), (6,0), (0,8) -> Area = 0.5 * 6 * 8 = 24 m^2
    tri_vertices = [Point2D(0.0, 0.0), Point2D(6.0, 0.0), Point2D(0.0, 8.0)]
    tri_zone = PolygonZone(id="tri", name="Triangle Zone", vertices=tri_vertices)
    assert tri_zone.floor_area == pytest.approx(24.0)


def test_building_geometry_serialization():
    z1 = PolygonZone(
        id="z1",
        name="Zone 1",
        vertices=[Point2D(0, 0), Point2D(4, 0), Point2D(4, 3), Point2D(0, 3)],
    )
    w1 = WallSegment(id="w1", p1=Point2D(0, 0), p2=Point2D(4, 0), zone_a="z1")
    w2 = WallSegment(id="w2", p1=Point2D(4, 0), p2=Point2D(4, 3), zone_a="z1")

    bg = BuildingGeometry(zones=[z1], walls=[w1, w2])
    bg_dict = bg.to_dict()

    assert len(bg_dict["zones"]) == 1
    assert len(bg_dict["walls"]) == 2
    assert bg_dict["zones"][0]["floor_area"] == pytest.approx(12.0)

    # Reconstruct from dict
    bg_restored = BuildingGeometry.from_dict(bg_dict)
    assert len(bg_restored.zones) == 1
    assert len(bg_restored.walls) == 2
    assert bg_restored.zones[0].floor_area == pytest.approx(12.0)


def test_stl_facet_area():
    facet = STLFacet(
        normal=(0.0, 0.0, 1.0),
        v1=(0.0, 0.0, 0.0),
        v2=(2.0, 0.0, 0.0),
        v3=(0.0, 3.0, 0.0),
    )
    # Right triangle with base 2 and height 3 -> area = 3.0
    assert facet.area == pytest.approx(3.0)


def test_stl_ascii_parser():
    ascii_stl = """solid simple_box
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 2.5
    vertex 5.0 0.0 2.5
    vertex 5.0 4.0 2.5
  endloop
endfacet
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 2.5
    vertex 5.0 4.0 2.5
    vertex 0.0 4.0 2.5
  endloop
endfacet
facet normal 0.0 -1.0 0.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 5.0 0.0 0.0
    vertex 5.0 0.0 2.5
  endloop
endfacet
endsolid simple_box
"""
    facets = STLParser.parse(ascii_stl)
    assert len(facets) == 3

    # Convert to building geometry
    bg = STLParser.to_building_geometry(facets)
    assert len(bg.zones) >= 1
    assert bg.metadata.get("source") == "stl_import"


def test_stl_binary_parser():
    # Build valid 84-byte header + 1 triangle binary STL
    header = b"Binary STL test file header" + b" " * (80 - len("Binary STL test file header"))
    num_triangles = 1
    # 50 bytes per triangle: normal (3f), v1 (3f), v2 (3f), v3 (3f), attribute (H)
    tri_data = struct.pack(
        "<3f 3f 3f 3f H",
        0.0, 0.0, 1.0,  # normal
        0.0, 0.0, 0.0,  # v1
        1.0, 0.0, 0.0,  # v2
        0.0, 1.0, 0.0,  # v3
        0,              # attribute
    )
    raw_binary = header + struct.pack("<I", num_triangles) + tri_data

    assert STLParser.is_binary(raw_binary) is True

    facets = STLParser.parse(raw_binary)
    assert len(facets) == 1
    assert facets[0].normal == (0.0, 0.0, 1.0)
    assert facets[0].area == pytest.approx(0.5)
