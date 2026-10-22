"""Geometry engine, 2D floorplan canvas format, and 3D STL mesh parser.

Supports:
1. 2D floorplan polygon representation with multi-zone definitions.
2. Binary and ASCII 3D STL mesh importer with facet normal classification.
3. Automated conversion from geometry into thermal network parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import struct
from typing import Any
import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Point2D:
        return cls(x=float(d["x"]), y=float(d["y"]))


@dataclass
class WallSegment:
    id: str
    p1: Point2D
    p2: Point2D
    zone_a: str
    zone_b: str | None = None      # None = exterior wall, otherwise adjacent zone
    height: float = 2.6
    u_value: float = 0.22          # W/(m^2*K) insulated exterior wall
    window_area: float = 0.0       # m^2 glazing aperture
    material: str = "brick_cavity"

    @property
    def length(self) -> float:
        return math.hypot(self.p2.x - self.p1.x, self.p2.y - self.p1.y)

    @property
    def gross_area(self) -> float:
        return self.length * self.height

    @property
    def net_wall_area(self) -> float:
        return max(0.1, self.gross_area - self.window_area)

    @property
    def azimuth(self) -> float:
        """Compass azimuth in degrees: North=0, East=90, South=180, West=270."""
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        # Outward normal is perpendicular to line (p1 -> p2)
        nx = dy
        ny = -dx
        angle_rad = math.atan2(nx, ny)
        angle_deg = math.degrees(angle_rad) % 360.0
        return round(angle_deg, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict(),
            "zone_a": self.zone_a,
            "zone_b": self.zone_b,
            "height": self.height,
            "length": round(self.length, 3),
            "area": round(self.gross_area, 2),
            "azimuth": self.azimuth,
            "u_value": self.u_value,
            "window_area": self.window_area,
            "material": self.material,
        }


@dataclass
class PolygonZone:
    id: str
    name: str
    vertices: list[Point2D]
    height: float = 2.6
    setpoint_temp_c: float = 21.0
    internal_gain_w: float = 120.0
    ventilation_ach: float = 0.5

    @property
    def floor_area(self) -> float:
        """Area computed via standard Shoelace formula."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) * 0.5

    @property
    def volume(self) -> float:
        return self.floor_area * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vertices": [v.to_dict() for v in self.vertices],
            "height": self.height,
            "floor_area": round(self.floor_area, 2),
            "volume": round(self.volume, 2),
            "setpoint_temp_c": self.setpoint_temp_c,
            "internal_gain_w": self.internal_gain_w,
            "ventilation_ach": self.ventilation_ach,
        }


@dataclass
class BuildingGeometry:
    """Complete building geometry with zones and wall segments."""
    zones: list[PolygonZone] = field(default_factory=list)
    walls: list[WallSegment] = field(default_factory=list)
    roof_u_value: float = 0.15
    floor_u_value: float = 0.18
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_floor_area(self) -> float:
        return sum(z.floor_area for z in self.zones)

    def total_volume(self) -> float:
        return sum(z.volume for z in self.zones)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zones": [z.to_dict() for z in self.zones],
            "walls": [w.to_dict() for w in self.walls],
            "roof_u_value": self.roof_u_value,
            "floor_u_value": self.floor_u_value,
            "total_floor_area": round(self.total_floor_area(), 2),
            "total_volume": round(self.total_volume(), 2),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildingGeometry:
        zones = [
            PolygonZone(
                id=zd["id"],
                name=zd.get("name", zd["id"]),
                vertices=[Point2D.from_dict(v) for v in zd["vertices"]],
                height=float(zd.get("height", 2.6)),
                setpoint_temp_c=float(zd.get("setpoint_temp_c", 21.0)),
                internal_gain_w=float(zd.get("internal_gain_w", 120.0)),
                ventilation_ach=float(zd.get("ventilation_ach", 0.5)),
            )
            for zd in data.get("zones", [])
        ]
        walls = [
            WallSegment(
                id=wd["id"],
                p1=Point2D.from_dict(wd["p1"]),
                p2=Point2D.from_dict(wd["p2"]),
                zone_a=wd["zone_a"],
                zone_b=wd.get("zone_b"),
                height=float(wd.get("height", 2.6)),
                u_value=float(wd.get("u_value", 0.22)),
                window_area=float(wd.get("window_area", 0.0)),
                material=wd.get("material", "brick_cavity"),
            )
            for wd in data.get("walls", [])
        ]
        return cls(
            zones=zones,
            walls=walls,
            roof_u_value=float(data.get("roof_u_value", 0.15)),
            floor_u_value=float(data.get("floor_u_value", 0.18)),
            metadata=data.get("metadata", {}),
        )


# -------------------------------------------------------------------------------------------------
# 3D STL Parser (ASCII and Binary)
# -------------------------------------------------------------------------------------------------

@dataclass
class STLFacet:
    normal: tuple[float, float, float]
    v1: tuple[float, float, float]
    v2: tuple[float, float, float]
    v3: tuple[float, float, float]

    @property
    def area(self) -> float:
        """Triangle area via cross product."""
        ax, ay, az = self.v2[0] - self.v1[0], self.v2[1] - self.v1[1], self.v2[2] - self.v1[2]
        bx, by, bz = self.v3[0] - self.v1[0], self.v3[1] - self.v1[1], self.v3[2] - self.v1[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        return 0.5 * math.hypot(cx, cy, cz)


class STLParser:
    """Parses 3D STL meshes (ASCII and Binary) and reconstructs building thermal zones."""

    @staticmethod
    def is_binary(raw_bytes: bytes) -> bool:
        if len(raw_bytes) < 84:
            return False
        num_triangles = struct.unpack("<I", raw_bytes[80:84])[0]
        expected_size = 84 + num_triangles * 50
        return len(raw_bytes) == expected_size

    @classmethod
    def parse(cls, content: bytes | str) -> list[STLFacet]:
        """Parse raw STL bytes or string into facet list."""
        if isinstance(content, str):
            raw_bytes = content.encode("utf-8")
        else:
            raw_bytes = content

        if cls.is_binary(raw_bytes):
            return cls._parse_binary(raw_bytes)
        return cls._parse_ascii(raw_bytes.decode("utf-8", errors="ignore"))

    @staticmethod
    def _parse_binary(data: bytes) -> list[STLFacet]:
        num_triangles = struct.unpack("<I", data[80:84])[0]
        facets = []
        offset = 84
        for _ in range(num_triangles):
            record = struct.unpack("<3f 3f 3f 3f H", data[offset:offset+50])
            normal = (record[0], record[1], record[2])
            v1 = (record[3], record[4], record[5])
            v2 = (record[6], record[7], record[8])
            v3 = (record[9], record[10], record[11])
            facets.append(STLFacet(normal=normal, v1=v1, v2=v2, v3=v3))
            offset += 50
        return facets

    @staticmethod
    def _parse_ascii(text: str) -> list[STLFacet]:
        facets = []
        lines = text.strip().splitlines()
        i = 0
        current_normal = (0.0, 0.0, 1.0)
        vertices: list[tuple[float, float, float]] = []

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("facet normal"):
                parts = line_str.split()
                if len(parts) >= 5:
                    current_normal = (float(parts[2]), float(parts[3]), float(parts[4]))
                vertices = []
            elif line_str.startswith("vertex"):
                parts = line_str.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line_str.startswith("endfacet"):
                if len(vertices) == 3:
                    facets.append(STLFacet(normal=current_normal, v1=vertices[0], v2=vertices[1], v3=vertices[2]))
        return facets

    @classmethod
    def to_building_geometry(cls, content: bytes | str | list[STLFacet], zone_name: str = "Imported 3D Building") -> BuildingGeometry:
        """Convert a 3D STL mesh into a synthesized multi-surface thermal building model."""
        if isinstance(content, list):
            facets = content
        else:
            facets = cls.parse(content)
        if not facets:
            raise ValueError("No valid 3D facets found in STL mesh")

        # Compute bounding box
        xs = [v[0] for f in facets for v in (f.v1, f.v2, f.v3)]
        ys = [v[1] for f in facets for v in (f.v1, f.v2, f.v3)]
        zs = [v[2] for f in facets for v in (f.v1, f.v2, f.v3)]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        dx = max(1.0, max_x - min_x)
        dy = max(1.0, max_y - min_y)
        dz = max(2.4, max_z - min_z)

        # Categorize surface areas by normal orientation
        area_roof = 0.0
        area_floor = 0.0
        area_north = 0.0
        area_south = 0.0
        area_east = 0.0
        area_west = 0.0

        for f in facets:
            a = f.area
            nz = f.normal[2]
            if nz > 0.7:
                area_roof += a
            elif nz < -0.7:
                area_floor += a
            else:
                nx, ny = f.normal[0], f.normal[1]
                angle = (math.degrees(math.atan2(nx, ny))) % 360.0
                if 45.0 <= angle < 135.0:
                    area_east += a
                elif 135.0 <= angle < 225.0:
                    area_south += a
                elif 225.0 <= angle < 315.0:
                    area_west += a
                else:
                    area_north += a

        # Fallback to bounding box footprint if degenerate normals
        if area_floor < 0.1:
            area_floor = dx * dy
        if area_roof < 0.1:
            area_roof = dx * dy

        floor_w = math.sqrt(area_floor * (dx / dy))
        floor_l = area_floor / max(0.5, floor_w)

        # Create 2D boundary polygon
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(floor_w, 0.0)
        p3 = Point2D(floor_w, floor_l)
        p4 = Point2D(0.0, floor_l)

        zone = PolygonZone(
            id="stl_zone_1",
            name=zone_name,
            vertices=[p1, p2, p3, p4],
            height=dz,
            setpoint_temp_c=21.0,
            internal_gain_w=150.0,
            ventilation_ach=0.5,
        )

        # Build 4 perimeter wall segments (South, East, North, West)
        walls = [
            WallSegment(id="wall_south", p1=p1, p2=p2, zone_a="stl_zone_1", height=dz, window_area=0.15 * (floor_w * dz)),
            WallSegment(id="wall_east", p1=p2, p2=p3, zone_a="stl_zone_1", height=dz, window_area=0.10 * (floor_l * dz)),
            WallSegment(id="wall_north", p1=p3, p2=p4, zone_a="stl_zone_1", height=dz, window_area=0.05 * (floor_w * dz)),
            WallSegment(id="wall_west", p1=p4, p2=p1, zone_a="stl_zone_1", height=dz, window_area=0.10 * (floor_l * dz)),
        ]

        return BuildingGeometry(
            zones=[zone],
            walls=walls,
            roof_u_value=0.15,
            floor_u_value=0.18,
            metadata={
                "stl_imported": True,
                "source": "stl_import",
                "facets_count": len(facets),
                "bounding_box": {
                    "width": round(dx, 2), "depth": round(dy, 2), "height": round(dz, 2),
                    "area_roof_m2": round(area_roof, 2),
                    "area_floor_m2": round(area_floor, 2),
                }
            }
        )
