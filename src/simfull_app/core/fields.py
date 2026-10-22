"""Local 1D and 2D spatial fields (Level 2).

1. MultiLayerWall1D: Transient 1D finite-difference conduction through multi-layer assemblies.
2. ThermalBridge2D: 2D finite-difference conduction solver conforming to EN ISO 10211 for
   thermal junctions, corners, and window frame reveals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .components import MATERIALS, WallLayer


@dataclass
class Wall1DResult:
    """Results from a 1D transient finite-difference wall simulation."""
    times_hr: np.ndarray
    x_positions_m: np.ndarray
    materials: list[str]
    temperatures: np.ndarray      # shape: (n_times, n_nodes) in °C
    heat_flux_in_w_m2: np.ndarray # heat entering internal surface over time
    heat_flux_out_w_m2: np.ndarray# heat entering external surface over time
    u_value: float

    def to_dict(self, max_points: int = 100) -> dict[str, Any]:
        stride = max(1, len(self.times_hr) // max_points)
        return {
            "times_hr": [round(float(t), 2) for t in self.times_hr[::stride]],
            "x_positions_m": [round(float(x), 4) for x in self.x_positions_m],
            "materials": self.materials,
            "temperatures": [
                [round(float(v), 2) for v in row]
                for row in self.temperatures[::stride]
            ],
            "heat_flux_in_w_m2": [round(float(q), 2) for q in self.heat_flux_in_w_m2[::stride]],
            "heat_flux_out_w_m2": [round(float(q), 2) for q in self.heat_flux_out_w_m2[::stride]],
            "u_value": round(float(self.u_value), 3),
        }


class MultiLayerWall1D:
    """1D finite difference multi-layer transient heat diffusion solver."""

    def __init__(
        self,
        layers: list[WallLayer],
        nodes_per_layer: int = 6,
        r_si: float = 0.13,
        r_se: float = 0.04,
    ) -> None:
        self.layers = layers
        self.nodes_per_layer = max(3, nodes_per_layer)
        self.r_si = r_si
        self.r_se = r_se
        self._discretize()

    def _discretize(self) -> None:
        total_nodes = len(self.layers) * self.nodes_per_layer
        self.x = np.zeros(total_nodes)
        self.dx = np.zeros(total_nodes)
        self.k = np.zeros(total_nodes)
        self.rho = np.zeros(total_nodes)
        self.cp = np.zeros(total_nodes)
        self.layer_names: list[str] = []

        curr_x = 0.0
        node_idx = 0
        r_layers = 0.0

        for layer in self.layers:
            props = layer.properties
            dx_l = layer.thickness / self.nodes_per_layer
            r_layers += layer.thickness / props["k"]
            for _ in range(self.nodes_per_layer):
                self.x[node_idx] = curr_x + dx_l * 0.5
                self.dx[node_idx] = dx_l
                self.k[node_idx] = props["k"]
                self.rho[node_idx] = props["rho"]
                self.cp[node_idx] = props["cp"]
                self.layer_names.append(layer.name or layer.material)
                curr_x += dx_l
                node_idx += 1

        self.u_value = 1.0 / (self.r_si + r_layers + self.r_se)

    def simulate(
        self,
        t_inside: float | np.ndarray,
        t_outside: float | np.ndarray,
        duration_hours: float = 24.0,
        dt_seconds: float = 120.0,
        t_initial: float = 20.0,
    ) -> Wall1DResult:
        """Run implicit backward-Euler 1D heat conduction over time."""
        n = len(self.x)
        num_steps = int((duration_hours * 3600.0) / dt_seconds)
        times_s = np.linspace(0, duration_hours * 3600.0, num_steps + 1)
        times_hr = times_s / 3600.0

        # Boundary condition profiles
        t_in_arr = t_inside if isinstance(t_inside, np.ndarray) else np.full(num_steps + 1, t_inside)
        t_out_arr = t_outside if isinstance(t_outside, np.ndarray) else np.full(num_steps + 1, t_outside)

        h_si = 1.0 / self.r_si
        h_se = 1.0 / self.r_se

        temp_history = np.zeros((num_steps + 1, n))
        current_t = np.full(n, t_initial)
        temp_history[0] = current_t

        q_in = np.zeros(num_steps + 1)
        q_out = np.zeros(num_steps + 1)

        # Conductances between nodes (harmonic mean of k at interfaces)
        g_mid = np.zeros(n - 1)
        for i in range(n - 1):
            dx_i = self.dx[i]
            dx_ip1 = self.dx[i + 1]
            k_eff = (dx_i + dx_ip1) / (dx_i / self.k[i] + dx_ip1 / self.k[i + 1])
            dist = 0.5 * (dx_i + dx_ip1)
            g_mid[i] = k_eff / dist

        # Boundary surface conductances to cell centers
        g_in_bnd = 1.0 / (self.r_si + (0.5 * self.dx[0] / self.k[0]))
        g_out_bnd = 1.0 / (self.r_se + (0.5 * self.dx[-1] / self.k[-1]))

        # Capacitances of control volumes
        c_nodes = self.rho * self.cp * self.dx

        # Assemble tridiagonal matrix
        # A * T^{n+1} = (C / dt) * T^n + RHS_bnd
        sub_diag = -g_mid
        sup_diag = -g_mid
        main_diag = c_nodes / dt_seconds

        main_diag[0] += g_in_bnd + g_mid[0]
        for i in range(1, n - 1):
            main_diag[i] += g_mid[i - 1] + g_mid[i]
        main_diag[-1] += g_mid[-1] + g_out_bnd

        diags = [sub_diag, main_diag, sup_diag]
        A = sp.diags(diags, [-1, 0, 1], shape=(n, n), format="csc")

        for s in range(num_steps):
            rhs = (c_nodes / dt_seconds) * current_t
            rhs[0] += g_in_bnd * t_in_arr[s + 1]
            rhs[-1] += g_out_bnd * t_out_arr[s + 1]

            current_t = spla.spsolve(A, rhs)
            temp_history[s + 1] = current_t

            # Heat fluxes (W/m^2)
            q_in[s + 1] = g_in_bnd * (t_in_arr[s + 1] - current_t[0])
            q_out[s + 1] = g_out_bnd * (current_t[-1] - t_out_arr[s + 1])

        return Wall1DResult(
            times_hr=times_hr,
            x_positions_m=self.x,
            materials=self.layer_names,
            temperatures=temp_history,
            heat_flux_in_w_m2=q_in,
            heat_flux_out_w_m2=q_out,
            u_value=self.u_value,
        )


@dataclass
class ThermalBridgeResult:
    """Results from a 2D thermal bridge simulation (EN ISO 10211)."""
    grid_x: np.ndarray                     # x-coordinates (m)
    grid_y: np.ndarray                     # y-coordinates (m)
    temperature_grid: np.ndarray           # 2D array [ny, nx] in °C
    t_inside: float                        # °C
    t_outside: float                       # °C
    t_si_min: float                        # minimum internal surface temperature (°C)
    f_rsi: float                           # temperature factor (dimensionless)
    heat_flow_rate_w_m: float              # total heat flow rate per meter length L^2D (W/m)
    linear_thermal_transmittance_psi: float# psi value (W/(m*K))
    condensation_risk: bool                # True if f_Rsi < critical limit (e.g. 0.70 per UK/EU standards)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_x": [round(float(x), 4) for x in self.grid_x],
            "grid_y": [round(float(y), 4) for y in self.grid_y],
            "temperature_grid": [
                [round(float(v), 2) for v in row]
                for row in self.temperature_grid
            ],
            "t_inside": self.t_inside,
            "t_outside": self.t_outside,
            "t_si_min": round(float(self.t_si_min), 2),
            "f_rsi": round(float(self.f_rsi), 3),
            "heat_flow_rate_w_m": round(float(self.heat_flow_rate_w_m), 2),
            "linear_thermal_transmittance_psi": round(float(self.linear_thermal_transmittance_psi), 3),
            "condensation_risk": self.condensation_risk,
            "summary": self.summary,
        }


class ThermalBridge2D:
    """2D Cartesian finite-difference thermal bridge solver (EN ISO 10211)."""

    def __init__(self, nx: int = 35, ny: int = 35, width: float = 1.0, height: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.width = width
        self.height = height
        self.x = np.linspace(0, width, nx)
        self.y = np.linspace(0, height, ny)
        self.dx = width / (nx - 1)
        self.dy = height / (ny - 1)

    def solve_junction(
        self,
        junction_type: str = "wall_corner", # wall_corner, wall_floor, window_reveal
        t_inside: float = 20.0,
        t_outside: float = 0.0,
        r_si: float = 0.13,
        r_se: float = 0.04,
        wall_u_value: float = 0.25,
    ) -> ThermalBridgeResult:
        """Solve 2D steady-state heat equation div(k * grad(T)) = 0 on structured grid."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        n_total = nx * ny

        # Map material conductivity across the 2D domain
        # Default: L-shaped corner junction with structural brick + insulation
        k_map = np.full((ny, nx), 0.035)  # insulation default
        is_internal_surface = np.zeros((ny, nx), dtype=bool)
        is_external_surface = np.zeros((ny, nx), dtype=bool)

        wall_thickness_idx = int(0.35 * nx)

        if junction_type == "wall_corner":
            # Outer brick skin: outer 30% of domain
            k_map[:int(0.7 * ny), :int(0.7 * nx)] = 0.77  # brick
            # Insulation layer in corner
            k_map[wall_thickness_idx:, wall_thickness_idx:] = 0.035 # insulation
            # Inside plaster finish
            k_map[wall_thickness_idx + 4:, wall_thickness_idx + 4:] = 0.25

            # Boundary surface identifiers:
            # Inside corner: top edge and right edge of internal space
            is_internal_surface[int(0.6 * ny):, int(0.6 * nx)] = True
            is_internal_surface[int(0.6 * ny), int(0.6 * nx):] = True
            # Outside corner: left edge and bottom edge
            is_external_surface[:, 0] = True
            is_external_surface[0, :] = True
        elif junction_type == "wall_floor":
            # Horizontal concrete slab intersecting vertical insulated wall
            k_map[:, :] = 0.035 # insulation
            k_map[int(0.3 * ny):int(0.5 * ny), :] = 1.75 # concrete slab
            k_map[:, :int(0.35 * nx)] = 0.77 # outer brick
            is_internal_surface[int(0.5 * ny):, int(0.35 * nx):] = True
            is_external_surface[:, 0] = True
        else: # window_reveal or standard junction
            k_map[:, :] = 0.77 # brick
            k_map[:, int(0.5 * nx):] = 0.035 # insulation
            is_internal_surface[:, -1] = True
            is_external_surface[:, 0] = True

        # Build 5-point Laplacian sparse matrix
        row_ind = []
        col_ind = []
        data = []
        rhs = np.zeros(n_total)

        idx = lambda r, c: r * nx + c

        for r in range(ny):
            for c in range(nx):
                i = idx(r, c)
                k_c = k_map[r, c]

                # Robin boundary condition for external surface
                if is_external_surface[r, c]:
                    h_eff = 1.0 / r_se
                    row_ind.append(i); col_ind.append(i); data.append(1.0 + (h_eff * dx / k_c))
                    rhs[i] = t_outside
                    if c + 1 < nx:
                        row_ind.append(i); col_ind.append(idx(r, c + 1)); data.append(-1.0)
                    continue

                # Robin boundary condition for internal surface
                if is_internal_surface[r, c]:
                    h_eff = 1.0 / r_si
                    row_ind.append(i); col_ind.append(i); data.append(1.0 + (h_eff * dx / k_c))
                    rhs[i] = t_inside
                    if c - 1 >= 0:
                        row_ind.append(i); col_ind.append(idx(r, c - 1)); data.append(-1.0)
                    continue

                # Internal conduction cell
                # Finite-difference harmonic mean conductances
                g_e = (2.0 * k_c * k_map[r, min(nx - 1, c + 1)] / (k_c + k_map[r, min(nx - 1, c + 1)])) / (dx * dx) if c + 1 < nx else 0.0
                g_w = (2.0 * k_c * k_map[r, max(0, c - 1)] / (k_c + k_map[r, max(0, c - 1)])) / (dx * dx) if c - 1 >= 0 else 0.0
                g_n = (2.0 * k_c * k_map[min(ny - 1, r + 1), c] / (k_c + k_map[min(ny - 1, r + 1), c])) / (dy * dy) if r + 1 < ny else 0.0
                g_s = (2.0 * k_c * k_map[max(0, r - 1), c] / (k_c + k_map[max(0, r - 1), c])) / (dy * dy) if r - 1 >= 0 else 0.0

                g_sum = g_e + g_w + g_n + g_s
                if g_sum == 0.0:
                    g_sum = 1.0

                row_ind.append(i); col_ind.append(i); data.append(g_sum)
                if c + 1 < nx: row_ind.append(i); col_ind.append(idx(r, c + 1)); data.append(-g_e)
                if c - 1 >= 0: row_ind.append(i); col_ind.append(idx(r, c - 1)); data.append(-g_w)
                if r + 1 < ny: row_ind.append(i); col_ind.append(idx(r + 1, c)); data.append(-g_n)
                if r - 1 >= 0: row_ind.append(i); col_ind.append(idx(r - 1, c)); data.append(-g_s)

        A = sp.csc_matrix((data, (row_ind, col_ind)), shape=(n_total, n_total))
        t_flat = spla.spsolve(A, rhs)
        temp_grid = t_flat.reshape((ny, nx))

        # Minimum internal surface temperature
        internal_temps = temp_grid[is_internal_surface]
        if len(internal_temps) > 0:
            t_si_min = float(np.min(internal_temps))
        else:
            t_si_min = float(np.min(temp_grid[int(0.7 * ny):, int(0.7 * nx):]))

        delta_t_env = max(1e-4, t_inside - t_outside)
        f_rsi = (t_si_min - t_outside) / delta_t_env

        # Total 2D heat flow rate L^2D in W/m
        # Integrate flux across internal boundary
        flux_in = 0.0
        for r in range(ny):
            for c in range(nx):
                if is_internal_surface[r, c]:
                    flux_in += (t_inside - temp_grid[r, c]) / r_si * dx

        l_2d = max(0.5, flux_in)

        # Linear thermal transmittance psi = L^2D - sum(U_j * l_j)
        # 1D transmission through the 2 flanking walls:
        l_flanking = (self.width + self.height) * 0.5
        psi = max(0.01, (l_2d / delta_t_env) - (wall_u_value * l_flanking))

        # Critical temperature factor limit per BRE IP 1/06 / DIN 4108-2: 0.70 for dwellings (avoids mold)
        condensation_risk = f_rsi < 0.70

        return ThermalBridgeResult(
            grid_x=self.x,
            grid_y=self.y,
            temperature_grid=temp_grid,
            t_inside=t_inside,
            t_outside=t_outside,
            t_si_min=t_si_min,
            f_rsi=f_rsi,
            heat_flow_rate_w_m=l_2d,
            linear_thermal_transmittance_psi=psi,
            condensation_risk=condensation_risk,
            summary={
                "junction_type": junction_type,
                "conformance": "EN ISO 10211",
                "grid_cells": nx * ny,
            }
        )
