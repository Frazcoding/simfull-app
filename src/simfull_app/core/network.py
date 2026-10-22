"""Multi-domain network simulation engine (Level 1).

A general graph-based state solver where nodes represent physical states (temperature,
voltage, pressure, displacement) with capacity, and links represent physical conductances
(conduction, convection, advection, radiation, electrical admittance).

Solved implicitly via backward Euler for unconditional numerical stability:
    (C / dt + G) * x^{n+1} = (C / dt) * x^n + q^{n+1}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Domain(str, Enum):
    THERMAL = "thermal"
    FLUID = "fluid"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"


@dataclass
class NetworkNode:
    """A physical state node in the simulation graph."""
    id: str
    domain: Domain = Domain.THERMAL
    name: str = ""
    capacity: float = 0.0          # J/K (thermal), F (electrical), kg (structural)
    value: float = 20.0            # current state value (e.g. temperature in °C)
    initial_value: float = 20.0
    fixed: bool = False            # True for boundary condition nodes (e.g. ambient)
    source: float = 0.0            # internal generation (W or A)
    zone: str | None = None        # parent thermal zone / room
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.value = self.initial_value


@dataclass
class NetworkLink:
    """A physical link between two nodes with conductance."""
    id: str
    node_a: str
    node_b: str
    conductance: float             # W/K (thermal), S (electrical), kg/(s*Pa) (fluid)
    domain: Domain = Domain.THERMAL
    kind: str = "conduction"       # conduction, convection, advection, radiation, emitter
    area: float = 0.0              # m^2
    length: float = 0.0            # m
    equation: str = ""             # primary governing formula / literature citation
    conductance_fn: Callable[[float, dict[str, float]], float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def current_conductance(self, t: float, states: dict[str, float]) -> float:
        if self.conductance_fn is not None:
            return float(self.conductance_fn(t, states))
        return self.conductance


@dataclass
class SimulationResult:
    """Time-series results from a simulation run."""
    times: np.ndarray                            # seconds
    hours: np.ndarray                            # hours from start
    node_ids: list[str]
    temperatures: dict[str, np.ndarray]          # node_id -> array of values
    flows: dict[str, np.ndarray]                 # link_id -> heat/mass flow array (W)
    total_energy_kwh: dict[str, float]           # aggregated cumulative energy (kWh)
    balance_error_max: float                     # maximum absolute energy conservation residual (W)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, sample_step: int = 1) -> dict[str, Any]:
        """Convert time-series to JSON-serializable dictionary."""
        indices = slice(None, None, max(1, sample_step))
        return {
            "hours": [round(float(h), 2) for h in self.hours[indices]],
            "node_ids": self.node_ids,
            "temperatures": {
                nid: [round(float(v), 2) for v in vals[indices]]
                for nid, vals in self.temperatures.items()
            },
            "flows": {
                lid: [round(float(f), 1) for f in fvals[indices]]
                for lid, fvals in self.flows.items()
            },
            "total_energy_kwh": {
                k: round(float(v), 3) for k, v in self.total_energy_kwh.items()
            },
            "balance_error_max": round(float(self.balance_error_max), 6),
            "summary": self.summary,
        }


class SimulationNetwork:
    """The central solver graph orchestrating nodes, links, and implicit time stepping."""

    def __init__(self) -> None:
        self.nodes: dict[str, NetworkNode] = {}
        self.links: dict[str, NetworkLink] = {}
        self._node_order: list[str] = []

    def add_node(self, node: NetworkNode) -> NetworkNode:
        self.nodes[node.id] = node
        if node.id not in self._node_order:
            self._node_order.append(node.id)
        return node

    def add_link(self, link: NetworkLink) -> NetworkLink:
        if link.node_a not in self.nodes:
            raise KeyError(f"Link {link.id} references unknown node_a: {link.node_a}")
        if link.node_b not in self.nodes:
            raise KeyError(f"Link {link.id} references unknown node_b: {link.node_b}")
        self.links[link.id] = link
        return link

    def get_node(self, node_id: str) -> NetworkNode:
        return self.nodes[node_id]

    def reset(self) -> None:
        for node in self.nodes.values():
            node.reset()

    def assemble_system(self, t: float, dt: float) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        """Assemble the system matrices (C / dt + G) and right-hand side vector."""
        n = len(self._node_order)
        id_to_idx = {nid: i for i, nid in enumerate(self._node_order)}
        states = {nid: self.nodes[nid].value for nid in self._node_order}

        diag_c = np.zeros(n)
        rhs = np.zeros(n)
        row_ind = []
        col_ind = []
        data = []

        # Populate capacitance and internal sources
        for i, nid in enumerate(self._node_order):
            node = self.nodes[nid]
            if node.fixed:
                # Fixed boundary condition node
                diag_c[i] = 0.0
                rhs[i] = node.value
            else:
                diag_c[i] = node.capacity
                # RHS gets C / dt * T^n + source
                rhs[i] = (node.capacity / dt) * node.value + node.source

        # Populate conductance matrix G
        g_diag = np.zeros(n)
        for link in self.links.values():
            cond = link.current_conductance(t, states)
            if cond == 0.0:
                continue
            i = id_to_idx[link.node_a]
            j = id_to_idx[link.node_b]

            node_a = self.nodes[link.node_a]
            node_b = self.nodes[link.node_b]

            # Link contributions: G * (T_i - T_j)
            if not node_a.fixed:
                g_diag[i] += cond
                row_ind.append(i)
                col_ind.append(j)
                data.append(-cond)
            else:
                pass  # fixed node row is handled separately

            if not node_b.fixed:
                g_diag[j] += cond
                row_ind.append(j)
                col_ind.append(i)
                data.append(-cond)
            else:
                pass

        # Combine diagonal: (C / dt + g_diag) for non-fixed, 1.0 for fixed
        for i, nid in enumerate(self._node_order):
            if self.nodes[nid].fixed:
                row_ind.append(i)
                col_ind.append(i)
                data.append(1.0)
            elif diag_c[i] == 0.0 and g_diag[i] == 0.0:
                # Decoupled / algebraic / monitor node without links or capacity
                row_ind.append(i)
                col_ind.append(i)
                data.append(1.0)
                rhs[i] = self.nodes[nid].value
            else:
                row_ind.append(i)
                col_ind.append(i)
                data.append(diag_c[i] / dt + g_diag[i])

        A = sp.csc_matrix((data, (row_ind, col_ind)), shape=(n, n))
        return A, rhs, diag_c

    def step(self, t: float, dt: float) -> tuple[dict[str, float], dict[str, float], float]:
        """Advance the simulation network by one implicit time step dt."""
        A, rhs, diag_c = self.assemble_system(t + dt, dt)
        new_values = spla.spsolve(A, rhs)

        old_values = np.array([self.nodes[nid].value for nid in self._node_order])
        id_to_idx = {nid: i for i, nid in enumerate(self._node_order)}

        # Update node values
        for i, nid in enumerate(self._node_order):
            if not self.nodes[nid].fixed:
                self.nodes[nid].value = float(new_values[i])

        state_dict = {nid: self.nodes[nid].value for nid in self._node_order}

        # Calculate link heat flows Q_ab = G * (T_a - T_b)
        flow_dict = {}
        for link in self.links.values():
            cond = link.current_conductance(t + dt, state_dict)
            ta = state_dict[link.node_a]
            tb = state_dict[link.node_b]
            flow_dict[link.id] = cond * (ta - tb)

        # Audit energy conservation residual
        stored_rate = np.sum(diag_c * (new_values - old_values) / dt)
        net_source = sum(node.source for node in self.nodes.values() if not node.fixed)
        net_boundary_flow = 0.0
        for link in self.links.values():
            a_fixed = self.nodes[link.node_a].fixed
            b_fixed = self.nodes[link.node_b].fixed
            if a_fixed and not b_fixed:
                net_boundary_flow += flow_dict[link.id]   # Flow from boundary A into B
            elif b_fixed and not a_fixed:
                net_boundary_flow -= flow_dict[link.id]   # Flow from B into A

        residual = abs(stored_rate - (net_source + net_boundary_flow))
        return state_dict, flow_dict, residual

    def simulate(
        self,
        duration: float,
        dt: float,
        weather_driver: Callable[[float], dict[str, float]] | None = None,
        hvac_controller: Callable[[float, dict[str, float]], None] | None = None,
    ) -> SimulationResult:
        """Run complete time-series simulation over duration seconds."""
        self.reset()
        num_steps = int(duration / dt)
        times = np.linspace(0, duration, num_steps + 1)
        hours = times / 3600.0

        num_nodes = len(self._node_order)
        temp_histories = {nid: np.zeros(num_steps + 1) for nid in self._node_order}
        flow_histories = {lid: np.zeros(num_steps + 1) for lid in self.links}

        # Record initial states
        for nid in self._node_order:
            temp_histories[nid][0] = self.nodes[nid].value

        max_residual = 0.0

        for step_idx in range(num_steps):
            t = times[step_idx]

            # 1. Update weather boundary conditions
            if weather_driver is not None:
                bc_values = weather_driver(t)
                for nid, val in bc_values.items():
                    if nid in self.nodes:
                        self.nodes[nid].value = val

            # 2. Update HVAC controllers
            if hvac_controller is not None:
                current_states = {nid: self.nodes[nid].value for nid in self._node_order}
                hvac_controller(t, current_states)

            # 3. Advance one step
            state_dict, flow_dict, residual = self.step(t, dt)
            max_residual = max(max_residual, residual)

            # Record
            for nid in self._node_order:
                temp_histories[nid][step_idx + 1] = state_dict[nid]
            for lid, q in flow_dict.items():
                flow_histories[lid][step_idx + 1] = q

        # Compute integrated energy totals in kWh
        total_energy_kwh = {}
        for lid, flow in flow_histories.items():
            # Integrate power (W) over time (s) -> Joules / 3.6e6 -> kWh
            joules = np.trapezoid(flow, times)
            total_energy_kwh[lid] = joules / 3.6e6

        return SimulationResult(
            times=times,
            hours=hours,
            node_ids=self._node_order,
            temperatures=temp_histories,
            flows=flow_histories,
            total_energy_kwh=total_energy_kwh,
            balance_error_max=max_residual,
        )
