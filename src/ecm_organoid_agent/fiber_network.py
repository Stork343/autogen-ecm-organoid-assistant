"""Research-oriented 3D ECM bead-spring fiber network simulator.

This module upgrades the ECM mechanics core to include:

- Nonlinear strain-stiffening fiber stretching:
  F = k * (exp(alpha * (l - l0)) - 1)
- Compression buckling surrogate:
  F = k_compress * (l - l0), with k_compress << k
- Bending energy:
  E_bend = k_bend * (theta - theta0)^2
- Nonlinear conjugate-gradient equilibrium solver with:
  - backtracking line search
  - adaptive step size
  - energy decrease checks
  - force residual logging
  - random restarts
- Monte Carlo aggregation for robust feature estimates
- Tensile-test nonlinearity validation
- Deterministic inverse design search for target mechanics

The module exposes the public API requested by the user:

- generate_network()
- apply_force()
- solve_equilibrium()
- compute_features()
- run_simulation()
- simulate_ecm()
- run_validation()
- design_ecm_candidates()
- visualize_network()
- visualize_stress_strain()
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import sqrt
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import qmc

AxisName = Literal["x", "y", "z"]
DESIGN_OBSERVABLES = {
    "stiffness",
    "anisotropy",
    "connectivity",
    "stress_propagation",
    "risk_index",
    "mesh_size_proxy",
    "permeability_proxy",
    "compressibility_proxy",
    "swelling_ratio_proxy",
    "loss_tangent_proxy",
    "poroelastic_time_constant_proxy",
    "strain_stiffening_proxy",
}


@dataclass
class SolverDiagnostics:
    """Diagnostics for a nonlinear equilibrium solve."""

    converged: bool
    attempts_used: int
    iterations: int
    final_energy: float
    final_max_force: float
    iteration_history: list[dict[str, float]] = field(default_factory=list)
    force_residual_curve: list[float] = field(default_factory=list)
    energy_curve: list[float] = field(default_factory=list)


@dataclass
class FiberNetwork:
    """Container for ECM fiber-network geometry, material parameters, and state."""

    rest_positions: np.ndarray
    positions: np.ndarray
    edges: np.ndarray
    rest_lengths: np.ndarray
    bending_triplets: np.ndarray
    rest_angles: np.ndarray
    fiber_stiffness: float
    bending_stiffness: float
    domain_size: float
    nonlinearity_alpha: float
    compression_ratio: float
    total_force: float = 0.0
    applied_axis: int = 0
    boundary_fraction: float = 0.15
    fixed_mask: np.ndarray | None = None
    loaded_mask: np.ndarray | None = None
    constrained_mask: np.ndarray | None = None
    constrained_positions: np.ndarray | None = None
    external_forces: np.ndarray | None = None
    loading_mode: str = "free"
    target_strain: float = 0.0
    solver: SolverDiagnostics | None = None


def generate_network(
    fiber_density: float,
    fiber_stiffness: float,
    bending_stiffness: float,
    crosslink_prob: float,
    domain_size: float,
    *,
    seed: int | None = None,
    target_nodes: int | None = None,
    nonlinearity_alpha: float = 6.0,
    compression_ratio: float = 0.05,
) -> FiberNetwork:
    """Generate a random 3D fiber network.

    Higher `fiber_density` increases node count and local connection radius.
    Higher `crosslink_prob` increases retained fiber connectivity.
    """

    if not (0.0 < fiber_density <= 1.0):
        raise ValueError("fiber_density must be in (0, 1].")
    if fiber_stiffness <= 0.0:
        raise ValueError("fiber_stiffness must be positive.")
    if bending_stiffness < 0.0:
        raise ValueError("bending_stiffness must be non-negative.")
    if not (0.0 <= crosslink_prob <= 1.0):
        raise ValueError("crosslink_prob must be in [0, 1].")
    if domain_size <= 0.0:
        raise ValueError("domain_size must be positive.")
    if nonlinearity_alpha <= 0.0:
        raise ValueError("nonlinearity_alpha must be positive.")
    if not (0.0 < compression_ratio < 1.0):
        raise ValueError("compression_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)
    node_count = int(target_nodes) if target_nodes is not None else 18
    rest_positions = rng.uniform(0.0, domain_size, size=(node_count, 3))

    tree = cKDTree(rest_positions)
    candidate_pairs: set[tuple[int, int]] = set()
    k_pool = min(node_count, max(4, int(3 + 10 * fiber_density)))
    _, indices = tree.query(rest_positions, k=k_pool)
    for i in range(node_count):
        for j in np.atleast_1d(indices[i])[1:]:
            edge = tuple(sorted((int(i), int(j))))
            if edge[0] != edge[1]:
                candidate_pairs.add(edge)

    if not candidate_pairs:
        candidate_pairs.update(_nearest_neighbor_edges(rest_positions, tree))

    scaffold_edges = _ensure_connected(rest_positions, np.asarray(_nearest_neighbor_edges(rest_positions, tree), dtype=int))
    scaffold_set = {tuple(edge) for edge in scaffold_edges}
    edges_set = set(scaffold_set)
    retain_prob = min(1.0, max(0.02, 0.1 + 0.9 * fiber_density * crosslink_prob + 0.25 * fiber_density))
    for edge in sorted(candidate_pairs):
        if edge in scaffold_set:
            continue
        if rng.random() <= retain_prob:
            edges_set.add(edge)

    edges = np.asarray(sorted(edges_set), dtype=int)
    rest_lengths = np.linalg.norm(rest_positions[edges[:, 1]] - rest_positions[edges[:, 0]], axis=1)
    bending_triplets, rest_angles = _build_bending_triplets(rest_positions, edges)

    node_count = rest_positions.shape[0]
    zeros = np.zeros((node_count, 3), dtype=float)
    false_mask = np.zeros(node_count, dtype=bool)
    return FiberNetwork(
        rest_positions=rest_positions,
        positions=rest_positions.copy(),
        edges=edges,
        rest_lengths=rest_lengths,
        bending_triplets=bending_triplets,
        rest_angles=rest_angles,
        fiber_stiffness=float(fiber_stiffness),
        bending_stiffness=float(bending_stiffness),
        domain_size=float(domain_size),
        nonlinearity_alpha=float(nonlinearity_alpha),
        compression_ratio=float(compression_ratio),
        fixed_mask=false_mask.copy(),
        loaded_mask=false_mask.copy(),
        constrained_mask=false_mask.copy(),
        constrained_positions=zeros.copy(),
        external_forces=zeros.copy(),
    )


def apply_force(
    network: FiberNetwork,
    *,
    total_force: float = 1.0,
    axis: AxisName = "x",
    boundary_fraction: float = 0.15,
) -> FiberNetwork:
    """Apply force-controlled loading with lower-boundary fixation."""

    if total_force <= 0.0:
        raise ValueError("total_force must be positive.")
    axis_index = _axis_to_index(axis)
    fixed_mask, loaded_mask = _boundary_masks(network.rest_positions, axis_index, network.domain_size, boundary_fraction)
    constrained_mask = fixed_mask.copy()
    constrained_positions = network.rest_positions.copy()
    external_forces = np.zeros_like(network.positions)
    external_forces[loaded_mask, axis_index] = total_force / max(int(loaded_mask.sum()), 1)

    network.positions = constrained_positions.copy()
    network.total_force = float(total_force)
    network.applied_axis = axis_index
    network.boundary_fraction = float(boundary_fraction)
    network.fixed_mask = fixed_mask
    network.loaded_mask = loaded_mask
    network.constrained_mask = constrained_mask
    network.constrained_positions = constrained_positions
    network.external_forces = external_forces
    network.loading_mode = "force"
    network.target_strain = 0.0
    network.solver = None
    return network


def apply_strain(
    network: FiberNetwork,
    *,
    strain: float,
    axis: AxisName = "x",
    boundary_fraction: float = 0.15,
) -> FiberNetwork:
    """Apply displacement-controlled tensile loading."""

    axis_index = _axis_to_index(axis)
    fixed_mask, loaded_mask = _boundary_masks(network.rest_positions, axis_index, network.domain_size, boundary_fraction)
    constrained_mask = fixed_mask | loaded_mask
    constrained_positions = network.rest_positions.copy()
    constrained_positions[loaded_mask, axis_index] += strain * network.domain_size

    network.positions = constrained_positions.copy()
    network.total_force = 0.0
    network.applied_axis = axis_index
    network.boundary_fraction = float(boundary_fraction)
    network.fixed_mask = fixed_mask
    network.loaded_mask = loaded_mask
    network.constrained_mask = constrained_mask
    network.constrained_positions = constrained_positions
    network.external_forces = np.zeros_like(network.positions)
    network.loading_mode = "strain"
    network.target_strain = float(strain)
    network.solver = None
    return network


def compute_energy(network: FiberNetwork, positions: np.ndarray | None = None) -> float:
    """Return internal elastic+bending energy, excluding external work."""

    positions_array = network.positions if positions is None else positions
    return _stretch_energy(network, positions_array) + _bending_energy(network, positions_array)


def compute_forces(
    network: FiberNetwork,
    positions: np.ndarray | None = None,
    *,
    include_external: bool = False,
    zero_constrained: bool = False,
) -> np.ndarray:
    """Return nodal forces.

    By default this is the internal restoring force, which is the negative
    gradient of the internal mechanical energy.
    """

    positions_array = network.positions if positions is None else positions
    forces = _internal_forces(network, positions_array)
    if include_external and network.external_forces is not None:
        forces = forces + network.external_forces
    if zero_constrained and network.constrained_mask is not None:
        forces = forces.copy()
        forces[network.constrained_mask] = 0.0
    return forces


def solve_equilibrium(
    network: FiberNetwork,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    max_restarts: int = 5,
    perturbation_scale: float = 0.01,
) -> FiberNetwork:
    """Solve static equilibrium using L-BFGS-B with random restarts."""

    if network.constrained_mask is None or not np.any(network.constrained_mask):
        raise RuntimeError("Apply loading or displacement boundary conditions before solving.")

    free_mask = ~network.constrained_mask
    if not np.any(free_mask):
        raise RuntimeError("No free degrees of freedom remain after applying constraints.")

    best_positions = network.positions.copy()
    best_energy = float("inf")
    best_residual = float("inf")
    best_history: list[dict[str, float]] = []
    best_force_curve: list[float] = []
    best_energy_curve: list[float] = []
    converged = False
    rng = np.random.default_rng(1234)

    for attempt in range(1, max_restarts + 1):
        positions = _initialize_positions_for_attempt(network, free_mask, rng, perturbation_scale, attempt)
        x = positions[free_mask].reshape(-1)
        iteration_history: list[dict[str, float]] = []
        force_curve: list[float] = []
        energy_curve: list[float] = []
        previous_x: np.ndarray | None = None

        def callback(xk: np.ndarray) -> None:
            nonlocal previous_x
            positions_local = _positions_from_vector(network, xk, free_mask)
            energy_local = _objective(network, xk, free_mask)
            total_forces = compute_forces(
                network,
                positions_local,
                include_external=(network.loading_mode == "force"),
                zero_constrained=False,
            )
            residual_local = float(np.linalg.norm(total_forces[free_mask], axis=1).max(initial=0.0))
            step_size_local = float(np.linalg.norm(xk - previous_x)) if previous_x is not None else 0.0
            iteration_history.append(
                {
                    "attempt": float(attempt),
                    "iteration": float(len(iteration_history) + 1),
                    "energy": float(energy_local),
                    "max_force": residual_local,
                    "step_size": step_size_local,
                }
            )
            force_curve.append(residual_local)
            energy_curve.append(float(energy_local))
            previous_x = xk.copy()

        result = minimize(
            fun=lambda x_vec: _objective(network, x_vec, free_mask),
            x0=x,
            jac=lambda x_vec: _objective_gradient(network, x_vec, free_mask),
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": max_iterations,
                "maxls": 40,
                "ftol": 1e-12,
                "gtol": tolerance * 0.1,
            },
        )

        final_positions = _positions_from_vector(network, result.x, free_mask)
        final_energy = float(_objective(network, result.x, free_mask))
        total_forces = compute_forces(
            network,
            final_positions,
            include_external=(network.loading_mode == "force"),
            zero_constrained=False,
        )
        final_residual = float(np.linalg.norm(total_forces[free_mask], axis=1).max(initial=0.0))

        if not iteration_history:
            iteration_history.append(
                {
                    "attempt": float(attempt),
                    "iteration": 1.0,
                    "energy": final_energy,
                    "max_force": final_residual,
                    "step_size": 0.0,
                }
            )
            force_curve.append(final_residual)
            energy_curve.append(final_energy)

        if final_residual < best_residual:
            best_positions = final_positions
            best_energy = final_energy
            best_residual = final_residual
            best_history = iteration_history
            best_force_curve = force_curve
            best_energy_curve = energy_curve

        if np.isfinite(final_residual) and final_residual < tolerance:
            converged = True
            break

    network.positions = best_positions
    network.solver = SolverDiagnostics(
        converged=converged,
        attempts_used=attempt,
        iterations=len(best_history),
        final_energy=float(best_energy),
        final_max_force=float(best_residual),
        iteration_history=best_history,
        force_residual_curve=best_force_curve,
        energy_curve=best_energy_curve,
    )
    network.last_energy = network.solver.final_energy
    network.last_max_force = network.solver.final_max_force
    network.iterations = network.solver.iterations
    return network


def compute_features(network: FiberNetwork) -> dict[str, float]:
    """Compute single-run emergent mechanics descriptors."""

    displacements = network.positions - network.rest_positions
    mean_displacement = float(np.mean(np.linalg.norm(displacements, axis=1)))

    axis_displacements = np.abs(displacements[:, network.applied_axis])
    loaded_disp = float(np.mean(axis_displacements[network.loaded_mask])) if np.any(network.loaded_mask) else 0.0

    if network.loading_mode == "force":
        stiffness = float(network.total_force / max(loaded_disp, 1e-12))
    else:
        reaction = _reaction_force(network)
        area = max(network.domain_size * network.domain_size, 1e-12)
        engineering_stress = reaction / area
        stiffness = float(engineering_stress / max(network.target_strain, 1e-12))

    edge_vectors = network.positions[network.edges[:, 1]] - network.positions[network.edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    valid = edge_lengths > 1e-12
    orientations = edge_vectors[valid] / edge_lengths[valid, None]
    orientation_tensor = orientations.T @ orientations / max(len(orientations), 1)
    eigenvalues = np.linalg.eigvalsh(orientation_tensor)
    anisotropy = float(max(eigenvalues[-1] - eigenvalues[0], 0.0))
    connectivity = _largest_component_fraction(network.edges, network.positions.shape[0])
    free_mask = ~network.constrained_mask
    stress_propagation = float(
        np.mean(axis_displacements[free_mask]) / max(loaded_disp, 1e-12)
    ) if np.any(free_mask) else 0.0
    edge_length_mean = float(np.mean(edge_lengths)) if edge_lengths.size else 0.0
    node_density = float(network.positions.shape[0] / max(network.domain_size**3, 1e-12))
    mesh_size_proxy = float(edge_length_mean / max(connectivity, 0.05))
    permeability_proxy = float((mesh_size_proxy**2) / max(node_density * (1.0 + 0.05 * network.fiber_stiffness), 1e-12))
    compressibility_proxy = float(
        1.0
        / (
            1.0
            + connectivity * network.fiber_stiffness
            + 2.0 * network.bending_stiffness
            + 0.25 * network.nonlinearity_alpha
        )
    )
    swelling_ratio_proxy = float(
        1.0
        + 1.5 * max(0.0, 1.0 - connectivity)
        + mesh_size_proxy / max(network.domain_size, 1e-12)
    )
    loss_tangent_proxy = float(
        min(
            max(
                0.05
                + 2.5 * network.bending_stiffness / max(network.fiber_stiffness, 1e-12)
                + 0.15 * mean_displacement / max(network.domain_size, 1e-12),
                0.01,
            ),
            5.0,
        )
    )
    poroelastic_time_constant_proxy = float(
        network.domain_size**2 / max(permeability_proxy * 1e3, 1e-12)
    )
    strain_stiffening_proxy = float(
        1.0 + 0.15 * network.nonlinearity_alpha * max(connectivity, 0.1)
    )

    return {
        "stiffness": stiffness,
        "anisotropy": anisotropy,
        "connectivity": connectivity,
        "mean_displacement": mean_displacement,
        "stress_propagation": stress_propagation,
        "mesh_size_proxy": mesh_size_proxy,
        "permeability_proxy": permeability_proxy,
        "compressibility_proxy": compressibility_proxy,
        "swelling_ratio_proxy": swelling_ratio_proxy,
        "loss_tangent_proxy": loss_tangent_proxy,
        "poroelastic_time_constant_proxy": poroelastic_time_constant_proxy,
        "strain_stiffening_proxy": strain_stiffening_proxy,
    }


def run_simulation(
    fiber_density: float,
    fiber_stiffness: float,
    bending_stiffness: float,
    crosslink_prob: float,
    domain_size: float,
    *,
    total_force: float = 1.0,
    axis: AxisName = "x",
    boundary_fraction: float = 0.15,
    seed: int | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    nonlinearity_alpha: float = 6.0,
    compression_ratio: float = 0.05,
    target_nodes: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic force-controlled ECM simulation."""

    network = generate_network(
        fiber_density=fiber_density,
        fiber_stiffness=fiber_stiffness,
        bending_stiffness=bending_stiffness,
        crosslink_prob=crosslink_prob,
        domain_size=domain_size,
        seed=seed,
        target_nodes=target_nodes,
        nonlinearity_alpha=nonlinearity_alpha,
        compression_ratio=compression_ratio,
    )
    apply_force(network, total_force=total_force, axis=axis, boundary_fraction=boundary_fraction)
    solve_equilibrium(network, max_iterations=max_iterations, tolerance=tolerance)
    features = compute_features(network)
    diagnostics = _solver_summary_dict(network.solver)
    property_proxy_keys = (
        "mesh_size_proxy",
        "permeability_proxy",
        "compressibility_proxy",
        "swelling_ratio_proxy",
        "loss_tangent_proxy",
        "poroelastic_time_constant_proxy",
        "strain_stiffening_proxy",
    )
    property_proxies = {
        key: float(features[key])
        for key in property_proxy_keys
        if key in features
    }
    return {
        **features,
        "material_property_proxies": property_proxies,
        "material_property_proxy_std": {key: 0.0 for key in property_proxies},
        "solver_converged": bool(network.solver.converged if network.solver else False),
        "final_residual": float(network.solver.final_max_force if network.solver else np.nan),
        "iteration_history": diagnostics.get("iteration_history", []),
        "force_residual_curve": diagnostics.get("force_residual_curve", []),
    }


def run_parameter_scan(
    fiber_density: float,
    fiber_stiffness: float,
    bending_stiffness: float,
    crosslink_prob: float,
    domain_size: float,
    *,
    total_force: float = 1.0,
    axis: AxisName = "x",
    boundary_fraction: float = 0.15,
    seed: int | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    monte_carlo_runs: int = 10,
    target_nodes: int | None = None,
    scan_factors: tuple[float, ...] = (0.7, 1.0, 1.3),
) -> dict[str, Any]:
    """Run one-factor-at-a-time sensitivity scans using Monte Carlo-aggregated metrics."""

    base_params = {
        "fiber_density": fiber_density,
        "fiber_stiffness": fiber_stiffness,
        "bending_stiffness": bending_stiffness,
        "crosslink_prob": crosslink_prob,
        "domain_size": domain_size,
        "total_force": total_force,
        "axis": axis,
        "boundary_fraction": boundary_fraction,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "monte_carlo_runs": monte_carlo_runs,
        "seed": seed if seed is not None else 1234,
        "target_nodes": target_nodes,
    }
    base_result = simulate_ecm(base_params)

    bounds = {
        "fiber_density": (0.05, 1.0),
        "fiber_stiffness": (1e-6, None),
        "bending_stiffness": (0.0, None),
        "crosslink_prob": (0.01, 1.0),
        "domain_size": (0.1, None),
    }
    scan_results: dict[str, list[dict[str, Any]]] = {}
    sensitivity = []

    for parameter_name in ("fiber_density", "fiber_stiffness", "bending_stiffness", "crosslink_prob", "domain_size"):
        baseline = float(base_params[parameter_name])
        rows = []
        stiffness_values = []
        for factor_index, factor in enumerate(scan_factors):
            varied = dict(base_params)
            candidate = baseline * factor
            lower, upper = bounds[parameter_name]
            if lower is not None:
                candidate = max(candidate, lower)
            if upper is not None:
                candidate = min(candidate, upper)
            varied[parameter_name] = float(candidate)
            varied["seed"] = int(base_params["seed"]) + factor_index + 101 * (len(sensitivity) + 1)
            result = simulate_ecm(varied)
            rows.append({"factor": float(factor), "parameter_value": float(candidate), "result": result})
            stiffness_values.append(float(result["stiffness_mean"]))
        scan_results[parameter_name] = rows
        base_stiffness = max(abs(float(base_result["stiffness_mean"])), 1e-12)
        sensitivity.append(
            {
                "parameter": parameter_name,
                "normalized_stiffness_span": float((max(stiffness_values) - min(stiffness_values)) / base_stiffness),
            }
        )

    sensitivity.sort(key=lambda item: item["normalized_stiffness_span"], reverse=True)
    return {
        "base_parameters": {k: v for k, v in base_params.items() if k not in {"seed", "monte_carlo_runs"}},
        "base_result": base_result,
        "scan_factors": list(scan_factors),
        "scan_results": scan_results,
        "sensitivity_ranking": sensitivity,
    }


def run_tensile_test(
    params: dict[str, Any],
    *,
    strains: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run a standard displacement-controlled tensile test with Monte Carlo averaging."""

    base_strains = np.linspace(0.0, 0.5, 7) if strains is None else np.asarray(strains, dtype=float)
    monte_carlo_runs = int(params.get("monte_carlo_runs", 10))
    seed = int(params.get("seed", 1234))
    target_nodes = params.get("target_nodes")

    mean_stresses = []
    std_stresses = []
    replicate_stress_matrix: list[list[float]] = []
    for replicate in range(monte_carlo_runs):
        network = generate_network(
            fiber_density=float(params["fiber_density"]),
            fiber_stiffness=float(params["fiber_stiffness"]),
            bending_stiffness=float(params["bending_stiffness"]),
            crosslink_prob=float(params["crosslink_prob"]),
            domain_size=float(params["domain_size"]),
            seed=seed + replicate,
            target_nodes=int(target_nodes) if target_nodes is not None else None,
            nonlinearity_alpha=float(params.get("nonlinearity_alpha", 6.0)),
            compression_ratio=float(params.get("compression_ratio", 0.05)),
        )
        replicate_stresses: list[float] = []
        previous_positions = network.positions.copy()
        for strain in base_strains:
            apply_strain(
                network,
                strain=float(strain),
                axis=str(params.get("axis", "x")),
                boundary_fraction=float(params.get("boundary_fraction", 0.15)),
            )
            free_mask = ~network.constrained_mask
            previous_positions = np.asarray(previous_positions, dtype=float)
            if previous_positions.shape == network.positions.shape and np.any(free_mask):
                network.positions[free_mask] = previous_positions[free_mask]
            solve_equilibrium(
                network,
                max_iterations=int(params.get("max_iterations", 500)),
                tolerance=float(params.get("tolerance", 1e-5)),
            )
            previous_positions = network.positions.copy()
            replicate_stresses.append(_reaction_force(network) / max(network.domain_size * network.domain_size, 1e-12))
        replicate_stress_matrix.append(replicate_stresses)

    for strain_index, _ in enumerate(base_strains):
        stresses = [row[strain_index] for row in replicate_stress_matrix]
        mean_stresses.append(float(np.mean(stresses)))
        std_stresses.append(float(np.std(stresses)))

    curve = [
        {"strain": float(strain), "stress": float(stress), "stress_std": float(std)}
        for strain, stress, std in zip(base_strains, mean_stresses, std_stresses)
    ]
    slopes = np.diff(mean_stresses) / np.diff(base_strains)
    low_strain_count = min(3, len(base_strains))
    replicate_small_strain_slopes = []
    for replicate_stresses in replicate_stress_matrix:
        coeff = np.polyfit(base_strains[:low_strain_count], replicate_stresses[:low_strain_count], 1)
        replicate_small_strain_slopes.append(float(coeff[0]))
    nonlinearity_valid = bool(np.all(np.diff(slopes) >= -1e-8) and slopes[-1] > slopes[0] * 1.1)
    return {
        "stress_strain_curve": curve,
        "tangent_slopes": slopes.tolist(),
        "small_strain_stiffness_mean": float(np.mean(replicate_small_strain_slopes)),
        "small_strain_stiffness_std": float(np.std(replicate_small_strain_slopes)),
        "nonlinearity_valid": nonlinearity_valid,
    }


def simulate_ecm(params: dict[str, Any]) -> dict[str, Any]:
    """Agent-safe ECM simulation API.

    Parameters are supplied and returned via JSON-friendly dictionaries.
    The function is deterministic for a fixed `seed`.
    """

    required = ["fiber_density", "fiber_stiffness", "bending_stiffness", "crosslink_prob", "domain_size"]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    monte_carlo_runs = int(params.get("monte_carlo_runs", 10))
    seed = int(params.get("seed", 1234))
    target_nodes = params.get("target_nodes")
    run_outputs = []
    warnings = []

    for replicate in range(monte_carlo_runs):
        result = run_simulation(
            fiber_density=float(params["fiber_density"]),
            fiber_stiffness=float(params["fiber_stiffness"]),
            bending_stiffness=float(params["bending_stiffness"]),
            crosslink_prob=float(params["crosslink_prob"]),
            domain_size=float(params["domain_size"]),
            total_force=float(params.get("total_force", 1.0)),
            axis=str(params.get("axis", "x")),
            boundary_fraction=float(params.get("boundary_fraction", 0.15)),
            seed=seed + replicate,
            max_iterations=int(params.get("max_iterations", 500)),
            tolerance=float(params.get("tolerance", 1e-5)),
            target_nodes=int(target_nodes) if target_nodes is not None else None,
            nonlinearity_alpha=float(params.get("nonlinearity_alpha", 6.0)),
            compression_ratio=float(params.get("compression_ratio", 0.05)),
        )
        run_outputs.append(result)
        if not result["solver_converged"]:
            warnings.append(f"solver_not_converged_run_{replicate}")

    stiffness_values = np.asarray([item["stiffness"] for item in run_outputs], dtype=float)
    anisotropy_values = np.asarray([item["anisotropy"] for item in run_outputs], dtype=float)
    connectivity_values = np.asarray([item["connectivity"] for item in run_outputs], dtype=float)
    stress_prop_values = np.asarray([item["stress_propagation"] for item in run_outputs], dtype=float)
    displacement_values = np.asarray([item["mean_displacement"] for item in run_outputs], dtype=float)
    residual_values = np.asarray([item["final_residual"] for item in run_outputs], dtype=float)
    proxy_keys = (
        "mesh_size_proxy",
        "permeability_proxy",
        "compressibility_proxy",
        "swelling_ratio_proxy",
        "loss_tangent_proxy",
        "poroelastic_time_constant_proxy",
        "strain_stiffening_proxy",
    )
    proxy_summary = {
        key: float(np.mean(np.asarray([item[key] for item in run_outputs], dtype=float)))
        for key in proxy_keys
    }
    proxy_uncertainty = {
        key: float(np.std(np.asarray([item[key] for item in run_outputs], dtype=float)))
        for key in proxy_keys
    }

    stiffness_mean = float(np.mean(stiffness_values))
    stiffness_std = float(np.std(stiffness_values))
    ci_radius = 1.96 * stiffness_std / sqrt(monte_carlo_runs)

    return {
        "stiffness_mean": stiffness_mean,
        "stiffness_std": stiffness_std,
        "confidence_interval": [float(stiffness_mean - ci_radius), float(stiffness_mean + ci_radius)],
        "anisotropy": float(np.mean(anisotropy_values)),
        "connectivity": float(np.mean(connectivity_values)),
        "stress_propagation": float(np.mean(stress_prop_values)),
        "mean_displacement": float(np.mean(displacement_values)),
        "stiffness": stiffness_mean,
        "solver_converged": bool(np.all(residual_values < float(params.get("tolerance", 1e-5)))),
        "risk_index": float(stiffness_std / max(abs(stiffness_mean), 1e-12)),
        "material_property_proxies": proxy_summary,
        "material_property_proxy_std": proxy_uncertainty,
        **proxy_summary,
        "warnings": warnings,
    }


def default_design_search_space() -> dict[str, dict[str, float]]:
    """Return default bounds for inverse ECM design search."""

    return {
        "fiber_density": {"min": 0.18, "max": 0.72},
        "fiber_stiffness": {"min": 4.0, "max": 14.0},
        "bending_stiffness": {"min": 0.05, "max": 0.45},
        "crosslink_prob": {"min": 0.2, "max": 0.85},
        "domain_size": {"min": 0.8, "max": 1.2},
    }


def design_ecm_candidates(
    targets: dict[str, Any],
    *,
    search_space: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    top_k: int = 3,
    candidate_budget: int = 12,
    monte_carlo_runs: int = 4,
    total_force: float = 0.2,
    axis: AxisName = "x",
    boundary_fraction: float = 0.15,
    seed: int = 1234,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    target_nodes: int = 8,
) -> dict[str, Any]:
    """Search ECM parameter space for candidates matching target mechanics.

    The search is deterministic for a fixed `targets`, `search_space`, and `seed`.
    It evaluates a curated anchor set plus a Halton sequence over the parameter space
    and ranks candidates by normalized target mismatch and robustness.
    """

    normalized_targets = _normalize_design_targets(targets)
    bounds = _normalize_design_search_space(search_space)
    normalized_constraints = _normalize_design_constraints(constraints)
    active_weights = _design_objective_weights(normalized_targets)
    candidate_budget = max(int(candidate_budget), max(int(top_k), 1), 6)
    top_k = max(1, int(top_k))

    evaluation_seed_base = 1234
    coarse_candidates = _build_design_candidates(bounds, candidate_budget, normalized_targets)
    coarse_evaluations = _evaluate_design_candidates(
        coarse_candidates,
        targets=normalized_targets,
        weights=active_weights,
        total_force=total_force,
        axis=axis,
        boundary_fraction=boundary_fraction,
        seed=evaluation_seed_base,
        max_iterations=max_iterations,
        tolerance=tolerance,
        monte_carlo_runs=monte_carlo_runs,
        target_nodes=target_nodes,
        constraints=normalized_constraints,
    )
    coarse_evaluations.sort(key=_design_sort_key)

    refinement_seeds = coarse_evaluations[: min(max(top_k, 2), 3)]
    refined_candidates = _refine_design_candidates(bounds, refinement_seeds)
    refined_evaluations = _evaluate_design_candidates(
        refined_candidates,
        targets=normalized_targets,
        weights=active_weights,
        total_force=total_force,
        axis=axis,
        boundary_fraction=boundary_fraction,
        seed=evaluation_seed_base + 100_000,
        max_iterations=max_iterations,
        tolerance=tolerance,
        monte_carlo_runs=monte_carlo_runs,
        target_nodes=target_nodes,
        constraints=normalized_constraints,
    )

    evaluations = _deduplicate_design_evaluations(coarse_evaluations + refined_evaluations)

    evaluations.sort(key=_design_sort_key)
    ranked = []
    for rank, row in enumerate(evaluations, start=1):
        ranked.append({**row, "rank": rank})

    return {
        "targets": normalized_targets,
        "search_space": bounds,
        "constraints": normalized_constraints,
        "objective_weights": active_weights,
        "candidate_budget": candidate_budget,
        "search_diagnostics": {
            "coarse_candidate_count": len(coarse_candidates),
            "refined_candidate_count": len(refined_candidates),
            "refinement_seed_count": len(refinement_seeds),
        },
        "evaluated_candidate_count": len(ranked),
        "feasible_candidate_count": sum(1 for row in ranked if row.get("feasible")),
        "top_k": top_k,
        "top_candidates": ranked[:top_k],
        "evaluated_candidates": ranked,
    }


def run_validation() -> dict[str, bool]:
    """Validate solver convergence, monotonicity, nonlinearity, and physics consistency."""

    base_params = default_validation_params()
    solver_probe = simulate_ecm(base_params)
    solver_converged = bool(solver_probe["solver_converged"])

    low_density = simulate_ecm({**base_params, "fiber_density": 0.2})
    high_density = simulate_ecm({**base_params, "fiber_density": 0.5})
    low_crosslink = simulate_ecm({**base_params, "crosslink_prob": 0.2})
    high_crosslink = simulate_ecm({**base_params, "crosslink_prob": 0.7})

    monotonicity_valid = bool(
        high_density["stiffness_mean"] > low_density["stiffness_mean"]
        and high_crosslink["stiffness_mean"] > low_crosslink["stiffness_mean"]
    )

    tensile = run_tensile_test(base_params)
    nonlinearity_valid = bool(tensile["nonlinearity_valid"])
    physics_valid = bool(solver_converged and monotonicity_valid and nonlinearity_valid)

    return {
        "solver_converged": solver_converged,
        "monotonicity_valid": monotonicity_valid,
        "nonlinearity_valid": nonlinearity_valid,
        "physics_valid": physics_valid,
    }


def default_validation_params() -> dict[str, Any]:
    """Return the validated reference parameter set used for physics checks."""

    return {
        "fiber_density": 0.35,
        "fiber_stiffness": 8.0,
        "bending_stiffness": 0.2,
        "crosslink_prob": 0.45,
        "domain_size": 1.0,
        "total_force": 0.2,
        "seed": 1234,
        "monte_carlo_runs": 10,
        "max_iterations": 500,
        "tolerance": 1e-5,
        "target_nodes": 8,
    }


def _normalize_design_targets(targets: dict[str, Any]) -> dict[str, float]:
    if not isinstance(targets, dict):
        raise ValueError("targets must be a dictionary.")

    normalized: dict[str, float] = {}
    if "stiffness" not in targets:
        raise ValueError("targets must include `stiffness`.")

    for key, raw_value in targets.items():
        if key not in DESIGN_OBSERVABLES:
            raise ValueError(f"Unsupported design target: {key}")
        if raw_value is None:
            continue
        value = float(raw_value)
        if key == "stiffness" and value <= 0.0:
            raise ValueError("Target stiffness must be positive.")
        if key in {"anisotropy", "connectivity", "stress_propagation"} and value < 0.0:
            raise ValueError(f"Target {key} must be non-negative.")
        if key.endswith("_proxy") and value < 0.0:
            raise ValueError(f"Target {key} must be non-negative.")
        normalized[key] = value

    if "connectivity" in normalized and normalized["connectivity"] > 1.0:
        raise ValueError("Target connectivity must be <= 1.0.")
    return normalized


def _normalize_design_search_space(search_space: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    defaults = default_design_search_space()
    if search_space is None:
        return defaults

    normalized = {name: values.copy() for name, values in defaults.items()}
    for parameter_name, candidate in search_space.items():
        if parameter_name not in normalized:
            raise ValueError(f"Unsupported design parameter in search_space: {parameter_name}")
        if isinstance(candidate, dict):
            lower = float(candidate.get("min"))
            upper = float(candidate.get("max"))
        elif isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            lower = float(candidate[0])
            upper = float(candidate[1])
        else:
            raise ValueError(
                f"search_space[{parameter_name}] must be a dict with min/max or a two-value sequence."
            )
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {parameter_name}: min must be < max.")
        normalized[parameter_name] = {"min": lower, "max": upper}
    return normalized


def _normalize_design_constraints(constraints: dict[str, Any] | None) -> dict[str, float]:
    if constraints is None:
        return {}
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a dictionary when provided.")

    normalized: dict[str, float] = {}
    for key, value in constraints.items():
        if not (key.startswith("min_") or key.startswith("max_")):
            raise ValueError(f"Unsupported design constraint: {key}")
        observable = key.removeprefix("min_").removeprefix("max_")
        if observable not in DESIGN_OBSERVABLES:
            raise ValueError(f"Unsupported design constraint: {key}")
        if value is None:
            continue
        normalized[key] = float(value)

    if "max_anisotropy" in normalized and normalized["max_anisotropy"] < 0.0:
        raise ValueError("max_anisotropy must be non-negative.")
    if "min_connectivity" in normalized:
        if not (0.0 <= normalized["min_connectivity"] <= 1.0):
            raise ValueError("min_connectivity must be in [0, 1].")
    if "max_risk_index" in normalized and normalized["max_risk_index"] < 0.0:
        raise ValueError("max_risk_index must be non-negative.")
    if "min_stress_propagation" in normalized and normalized["min_stress_propagation"] < 0.0:
        raise ValueError("min_stress_propagation must be non-negative.")
    for key, value in normalized.items():
        if key.startswith("max_") and value < 0.0:
            raise ValueError(f"{key} must be non-negative.")
    return normalized


def _design_objective_weights(targets: dict[str, float]) -> dict[str, float]:
    raw_weights = {
        "stiffness": 0.45,
        "anisotropy": 0.12 if "anisotropy" in targets else 0.0,
        "connectivity": 0.12 if "connectivity" in targets else 0.0,
        "stress_propagation": 0.1 if "stress_propagation" in targets else 0.0,
        "risk": 0.05,
    }
    extra_property_targets = [
        key
        for key in targets
        if key
        not in {"stiffness", "anisotropy", "connectivity", "stress_propagation"}
    ]
    for key in extra_property_targets:
        raw_weights[key] = 0.06
    total = sum(raw_weights.values())
    return {key: float(value / total) for key, value in raw_weights.items() if value > 0.0}


def _build_design_candidates(
    search_space: dict[str, dict[str, float]],
    candidate_budget: int,
    targets: dict[str, float],
) -> list[dict[str, float]]:
    parameter_names = list(search_space.keys())
    anchors = []
    default_params = default_validation_params()
    midpoint = {}
    for name in parameter_names:
        lower = search_space[name]["min"]
        upper = search_space[name]["max"]
        midpoint[name] = float(0.5 * (lower + upper))
    anchors.append(midpoint)
    anchors.append(
        {
            name: float(
                min(max(float(default_params.get(name, midpoint[name])), search_space[name]["min"]), search_space[name]["max"])
            )
            for name in parameter_names
        }
    )
    anchors.append(
        {
            "fiber_density": search_space["fiber_density"]["max"],
            "fiber_stiffness": midpoint["fiber_stiffness"],
            "bending_stiffness": midpoint["bending_stiffness"],
            "crosslink_prob": midpoint["crosslink_prob"],
            "domain_size": midpoint["domain_size"],
        }
    )
    anchors.append(
        {
            "fiber_density": midpoint["fiber_density"],
            "fiber_stiffness": midpoint["fiber_stiffness"],
            "bending_stiffness": midpoint["bending_stiffness"],
            "crosslink_prob": search_space["crosslink_prob"]["max"],
            "domain_size": midpoint["domain_size"],
        }
    )
    anchors.extend(_target_guided_design_anchors(search_space, targets))

    candidates: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    def add_candidate(params: dict[str, float]) -> None:
        key = tuple(round(float(params[name]), 6) for name in parameter_names)
        if key in seen:
            return
        seen.add(key)
        candidates.append({name: float(params[name]) for name in parameter_names})

    for anchor in anchors:
        add_candidate(anchor)

    remaining = max(candidate_budget - len(candidates), 0)
    if remaining > 0:
        sampler = qmc.Halton(d=len(parameter_names), scramble=False)
        for sample in sampler.random(remaining * 3):
            params = {}
            for index, name in enumerate(parameter_names):
                lower = search_space[name]["min"]
                upper = search_space[name]["max"]
                params[name] = float(lower + (upper - lower) * sample[index])
            add_candidate(params)
            if len(candidates) >= candidate_budget:
                break

    return candidates[:candidate_budget]


def _target_guided_design_anchors(
    search_space: dict[str, dict[str, float]],
    targets: dict[str, float],
) -> list[dict[str, float]]:
    reference = default_validation_params()
    reference_stiffness = 6.086052061691116
    target_ratio = max(float(targets["stiffness"]) / max(reference_stiffness, 1e-6), 0.2)
    stiffness_scale = min(max(target_ratio, 0.6), 1.7)
    density_scale = min(max(sqrt(target_ratio), 0.75), 1.35)

    heuristics = [
        {
            "fiber_density": float(reference["fiber_density"]) * density_scale,
            "fiber_stiffness": float(reference["fiber_stiffness"]) * stiffness_scale,
            "bending_stiffness": float(reference["bending_stiffness"]) * max(0.8, density_scale),
            "crosslink_prob": float(reference["crosslink_prob"]) * min(max(target_ratio, 0.75), 1.4),
            "domain_size": float(reference["domain_size"]),
        },
        {
            "fiber_density": float(reference["fiber_density"]) * max(0.9, density_scale * 0.95),
            "fiber_stiffness": float(reference["fiber_stiffness"]) * min(max(stiffness_scale * 1.1, 0.8), 1.8),
            "bending_stiffness": float(reference["bending_stiffness"]) * min(max(stiffness_scale, 0.8), 1.6),
            "crosslink_prob": float(reference["crosslink_prob"]) * min(max(target_ratio * 0.9, 0.7), 1.5),
            "domain_size": float(reference["domain_size"]) * 0.95,
        },
    ]
    return [_clamp_design_params(params, search_space) for params in heuristics]


def _evaluate_design_candidates(
    candidate_params: Sequence[dict[str, float]],
    *,
    targets: dict[str, float],
    weights: dict[str, float],
    total_force: float,
    axis: AxisName,
    boundary_fraction: float,
    seed: int,
    max_iterations: int,
    tolerance: float,
    monte_carlo_runs: int,
    target_nodes: int,
    constraints: dict[str, float],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for index, parameters in enumerate(candidate_params):
        simulation_payload = simulate_ecm(
            {
                **parameters,
                "total_force": float(total_force),
                "axis": axis,
                "boundary_fraction": float(boundary_fraction),
                "seed": int(seed + 97 * index),
                "max_iterations": int(max_iterations),
                "tolerance": float(tolerance),
                "monte_carlo_runs": int(monte_carlo_runs),
                "target_nodes": int(target_nodes),
            }
        )
        evaluations.append(
            _score_design_candidate(
                parameters=parameters,
                simulation=simulation_payload,
                targets=targets,
                weights=weights,
                constraints=constraints,
            )
        )
    return evaluations


def _refine_design_candidates(
    search_space: dict[str, dict[str, float]],
    seed_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, float]]:
    refined: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    parameter_names = list(search_space.keys())

    def add(params: dict[str, float]) -> None:
        clamped = _clamp_design_params(params, search_space)
        key = tuple(round(float(clamped[name]), 6) for name in parameter_names)
        if key in seen:
            return
        seen.add(key)
        refined.append(clamped)

    for row in seed_candidates:
        base = dict(row["parameters"])
        add(base)
        density_span = search_space["fiber_density"]["max"] - search_space["fiber_density"]["min"]
        stiffness_span = search_space["fiber_stiffness"]["max"] - search_space["fiber_stiffness"]["min"]
        bending_span = search_space["bending_stiffness"]["max"] - search_space["bending_stiffness"]["min"]
        crosslink_span = search_space["crosslink_prob"]["max"] - search_space["crosslink_prob"]["min"]
        domain_span = search_space["domain_size"]["max"] - search_space["domain_size"]["min"]

        tweaks = [
            {"fiber_stiffness": base["fiber_stiffness"] + 0.12 * stiffness_span},
            {"fiber_stiffness": base["fiber_stiffness"] - 0.12 * stiffness_span},
            {"fiber_density": base["fiber_density"] + 0.1 * density_span},
            {"fiber_density": base["fiber_density"] - 0.1 * density_span},
            {"crosslink_prob": base["crosslink_prob"] + 0.1 * crosslink_span},
            {"crosslink_prob": base["crosslink_prob"] - 0.1 * crosslink_span},
            {
                "fiber_density": base["fiber_density"] + 0.06 * density_span,
                "fiber_stiffness": base["fiber_stiffness"] + 0.08 * stiffness_span,
                "crosslink_prob": base["crosslink_prob"] + 0.08 * crosslink_span,
                "bending_stiffness": base["bending_stiffness"] + 0.08 * bending_span,
            },
            {
                "fiber_density": base["fiber_density"] - 0.06 * density_span,
                "fiber_stiffness": base["fiber_stiffness"] - 0.08 * stiffness_span,
                "crosslink_prob": base["crosslink_prob"] - 0.08 * crosslink_span,
                "domain_size": base["domain_size"] + 0.08 * domain_span,
            },
        ]
        for tweak in tweaks:
            add({**base, **tweak})

    return refined


def _clamp_design_params(
    params: dict[str, float],
    search_space: dict[str, dict[str, float]],
) -> dict[str, float]:
    clamped = {}
    for name, bounds in search_space.items():
        clamped[name] = float(min(max(float(params[name]), bounds["min"]), bounds["max"]))
    return clamped


def _deduplicate_design_evaluations(evaluations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[float, ...], dict[str, Any]] = {}
    for row in evaluations:
        params = row["parameters"]
        key = tuple(
            round(float(params[name]), 6)
            for name in ("fiber_density", "fiber_stiffness", "bending_stiffness", "crosslink_prob", "domain_size")
        )
        existing = deduped.get(key)
        if existing is None or _design_sort_key(row) < _design_sort_key(existing):
            deduped[key] = row
    return list(deduped.values())


def _score_design_candidate(
    *,
    parameters: dict[str, float],
    simulation: dict[str, Any],
    targets: dict[str, float],
    weights: dict[str, float],
    constraints: dict[str, float],
) -> dict[str, Any]:
    components = {
        "stiffness": abs(float(simulation["stiffness_mean"]) - float(targets["stiffness"]))
        / max(abs(float(targets["stiffness"])), 1e-12),
        "risk": float(simulation["risk_index"]),
    }
    for observable in targets:
        if observable == "stiffness":
            continue
        if observable == "risk_index":
            components["risk_index"] = abs(float(simulation["risk_index"]) - float(targets["risk_index"])) / max(
                float(targets["risk_index"]),
                0.05,
            )
            continue
        if observable in simulation:
            components[observable] = abs(float(simulation[observable]) - float(targets[observable])) / max(
                abs(float(targets[observable])),
                0.05,
            )

    constraint_violations = _design_constraint_violations(simulation, constraints)
    feasible = bool(not constraint_violations and simulation.get("solver_converged", False))
    score = float(sum(weights[key] * components.get(key, 0.0) for key in weights))
    if not bool(simulation.get("solver_converged", False)):
        score += 10.0
    score += 3.0 * sum(constraint_violations.values())
    score += 0.25 * len(simulation.get("warnings", []))

    summary_bits = [
        f"stiffness={simulation['stiffness_mean']:.3f}",
        f"anisotropy={simulation['anisotropy']:.3f}",
        f"connectivity={simulation['connectivity']:.3f}",
        f"risk={simulation['risk_index']:.3f}",
    ]
    return {
        "parameters": parameters,
        "features": {
            "stiffness_mean": float(simulation["stiffness_mean"]),
            "stiffness_std": float(simulation["stiffness_std"]),
            "confidence_interval": list(simulation["confidence_interval"]),
            "anisotropy": float(simulation["anisotropy"]),
            "connectivity": float(simulation["connectivity"]),
            "stress_propagation": float(simulation["stress_propagation"]),
            "mean_displacement": float(simulation["mean_displacement"]),
            "risk_index": float(simulation["risk_index"]),
            "solver_converged": bool(simulation["solver_converged"]),
            "material_property_proxies": dict(simulation.get("material_property_proxies", {})),
            "material_property_proxy_std": dict(simulation.get("material_property_proxy_std", {})),
            **{
                key: float(simulation[key])
                for key in (
                    "mesh_size_proxy",
                    "permeability_proxy",
                    "compressibility_proxy",
                    "swelling_ratio_proxy",
                    "loss_tangent_proxy",
                    "poroelastic_time_constant_proxy",
                    "strain_stiffening_proxy",
                )
                if key in simulation
            },
            "warnings": list(simulation.get("warnings", [])),
        },
        "errors": {key: float(value) for key, value in components.items()},
        "feasible": feasible,
        "constraint_violations": {key: float(value) for key, value in constraint_violations.items()},
        "score": score,
        "match_summary": ", ".join(summary_bits),
    }


def _design_constraint_violations(
    simulation: dict[str, Any],
    constraints: dict[str, float],
) -> dict[str, float]:
    violations: dict[str, float] = {}
    for constraint_name, limit_value in constraints.items():
        if constraint_name.startswith("max_"):
            observable = constraint_name.removeprefix("max_")
            if observable not in simulation:
                continue
            observed = float(simulation[observable])
            limit = float(limit_value)
            if observed > limit:
                violations[constraint_name] = (observed - limit) / max(abs(limit), 0.05)
        elif constraint_name.startswith("min_"):
            observable = constraint_name.removeprefix("min_")
            if observable not in simulation:
                continue
            observed = float(simulation[observable])
            floor = float(limit_value)
            if observed < floor:
                violations[constraint_name] = (floor - observed) / max(abs(floor), 0.05)
    return violations


def _design_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    params = candidate["parameters"]
    features = candidate["features"]
    errors = candidate["errors"]
    return (
        0.0 if candidate.get("feasible") else 1.0,
        float(candidate["score"]),
        float(features["risk_index"]),
        -float(features["connectivity"]),
        float(errors.get("stiffness", 0.0)),
        float(params["fiber_density"]),
        float(params["crosslink_prob"]),
        float(params["fiber_stiffness"]),
    )


def visualize_network(
    network: FiberNetwork,
    *,
    deformed: bool = True,
    show_boundaries: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Render the fiber network in 3D."""

    positions = network.positions if deformed else network.rest_positions
    if ax is None:
        figure = plt.figure(figsize=(8, 7))
        ax = figure.add_subplot(111, projection="3d")

    for i, j in network.edges:
        segment = positions[[i, j]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#1f6f6a", alpha=0.5, linewidth=0.9)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=10, color="#1e3a5f", alpha=0.85)
    if show_boundaries and np.any(network.fixed_mask):
        fixed = positions[network.fixed_mask]
        ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], s=24, color="#b54708", label="fixed")
    if show_boundaries and np.any(network.loaded_mask):
        loaded = positions[network.loaded_mask]
        ax.scatter(loaded[:, 0], loaded[:, 1], loaded[:, 2], s=24, color="#0f766e", label="loaded")

    ax.set_xlim(0.0, network.domain_size)
    ax.set_ylim(0.0, network.domain_size)
    ax.set_zlim(0.0, network.domain_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Deformed ECM Fiber Network" if deformed else "Reference ECM Fiber Network")
    if show_boundaries:
        ax.legend(loc="upper right")
    return ax


def visualize_stress_strain(
    stress_strain_curve: Sequence[dict[str, float]],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a stress-strain curve with optional uncertainty shading."""

    strains = np.asarray([point["strain"] for point in stress_strain_curve], dtype=float)
    stresses = np.asarray([point["stress"] for point in stress_strain_curve], dtype=float)
    stds = np.asarray([point.get("stress_std", 0.0) for point in stress_strain_curve], dtype=float)

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 5))

    ax.plot(strains, stresses, color="#0f766e", linewidth=2.0, marker="o")
    if np.any(stds > 0.0):
        ax.fill_between(strains, stresses - stds, stresses + stds, color="#0f766e", alpha=0.18)
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress")
    ax.set_title("ECM Stress-Strain Response")
    return ax


def main() -> None:
    """Run a research-grade example and display the stress-strain response."""

    params = {
        "fiber_density": 0.35,
        "fiber_stiffness": 8.0,
        "bending_stiffness": 0.2,
        "crosslink_prob": 0.45,
        "domain_size": 1.0,
        "seed": 1234,
        "monte_carlo_runs": 10,
        "max_iterations": 500,
        "tolerance": 1e-5,
    }
    result = simulate_ecm(params)
    tensile = run_tensile_test(params)
    validation = run_validation()

    print(json.dumps({"simulation": result, "validation": validation}, ensure_ascii=False, indent=2))
    visualize_stress_strain(tensile["stress_strain_curve"])
    plt.tight_layout()
    plt.show()


def _solver_summary_dict(diagnostics: SolverDiagnostics | None) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    return {
        "converged": diagnostics.converged,
        "attempts_used": diagnostics.attempts_used,
        "iterations": diagnostics.iterations,
        "final_energy": diagnostics.final_energy,
        "final_max_force": diagnostics.final_max_force,
        "iteration_history": diagnostics.iteration_history,
        "force_residual_curve": diagnostics.force_residual_curve,
        "energy_curve": diagnostics.energy_curve,
    }


def _axis_to_index(axis: AxisName | str) -> int:
    mapping = {"x": 0, "y": 1, "z": 2}
    if axis not in mapping:
        raise ValueError(f"Unsupported axis: {axis}")
    return mapping[axis]


def _boundary_masks(
    positions: np.ndarray,
    axis_index: int,
    domain_size: float,
    boundary_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.02 <= boundary_fraction < 0.45):
        raise ValueError("boundary_fraction must be in [0.02, 0.45).")
    threshold = boundary_fraction * domain_size
    axis_values = positions[:, axis_index]
    fixed_mask = axis_values <= (axis_values.min() + threshold)
    loaded_mask = axis_values >= (axis_values.max() - threshold)
    if fixed_mask.sum() < 2:
        fixed_mask[np.argsort(axis_values)[:2]] = True
    if loaded_mask.sum() < 2:
        loaded_mask[np.argsort(axis_values)[-2:]] = True
    return fixed_mask, loaded_mask

def _nearest_neighbor_edges(positions: np.ndarray, tree: cKDTree) -> list[tuple[int, int]]:
    _, indices = tree.query(positions, k=min(4, len(positions)))
    edges = set()
    for i in range(len(positions)):
        for j in np.atleast_1d(indices[i])[1:]:
            edge = tuple(sorted((int(i), int(j))))
            if edge[0] != edge[1]:
                edges.add(edge)
    return sorted(edges)


def _ensure_connected(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    node_count = positions.shape[0]
    edges_list = {tuple(edge) for edge in np.asarray(edges, dtype=int)}
    while True:
        adjacency = _adjacency_matrix(np.asarray(sorted(edges_list), dtype=int), node_count)
        n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
        if n_components <= 1:
            break
        component_ids = np.unique(labels)
        primary_nodes = np.where(labels == component_ids[0])[0]
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for component in component_ids[1:]:
            component_nodes = np.where(labels == component)[0]
            diffs = positions[primary_nodes][:, None, :] - positions[component_nodes][None, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            loc = np.unravel_index(int(np.argmin(distances)), distances.shape)
            distance = float(distances[loc])
            if distance < best_distance:
                best_distance = distance
                best_pair = tuple(sorted((int(primary_nodes[loc[0]]), int(component_nodes[loc[1]]))))
        if best_pair is None:
            break
        edges_list.add(best_pair)
    return np.asarray(sorted(edges_list), dtype=int)


def _adjacency_matrix(edges: np.ndarray, node_count: int) -> csr_matrix:
    if edges.size == 0:
        return csr_matrix((node_count, node_count))
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones_like(row, dtype=float)
    return csr_matrix((data, (row, col)), shape=(node_count, node_count))


def _build_bending_triplets(positions: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(len(positions))]
    for i, j in edges:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    triplets = []
    rest_angles = []
    for center, neighbors in enumerate(adjacency):
        unique_neighbors = sorted(set(neighbors))
        if len(unique_neighbors) < 2:
            continue
        for left_index in range(len(unique_neighbors)):
            for right_index in range(left_index + 1, len(unique_neighbors)):
                i = unique_neighbors[left_index]
                k = unique_neighbors[right_index]
                a = positions[i] - positions[center]
                b = positions[k] - positions[center]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a < 1e-12 or norm_b < 1e-12:
                    continue
                cos_theta = float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
                theta = float(np.arccos(cos_theta))
                triplets.append((i, center, k))
                rest_angles.append(theta)
    return np.asarray(triplets, dtype=int), np.asarray(rest_angles, dtype=float)


def _spring_force_and_energy(network: FiberNetwork, length: float, rest_length: float) -> tuple[float, float]:
    effective_stiffness = network.fiber_stiffness / max(rest_length, 1e-8)
    delta = length - rest_length
    if delta >= 0.0:
        exponent = float(np.clip(network.nonlinearity_alpha * delta, -50.0, 50.0))
        exp_term = float(np.exp(exponent))
        force = effective_stiffness * (exp_term - 1.0)
        energy = (effective_stiffness / network.nonlinearity_alpha) * (
            exp_term - 1.0 - exponent
        )
        return float(force), float(energy)

    k_compress = network.compression_ratio * effective_stiffness
    force = k_compress * delta
    energy = 0.5 * k_compress * delta * delta
    return float(force), float(energy)


def _stretch_energy(network: FiberNetwork, positions: np.ndarray) -> float:
    total = 0.0
    edge_vectors = positions[network.edges[:, 1]] - positions[network.edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    for length, rest_length in zip(lengths, network.rest_lengths):
        _, energy = _spring_force_and_energy(network, float(length), float(rest_length))
        total += energy
    return float(total)


def _bending_energy(network: FiberNetwork, positions: np.ndarray) -> float:
    if network.bending_stiffness <= 0.0 or network.bending_triplets.size == 0:
        return 0.0
    total = 0.0
    for triplet, theta0 in zip(network.bending_triplets, network.rest_angles):
        i, j, k = triplet
        a = positions[i] - positions[j]
        b = positions[k] - positions[j]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            continue
        cos_theta = float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        delta = theta - theta0
        total += network.bending_stiffness * delta * delta
    return float(total)


def _external_work(network: FiberNetwork, positions: np.ndarray) -> float:
    if network.external_forces is None:
        return 0.0
    displacement = positions - network.rest_positions
    return float(np.sum(network.external_forces * displacement))


def _internal_forces(network: FiberNetwork, positions: np.ndarray) -> np.ndarray:
    forces = np.zeros_like(positions)

    edge_vectors = positions[network.edges[:, 1]] - positions[network.edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    safe_lengths = np.maximum(lengths, 1e-12)
    directions = edge_vectors / safe_lengths[:, None]
    for edge_index, (i, j) in enumerate(network.edges):
        magnitude, _ = _spring_force_and_energy(network, float(lengths[edge_index]), float(network.rest_lengths[edge_index]))
        force_vector = magnitude * directions[edge_index]
        forces[i] += force_vector
        forces[j] -= force_vector

    if network.bending_stiffness > 0.0 and network.bending_triplets.size:
        for idx, (i, j, k) in enumerate(network.bending_triplets):
            a = positions[i] - positions[j]
            b = positions[k] - positions[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-12 or norm_b < 1e-12:
                continue
            u = a / norm_a
            v = b / norm_b
            cos_theta = float(np.clip(np.dot(u, v), -1.0, 1.0))
            theta = float(np.arccos(cos_theta))
            sin_theta = max(sqrt(max(1.0 - cos_theta * cos_theta, 0.0)), 1e-8)
            delta = theta - float(network.rest_angles[idx])
            grad_i_cos = (v - cos_theta * u) / norm_a
            grad_k_cos = (u - cos_theta * v) / norm_b
            grad_j_cos = -(grad_i_cos + grad_k_cos)
            factor = 2.0 * network.bending_stiffness * delta / sin_theta
            forces[i] += factor * grad_i_cos
            forces[j] += factor * grad_j_cos
            forces[k] += factor * grad_k_cos

    return forces


def _reaction_force(network: FiberNetwork) -> float:
    internal = _internal_forces(network, network.positions)
    return float(abs(np.sum(internal[network.loaded_mask, network.applied_axis])))


def _net_forces(network: FiberNetwork, positions: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for total nodal forces."""

    return compute_forces(network, positions, include_external=True, zero_constrained=False)


def _total_potential_energy(network: FiberNetwork, positions: np.ndarray) -> float:
    """Backward-compatible alias for the loaded objective."""

    return compute_energy(network, positions) - (0.0 if network.loading_mode == "strain" else _external_work(network, positions))


def _objective(network: FiberNetwork, free_vector: np.ndarray, free_mask: np.ndarray) -> float:
    positions = _positions_from_vector(network, free_vector, free_mask)
    return compute_energy(network, positions) - (0.0 if network.loading_mode == "strain" else _external_work(network, positions))


def _objective_gradient(network: FiberNetwork, free_vector: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    positions = _positions_from_vector(network, free_vector, free_mask)
    total_forces = compute_forces(network, positions, include_external=(network.loading_mode == "force"), zero_constrained=False)
    return -total_forces[free_mask].reshape(-1)


def _positions_from_vector(network: FiberNetwork, free_vector: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    positions = network.constrained_positions.copy()
    positions[free_mask] = free_vector.reshape((-1, 3))
    return positions


def _initialize_positions_for_attempt(
    network: FiberNetwork,
    free_mask: np.ndarray,
    rng: np.random.Generator,
    perturbation_scale: float,
    attempt: int,
) -> np.ndarray:
    positions = network.constrained_positions.copy()
    positions[free_mask] = network.rest_positions[free_mask]
    if attempt > 1:
        positions[free_mask] += rng.normal(
            scale=perturbation_scale * network.domain_size * min(attempt, 3),
            size=positions[free_mask].shape,
        )
    return positions


def _largest_component_fraction(edges: np.ndarray, node_count: int) -> float:
    adjacency = _adjacency_matrix(edges, node_count)
    component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
    if component_count == 0:
        return 0.0
    counts = np.bincount(labels, minlength=component_count)
    return float(counts.max() / max(node_count, 1))


__all__ = [
    "FiberNetwork",
    "SolverDiagnostics",
    "apply_force",
    "apply_strain",
    "compute_energy",
    "compute_features",
    "compute_forces",
    "generate_network",
    "run_parameter_scan",
    "run_simulation",
    "run_tensile_test",
    "run_validation",
    "default_validation_params",
    "simulate_ecm",
    "solve_equilibrium",
    "visualize_network",
    "visualize_stress_strain",
]


if __name__ == "__main__":
    main()
