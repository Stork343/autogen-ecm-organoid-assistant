from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.fiber_network import (
    _net_forces,
    _total_potential_energy,
    apply_force,
    compute_features,
    default_validation_params,
    design_ecm_candidates,
    generate_network,
    run_parameter_scan,
    run_simulation,
    simulate_ecm,
    solve_equilibrium,
)


class FiberNetworkTests(unittest.TestCase):
    def test_generate_network_produces_edges_and_positions(self) -> None:
        network = generate_network(
            fiber_density=0.3,
            fiber_stiffness=10.0,
            bending_stiffness=0.2,
            crosslink_prob=0.4,
            domain_size=1.0,
            seed=1,
        )
        self.assertEqual(network.rest_positions.shape[1], 3)
        self.assertGreater(network.edges.shape[0], 0)
        self.assertEqual(network.edges.shape[1], 2)
        self.assertGreaterEqual(network.edges.min(), 0)
        self.assertLess(network.edges.max(), network.rest_positions.shape[0])
        self.assertFalse(np.any(network.edges[:, 0] == network.edges[:, 1]))

    def test_compute_energy_positive_and_increases_with_deformation(self) -> None:
        network = generate_network(
            fiber_density=0.3,
            fiber_stiffness=10.0,
            bending_stiffness=0.2,
            crosslink_prob=0.4,
            domain_size=1.0,
            seed=2,
        )
        reference = network.rest_positions.copy()
        small_def = reference.copy()
        large_def = reference.copy()
        small_def[0, 0] += 0.01
        large_def[0, 0] += 0.05

        e0 = _total_potential_energy(network, reference)
        e1 = _total_potential_energy(network, small_def)
        e2 = _total_potential_energy(network, large_def)

        self.assertGreaterEqual(e0, 0.0)
        self.assertGreater(e1, 0.0)
        self.assertGreater(e2, e1)

    def test_compute_forces_match_negative_energy_gradient(self) -> None:
        network = generate_network(
            fiber_density=0.3,
            fiber_stiffness=8.0,
            bending_stiffness=0.1,
            crosslink_prob=0.45,
            domain_size=1.0,
            seed=4,
        )
        positions = network.rest_positions.copy()
        positions[3, 1] += 0.02
        analytic_forces = _net_forces(network, positions)

        eps = 1e-6
        plus = positions.copy()
        minus = positions.copy()
        plus[3, 1] += eps
        minus[3, 1] -= eps
        energy_plus = _total_potential_energy(network, plus)
        energy_minus = _total_potential_energy(network, minus)
        numerical_gradient = (energy_plus - energy_minus) / (2.0 * eps)

        self.assertAlmostEqual(analytic_forces[3, 1], -numerical_gradient, places=4)

    def test_solve_equilibrium_runs_and_features_are_finite(self) -> None:
        network = generate_network(
            fiber_density=0.35,
            fiber_stiffness=12.0,
            bending_stiffness=0.1,
            crosslink_prob=0.5,
            domain_size=1.0,
            seed=7,
        )
        apply_force(network, total_force=1.0, axis="x")
        solve_equilibrium(network, max_iterations=120, tolerance=1e-5)
        features = compute_features(network)
        self.assertTrue(np.isfinite(network.last_energy))
        self.assertTrue(np.isfinite(network.last_max_force))
        self.assertLess(network.last_max_force, 5e-2)
        self.assertFalse(np.isnan(network.positions).any())

        self.assertTrue(
            {
                "stiffness",
                "anisotropy",
                "connectivity",
                "mean_displacement",
                "stress_propagation",
                "mesh_size_proxy",
                "permeability_proxy",
                "compressibility_proxy",
                "swelling_ratio_proxy",
                "loss_tangent_proxy",
                "poroelastic_time_constant_proxy",
                "strain_stiffening_proxy",
            }.issubset(features.keys())
        )
        for value in features.values():
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_run_simulation_returns_required_feature_dict(self) -> None:
        result = run_simulation(
            fiber_density=0.25,
            fiber_stiffness=8.0,
            bending_stiffness=0.0,
            crosslink_prob=0.35,
            domain_size=1.0,
            seed=11,
            max_iterations=80,
        )
        self.assertTrue(
            {
                "stiffness",
                "anisotropy",
                "connectivity",
                "mean_displacement",
                "stress_propagation",
                "material_property_proxies",
                "material_property_proxy_std",
                "mesh_size_proxy",
                "permeability_proxy",
                "compressibility_proxy",
                "swelling_ratio_proxy",
                "loss_tangent_proxy",
                "poroelastic_time_constant_proxy",
                "strain_stiffening_proxy",
                "solver_converged",
                "final_residual",
                "iteration_history",
                "force_residual_curve",
            }.issubset(result.keys())
        )

    def test_run_parameter_scan_returns_sensitivity_summary(self) -> None:
        result = run_parameter_scan(
            fiber_density=0.25,
            fiber_stiffness=8.0,
            bending_stiffness=0.1,
            crosslink_prob=0.35,
            domain_size=1.0,
            seed=13,
            max_iterations=60,
            monte_carlo_runs=3,
        )
        self.assertIn("base_result", result)
        self.assertIn("scan_results", result)
        self.assertIn("sensitivity_ranking", result)
        self.assertIn("fiber_density", result["scan_results"])
        self.assertGreater(len(result["sensitivity_ranking"]), 0)

    def test_design_ecm_candidates_returns_ranked_top_k(self) -> None:
        result = design_ecm_candidates(
            {"stiffness": 8.0, "anisotropy": 0.1, "connectivity": 0.95},
            constraints={"max_anisotropy": 0.3, "min_connectivity": 0.9},
            top_k=3,
            candidate_budget=6,
            monte_carlo_runs=2,
            target_nodes=8,
            max_iterations=200,
        )
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(len(result["top_candidates"]), 3)
        ranks = [candidate["rank"] for candidate in result["top_candidates"]]
        self.assertEqual(ranks, [1, 2, 3])
        self.assertIn("search_diagnostics", result)
        self.assertGreaterEqual(result["search_diagnostics"]["coarse_candidate_count"], 6)
        self.assertGreater(result["search_diagnostics"]["refined_candidate_count"], 0)
        self.assertIn("constraints", result)
        self.assertIn("feasible_candidate_count", result)
        self.assertLessEqual(
            result["top_candidates"][0]["score"],
            result["top_candidates"][1]["score"],
        )
        self.assertIn("parameters", result["top_candidates"][0])
        self.assertIn("features", result["top_candidates"][0])
        self.assertIn("feasible", result["top_candidates"][0])
        self.assertIn("constraint_violations", result["top_candidates"][0])
        self.assertLess(
            abs(result["top_candidates"][0]["features"]["stiffness_mean"] - 8.0),
            2.5,
        )

    def test_design_ecm_candidates_accepts_property_targets_and_constraints(self) -> None:
        baseline = simulate_ecm({**default_validation_params(), "monte_carlo_runs": 1})
        result = design_ecm_candidates(
            {
                "stiffness": float(baseline["stiffness_mean"]),
                "loss_tangent_proxy": float(baseline["loss_tangent_proxy"]),
                "mesh_size_proxy": float(baseline["mesh_size_proxy"]),
            },
            constraints={
                "max_loss_tangent_proxy": float(baseline["loss_tangent_proxy"]) * 1.2,
                "min_mesh_size_proxy": float(baseline["mesh_size_proxy"]) * 0.8,
            },
            top_k=2,
            candidate_budget=6,
            monte_carlo_runs=1,
            target_nodes=8,
            max_iterations=150,
        )
        self.assertIn("loss_tangent_proxy", result["targets"])
        self.assertIn("mesh_size_proxy", result["targets"])
        self.assertIn("max_loss_tangent_proxy", result["constraints"])
        self.assertIn("min_mesh_size_proxy", result["constraints"])
        self.assertIn("loss_tangent_proxy", result["top_candidates"][0]["features"])
        self.assertIn("mesh_size_proxy", result["top_candidates"][0]["features"])


if __name__ == "__main__":
    unittest.main()
