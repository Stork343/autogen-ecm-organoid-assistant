from __future__ import annotations

import unittest

from tests.febio_test_support import PROJECT_ROOT  # noqa: F401

from ecm_organoid_agent.febio.mapping import CandidateSimulationMappingOptions, candidate_to_simulation_request
from ecm_organoid_agent.febio.schemas import (
    OrganoidSpheroidRequest,
    SingleCellContractionRequest,
    simulation_request_from_dict,
)


class FEBioSchemaValidationTests(unittest.TestCase):
    def test_simulation_request_from_dict_validates_bulk_payload(self) -> None:
        request = simulation_request_from_dict(
            {
                "scenario": "bulk_mechanics",
                "matrix_youngs_modulus": 8.0,
                "matrix_poisson_ratio": 0.3,
                "sample_dimensions": [1.0, 1.0, 1.0],
                "prescribed_displacement": -0.1,
            }
        )
        self.assertEqual(request.scenario, "bulk_mechanics")
        self.assertEqual(request.sample_dimensions, (1.0, 1.0, 1.0))

    def test_invalid_poisson_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            simulation_request_from_dict(
                {
                    "scenario": "organoid_spheroid",
                    "matrix_youngs_modulus": 8.0,
                    "matrix_poisson_ratio": 0.6,
                }
            )

    def test_invalid_mesh_resolution_raises(self) -> None:
        with self.assertRaises(ValueError):
            simulation_request_from_dict(
                {
                    "scenario": "bulk_mechanics",
                    "matrix_youngs_modulus": 8.0,
                    "matrix_poisson_ratio": 0.3,
                    "mesh_resolution": [0, 4, 4],
                }
            )

    def test_invalid_single_cell_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            SingleCellContractionRequest(matrix_extent=1.0, cell_radius=0.6)

    def test_invalid_organoid_displacement_raises(self) -> None:
        with self.assertRaises(ValueError):
            OrganoidSpheroidRequest(organoid_radial_displacement=0.0)

    def test_single_cell_target_fields_are_validated(self) -> None:
        request = SingleCellContractionRequest(
            matrix_youngs_modulus=8.0,
            matrix_poisson_ratio=0.3,
            target_stress_propagation_distance=0.25,
            target_strain_heterogeneity=0.35,
        )
        self.assertEqual(request.loading_mode, "radial_contractility")


class FEBioMappingTests(unittest.TestCase):
    def test_candidate_to_simulation_request_maps_single_cell_request(self) -> None:
        candidate = {
            "rank": 1,
            "score": 0.21,
            "parameters": {"domain_size": 1.2, "fiber_density": 0.35},
            "features": {"stiffness_mean": 8.5, "stress_propagation": 0.7, "anisotropy": 0.12, "connectivity": 0.95},
        }
        request = candidate_to_simulation_request(
            candidate,
            options=CandidateSimulationMappingOptions(
                scenario="single_cell_contraction",
                target_stiffness=8.0,
                cell_contractility=0.03,
            ),
        )
        self.assertEqual(request.scenario, "single_cell_contraction")
        self.assertEqual(request.cell_contractility, 0.03)
        self.assertIn("mapping_version", request.metadata)


if __name__ == "__main__":
    unittest.main()
