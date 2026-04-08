from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.febio_test_support import PROJECT_ROOT, write_fake_bulk_output_bundle  # noqa: F401

from ecm_organoid_agent.febio.builder import build_simulation_input
from ecm_organoid_agent.febio.metrics import calculate_simulation_metrics
from ecm_organoid_agent.febio.parser import parse_simulation_outputs
from ecm_organoid_agent.febio.schemas import OrganoidSpheroidRequest


class FEBioMetricsTests(unittest.TestCase):
    def test_bulk_metrics_include_stiffness_peak_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build, runner = write_fake_bulk_output_bundle(simulation_dir)
            result = parse_simulation_outputs(build_artifacts=build, runner_result=runner)
            metrics = calculate_simulation_metrics(result, simulation_dir=simulation_dir)
            self.assertAlmostEqual(metrics["effective_stiffness"], 8.0, places=6)
            self.assertAlmostEqual(metrics["peak_stress"], 0.4, places=6)
            self.assertTrue(metrics["feasibility_flags"]["solver_converged"])
            self.assertEqual(metrics["target_mismatch_score"], 0.0)

    def test_organoid_metrics_include_suitability_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build = build_simulation_input(
                OrganoidSpheroidRequest(
                    matrix_youngs_modulus=8.0,
                    matrix_poisson_ratio=0.3,
                    matrix_extent=1.0,
                    organoid_radius=0.18,
                    organoid_radial_displacement=0.03,
                    target_stiffness=8.0,
                ),
                simulation_dir,
            )
            metadata = json.loads(build.metadata_path.read_text(encoding="utf-8"))
            node_ids = sorted(int(key) for key in metadata["mesh"]["nodes"].keys())
            interface_ids = set(metadata["node_sets"]["interface_nodes"])
            matrix_element_ids = metadata["element_sets"]["matrix_domain"]
            result_payload = {
                "status": "succeeded",
                "request": build.request.to_dict(),
                "mesh_metadata": metadata["mesh"],
                "node_sets": metadata["node_sets"],
                "element_sets": metadata["element_sets"],
                "warnings": [],
                "extracted_fields": {
                    "solver_converged": True,
                    "node_displacement": {
                        "final": {
                            "ids": node_ids,
                            "values": [
                                [0.0, 0.0, 0.03 if node_id in interface_ids else 0.002]
                                for node_id in node_ids
                            ],
                        }
                    },
                    "element_principal_stress": {
                        "final": {
                            "ids": matrix_element_ids,
                            "values": [[0.05, -0.12, -0.3] for _ in matrix_element_ids],
                        }
                    },
                    "top_reaction": {"final": {"ids": [], "values": []}},
                },
            }
            metrics = calculate_simulation_metrics(result_payload, simulation_dir=simulation_dir)
            self.assertIn("candidate_suitability_score_components", metrics)
            self.assertIsNotNone(metrics["candidate_suitability_score_components"])
            self.assertIn("suitability_score", metrics["candidate_suitability_score_components"])
            self.assertIsNotNone(metrics["stress_propagation_distance"])
            self.assertIn("compression_tension_region_summary", metrics)


if __name__ == "__main__":
    unittest.main()
