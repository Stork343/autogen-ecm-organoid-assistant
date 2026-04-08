from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.febio_test_support import PROJECT_ROOT, load_build_metadata  # noqa: F401

from ecm_organoid_agent.febio.builder import build_simulation_input
from ecm_organoid_agent.febio.schemas import BulkMechanicsRequest, OrganoidSpheroidRequest, SingleCellContractionRequest


class FEBioBuilderTests(unittest.TestCase):
    def test_build_bulk_input_writes_fixed_template_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build = build_simulation_input(
                BulkMechanicsRequest(
                    matrix_youngs_modulus=8.0,
                    matrix_poisson_ratio=0.3,
                    sample_dimensions=(1.0, 1.0, 1.0),
                    prescribed_displacement=-0.1,
                ),
                simulation_dir,
            )
            self.assertTrue(build.input_path.exists())
            self.assertTrue(build.metadata_path.exists())
            self.assertTrue(build.scenario_summary_path.exists())
            input_text = build.input_path.read_text(encoding="utf-8")
            self.assertIn("<febio_spec", input_text)
            self.assertIn("node_displacement.log", input_text)
            self.assertIn("element_principal_stress.log", input_text)

    def test_build_single_cell_input_contains_interface_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build = build_simulation_input(
                SingleCellContractionRequest(
                    matrix_youngs_modulus=8.0,
                    matrix_poisson_ratio=0.3,
                    cell_contractility=0.02,
                ),
                simulation_dir,
            )
            input_text = build.input_path.read_text(encoding="utf-8")
            self.assertIn("interface_disp_x", input_text)
            self.assertIn("cell_material", input_text)
            metadata = load_build_metadata(build)
            self.assertIn("interface_nodes", metadata["node_sets"])

    def test_build_organoid_input_contains_organoid_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build = build_simulation_input(
                OrganoidSpheroidRequest(
                    matrix_youngs_modulus=8.0,
                    matrix_poisson_ratio=0.3,
                    organoid_radius=0.18,
                    organoid_radial_displacement=0.03,
                ),
                simulation_dir,
            )
            input_text = build.input_path.read_text(encoding="utf-8")
            self.assertIn("organoid_domain", input_text)
            self.assertIn("organoid_material", input_text)
            metadata = load_build_metadata(build)
            self.assertIn("matrix_domain", metadata["element_sets"])


if __name__ == "__main__":
    unittest.main()
