from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.febio.builder import BuildArtifacts, build_simulation_input
from ecm_organoid_agent.febio.config import FEBioConfig
from ecm_organoid_agent.febio.mapping import CandidateSimulationMappingOptions, candidate_to_simulation_request
from ecm_organoid_agent.febio.metrics import calculate_simulation_metrics
from ecm_organoid_agent.febio.parser import parse_simulation_outputs
from ecm_organoid_agent.febio.runner import RunnerResult, build_febio_command, run_febio_job
from ecm_organoid_agent.febio.schemas import (
    BulkMechanicsRequest,
    OrganoidSpheroidRequest,
    SingleCellContractionRequest,
    simulation_request_from_dict,
)
from ecm_organoid_agent.runner import run_research_agent_sync


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


class FEBioTemplateBuildTests(unittest.TestCase):
    def test_build_simulation_input_writes_fixed_template_artifacts(self) -> None:
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


class FEBioRunnerTests(unittest.TestCase):
    def test_build_febio_command_uses_fixed_safe_arguments(self) -> None:
        command = build_febio_command("/path/to/febio4", Path("/tmp/input.feb"))
        self.assertEqual(command, ["/path/to/febio4", "-i", "input.feb", "-silent"])

    def test_run_febio_job_handles_unavailable_runtime_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            input_path = simulation_dir / "input.feb"
            input_path.write_text("<febio_spec version='4.0'/>", encoding="utf-8")
            config = FEBioConfig(
                enabled=True,
                executable=None,
                timeout_seconds=30,
                default_tmp_dir=simulation_dir,
                available=False,
                status_message="FEBio missing for test.",
            )
            result = run_febio_job(
                febio_config=config,
                simulation_dir=simulation_dir,
                input_path=input_path,
            )
            self.assertEqual(result.status, "unavailable")
            self.assertTrue((simulation_dir / "runner_metadata.json").exists())

    def test_run_febio_job_constructs_expected_subprocess_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            input_path = simulation_dir / "input.feb"
            input_path.write_text("<febio_spec version='4.0'/>", encoding="utf-8")
            config = FEBioConfig(
                enabled=True,
                executable="/path/to/febio4",
                timeout_seconds=45,
                default_tmp_dir=simulation_dir,
                available=True,
                status_message="ok",
            )
            completed = type("Completed", (), {"stdout": "", "stderr": "", "returncode": 0})()
            with patch("ecm_organoid_agent.febio.runner.subprocess.run", return_value=completed) as mock_run:
                result = run_febio_job(
                    febio_config=config,
                    simulation_dir=simulation_dir,
                    input_path=input_path,
                )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(mock_run.call_args.kwargs["cwd"], simulation_dir)
            self.assertEqual(mock_run.call_args.kwargs["timeout"], 45)
            self.assertEqual(mock_run.call_args.args[0], ["/path/to/febio4", "-i", "input.feb", "-silent"])


class FEBioParserAndMetricsTests(unittest.TestCase):
    def _write_fake_output_bundle(self, simulation_dir: Path) -> tuple[BuildArtifacts, RunnerResult]:
        request = BulkMechanicsRequest(
            matrix_youngs_modulus=8.0,
            matrix_poisson_ratio=0.3,
            sample_dimensions=(1.0, 1.0, 1.0),
            prescribed_displacement=-0.1,
            target_stiffness=8.0,
        )
        build = build_simulation_input(request, simulation_dir)
        (simulation_dir / "node_displacement.log").write_text(
            "\n".join(
                [
                    "*Step  = 0",
                    "*Time  = 0",
                    "*Data  = ux;uy;uz",
                    "1,0,0,0",
                    "2,0,0,0",
                    "*Step  = 1",
                    "*Time  = 1",
                    "*Data  = ux;uy;uz",
                    "1,0,0,0",
                    "2,0,0,-0.1",
                ]
            ),
            encoding="utf-8",
        )
        (simulation_dir / "top_reaction.log").write_text(
            "\n".join(
                [
                    "*Step  = 1",
                    "*Time  = 1",
                    "*Data  = Rx;Ry;Rz",
                    "2,0,0,-0.8",
                ]
            ),
            encoding="utf-8",
        )
        (simulation_dir / "element_principal_stress.log").write_text(
            "\n".join(
                [
                    "*Step  = 1",
                    "*Time  = 1",
                    "*Data  = s1;s2;s3",
                    "1,0.1,0.0,-0.4",
                ]
            ),
            encoding="utf-8",
        )
        (simulation_dir / "input.log").write_text(
            "N O R M A L   T E R M I N A T I O N\n",
            encoding="utf-8",
        )
        runner = RunnerResult(
            status="succeeded",
            returncode=0,
            command=["febio4", "-i", "input.feb", "-silent"],
            duration_seconds=0.1,
            stdout_path=simulation_dir / "febio_stdout.txt",
            stderr_path=simulation_dir / "febio_stderr.txt",
            log_path=simulation_dir / "input.log",
            xplt_path=simulation_dir / "input.xplt",
            dmp_path=simulation_dir / "input.dmp",
            metadata_path=simulation_dir / "runner_metadata.json",
        )
        runner.stdout_path.write_text("", encoding="utf-8")
        runner.stderr_path.write_text("", encoding="utf-8")
        return build, runner

    def test_parser_extracts_last_step_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build, runner = self._write_fake_output_bundle(simulation_dir)
            result = parse_simulation_outputs(
                build_artifacts=build,
                runner_result=runner,
            )
            self.assertTrue(result["extracted_fields"]["solver_converged"])
            self.assertEqual(result["extracted_fields"]["top_reaction"]["final"]["ids"], [2])
            metrics = calculate_simulation_metrics(result, simulation_dir=simulation_dir)
            self.assertAlmostEqual(metrics["effective_stiffness"], 8.0, places=6)
            self.assertAlmostEqual(metrics["peak_stress"], 0.4, places=6)
            self.assertTrue(metrics["feasibility_flags"]["solver_converged"])

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


class FEBioWorkflowSmokeTests(unittest.TestCase):
    def test_simulation_workflow_smoke_with_mocked_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            fake_payload = {
                "status": "succeeded",
                "request": {"scenario": "bulk_mechanics"},
                "runner": {"command": ["febio4", "-i", "input.feb", "-silent"]},
                "simulation_result": {"status": "succeeded", "warnings": []},
                "simulation_metrics": {
                    "effective_stiffness": 8.2,
                    "peak_stress": 0.4,
                    "displacement_decay_length": None,
                    "strain_heterogeneity": 0.12,
                    "target_mismatch_score": 0.025,
                    "feasibility_flags": {"solver_converged": True},
                },
                "final_summary_path": str(project_dir / "runs" / "fake" / "simulation" / "final_summary.md"),
                "final_summary": "ok",
                "simulation_dir": str(project_dir / "runs" / "fake" / "simulation"),
                "febio": {"available": True},
            }

            with patch("ecm_organoid_agent.tools.run_simulation_request", return_value=fake_payload):
                result = run_research_agent_sync(
                    project_dir=project_dir,
                    query="Run FEBio bulk verification",
                    report_name="simulation_report.md",
                    workflow="simulation",
                    simulation_scenario="bulk_mechanics",
                    target_stiffness=8.0,
                    matrix_youngs_modulus=8.0,
                )

            self.assertEqual(result.workflow, "simulation")
            self.assertTrue(result.report_path.exists())
            self.assertIsNotNone(result.run_dir)
            metadata = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["workflow"], "simulation")
            self.assertIn("simulation_agent", metadata["stage_files"])

    def test_simulation_workflow_gracefully_degrades_without_febio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            with patch.dict(os.environ, {"FEBIO_ENABLED": "0"}, clear=False):
                result = run_research_agent_sync(
                    project_dir=project_dir,
                    query="Run FEBio bulk verification",
                    report_name="simulation_report.md",
                    workflow="simulation",
                    simulation_scenario="bulk_mechanics",
                    target_stiffness=8.0,
                    matrix_youngs_modulus=8.0,
                )

            self.assertEqual(result.workflow, "simulation")
            self.assertTrue(result.report_path.exists())
            self.assertIn("status=unavailable", result.final_summary)


if __name__ == "__main__":
    unittest.main()
