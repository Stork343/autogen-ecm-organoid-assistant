from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.febio_test_support import PROJECT_ROOT, make_workspace  # noqa: F401

from ecm_organoid_agent.runner import run_research_agent_sync


class SimulationWorkflowTests(unittest.TestCase):
    def test_simulation_workflow_smoke_with_mocked_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = make_workspace(tmp_dir)
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
            project_dir = make_workspace(tmp_dir)
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

    def test_design_plus_simulation_minimal_closed_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = make_workspace(tmp_dir)
            fake_simulation = {
                "status": "succeeded",
                "request": {"scenario": "bulk_mechanics"},
                "runner": {"command": ["febio4", "-i", "input.feb", "-silent"]},
                "simulation_result": {"status": "succeeded", "warnings": []},
                "simulation_metrics": {
                    "effective_stiffness": 8.1,
                    "peak_stress": 0.42,
                    "displacement_decay_length": None,
                    "strain_heterogeneity": 0.1,
                    "target_mismatch_score": 0.01,
                    "feasibility_flags": {"solver_converged": True},
                },
                "final_summary_path": str(project_dir / "runs" / "fake" / "simulation" / "final_summary.md"),
                "final_summary": "ok",
                "simulation_dir": str(project_dir / "runs" / "fake" / "simulation"),
                "febio": {"available": True},
            }

            with patch("ecm_organoid_agent.tools.run_simulation_request", return_value=fake_simulation):
                result = run_research_agent_sync(
                    project_dir=project_dir,
                    query="Design an ECM near stiffness 8 Pa and verify with FEBio",
                    report_name="design_report.md",
                    workflow="design",
                    target_stiffness=8.0,
                    target_anisotropy=0.1,
                    target_connectivity=0.95,
                    target_stress_propagation=0.5,
                    design_top_k=2,
                    design_candidate_budget=4,
                    design_monte_carlo_runs=1,
                    design_run_simulation=True,
                    design_simulation_scenario="bulk_mechanics",
                    design_simulation_top_k=1,
                )

            design_summary = json.loads((result.run_dir / "design_summary.json").read_text(encoding="utf-8"))
            self.assertIn("design_simulation", design_summary)
            self.assertEqual(design_summary["design_simulation"]["scenario"], "bulk_mechanics")
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Simulation Evidence", report_text)
            self.assertIn("Final Recommendation", report_text)


if __name__ == "__main__":
    unittest.main()
