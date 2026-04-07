from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.runner import (
    _needs_mechanics_fallback,
    _needs_search_fallback,
    build_single_agent_task_prompt,
    build_team_task_prompt,
    run_research_agent_sync,
)


class PromptBuilderTests(unittest.TestCase):
    def test_single_prompt_contains_save_instruction(self) -> None:
        prompt = build_single_agent_task_prompt(
            "synthetic ECM for intestinal organoid culture",
            "weekly.md",
            5,
            "# Template",
        )
        self.assertIn("save_report", prompt)
        self.assertIn("weekly.md", prompt)

    def test_team_prompt_mentions_roles_and_completion_token(self) -> None:
        prompt = build_team_task_prompt(
            "synthetic ECM for intestinal organoid culture",
            "weekly.md",
            5,
            "# Template",
        )
        self.assertIn("SearchAgent", prompt)
        self.assertIn("WriterAgent", prompt)
        self.assertIn("REPORT_COMPLETE", prompt)
        self.assertIn("weekly.md", prompt)

    def test_search_fallback_detection_handles_raw_function_markup(self) -> None:
        self.assertTrue(_needs_search_fallback("<｜DSML｜function_calls>"))
        self.assertFalse(
            _needs_search_fallback("## Search Coverage\n## Candidate Studies\n- example")
        )

    def test_mechanics_fallback_detection_handles_raw_function_markup(self) -> None:
        self.assertTrue(_needs_mechanics_fallback("<｜DSML｜function_calls>"))
        self.assertFalse(
            _needs_mechanics_fallback("## Dataset Summary\n## Fitted Model\n- creep")
        )

    def test_design_workflow_writes_report_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            result = run_research_agent_sync(
                project_dir=project_dir,
                query="Design an ECM near stiffness 8 Pa with low anisotropy",
                report_name="design_report.md",
                workflow="design",
                target_stiffness=8.0,
                target_anisotropy=0.1,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                constraint_max_anisotropy=0.3,
                constraint_min_connectivity=0.9,
                design_top_k=3,
                design_candidate_budget=6,
                design_monte_carlo_runs=2,
            )

            self.assertEqual(result.workflow, "design")
            self.assertIsNotNone(result.report_path)
            self.assertTrue(result.report_path.exists())
            self.assertIn("physics_valid=", result.final_summary)
            self.assertIsNotNone(result.run_dir)

            design_summary_path = result.run_dir / "design_summary.json"
            self.assertTrue(design_summary_path.exists())
            design_summary = json.loads(design_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(design_summary["workflow"], "design")
            self.assertEqual(design_summary["targets"]["stiffness"], 8.0)
            self.assertEqual(design_summary["constraints"]["max_anisotropy"], 0.3)
            self.assertIn("top_candidates", design_summary)
            self.assertIn("formulation_recommendations", design_summary)
            self.assertIn("sensitivity_summary", design_summary)
            self.assertIn("design_assessment", design_summary)
            self.assertIn(design_summary["design_assessment"]["status"], {"ready_for_screening", "caution", "not_ready"})
            self.assertIn("recommended_for_screening=", result.final_summary)

            metadata = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
            stage_files = metadata.get("stage_files", {})
            self.assertIn("design_summary", stage_files)
            self.assertTrue(Path(stage_files["design_summary"]).exists())

    def test_design_workflow_accepts_extra_targets_and_constraints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            result = run_research_agent_sync(
                project_dir=project_dir,
                query="Design an ECM near stiffness 8 Pa with bounded damping",
                report_name="design_report.md",
                workflow="design",
                target_stiffness=8.0,
                target_anisotropy=0.1,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                design_extra_targets_json=json.dumps({"loss_tangent_proxy": 0.3}),
                design_extra_constraints_json=json.dumps({"max_loss_tangent_proxy": 1.0}),
                design_top_k=2,
                design_candidate_budget=6,
                design_monte_carlo_runs=1,
            )

            design_summary = json.loads((result.run_dir / "design_summary.json").read_text(encoding="utf-8"))
            self.assertIn("loss_tangent_proxy", design_summary["targets"])
            self.assertIn("max_loss_tangent_proxy", design_summary["constraints"])

    def test_design_campaign_workflow_writes_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            cal_run = project_dir / "runs" / "seed_calibration"
            cal_run.mkdir(parents=True)
            (cal_run / "calibration_results.json").write_text(
                json.dumps(
                    {
                        "family_priors": [
                            {
                                "material_family": "GelMA",
                                "sample_count": 2,
                                "mean_abs_error": 18.0,
                                "parameter_priors": {
                                    "fiber_density": {"mean": 0.55, "std": 0.05},
                                    "fiber_stiffness": {"mean": 9.5, "std": 1.0},
                                    "bending_stiffness": {"mean": 0.24, "std": 0.02},
                                    "crosslink_prob": {"mean": 0.62, "std": 0.08},
                                    "domain_size": {"mean": 1.0, "std": 0.0},
                                },
                            }
                        ],
                        "condition_priors": [
                            {
                                "material_family": "GelMA",
                                "concentration_fraction": 0.15,
                                "curing_seconds": 120.0,
                                "target_stiffness_mean": 8.0,
                                "sample_count": 1,
                                "mean_abs_error": 10.0,
                                "parameter_priors": {
                                    "fiber_density": {"mean": 0.42, "std": 0.01},
                                    "fiber_stiffness": {"mean": 10.0, "std": 0.2},
                                    "bending_stiffness": {"mean": 0.24, "std": 0.01},
                                    "crosslink_prob": {"mean": 0.58, "std": 0.03},
                                    "domain_size": {"mean": 1.0, "std": 0.0},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = run_research_agent_sync(
                project_dir=project_dir,
                query="Compare GelMA ECM targets across 6, 8, and 10 Pa windows",
                report_name="campaign_report.md",
                workflow="design_campaign",
                campaign_target_stiffnesses="6,8,10",
                target_anisotropy=0.12,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                constraint_max_anisotropy=0.35,
                design_top_k=3,
                design_candidate_budget=6,
                design_monte_carlo_runs=2,
                condition_concentration_fraction=0.15,
                condition_curing_seconds=120.0,
            )

            self.assertEqual(result.workflow, "design_campaign")
            self.assertTrue(result.report_path.exists())
            self.assertIsNotNone(result.run_dir)
            campaign_summary_path = result.run_dir / "campaign_summary.json"
            self.assertTrue(campaign_summary_path.exists())
            campaign_summary = json.loads(campaign_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(campaign_summary["workflow"], "design_campaign")
            self.assertEqual(len(campaign_summary["campaign_results"]), 3)
            self.assertEqual(campaign_summary["constraints"]["max_anisotropy"], 0.35)
            self.assertIn("formulation_recommendations", campaign_summary)
            self.assertIn("calibration_context", campaign_summary)
            self.assertEqual(campaign_summary["calibration_context"]["prior_level"], "condition")
            self.assertIn("calibration_context", campaign_summary["campaign_results"][0])
            self.assertIn("design_assessment", campaign_summary["campaign_results"][0])
            self.assertIn("campaign_assessment", campaign_summary)
            self.assertEqual(campaign_summary["report_path"], str(result.report_path))
            self.assertIn("recommended_targets=", result.final_summary)

    def test_benchmark_workflow_writes_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            fake_benchmark_payload = {
                "workflow": "benchmark",
                "solver_benchmark": {"summary": {"pass_rate": 1.0}},
                "load_ladder_benchmark": {"summary": {"displacement_monotonic": True}},
                "scaling_benchmark": {"summary": {"pass_count": 3}},
                "inverse_design_benchmark": {"summary": {"mean_abs_error": 0.5}},
                "property_target_design_benchmark": {"summary": {"mean_property_error": 0.12, "overall_pass": True}},
                "repeatability_benchmark": {"summary": {"stiffness_std": 0.0}},
                "identifiability_proxy_benchmark": {"summary": {"identifiability_risk": "high"}},
                "mechanics_fit_benchmark": {"summary": {"mean_relative_error": 0.01}},
                "calibration_design_benchmark": {"summary": {"available": False, "mean_combined_error_improvement": 0.0}},
                "summary": {
                    "overall_pass": False,
                    "solver_pass_rate": 1.0,
                    "load_ladder_monotonic": True,
                    "scaling_pass_count": 3,
                    "inverse_design_mean_abs_error": 0.5,
                    "property_target_mean_error": 0.12,
                    "repeatability_stiffness_std": 0.0,
                    "identifiability_risk": "high",
                    "fit_mean_relative_error": 0.01,
                    "calibration_benchmark_available": False,
                    "calibration_design_improvement": 0.0,
                },
            }
            with patch("ecm_organoid_agent.runner.run_mechanics_benchmark_suite", return_value=fake_benchmark_payload):
                result = run_research_agent_sync(
                    project_dir=project_dir,
                    query="Benchmark the current ECM mechanics core",
                    report_name="benchmark_report.md",
                    workflow="benchmark",
                )

            self.assertEqual(result.workflow, "benchmark")
            self.assertTrue(result.report_path.exists())
            self.assertIsNotNone(result.run_dir)
            benchmark_summary_path = result.run_dir / "benchmark_summary.json"
            self.assertTrue(benchmark_summary_path.exists())
            benchmark_summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(benchmark_summary["workflow"], "benchmark")
            self.assertIn("solver_benchmark", benchmark_summary)
            self.assertIn("load_ladder_benchmark", benchmark_summary)
            self.assertIn("scaling_benchmark", benchmark_summary)
            self.assertIn("inverse_design_benchmark", benchmark_summary)
            self.assertIn("property_target_design_benchmark", benchmark_summary)
            self.assertIn("repeatability_benchmark", benchmark_summary)
            self.assertIn("identifiability_proxy_benchmark", benchmark_summary)
            self.assertIn("mechanics_fit_benchmark", benchmark_summary)
            self.assertIn("calibration_design_benchmark", benchmark_summary)
            self.assertIn("calibration_design_improvement", benchmark_summary["summary"])
            self.assertIn("property_target_mean_error", benchmark_summary["summary"])

    def test_dataset_workflow_writes_manifest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            result = run_research_agent_sync(
                project_dir=project_dir,
                query="hydrogel rheology",
                report_name="dataset_report.md",
                workflow="datasets",
            )

            self.assertEqual(result.workflow, "datasets")
            self.assertTrue(result.report_path.exists())
            self.assertIsNotNone(result.run_dir)
            snapshot_path = result.run_dir / "dataset_manifest_snapshot.json"
            self.assertTrue(snapshot_path.exists())
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["datasets"]), 1)

    def test_calibration_workflow_writes_results(self) -> None:
        project_dir = PROJECT_ROOT
        result = run_research_agent_sync(
            project_dir=project_dir,
            query="Calibrate ECM mechanics priors from hydrogel characterization data",
            report_name="calibration_report_test.md",
            workflow="calibration",
            dataset_id="hydrogel_characterization_data",
            calibration_max_samples=1,
        )
        self.assertEqual(result.workflow, "calibration")
        self.assertTrue(result.report_path.exists())
        self.assertIsNotNone(result.run_dir)
        targets_path = result.run_dir / "calibration_targets.json"
        results_path = result.run_dir / "calibration_results.json"
        impact_path = result.run_dir / "calibration_impact.json"
        self.assertTrue(targets_path.exists())
        self.assertTrue(results_path.exists())
        self.assertTrue(impact_path.exists())

    def test_design_workflow_uses_calibrated_search_space_when_priors_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
                (project_dir / dirname).mkdir()

            cal_run = project_dir / "runs" / "seed_calibration"
            cal_run.mkdir(parents=True)
            (cal_run / "calibration_results.json").write_text(
                json.dumps(
                    {
                        "family_priors": [
                            {
                                "material_family": "GelMA",
                                "parameter_priors": {
                                    "fiber_density": {"mean": 0.55, "std": 0.05},
                                    "fiber_stiffness": {"mean": 9.5, "std": 1.0},
                                    "bending_stiffness": {"mean": 0.24, "std": 0.02},
                                    "crosslink_prob": {"mean": 0.62, "std": 0.08},
                                    "domain_size": {"mean": 1.0, "std": 0.0},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = run_research_agent_sync(
                project_dir=project_dir,
                query="Design a GelMA-like ECM near stiffness 8 Pa for 15% GelMA cured 120 s",
                report_name="design_report.md",
                workflow="design",
                target_stiffness=8.0,
                target_anisotropy=0.1,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                design_top_k=3,
                design_candidate_budget=6,
                design_monte_carlo_runs=2,
                condition_concentration_fraction=0.15,
                condition_curing_seconds=120.0,
            )

            design_summary = json.loads((result.run_dir / "design_summary.json").read_text(encoding="utf-8"))
            self.assertIn("calibrated_search_space", design_summary)
            self.assertTrue(design_summary["calibrated_search_space"])
            self.assertIn("calibration_context", design_summary)
            self.assertIn(design_summary["calibration_context"]["prior_level"], {"condition", "family"})


if __name__ == "__main__":
    unittest.main()
