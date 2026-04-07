from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.benchmarks import (
    run_calibration_design_benchmark,
    run_inverse_design_benchmark,
    run_inverse_design_repeatability_benchmark,
    run_identifiability_proxy_benchmark,
    run_load_ladder_benchmark,
    run_mechanics_benchmark_suite,
    run_mechanics_fit_benchmark,
    run_property_target_design_benchmark,
    run_scaling_benchmark,
    run_solver_benchmark,
)


class MechanicsBenchmarkTests(unittest.TestCase):
    def test_solver_benchmark_returns_case_and_summary_fields(self) -> None:
        payload = run_solver_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertGreater(len(payload["cases"]), 0)
        self.assertIn("pass_rate", payload["summary"])

    def test_inverse_design_benchmark_returns_error_metrics(self) -> None:
        payload = run_inverse_design_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertGreater(len(payload["cases"]), 0)
        self.assertIn("mean_abs_error", payload["summary"])
        self.assertIn("top_candidate", payload["cases"][0])

    def test_load_ladder_benchmark_returns_monotonicity_summary(self) -> None:
        payload = run_load_ladder_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertIn("displacement_monotonic", payload["summary"])
        self.assertGreater(len(payload["cases"]), 0)

    def test_scaling_benchmark_returns_scaling_summary(self) -> None:
        payload = run_scaling_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertIn("node_count_levels", payload["summary"])
        self.assertGreater(len(payload["cases"]), 0)

    def test_repeatability_benchmark_returns_variability_metrics(self) -> None:
        payload = run_inverse_design_repeatability_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertIn("stiffness_std", payload["summary"])
        self.assertGreater(len(payload["cases"]), 0)

    def test_property_target_design_benchmark_returns_property_error_summary(self) -> None:
        payload = run_property_target_design_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertIn("mean_property_error", payload["summary"])
        self.assertIn("top_candidate", payload["cases"][0])
        self.assertIn("property_errors", payload["cases"][0])

    def test_identifiability_proxy_returns_risk_summary(self) -> None:
        payload = run_identifiability_proxy_benchmark()
        self.assertIn("cases", payload)
        self.assertIn("summary", payload)
        self.assertIn("identifiability_risk", payload["summary"])
        self.assertIn("recommended_measurements", payload["summary"])
        self.assertIn("dominant_degenerate_parameters", payload["summary"])

    def test_mechanics_fit_benchmark_returns_relative_errors(self) -> None:
        payload = run_mechanics_fit_benchmark()
        self.assertIn("cases", payload)
        self.assertGreater(len(payload["cases"]), 0)
        self.assertIn("relative_errors", payload["cases"][0])
        self.assertIn("mean_relative_error", payload["summary"])

    def test_calibration_design_benchmark_reports_improvement_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = run_calibration_design_benchmark(project_dir=Path(tmp_dir), max_samples=2)
        self.assertIn("summary", payload)
        self.assertIn("available", payload["summary"])
        self.assertIn("mean_abs_error_baseline", payload["summary"])
        self.assertIn("mean_abs_error_calibrated", payload["summary"])
        self.assertIn("mean_total_error_delta", payload["summary"])
        self.assertIn("improved_abs_case_count", payload["summary"])

    def test_calibration_design_benchmark_reuses_cached_calibration_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            run_dir = project_dir / "runs" / "20260401_cached_calibration"
            run_dir.mkdir(parents=True)
            (run_dir / "calibration_targets.json").write_text(
                json.dumps(
                    {
                        "calibration_targets": [
                            {
                                "sample_key": "gelma_case_1",
                                "dataset_id": "hydrogel_characterization_data",
                                "target_stiffness": 8.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "calibration_results.json").write_text(
                json.dumps({"family_priors": [{"material_family": "GelMA", "sample_count": 1}]}),
                encoding="utf-8",
            )
            (run_dir / "calibration_impact.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "sample_key": "gelma_case_1",
                                "material_family": "GelMA",
                                "baseline": {"abs_error": 2.0, "total_error": 1.2, "auxiliary_errors": {}, "candidate": {}},
                                "calibrated": {"abs_error": 1.5, "total_error": 1.0, "auxiliary_errors": {}, "candidate": {}},
                                "calibrated_search_space": {"fiber_density": {"min": 0.3, "max": 0.6}},
                                "improved_total_error": True,
                                "improved_abs_error": True,
                            }
                        ],
                        "summary": {
                            "available": True,
                            "eligible_case_count": 1,
                            "improved_abs_case_count": 1,
                            "improved_total_case_count": 1,
                            "mean_abs_error_baseline": 2.0,
                            "mean_abs_error_calibrated": 1.5,
                            "mean_abs_error_delta": 0.5,
                            "mean_total_error_baseline": 1.2,
                            "mean_total_error_calibrated": 1.0,
                            "mean_total_error_delta": 0.2,
                            "baseline_feasible_count": 1,
                            "calibrated_feasible_count": 1,
                            "evaluation_mode": "leave_one_out_when_possible",
                            "overall_pass": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            payload = run_calibration_design_benchmark(project_dir=project_dir, max_samples=2)
            self.assertTrue(payload["summary"]["available"])
            self.assertEqual(payload["summary"]["cached_from_run"], "20260401_cached_calibration")
            self.assertEqual(payload["summary"]["mean_combined_error_improvement"], 0.2)

    def test_mechanics_benchmark_suite_returns_overall_summary(self) -> None:
        fake_payload = {"summary": {"overall_pass": True}}
        with patch("ecm_organoid_agent.benchmarks.run_solver_benchmark", return_value={"summary": {"overall_pass": True, "pass_rate": 1.0}}), patch(
            "ecm_organoid_agent.benchmarks.run_load_ladder_benchmark", return_value={"summary": {"overall_pass": True, "displacement_monotonic": True}}
        ), patch(
            "ecm_organoid_agent.benchmarks.run_scaling_benchmark", return_value={"summary": {"overall_pass": True, "pass_count": 3}}
        ), patch(
            "ecm_organoid_agent.benchmarks.run_inverse_design_benchmark",
            return_value={"summary": {"overall_pass": True, "mean_abs_error": 0.2}},
        ), patch(
            "ecm_organoid_agent.benchmarks.run_property_target_design_benchmark",
            return_value={"summary": {"overall_pass": True, "mean_property_error": 0.12}},
        ), patch(
            "ecm_organoid_agent.benchmarks.run_inverse_design_repeatability_benchmark",
            return_value={"summary": {"overall_pass": True, "stiffness_std": 0.01}},
        ), patch(
            "ecm_organoid_agent.benchmarks.run_identifiability_proxy_benchmark",
            return_value={"summary": {"overall_pass": True, "identifiability_risk": "low"}},
        ), patch(
            "ecm_organoid_agent.benchmarks.run_mechanics_fit_benchmark",
            return_value={"summary": {"overall_pass": True, "mean_relative_error": 0.01}},
        ), patch(
            "ecm_organoid_agent.benchmarks.run_calibration_design_benchmark",
            return_value={"summary": {"available": False, "overall_pass": False, "mean_combined_error_improvement": 0.0}},
        ):
            payload = run_mechanics_benchmark_suite(project_dir=PROJECT_ROOT, include_calibration_design=False)
            self.assertEqual(payload["workflow"], "benchmark")
            self.assertIn("solver_benchmark", payload)
            self.assertIn("load_ladder_benchmark", payload)
            self.assertIn("scaling_benchmark", payload)
            self.assertIn("inverse_design_benchmark", payload)
            self.assertIn("property_target_design_benchmark", payload)
            self.assertIn("repeatability_benchmark", payload)
            self.assertIn("identifiability_proxy_benchmark", payload)
            self.assertIn("mechanics_fit_benchmark", payload)
            self.assertIn("calibration_design_benchmark", payload)
            self.assertIn("calibration_design_improvement", payload["summary"])
            self.assertIn("property_target_mean_error", payload["summary"])
            self.assertFalse(payload["summary"]["calibration_benchmark_available"])
            self.assertIn("overall_pass", payload["summary"])


if __name__ == "__main__":
    unittest.main()
