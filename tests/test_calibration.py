from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.calibration import (
    build_calibration_targets,
    calibrated_search_space_from_calibration_results,
    calibrated_search_space_from_priors,
    condition_prior_for_material,
    extract_summary_measurements,
    interpolated_condition_prior_for_material,
    load_latest_family_priors,
    run_calibration_pipeline,
    search_space_from_family_prior,
)


class CalibrationTests(unittest.TestCase):
    def test_extract_summary_measurements_reads_real_dataset(self) -> None:
        measurements = extract_summary_measurements(
            project_dir=PROJECT_ROOT,
            dataset_id="hydrogel_characterization_data",
        )
        self.assertGreater(len(measurements), 0)
        self.assertIn(measurements[0].material_family, {"GelMA", "PEGDA", "unknown"})
        metric_names = {row.metric_name for row in measurements}
        self.assertIn("speed_of_sound_m_s", metric_names)
        self.assertIn("acoustic_impedance_mrayl", metric_names)

    def test_build_calibration_targets_from_real_measurements(self) -> None:
        measurements = extract_summary_measurements(
            project_dir=PROJECT_ROOT,
            dataset_id="hydrogel_characterization_data",
        )
        targets = build_calibration_targets(measurements, max_samples=4)
        self.assertGreater(len(targets), 0)
        self.assertIn("target_stiffness", targets[0])
        self.assertIn("material_family", targets[0])
        self.assertIn("condition_signature", targets[0])
        self.assertIn("target_property_hints", targets[0])
        self.assertTrue(
            any(
                "viscosity_low_shear_pas" in target["measurement_bundle"]
                or "shear_thinning_ratio" in target["measurement_bundle"]
                for target in targets
            )
        )

    def test_build_calibration_targets_uses_diverse_family_sampling_when_limited(self) -> None:
        measurements = extract_summary_measurements(
            project_dir=PROJECT_ROOT,
            dataset_id="hydrogel_characterization_data",
        )
        targets = build_calibration_targets(measurements, max_samples=10)
        families = {target["material_family"] for target in targets}
        self.assertIn("GelMA", families)
        self.assertIn("PEGDA", families)

    def test_run_calibration_pipeline_returns_family_priors(self) -> None:
        payload = run_calibration_pipeline(
            project_dir=PROJECT_ROOT,
            dataset_id="hydrogel_characterization_data",
            max_samples=1,
        )
        self.assertEqual(payload["workflow"], "calibration")
        self.assertGreater(payload["summary"]["target_count"], 0)
        self.assertGreater(len(payload["calibration_results"]["family_priors"]), 0)
        self.assertIn("speed_of_sound_m_s", payload["summary"]["metrics_covered"])
        self.assertIn("acoustic_impedance_mrayl", payload["summary"]["metrics_covered"])
        self.assertIn("metrics_covered", payload["summary"])
        self.assertIn("youngs_modulus_kpa", payload["summary"]["metrics_covered"])
        self.assertIn("viscosity_low_shear_pas", payload["summary"]["metrics_covered"])
        self.assertIn("condition_signature", payload["calibration_results"]["condition_priors"][0])
        self.assertIn("target_property_hints", payload["calibration_targets"][0])
        self.assertIn("calibration_impact_assessment", payload)
        self.assertTrue(payload["calibration_impact_assessment"]["summary"]["available"])
        self.assertIn("mean_total_error_delta", payload["calibration_impact_assessment"]["summary"])
        self.assertIn("evaluation_mode", payload["calibration_impact_assessment"]["summary"])

    def test_search_space_from_family_prior_returns_bounded_ranges(self) -> None:
        prior = {
            "material_family": "GelMA",
            "sample_count": 2,
            "mean_abs_error": 20.0,
            "parameter_priors": {
                "fiber_density": {"mean": 0.55, "std": 0.05},
                "fiber_stiffness": {"mean": 9.5, "std": 1.0},
                "bending_stiffness": {"mean": 0.24, "std": 0.02},
                "crosslink_prob": {"mean": 0.62, "std": 0.08},
                "domain_size": {"mean": 1.0, "std": 0.0},
            },
        }
        bounds = search_space_from_family_prior(prior)
        self.assertIsNotNone(bounds)
        self.assertLess(bounds["fiber_density"]["min"], bounds["fiber_density"]["max"])
        self.assertGreaterEqual(bounds["crosslink_prob"]["min"], 0.01)
        self.assertGreater(bounds["fiber_stiffness"]["max"] - bounds["fiber_stiffness"]["min"], 4.0)

    def test_calibrated_search_space_from_priors_matches_material(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "sample_count": 2,
                "mean_abs_error": 20.0,
                "parameter_priors": {
                    "fiber_density": {"mean": 0.55, "std": 0.05},
                    "fiber_stiffness": {"mean": 9.5, "std": 1.0},
                    "bending_stiffness": {"mean": 0.24, "std": 0.02},
                    "crosslink_prob": {"mean": 0.62, "std": 0.08},
                    "domain_size": {"mean": 1.0, "std": 0.0},
                },
            }
        ]
        bounds = calibrated_search_space_from_priors(priors, "GelMA")
        self.assertIsNotNone(bounds)
        self.assertIn("fiber_stiffness", bounds)

    def test_condition_prior_for_material_prefers_exact_match(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.10,
                "curing_seconds": 60.0,
                "target_stiffness_mean": 4.0,
                "sample_count": 2,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 120.0,
                "target_stiffness_mean": 10.0,
                "sample_count": 3,
            },
        ]
        matched = condition_prior_for_material(
            priors,
            "GelMA",
            concentration_fraction=0.15,
            curing_seconds=120.0,
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["concentration_fraction"], 0.15)
        self.assertEqual(matched["curing_seconds"], 120.0)

    def test_condition_prior_for_material_can_use_target_stiffness_when_conditions_missing(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.10,
                "curing_seconds": 60.0,
                "target_stiffness_mean": 4.0,
                "achieved_feature_summary": {"anisotropy_mean": 0.2, "connectivity_mean": 1.0, "stress_propagation_mean": 0.6},
                "sample_count": 2,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 120.0,
                "target_stiffness_mean": 10.0,
                "achieved_feature_summary": {"anisotropy_mean": 0.12, "connectivity_mean": 1.0, "stress_propagation_mean": 0.4},
                "sample_count": 2,
            },
        ]
        matched = condition_prior_for_material(priors, "GelMA", target_stiffness=9.0)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["curing_seconds"], 120.0)

    def test_condition_prior_can_use_additional_targets(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.10,
                "curing_seconds": 60.0,
                "target_stiffness_mean": 8.0,
                "achieved_feature_summary": {"anisotropy_mean": 0.25, "connectivity_mean": 1.0, "stress_propagation_mean": 1.3},
                "sample_count": 1,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 120.0,
                "target_stiffness_mean": 8.0,
                "achieved_feature_summary": {"anisotropy_mean": 0.11, "connectivity_mean": 1.0, "stress_propagation_mean": 0.55},
                "sample_count": 1,
            },
        ]
        matched = condition_prior_for_material(
            priors,
            "GelMA",
            target_stiffness=8.0,
            target_anisotropy=0.1,
            target_connectivity=0.95,
            target_stress_propagation=0.5,
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["curing_seconds"], 120.0)

    def test_condition_prior_can_use_generic_condition_signature(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 60.0,
                "condition_signature": {"temperature_c": 25.0, "photoinitiator_fraction": 0.01},
                "target_stiffness_mean": 8.0,
                "sample_count": 1,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 60.0,
                "condition_signature": {"temperature_c": 37.0, "photoinitiator_fraction": 0.02},
                "target_stiffness_mean": 8.0,
                "sample_count": 1,
            },
        ]
        matched = condition_prior_for_material(
            priors,
            "GelMA",
            condition_overrides={"temperature_c": 37.0, "photoinitiator_fraction": 0.02},
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["condition_signature"]["temperature_c"], 37.0)
        self.assertEqual(matched["condition_signature"]["photoinitiator_fraction"], 0.02)

    def test_condition_prior_can_use_generic_condition_signature(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 60.0,
                "condition_signature": {"concentration_fraction": 0.15, "curing_seconds": 60.0, "temperature_c": 25.0},
                "target_stiffness_mean": 12.0,
                "sample_count": 1,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 60.0,
                "condition_signature": {"concentration_fraction": 0.15, "curing_seconds": 60.0, "temperature_c": 37.0},
                "target_stiffness_mean": 12.0,
                "sample_count": 1,
            },
        ]
        matched = condition_prior_for_material(
            priors,
            "GelMA",
            concentration_fraction=0.15,
            curing_seconds=60.0,
            condition_overrides={"temperature_c": 37.0},
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["condition_signature"]["temperature_c"], 37.0)

    def test_interpolated_condition_prior_for_material_returns_interpolated_prior(self) -> None:
        priors = [
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 30.0,
                "condition_signature": {"concentration_fraction": 0.15, "curing_seconds": 30.0},
                "target_stiffness_mean": 33.0,
                "sample_count": 1,
                "parameter_priors": {"fiber_stiffness": {"mean": 8.0, "std": 0.0}},
                "auxiliary_error_summary": {"density": {"mean": 0.01, "std": 0.0}},
                "achieved_feature_summary": {"anisotropy_mean": 0.18},
                "mean_abs_error": 3.0,
            },
            {
                "material_family": "GelMA",
                "concentration_fraction": 0.15,
                "curing_seconds": 120.0,
                "condition_signature": {"concentration_fraction": 0.15, "curing_seconds": 120.0},
                "target_stiffness_mean": 95.0,
                "sample_count": 1,
                "parameter_priors": {"fiber_stiffness": {"mean": 14.0, "std": 0.0}},
                "auxiliary_error_summary": {"density": {"mean": 0.02, "std": 0.0}},
                "achieved_feature_summary": {"anisotropy_mean": 0.24},
                "mean_abs_error": 30.0,
            },
        ]
        prior = interpolated_condition_prior_for_material(priors, "GelMA", concentration_fraction=0.15, target_stiffness=50.0)
        self.assertIsNotNone(prior)
        self.assertIn("interpolation", prior)
        self.assertGreater(prior["parameter_priors"]["fiber_stiffness"]["mean"], 8.0)
        self.assertLess(prior["parameter_priors"]["fiber_stiffness"]["mean"], 14.0)

    def test_calibrated_search_space_from_results_returns_condition_signature_context(self) -> None:
        calibration_results = {
            "condition_priors": [
                {
                    "material_family": "GelMA",
                    "concentration_fraction": 0.15,
                    "curing_seconds": 60.0,
                    "condition_signature": {"temperature_c": 37.0, "photoinitiator_fraction": 0.02},
                    "target_stiffness_mean": 8.0,
                    "sample_count": 1,
                    "parameter_priors": {
                        "fiber_density": {"mean": 0.42, "std": 0.01},
                        "fiber_stiffness": {"mean": 10.0, "std": 0.2},
                        "bending_stiffness": {"mean": 0.24, "std": 0.01},
                        "crosslink_prob": {"mean": 0.58, "std": 0.03},
                        "domain_size": {"mean": 1.0, "std": 0.0},
                    },
                    "achieved_feature_summary": {"anisotropy_mean": 0.1, "connectivity_mean": 1.0, "stress_propagation_mean": 0.5},
                }
            ],
            "family_priors": [],
        }
        bounds, context = calibrated_search_space_from_calibration_results(
            calibration_results,
            material_family="GelMA",
            condition_overrides={"temperature_c": 37.0, "photoinitiator_fraction": 0.02},
            target_stiffness=8.0,
        )
        self.assertIsNotNone(bounds)
        self.assertIsNotNone(context)
        self.assertEqual(context["prior_level"], "condition")
        self.assertEqual(context["condition_signature"]["temperature_c"], 37.0)


if __name__ == "__main__":
    unittest.main()
