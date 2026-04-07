from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.mechanics import (
    analyze_cyclic_response,
    burgers_creep_strain,
    fit_burgers_creep_coarse,
    fit_frequency_sweep_coarse,
    fit_kelvin_voigt_coarse,
    fit_linear_elastic_through_origin,
    fit_maxwell_coarse,
    fit_mechanics_dataset,
    fit_power_law_elastic,
    fit_sls_creep_coarse,
    fit_sls_relaxation_coarse,
    generalized_maxwell_frequency_response,
    kelvin_voigt_creep_strain,
    linear_elastic_stress,
    maxwell_stress_relaxation,
    power_law_stiffness,
    power_law_stress,
    simulate_mechanics_curve,
    standard_linear_solid_creep_strain,
    standard_linear_solid_stress_relaxation,
)


class LinearElasticMechanicsTests(unittest.TestCase):
    def test_linear_elastic_fit_through_origin(self) -> None:
        strain = np.array([0.0, 0.1, 0.2, 0.4], dtype=float)
        stress = linear_elastic_stress(strain, 3.0)
        result = fit_linear_elastic_through_origin(strain, stress)
        self.assertAlmostEqual(result.modulus, 3.0)
        self.assertAlmostEqual(result.mse, 0.0)
        self.assertAlmostEqual(result.r2, 1.0)
        self.assertEqual(result.n_points, 4)

    def test_power_law_fit_recovers_coefficient_and_exponent(self) -> None:
        strain = np.array([0.1, 0.2, 0.4, 0.8], dtype=float)
        stress = power_law_stress(strain, coefficient=2.5, exponent=1.4)
        result = fit_power_law_elastic(strain, stress)
        self.assertAlmostEqual(result.coefficient, 2.5, places=3)
        self.assertAlmostEqual(result.exponent, 1.4, places=3)
        self.assertLess(result.mse, 1e-10)


class KelvinVoigtMechanicsTests(unittest.TestCase):
    def test_kelvin_voigt_creep_response_and_coarse_fit(self) -> None:
        time = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        applied_stress = 10.0
        elastic_modulus = 5.0
        viscosity = 10.0
        strain = kelvin_voigt_creep_strain(time, applied_stress, elastic_modulus, viscosity)
        expected = np.array(
            [
                0.0,
                2.0 * (1.0 - math.exp(-0.5)),
                2.0 * (1.0 - math.exp(-1.0)),
                2.0 * (1.0 - math.exp(-1.5)),
            ],
            dtype=float,
        )
        np.testing.assert_allclose(strain, expected)
        result = fit_kelvin_voigt_coarse(
            time,
            strain,
            applied_stress,
            modulus_grid=np.array([4.0, 5.0, 6.0], dtype=float),
            tau_grid=np.array([1.0, 2.0, 3.0], dtype=float),
        )
        self.assertAlmostEqual(result.elastic_modulus, 5.0)
        self.assertAlmostEqual(result.relaxation_time, 2.0)
        self.assertAlmostEqual(result.viscosity, 10.0)
        self.assertAlmostEqual(result.mse, 0.0)


class StandardLinearSolidMechanicsTests(unittest.TestCase):
    def test_sls_creep_response_and_fit(self) -> None:
        time = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        strain = standard_linear_solid_creep_strain(time, 10.0, 8.0, 4.0, 1.5)
        result = fit_sls_creep_coarse(
            time,
            strain,
            10.0,
            instantaneous_modulus_grid=np.array([6.0, 8.0, 10.0], dtype=float),
            equilibrium_fraction_grid=np.array([0.4, 0.5, 0.6], dtype=float),
            tau_grid=np.array([1.0, 1.5, 2.0], dtype=float),
        )
        self.assertAlmostEqual(result.instantaneous_modulus, 8.0)
        self.assertAlmostEqual(result.equilibrium_modulus, 4.0)
        self.assertAlmostEqual(result.relaxation_time, 1.5)
        self.assertLess(result.mse, 1e-10)

    def test_sls_relaxation_response_and_fit(self) -> None:
        time = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        stress = standard_linear_solid_stress_relaxation(time, 0.25, 12.0, 5.0, 2.0)
        result = fit_sls_relaxation_coarse(
            time,
            stress,
            0.25,
            instantaneous_modulus_grid=np.array([10.0, 12.0, 14.0], dtype=float),
            equilibrium_fraction_grid=np.array([0.3, 5.0 / 12.0, 0.6], dtype=float),
            tau_grid=np.array([1.0, 2.0, 3.0], dtype=float),
        )
        self.assertAlmostEqual(result.instantaneous_modulus, 12.0)
        self.assertAlmostEqual(result.equilibrium_modulus, 5.0)
        self.assertAlmostEqual(result.relaxation_time, 2.0)
        self.assertLess(result.mse, 1e-10)


class BurgersMechanicsTests(unittest.TestCase):
    def test_burgers_creep_response_and_fit(self) -> None:
        time = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        strain = burgers_creep_strain(time, 5.0, 20.0, 10.0, 100.0, 20.0)
        result = fit_burgers_creep_coarse(
            time,
            strain,
            5.0,
            instantaneous_modulus_grid=np.array([15.0, 20.0, 25.0], dtype=float),
            delayed_modulus_grid=np.array([8.0, 10.0, 12.0], dtype=float),
            retardation_time_grid=np.array([1.0, 2.0, 3.0], dtype=float),
            maxwell_viscosity_grid=np.array([80.0, 100.0, 120.0], dtype=float),
        )
        self.assertAlmostEqual(result.instantaneous_modulus, 20.0)
        self.assertAlmostEqual(result.delayed_modulus, 10.0)
        self.assertAlmostEqual(result.maxwell_viscosity, 100.0)
        self.assertAlmostEqual(result.kelvin_viscosity, 20.0)
        self.assertAlmostEqual(result.retardation_time, 2.0)
        self.assertLess(result.mse, 1e-10)


class MaxwellMechanicsTests(unittest.TestCase):
    def test_maxwell_relaxation_response_and_coarse_fit(self) -> None:
        time = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        applied_strain = 0.2
        elastic_modulus = 50.0
        viscosity = 100.0
        stress = maxwell_stress_relaxation(time, applied_strain, elastic_modulus, viscosity)
        expected = np.array(
            [
                10.0,
                10.0 * math.exp(-0.5),
                10.0 * math.exp(-1.0),
                10.0 * math.exp(-1.5),
            ],
            dtype=float,
        )
        np.testing.assert_allclose(stress, expected)
        result = fit_maxwell_coarse(
            time,
            stress,
            applied_strain,
            modulus_grid=np.array([40.0, 50.0, 60.0], dtype=float),
            tau_grid=np.array([1.0, 2.0, 3.0], dtype=float),
        )
        self.assertAlmostEqual(result.elastic_modulus, 50.0)
        self.assertAlmostEqual(result.relaxation_time, 2.0)
        self.assertAlmostEqual(result.viscosity, 100.0)
        self.assertAlmostEqual(result.mse, 0.0)


class FrequencySweepMechanicsTests(unittest.TestCase):
    def test_frequency_sweep_response_and_fit(self) -> None:
        frequency = np.array([0.1, 0.3, 1.0, 3.0], dtype=float)
        storage, loss = generalized_maxwell_frequency_response(frequency, 2.0, 6.0, 1.2)
        result = fit_frequency_sweep_coarse(
            frequency,
            storage,
            loss,
            equilibrium_modulus_grid=np.array([1.5, 2.0, 2.5], dtype=float),
            dynamic_modulus_grid=np.array([5.0, 6.0, 7.0], dtype=float),
            tau_grid=np.array([0.8, 1.2, 1.6], dtype=float),
        )
        self.assertAlmostEqual(result.equilibrium_modulus, 2.0)
        self.assertAlmostEqual(result.dynamic_modulus, 6.0)
        self.assertAlmostEqual(result.relaxation_time, 1.2)
        self.assertLess(result.mse, 1e-10)


class CyclicMechanicsTests(unittest.TestCase):
    def test_cyclic_analysis_returns_hysteresis_metrics(self) -> None:
        strain = np.array([0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.08, 0.16, 0.08, 0.0], dtype=float)
        stress = np.array([0.0, 1.0, 2.0, 1.1, 0.1, 0.0, 0.8, 1.6, 0.9, 0.05], dtype=float)
        cycle = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
        result = analyze_cyclic_response(strain, stress, cycle=cycle)
        self.assertGreater(result.secant_modulus, 0.0)
        self.assertGreaterEqual(result.hysteresis_area, 0.0)
        self.assertGreaterEqual(result.loss_factor, 0.0)
        self.assertEqual(result.cycle_count, 2)


class DatasetLevelMechanicsTests(unittest.TestCase):
    def test_fit_mechanics_dataset_auto_detects_frequency_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "freq.csv"
            frequency = np.array([0.1, 0.3, 1.0, 3.0], dtype=float)
            storage, loss = generalized_maxwell_frequency_response(frequency, 2.0, 6.0, 1.2)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["frequency", "storage_modulus", "loss_modulus"])
                writer.writeheader()
                for f, gp, gpp in zip(frequency, storage, loss):
                    writer.writerow({"frequency": f, "storage_modulus": gp, "loss_modulus": gpp})

            payload = fit_mechanics_dataset(path, experiment_type="auto")
            self.assertEqual(payload["experiment_type"], "frequency_sweep")
            self.assertEqual(payload["selected_model"], "generalized_maxwell_frequency_sweep")
            self.assertIn("candidate_models", payload)
            self.assertIn("parameter_intervals", payload)

    def test_simulate_mechanics_curve_supports_frequency_sweep(self) -> None:
        payload = simulate_mechanics_curve(
            model_type="generalized_maxwell_frequency_sweep",
            x_values=[0.1, 1.0, 10.0],
            parameters={"equilibrium_modulus": 2.0, "dynamic_modulus": 5.0, "relaxation_time": 1.5},
        )
        self.assertEqual(payload["model_type"], "generalized_maxwell_frequency_sweep")
        self.assertIn("secondary_y_values", payload)
        self.assertEqual(len(payload["x_values"]), len(payload["y_values"]))


class PowerLawTests(unittest.TestCase):
    def test_power_law_stiffness_utility(self) -> None:
        strain = np.array([1.0, 2.0, 4.0], dtype=float)
        stress = power_law_stress(strain, coefficient=2.0, exponent=1.5)
        stiffness = power_law_stiffness(strain, coefficient=2.0, exponent=1.5)
        np.testing.assert_allclose(stress, 2.0 * np.power(strain, 1.5))
        np.testing.assert_allclose(stiffness, 3.0 * np.power(strain, 0.5))


if __name__ == "__main__":
    unittest.main()
