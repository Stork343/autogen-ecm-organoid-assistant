"""ECM mechanics models, dataset loaders, fitting helpers, and uncertainty summaries."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

ArrayLike = Sequence[float] | Iterable[float] | np.ndarray | float
ExperimentType = Literal["auto", "elastic", "creep", "relaxation", "frequency_sweep", "cyclic"]
GRID_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class LinearElasticFitResult:
    modulus: float
    mse: float
    r2: float
    n_points: int


@dataclass(frozen=True)
class PowerLawElasticFitResult:
    coefficient: float
    exponent: float
    mse: float
    r2: float
    n_points: int


@dataclass(frozen=True)
class KelvinVoigtFitResult:
    elastic_modulus: float
    viscosity: float
    relaxation_time: float
    mse: float
    n_points: int
    modulus_grid_size: int
    tau_grid_size: int


@dataclass(frozen=True)
class StandardLinearSolidFitResult:
    instantaneous_modulus: float
    equilibrium_modulus: float
    branch_viscosity: float
    relaxation_time: float
    mse: float
    n_points: int
    instantaneous_grid_size: int
    equilibrium_fraction_grid_size: int
    tau_grid_size: int


@dataclass(frozen=True)
class BurgersFitResult:
    instantaneous_modulus: float
    delayed_modulus: float
    maxwell_viscosity: float
    kelvin_viscosity: float
    retardation_time: float
    mse: float
    n_points: int
    instantaneous_grid_size: int
    delayed_grid_size: int
    retardation_grid_size: int
    maxwell_viscosity_grid_size: int


@dataclass(frozen=True)
class MaxwellFitResult:
    elastic_modulus: float
    viscosity: float
    relaxation_time: float
    mse: float
    n_points: int
    modulus_grid_size: int
    tau_grid_size: int


@dataclass(frozen=True)
class FrequencySweepFitResult:
    equilibrium_modulus: float
    dynamic_modulus: float
    relaxation_time: float
    crossover_frequency_hz: float
    mse: float
    n_points: int
    modulus_grid_size: int
    tau_grid_size: int


@dataclass(frozen=True)
class CyclicLoopFitResult:
    secant_modulus: float
    hysteresis_area: float
    loss_factor: float
    residual_strain: float
    peak_stress: float
    peak_strain: float
    cycle_count: int
    n_points: int


def _as_1d_array(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _broadcast_step_input(name: str, values: ArrayLike, n: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(n, float(array), dtype=float)
    array = _as_1d_array(name, array)
    if array.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have length {n}.")
    return array


def _default_positive_grid(guess: float, *, size: int = 17, factor: float = 10.0) -> np.ndarray:
    guess = float(abs(guess))
    if not np.isfinite(guess) or guess <= 0.0:
        guess = 1.0
    lower = guess / factor
    upper = guess * factor
    if lower <= 0.0:
        lower = np.finfo(float).tiny
    return np.logspace(np.log10(lower), np.log10(upper), size)


def _bounded_linear_grid(center: float, *, lower: float, upper: float, size: int = 17, span: float = 0.5) -> np.ndarray:
    center = float(center)
    if not np.isfinite(center):
        center = (lower + upper) / 2.0
    start = max(lower, center * (1.0 - span))
    stop = min(upper, center * (1.0 + span))
    if stop <= start:
        stop = max(start + np.finfo(float).eps, upper)
    return np.linspace(start, stop, size)


def _estimate_time_scale(time: np.ndarray) -> float:
    positive = time[time > 0.0]
    if positive.size:
        return float(np.median(positive))
    span = float(np.max(time) - np.min(time))
    if span > 0.0:
        return span / 2.0
    return 1.0


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = observed - float(np.mean(observed))
    total_ss = float(np.sum(centered**2))
    residual_ss = float(np.sum((observed - predicted) ** 2))
    if total_ss == 0.0:
        return 1.0 if residual_ss == 0.0 else 0.0
    return float(1.0 - residual_ss / total_ss)


def _selection_score(mse: float, n_points: int, n_params: int) -> float:
    return float(n_points * np.log(max(mse, np.finfo(float).tiny)) + 2.0 * n_params)


def _grid_diagnostics(
    *,
    best_score: float,
    score_grid: np.ndarray,
    grids: dict[str, np.ndarray],
    best_values: dict[str, float],
) -> dict[str, Any]:
    threshold = float(best_score * (1.0 + GRID_RELATIVE_TOLERANCE) + np.finfo(float).eps)
    mask = np.asarray(score_grid <= threshold)
    if not np.any(mask):
        mask = np.asarray(score_grid == np.min(score_grid))

    indices = np.argwhere(mask)
    intervals: dict[str, dict[str, float]] = {}
    normalized_widths: list[float] = []
    for axis, (name, values) in enumerate(grids.items()):
        if indices.size == 0:
            selected = np.asarray([best_values[name]], dtype=float)
        else:
            selected = np.asarray(values[indices[:, axis]], dtype=float)
        lower = float(np.min(selected))
        upper = float(np.max(selected))
        best = float(best_values[name])
        intervals[name] = {"min": lower, "max": upper, "best": best}
        normalized_widths.append((upper - lower) / max(abs(best), 1e-12))

    max_width = float(max(normalized_widths) if normalized_widths else 0.0)
    if max_width > 0.75:
        risk = "high"
    elif max_width > 0.25:
        risk = "medium"
    else:
        risk = "low"

    return {
        "parameter_intervals": intervals,
        "identifiability": {
            "near_optimal_count": int(indices.shape[0]) if indices.size else 1,
            "max_normalized_interval_width": max_width,
            "risk": risk,
        },
    }


def linear_elastic_stress(strain: ArrayLike, modulus: float) -> np.ndarray:
    return float(modulus) * np.asarray(strain, dtype=float)


def power_law_stress(
    strain: ArrayLike,
    coefficient: float,
    exponent: float,
    *,
    reference_strain: float = 1.0,
) -> np.ndarray:
    strain_array = np.asarray(strain, dtype=float)
    scaled = np.maximum(strain_array / max(reference_strain, np.finfo(float).eps), np.finfo(float).tiny)
    return float(coefficient) * np.power(scaled, float(exponent))


def power_law_stiffness(
    strain: ArrayLike,
    coefficient: float,
    exponent: float,
    *,
    reference_strain: float = 1.0,
) -> np.ndarray:
    strain_array = np.asarray(strain, dtype=float)
    scaled = np.maximum(strain_array / max(reference_strain, np.finfo(float).eps), np.finfo(float).tiny)
    return float(coefficient) * float(exponent) / max(reference_strain, np.finfo(float).eps) * np.power(
        scaled, float(exponent) - 1.0
    )


def kelvin_voigt_creep_strain(
    time: ArrayLike,
    applied_stress: ArrayLike,
    elastic_modulus: float,
    viscosity: float,
) -> np.ndarray:
    time_array = _as_1d_array("time", time)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    return stress_array / elastic_modulus * (1.0 - np.exp(-(elastic_modulus * time_array) / viscosity))


def standard_linear_solid_creep_strain(
    time: ArrayLike,
    applied_stress: ArrayLike,
    instantaneous_modulus: float,
    equilibrium_modulus: float,
    relaxation_time: float,
) -> np.ndarray:
    time_array = _as_1d_array("time", time)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    e0 = max(float(instantaneous_modulus), np.finfo(float).eps)
    einf = max(min(float(equilibrium_modulus), e0), np.finfo(float).eps)
    tau = max(float(relaxation_time), np.finfo(float).eps)
    compliance_jump = max((1.0 / einf) - (1.0 / e0), 0.0)
    return stress_array / einf - stress_array * compliance_jump * np.exp(-time_array / tau)


def burgers_creep_strain(
    time: ArrayLike,
    applied_stress: ArrayLike,
    instantaneous_modulus: float,
    delayed_modulus: float,
    maxwell_viscosity: float,
    kelvin_viscosity: float,
) -> np.ndarray:
    time_array = _as_1d_array("time", time)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    e1 = max(float(instantaneous_modulus), np.finfo(float).eps)
    e2 = max(float(delayed_modulus), np.finfo(float).eps)
    eta1 = max(float(maxwell_viscosity), np.finfo(float).eps)
    eta2 = max(float(kelvin_viscosity), np.finfo(float).eps)
    tau = eta2 / e2
    return stress_array / e1 + stress_array / e2 * (1.0 - np.exp(-time_array / tau)) + stress_array * time_array / eta1


def maxwell_stress_relaxation(
    time: ArrayLike,
    applied_strain: ArrayLike,
    elastic_modulus: float,
    viscosity: float,
) -> np.ndarray:
    time_array = _as_1d_array("time", time)
    strain_array = _broadcast_step_input("applied_strain", applied_strain, time_array.size)
    return elastic_modulus * strain_array * np.exp(-(elastic_modulus * time_array) / viscosity)


def standard_linear_solid_stress_relaxation(
    time: ArrayLike,
    applied_strain: ArrayLike,
    instantaneous_modulus: float,
    equilibrium_modulus: float,
    relaxation_time: float,
) -> np.ndarray:
    time_array = _as_1d_array("time", time)
    strain_array = _broadcast_step_input("applied_strain", applied_strain, time_array.size)
    e0 = max(float(instantaneous_modulus), np.finfo(float).eps)
    einf = max(min(float(equilibrium_modulus), e0), np.finfo(float).eps)
    tau = max(float(relaxation_time), np.finfo(float).eps)
    return strain_array * (einf + (e0 - einf) * np.exp(-time_array / tau))


def generalized_maxwell_frequency_response(
    frequency_hz: ArrayLike,
    equilibrium_modulus: float,
    dynamic_modulus: float,
    relaxation_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_array = _as_1d_array("frequency_hz", frequency_hz)
    angular_frequency = 2.0 * np.pi * frequency_array
    tau = max(float(relaxation_time), np.finfo(float).eps)
    omega_tau = angular_frequency * tau
    g_inf = float(equilibrium_modulus)
    g_1 = float(dynamic_modulus)
    storage = g_inf + g_1 * (omega_tau**2) / (1.0 + omega_tau**2)
    loss = g_1 * omega_tau / (1.0 + omega_tau**2)
    return storage, loss


def fit_linear_elastic_through_origin(strain: ArrayLike, stress: ArrayLike) -> LinearElasticFitResult:
    strain_array = _as_1d_array("strain", strain)
    stress_array = _as_1d_array("stress", stress)
    if strain_array.shape != stress_array.shape:
        raise ValueError("strain and stress must have the same length.")
    denominator = float(np.dot(strain_array, strain_array))
    if denominator == 0.0:
        raise ValueError("strain must contain at least one non-zero value.")
    modulus = float(np.dot(strain_array, stress_array) / denominator)
    predicted = modulus * strain_array
    residuals = stress_array - predicted
    mse = float(np.mean(residuals**2))
    return LinearElasticFitResult(modulus=modulus, mse=mse, r2=_r2_score(stress_array, predicted), n_points=int(strain_array.size))


def fit_power_law_elastic(strain: ArrayLike, stress: ArrayLike, *, reference_strain: float = 1.0) -> PowerLawElasticFitResult:
    strain_array = _as_1d_array("strain", strain)
    stress_array = _as_1d_array("stress", stress)
    if strain_array.shape != stress_array.shape:
        raise ValueError("strain and stress must have the same length.")
    positive_mask = (strain_array > 0.0) & (stress_array > 0.0)
    if int(np.sum(positive_mask)) < 2:
        raise ValueError("power-law fit requires at least two strictly positive stress/strain points.")
    x = np.log(strain_array[positive_mask] / max(reference_strain, np.finfo(float).eps))
    y = np.log(stress_array[positive_mask])
    slope, intercept = np.polyfit(x, y, 1)
    coefficient = float(np.exp(intercept))
    exponent = float(slope)
    predicted = power_law_stress(strain_array, coefficient, exponent, reference_strain=reference_strain)
    mse = float(np.mean((predicted - stress_array) ** 2))
    return PowerLawElasticFitResult(
        coefficient=coefficient,
        exponent=exponent,
        mse=mse,
        r2=_r2_score(stress_array, predicted),
        n_points=int(strain_array.size),
    )


def fit_kelvin_voigt_coarse(
    time: ArrayLike,
    strain: ArrayLike,
    applied_stress: ArrayLike,
    *,
    modulus_grid: ArrayLike | None = None,
    tau_grid: ArrayLike | None = None,
) -> KelvinVoigtFitResult:
    time_array = _as_1d_array("time", time)
    strain_array = _as_1d_array("strain", strain)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    if strain_array.shape != time_array.shape:
        raise ValueError("time and strain must have the same length.")
    stress_scale = float(np.max(np.abs(stress_array))) or 1.0
    strain_scale = float(np.max(np.abs(strain_array))) or np.finfo(float).eps
    modulus_guess = stress_scale / max(strain_scale, np.finfo(float).eps)
    tau_guess = _estimate_time_scale(time_array)
    modulus_values = _as_1d_array("modulus_grid", modulus_grid) if modulus_grid is not None else _default_positive_grid(modulus_guess)
    tau_values = _as_1d_array("tau_grid", tau_grid) if tau_grid is not None else _default_positive_grid(tau_guess)
    preds = stress_array[:, None, None] / modulus_values[None, :, None]
    preds = preds * (1.0 - np.exp(-time_array[:, None, None] / tau_values[None, None, :]))
    mse_matrix = np.mean((preds - strain_array[:, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_matrix)), mse_matrix.shape)
    best_modulus = float(modulus_values[best_index[0]])
    best_tau = float(tau_values[best_index[1]])
    return KelvinVoigtFitResult(
        elastic_modulus=best_modulus,
        viscosity=float(best_modulus * best_tau),
        relaxation_time=best_tau,
        mse=float(mse_matrix[best_index]),
        n_points=int(time_array.size),
        modulus_grid_size=int(modulus_values.size),
        tau_grid_size=int(tau_values.size),
    )


def fit_sls_creep_coarse(
    time: ArrayLike,
    strain: ArrayLike,
    applied_stress: ArrayLike,
    *,
    instantaneous_modulus_grid: ArrayLike | None = None,
    equilibrium_fraction_grid: ArrayLike | None = None,
    tau_grid: ArrayLike | None = None,
) -> StandardLinearSolidFitResult:
    time_array = _as_1d_array("time", time)
    strain_array = _as_1d_array("strain", strain)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    if strain_array.shape != time_array.shape:
        raise ValueError("time and strain must have the same length.")
    stress_scale = float(np.max(np.abs(stress_array))) or 1.0
    strain_scale = float(np.max(np.abs(strain_array))) or np.finfo(float).eps
    modulus_guess = stress_scale / max(strain_scale, np.finfo(float).eps)
    tau_guess = _estimate_time_scale(time_array)
    instantaneous_values = (
        _as_1d_array("instantaneous_modulus_grid", instantaneous_modulus_grid)
        if instantaneous_modulus_grid is not None
        else _default_positive_grid(modulus_guess)
    )
    equilibrium_fractions = (
        _as_1d_array("equilibrium_fraction_grid", equilibrium_fraction_grid)
        if equilibrium_fraction_grid is not None
        else np.linspace(0.2, 0.95, 11)
    )
    tau_values = _as_1d_array("tau_grid", tau_grid) if tau_grid is not None else _default_positive_grid(tau_guess)
    equilibrium_values = instantaneous_values[:, None] * equilibrium_fractions[None, :]
    compliance_jump = np.maximum((1.0 / equilibrium_values[:, :, None]) - (1.0 / instantaneous_values[:, None, None]), 0.0)
    preds = stress_array[:, None, None, None] / equilibrium_values[None, :, :, None]
    preds = preds - stress_array[:, None, None, None] * compliance_jump[None, :, :, :] * np.exp(
        -time_array[:, None, None, None] / tau_values[None, None, None, :]
    )
    mse_tensor = np.mean((preds - strain_array[:, None, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_tensor)), mse_tensor.shape)
    e0 = float(instantaneous_values[best_index[0]])
    einf = float(equilibrium_values[best_index[0], best_index[1]])
    tau = float(tau_values[best_index[2]])
    return StandardLinearSolidFitResult(
        instantaneous_modulus=e0,
        equilibrium_modulus=einf,
        branch_viscosity=float(max((e0 - einf) * tau, np.finfo(float).eps)),
        relaxation_time=tau,
        mse=float(mse_tensor[best_index]),
        n_points=int(time_array.size),
        instantaneous_grid_size=int(instantaneous_values.size),
        equilibrium_fraction_grid_size=int(equilibrium_fractions.size),
        tau_grid_size=int(tau_values.size),
    )


def fit_burgers_creep_coarse(
    time: ArrayLike,
    strain: ArrayLike,
    applied_stress: ArrayLike,
    *,
    instantaneous_modulus_grid: ArrayLike | None = None,
    delayed_modulus_grid: ArrayLike | None = None,
    retardation_time_grid: ArrayLike | None = None,
    maxwell_viscosity_grid: ArrayLike | None = None,
) -> BurgersFitResult:
    time_array = _as_1d_array("time", time)
    strain_array = _as_1d_array("strain", strain)
    stress_array = _broadcast_step_input("applied_stress", applied_stress, time_array.size)
    if strain_array.shape != time_array.shape:
        raise ValueError("time and strain must have the same length.")
    stress_scale = float(np.max(np.abs(stress_array))) or 1.0
    strain_scale = float(np.max(np.abs(strain_array))) or np.finfo(float).eps
    modulus_guess = stress_scale / max(strain_scale, np.finfo(float).eps)
    tau_guess = _estimate_time_scale(time_array)
    eta_guess = stress_scale * max(tau_guess, 1.0) / max(strain_scale, np.finfo(float).eps)
    instantaneous_values = (
        _as_1d_array("instantaneous_modulus_grid", instantaneous_modulus_grid)
        if instantaneous_modulus_grid is not None
        else _default_positive_grid(modulus_guess, size=9)
    )
    delayed_values = (
        _as_1d_array("delayed_modulus_grid", delayed_modulus_grid)
        if delayed_modulus_grid is not None
        else _default_positive_grid(modulus_guess * 0.7, size=9)
    )
    retardation_values = (
        _as_1d_array("retardation_time_grid", retardation_time_grid)
        if retardation_time_grid is not None
        else _default_positive_grid(tau_guess, size=7)
    )
    maxwell_viscosity_values = (
        _as_1d_array("maxwell_viscosity_grid", maxwell_viscosity_grid)
        if maxwell_viscosity_grid is not None
        else _default_positive_grid(eta_guess, size=7)
    )
    kelvin_viscosity = delayed_values[None, :, None, None] * retardation_values[None, None, :, None]
    preds = stress_array[:, None, None, None, None] / instantaneous_values[None, :, None, None, None]
    preds = preds + stress_array[:, None, None, None, None] / delayed_values[None, None, :, None, None] * (
        1.0 - np.exp(-time_array[:, None, None, None, None] / retardation_values[None, None, None, :, None])
    )
    preds = preds + stress_array[:, None, None, None, None] * time_array[:, None, None, None, None] / maxwell_viscosity_values[
        None, None, None, None, :
    ]
    mse_tensor = np.mean((preds - strain_array[:, None, None, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_tensor)), mse_tensor.shape)
    e1 = float(instantaneous_values[best_index[0]])
    e2 = float(delayed_values[best_index[1]])
    tau = float(retardation_values[best_index[2]])
    eta1 = float(maxwell_viscosity_values[best_index[3]])
    return BurgersFitResult(
        instantaneous_modulus=e1,
        delayed_modulus=e2,
        maxwell_viscosity=eta1,
        kelvin_viscosity=float(e2 * tau),
        retardation_time=tau,
        mse=float(mse_tensor[best_index]),
        n_points=int(time_array.size),
        instantaneous_grid_size=int(instantaneous_values.size),
        delayed_grid_size=int(delayed_values.size),
        retardation_grid_size=int(retardation_values.size),
        maxwell_viscosity_grid_size=int(maxwell_viscosity_values.size),
    )


def fit_maxwell_coarse(
    time: ArrayLike,
    stress: ArrayLike,
    applied_strain: ArrayLike,
    *,
    modulus_grid: ArrayLike | None = None,
    tau_grid: ArrayLike | None = None,
) -> MaxwellFitResult:
    time_array = _as_1d_array("time", time)
    stress_array = _as_1d_array("stress", stress)
    strain_array = _broadcast_step_input("applied_strain", applied_strain, time_array.size)
    if stress_array.shape != time_array.shape:
        raise ValueError("time and stress must have the same length.")
    stress_scale = float(np.max(np.abs(stress_array))) or 1.0
    strain_scale = float(np.median(np.abs(strain_array))) or np.finfo(float).eps
    modulus_guess = stress_scale / max(strain_scale, np.finfo(float).eps)
    tau_guess = _estimate_time_scale(time_array)
    modulus_values = _as_1d_array("modulus_grid", modulus_grid) if modulus_grid is not None else _default_positive_grid(modulus_guess)
    tau_values = _as_1d_array("tau_grid", tau_grid) if tau_grid is not None else _default_positive_grid(tau_guess)
    preds = modulus_values[None, :, None] * strain_array[:, None, None] * np.exp(-time_array[:, None, None] / tau_values[None, None, :])
    mse_matrix = np.mean((preds - stress_array[:, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_matrix)), mse_matrix.shape)
    best_modulus = float(modulus_values[best_index[0]])
    best_tau = float(tau_values[best_index[1]])
    return MaxwellFitResult(
        elastic_modulus=best_modulus,
        viscosity=float(best_modulus * best_tau),
        relaxation_time=best_tau,
        mse=float(mse_matrix[best_index]),
        n_points=int(time_array.size),
        modulus_grid_size=int(modulus_values.size),
        tau_grid_size=int(tau_values.size),
    )


def fit_sls_relaxation_coarse(
    time: ArrayLike,
    stress: ArrayLike,
    applied_strain: ArrayLike,
    *,
    instantaneous_modulus_grid: ArrayLike | None = None,
    equilibrium_fraction_grid: ArrayLike | None = None,
    tau_grid: ArrayLike | None = None,
) -> StandardLinearSolidFitResult:
    time_array = _as_1d_array("time", time)
    stress_array = _as_1d_array("stress", stress)
    strain_array = _broadcast_step_input("applied_strain", applied_strain, time_array.size)
    if stress_array.shape != time_array.shape:
        raise ValueError("time and stress must have the same length.")
    stress_scale = float(np.max(np.abs(stress_array))) or 1.0
    strain_scale = float(np.max(np.abs(strain_array))) or np.finfo(float).eps
    modulus_guess = stress_scale / max(strain_scale, np.finfo(float).eps)
    tau_guess = _estimate_time_scale(time_array)
    instantaneous_values = (
        _as_1d_array("instantaneous_modulus_grid", instantaneous_modulus_grid)
        if instantaneous_modulus_grid is not None
        else _default_positive_grid(modulus_guess)
    )
    equilibrium_fractions = (
        _as_1d_array("equilibrium_fraction_grid", equilibrium_fraction_grid)
        if equilibrium_fraction_grid is not None
        else np.linspace(0.1, 0.9, 11)
    )
    tau_values = _as_1d_array("tau_grid", tau_grid) if tau_grid is not None else _default_positive_grid(tau_guess)
    equilibrium_values = instantaneous_values[:, None] * equilibrium_fractions[None, :]
    preds = strain_array[:, None, None, None] * (
        equilibrium_values[None, :, :, None]
        + (instantaneous_values[None, :, None, None] - equilibrium_values[None, :, :, None])
        * np.exp(-time_array[:, None, None, None] / tau_values[None, None, None, :])
    )
    mse_tensor = np.mean((preds - stress_array[:, None, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_tensor)), mse_tensor.shape)
    e0 = float(instantaneous_values[best_index[0]])
    einf = float(equilibrium_values[best_index[0], best_index[1]])
    tau = float(tau_values[best_index[2]])
    return StandardLinearSolidFitResult(
        instantaneous_modulus=e0,
        equilibrium_modulus=einf,
        branch_viscosity=float(max((e0 - einf) * tau, np.finfo(float).eps)),
        relaxation_time=tau,
        mse=float(mse_tensor[best_index]),
        n_points=int(time_array.size),
        instantaneous_grid_size=int(instantaneous_values.size),
        equilibrium_fraction_grid_size=int(equilibrium_fractions.size),
        tau_grid_size=int(tau_values.size),
    )


def fit_frequency_sweep_coarse(
    frequency_hz: ArrayLike,
    storage_modulus: ArrayLike,
    loss_modulus: ArrayLike,
    *,
    equilibrium_modulus_grid: ArrayLike | None = None,
    dynamic_modulus_grid: ArrayLike | None = None,
    tau_grid: ArrayLike | None = None,
) -> FrequencySweepFitResult:
    frequency_array = _as_1d_array("frequency_hz", frequency_hz)
    storage_array = _as_1d_array("storage_modulus", storage_modulus)
    loss_array = _as_1d_array("loss_modulus", loss_modulus)
    if storage_array.shape != frequency_array.shape or loss_array.shape != frequency_array.shape:
        raise ValueError("frequency, storage_modulus, and loss_modulus must have the same length.")
    equilibrium_guess = max(float(np.min(storage_array)), np.finfo(float).eps)
    dynamic_guess = max(float(np.max(storage_array) - np.min(storage_array)), equilibrium_guess * 0.25)
    tau_guess = 1.0 / max(2.0 * np.pi * float(np.median(frequency_array[frequency_array > 0.0])) if np.any(frequency_array > 0.0) else 1.0, np.finfo(float).eps)
    equilibrium_values = (
        _as_1d_array("equilibrium_modulus_grid", equilibrium_modulus_grid)
        if equilibrium_modulus_grid is not None
        else _default_positive_grid(equilibrium_guess, size=13, factor=4.0)
    )
    dynamic_values = (
        _as_1d_array("dynamic_modulus_grid", dynamic_modulus_grid)
        if dynamic_modulus_grid is not None
        else _default_positive_grid(dynamic_guess, size=13, factor=6.0)
    )
    tau_values = _as_1d_array("tau_grid", tau_grid) if tau_grid is not None else _default_positive_grid(tau_guess, size=13, factor=10.0)

    angular_frequency = 2.0 * np.pi * frequency_array[:, None, None, None]
    omega_tau = angular_frequency * tau_values[None, None, None, :]
    storage_pred = equilibrium_values[None, :, None, None] + dynamic_values[None, None, :, None] * (omega_tau**2) / (1.0 + omega_tau**2)
    loss_pred = dynamic_values[None, None, :, None] * omega_tau / (1.0 + omega_tau**2)
    mse_tensor = np.mean((storage_pred - storage_array[:, None, None, None]) ** 2 + (loss_pred - loss_array[:, None, None, None]) ** 2, axis=0)
    best_index = np.unravel_index(int(np.argmin(mse_tensor)), mse_tensor.shape)
    g_inf = float(equilibrium_values[best_index[0]])
    g_dyn = float(dynamic_values[best_index[1]])
    tau = float(tau_values[best_index[2]])
    return FrequencySweepFitResult(
        equilibrium_modulus=g_inf,
        dynamic_modulus=g_dyn,
        relaxation_time=tau,
        crossover_frequency_hz=float(1.0 / max(2.0 * np.pi * tau, np.finfo(float).eps)),
        mse=float(mse_tensor[best_index]),
        n_points=int(frequency_array.size),
        modulus_grid_size=int(equilibrium_values.size * dynamic_values.size),
        tau_grid_size=int(tau_values.size),
    )


def analyze_cyclic_response(
    strain: ArrayLike,
    stress: ArrayLike,
    *,
    cycle: ArrayLike | None = None,
) -> CyclicLoopFitResult:
    strain_array = _as_1d_array("strain", strain)
    stress_array = _as_1d_array("stress", stress)
    if strain_array.shape != stress_array.shape:
        raise ValueError("strain and stress must have the same length.")
    if cycle is None:
        cycle_array = np.zeros_like(strain_array, dtype=int)
    else:
        cycle_array = np.asarray(cycle, dtype=int)
        if cycle_array.shape != strain_array.shape:
            raise ValueError("cycle must have the same length as strain/stress.")

    secant_values = []
    hysteresis_values = []
    residual_values = []
    peak_stresses = []
    peak_strains = []
    for cycle_id in np.unique(cycle_array):
        mask = cycle_array == cycle_id
        local_strain = strain_array[mask]
        local_stress = stress_array[mask]
        if local_strain.size < 2:
            continue
        strain_span = float(np.max(local_strain) - np.min(local_strain))
        stress_span = float(np.max(local_stress) - np.min(local_stress))
        secant_values.append(stress_span / max(strain_span, np.finfo(float).eps))
        hysteresis_values.append(float(abs(np.trapezoid(local_stress, local_strain))))
        zero_idx = int(np.argmin(np.abs(local_stress)))
        residual_values.append(float(local_strain[zero_idx]))
        peak_stresses.append(float(np.max(np.abs(local_stress))))
        peak_strains.append(float(np.max(np.abs(local_strain))))

    if not secant_values:
        raise ValueError("cyclic analysis requires at least one cycle with more than one point.")
    secant_modulus = float(np.mean(secant_values))
    hysteresis_area = float(np.mean(hysteresis_values))
    peak_stress = float(np.mean(peak_stresses))
    peak_strain = float(np.mean(peak_strains))
    stored_energy = 0.5 * peak_stress * peak_strain
    loss_factor = float(hysteresis_area / max(2.0 * np.pi * stored_energy, np.finfo(float).eps))
    return CyclicLoopFitResult(
        secant_modulus=secant_modulus,
        hysteresis_area=hysteresis_area,
        loss_factor=loss_factor,
        residual_strain=float(np.mean(residual_values)),
        peak_stress=peak_stress,
        peak_strain=peak_strain,
        cycle_count=int(len(secant_values)),
        n_points=int(strain_array.size),
    )


def load_mechanics_dataset(
    data_path: Path | str,
    *,
    delimiter: str = ",",
    time_column: str = "time",
    stress_column: str = "stress",
    strain_column: str = "strain",
) -> dict[str, list[float]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    alias_lookup = {
        "time": [time_column, "time", "t"],
        "stress": [stress_column, "stress", "sigma"],
        "strain": [strain_column, "strain", "epsilon"],
        "frequency_hz": ["frequency", "frequency_hz", "freq", "hz"],
        "storage_modulus": ["storage_modulus", "g_prime", "gstorage", "storage", "g'"],
        "loss_modulus": ["loss_modulus", "g_double_prime", "gloss", "loss", "g''"],
        "cycle": ["cycle", "cycle_index", "cycle_id"],
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return {}
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field is not None}
        resolved_columns: dict[str, str] = {}
        for canonical, aliases in alias_lookup.items():
            for alias in aliases:
                candidate = field_map.get(alias.strip().lower())
                if candidate is not None:
                    resolved_columns[canonical] = candidate
                    break

        columns: dict[str, list[float]] = {key: [] for key in resolved_columns}
        for row in reader:
            for canonical, source in resolved_columns.items():
                value = row.get(source)
                if value in (None, ""):
                    continue
                try:
                    columns[canonical].append(float(value))
                except ValueError:
                    continue
    return {key: values for key, values in columns.items() if values}


def infer_experiment_type(
    dataset: dict[str, list[float]],
    *,
    experiment_type: ExperimentType = "auto",
    applied_stress: float | None = None,
    applied_strain: float | None = None,
) -> str:
    if experiment_type != "auto":
        return experiment_type
    has_time = "time" in dataset
    has_stress = "stress" in dataset
    has_strain = "strain" in dataset
    has_frequency = "frequency_hz" in dataset
    has_storage = "storage_modulus" in dataset
    has_loss = "loss_modulus" in dataset
    has_cycle = "cycle" in dataset
    if has_frequency and has_storage and has_loss:
        return "frequency_sweep"
    if has_cycle and has_stress and has_strain:
        return "cyclic"
    if has_time and has_strain and applied_stress is not None and applied_stress > 0:
        return "creep"
    if has_time and has_stress and applied_strain is not None and applied_strain > 0:
        return "relaxation"
    if has_stress and has_strain:
        return "elastic"
    raise ValueError("Unable to infer experiment type from dataset columns and loading conditions.")


def _fit_payload(
    *,
    model_type: str,
    fit_result: Any,
    best_score: float,
    score_grid: np.ndarray | None,
    grids: dict[str, np.ndarray] | None,
    best_values: dict[str, float],
) -> dict[str, Any]:
    payload = {
        "model_type": model_type,
        "fit": _fit_to_dict(fit_result),
        "selection_score": float(best_score),
    }
    if score_grid is not None and grids is not None:
        payload.update(
            _grid_diagnostics(
                best_score=float(best_score),
                score_grid=np.asarray(score_grid, dtype=float),
                grids=grids,
                best_values=best_values,
            )
        )
    return payload


def fit_mechanics_dataset(
    data_path: Path | str,
    *,
    experiment_type: ExperimentType = "auto",
    delimiter: str = ",",
    time_column: str = "time",
    stress_column: str = "stress",
    strain_column: str = "strain",
    applied_stress: float | None = None,
    applied_strain: float | None = None,
) -> dict[str, Any]:
    dataset = load_mechanics_dataset(
        data_path,
        delimiter=delimiter,
        time_column=time_column,
        stress_column=stress_column,
        strain_column=strain_column,
    )
    resolved_type = infer_experiment_type(
        dataset,
        experiment_type=experiment_type,
        applied_stress=applied_stress,
        applied_strain=applied_strain,
    )

    candidate_models: list[dict[str, Any]] = []
    selected_model = ""
    selected_fit: dict[str, Any] = {}

    if resolved_type == "elastic":
        strain_array = np.asarray(dataset["strain"], dtype=float)
        stress_array = np.asarray(dataset["stress"], dtype=float)
        linear_fit = fit_linear_elastic_through_origin(strain_array, stress_array)
        linear_grid = _default_positive_grid(linear_fit.modulus, size=31, factor=2.0)
        linear_preds = linear_grid[None, :] * strain_array[:, None]
        linear_mse = np.mean((linear_preds - stress_array[:, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="linear_elastic",
                fit_result=linear_fit,
                best_score=_selection_score(linear_fit.mse, linear_fit.n_points, 1),
                score_grid=linear_mse,
                grids={"modulus": linear_grid},
                best_values={"modulus": linear_fit.modulus},
            )
        )

        positive_mask = (strain_array > 0.0) & (stress_array > 0.0)
        if int(np.sum(positive_mask)) >= 2:
            power_fit = fit_power_law_elastic(strain_array, stress_array)
            coefficient_grid = _default_positive_grid(power_fit.coefficient, size=21, factor=3.0)
            exponent_grid = _bounded_linear_grid(power_fit.exponent, lower=0.25, upper=4.0, size=21, span=0.4)
            preds = power_law_stress(
                strain_array[:, None, None],
                coefficient_grid[None, :, None],
                exponent_grid[None, None, :],
            )
            power_mse = np.mean((preds - stress_array[:, None, None]) ** 2, axis=0)
            candidate_models.append(
                _fit_payload(
                    model_type="power_law_elastic",
                    fit_result=power_fit,
                    best_score=_selection_score(power_fit.mse, power_fit.n_points, 2),
                    score_grid=power_mse,
                    grids={"coefficient": coefficient_grid, "exponent": exponent_grid},
                    best_values={"coefficient": power_fit.coefficient, "exponent": power_fit.exponent},
                )
            )
    elif resolved_type == "creep":
        time_array = np.asarray(dataset["time"], dtype=float)
        strain_array = np.asarray(dataset["strain"], dtype=float)
        applied = float(applied_stress if applied_stress is not None else 0.0)

        kv_fit = fit_kelvin_voigt_coarse(time_array, strain_array, applied)
        kv_modulus_grid = _default_positive_grid(kv_fit.elastic_modulus, size=17, factor=3.0)
        kv_tau_grid = _default_positive_grid(kv_fit.relaxation_time, size=17, factor=4.0)
        kv_preds = applied / kv_modulus_grid[None, :, None] * (1.0 - np.exp(-time_array[:, None, None] / kv_tau_grid[None, None, :]))
        kv_mse = np.mean((kv_preds - strain_array[:, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="kelvin_voigt_creep",
                fit_result=kv_fit,
                best_score=_selection_score(kv_fit.mse, kv_fit.n_points, 2),
                score_grid=kv_mse,
                grids={"elastic_modulus": kv_modulus_grid, "relaxation_time": kv_tau_grid},
                best_values={"elastic_modulus": kv_fit.elastic_modulus, "relaxation_time": kv_fit.relaxation_time},
            )
        )

        sls_fit = fit_sls_creep_coarse(time_array, strain_array, applied)
        sls_e0_grid = _default_positive_grid(sls_fit.instantaneous_modulus, size=13, factor=3.0)
        sls_frac_grid = np.linspace(0.2, 0.95, 11)
        sls_tau_grid = _default_positive_grid(sls_fit.relaxation_time, size=13, factor=4.0)
        sls_einf = sls_e0_grid[:, None] * sls_frac_grid[None, :]
        compliance_jump = np.maximum((1.0 / sls_einf[:, :, None]) - (1.0 / sls_e0_grid[:, None, None]), 0.0)
        sls_preds = applied / sls_einf[None, :, :, None] - applied * compliance_jump[None, :, :, :] * np.exp(
            -time_array[:, None, None, None] / sls_tau_grid[None, None, None, :]
        )
        sls_mse = np.mean((sls_preds - strain_array[:, None, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="standard_linear_solid_creep",
                fit_result=sls_fit,
                best_score=_selection_score(sls_fit.mse, sls_fit.n_points, 3),
                score_grid=sls_mse,
                grids={
                    "instantaneous_modulus": sls_e0_grid,
                    "equilibrium_fraction": sls_frac_grid,
                    "relaxation_time": sls_tau_grid,
                },
                best_values={
                    "instantaneous_modulus": sls_fit.instantaneous_modulus,
                    "equilibrium_fraction": sls_fit.equilibrium_modulus / max(sls_fit.instantaneous_modulus, np.finfo(float).eps),
                    "relaxation_time": sls_fit.relaxation_time,
                },
            )
        )

        burgers_fit = fit_burgers_creep_coarse(time_array, strain_array, applied)
        e1_grid = _default_positive_grid(burgers_fit.instantaneous_modulus, size=9, factor=2.5)
        e2_grid = _default_positive_grid(burgers_fit.delayed_modulus, size=9, factor=2.5)
        tau_grid = _default_positive_grid(burgers_fit.retardation_time, size=7, factor=3.0)
        eta1_grid = _default_positive_grid(burgers_fit.maxwell_viscosity, size=7, factor=3.0)
        burgers_preds = applied / e1_grid[None, :, None, None, None]
        burgers_preds = burgers_preds + applied / e2_grid[None, None, :, None, None] * (
            1.0 - np.exp(-time_array[:, None, None, None, None] / tau_grid[None, None, None, :, None])
        )
        burgers_preds = burgers_preds + applied * time_array[:, None, None, None, None] / eta1_grid[None, None, None, None, :]
        burgers_mse = np.mean((burgers_preds - strain_array[:, None, None, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="burgers_creep",
                fit_result=burgers_fit,
                best_score=_selection_score(burgers_fit.mse, burgers_fit.n_points, 4),
                score_grid=burgers_mse,
                grids={
                    "instantaneous_modulus": e1_grid,
                    "delayed_modulus": e2_grid,
                    "retardation_time": tau_grid,
                    "maxwell_viscosity": eta1_grid,
                },
                best_values={
                    "instantaneous_modulus": burgers_fit.instantaneous_modulus,
                    "delayed_modulus": burgers_fit.delayed_modulus,
                    "retardation_time": burgers_fit.retardation_time,
                    "maxwell_viscosity": burgers_fit.maxwell_viscosity,
                },
            )
        )
    elif resolved_type == "relaxation":
        time_array = np.asarray(dataset["time"], dtype=float)
        stress_array = np.asarray(dataset["stress"], dtype=float)
        applied = float(applied_strain if applied_strain is not None else 0.0)

        maxwell_fit = fit_maxwell_coarse(time_array, stress_array, applied)
        mx_modulus_grid = _default_positive_grid(maxwell_fit.elastic_modulus, size=17, factor=3.0)
        mx_tau_grid = _default_positive_grid(maxwell_fit.relaxation_time, size=17, factor=4.0)
        mx_preds = mx_modulus_grid[None, :, None] * applied * np.exp(-time_array[:, None, None] / mx_tau_grid[None, None, :])
        mx_mse = np.mean((mx_preds - stress_array[:, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="maxwell_relaxation",
                fit_result=maxwell_fit,
                best_score=_selection_score(maxwell_fit.mse, maxwell_fit.n_points, 2),
                score_grid=mx_mse,
                grids={"elastic_modulus": mx_modulus_grid, "relaxation_time": mx_tau_grid},
                best_values={"elastic_modulus": maxwell_fit.elastic_modulus, "relaxation_time": maxwell_fit.relaxation_time},
            )
        )

        sls_fit = fit_sls_relaxation_coarse(time_array, stress_array, applied)
        sls_e0_grid = _default_positive_grid(sls_fit.instantaneous_modulus, size=13, factor=3.0)
        sls_frac_grid = np.linspace(0.1, 0.9, 11)
        sls_tau_grid = _default_positive_grid(sls_fit.relaxation_time, size=13, factor=4.0)
        sls_einf = sls_e0_grid[:, None] * sls_frac_grid[None, :]
        sls_preds = applied * (
            sls_einf[None, :, :, None]
            + (sls_e0_grid[None, :, None, None] - sls_einf[None, :, :, None]) * np.exp(-time_array[:, None, None, None] / sls_tau_grid[None, None, None, :])
        )
        sls_mse = np.mean((sls_preds - stress_array[:, None, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="standard_linear_solid_relaxation",
                fit_result=sls_fit,
                best_score=_selection_score(sls_fit.mse, sls_fit.n_points, 3),
                score_grid=sls_mse,
                grids={
                    "instantaneous_modulus": sls_e0_grid,
                    "equilibrium_fraction": sls_frac_grid,
                    "relaxation_time": sls_tau_grid,
                },
                best_values={
                    "instantaneous_modulus": sls_fit.instantaneous_modulus,
                    "equilibrium_fraction": sls_fit.equilibrium_modulus / max(sls_fit.instantaneous_modulus, np.finfo(float).eps),
                    "relaxation_time": sls_fit.relaxation_time,
                },
            )
        )
    elif resolved_type == "frequency_sweep":
        frequency_array = np.asarray(dataset["frequency_hz"], dtype=float)
        storage_array = np.asarray(dataset["storage_modulus"], dtype=float)
        loss_array = np.asarray(dataset["loss_modulus"], dtype=float)
        frequency_fit = fit_frequency_sweep_coarse(frequency_array, storage_array, loss_array)
        eq_grid = _default_positive_grid(frequency_fit.equilibrium_modulus, size=13, factor=3.0)
        dyn_grid = _default_positive_grid(frequency_fit.dynamic_modulus, size=13, factor=3.0)
        tau_grid = _default_positive_grid(frequency_fit.relaxation_time, size=13, factor=4.0)
        angular_frequency = 2.0 * np.pi * frequency_array[:, None, None, None]
        omega_tau = angular_frequency * tau_grid[None, None, None, :]
        storage_pred = eq_grid[None, :, None, None] + dyn_grid[None, None, :, None] * (omega_tau**2) / (1.0 + omega_tau**2)
        loss_pred = dyn_grid[None, None, :, None] * omega_tau / (1.0 + omega_tau**2)
        freq_mse = np.mean((storage_pred - storage_array[:, None, None, None]) ** 2 + (loss_pred - loss_array[:, None, None, None]) ** 2, axis=0)
        candidate_models.append(
            _fit_payload(
                model_type="generalized_maxwell_frequency_sweep",
                fit_result=frequency_fit,
                best_score=_selection_score(frequency_fit.mse, frequency_fit.n_points, 3),
                score_grid=freq_mse,
                grids={
                    "equilibrium_modulus": eq_grid,
                    "dynamic_modulus": dyn_grid,
                    "relaxation_time": tau_grid,
                },
                best_values={
                    "equilibrium_modulus": frequency_fit.equilibrium_modulus,
                    "dynamic_modulus": frequency_fit.dynamic_modulus,
                    "relaxation_time": frequency_fit.relaxation_time,
                },
            )
        )
    elif resolved_type == "cyclic":
        cyclic_fit = analyze_cyclic_response(
            dataset["strain"],
            dataset["stress"],
            cycle=dataset.get("cycle"),
        )
        candidate_models.append(
            {
                "model_type": "cyclic_loop_metrics",
                "fit": _fit_to_dict(cyclic_fit),
                "selection_score": 0.0,
                "parameter_intervals": {},
                "identifiability": {"near_optimal_count": 1, "max_normalized_interval_width": 0.0, "risk": "low"},
            }
        )
    else:
        raise ValueError(f"Unsupported experiment type: {resolved_type}")

    selected = min(candidate_models, key=lambda item: float(item.get("selection_score", np.inf)))
    selected_model = str(selected["model_type"])
    selected_fit = selected["fit"]
    return {
        "experiment_type": resolved_type,
        "selected_model": selected_model,
        "data_path": str(Path(data_path).resolve()),
        "available_columns": sorted(dataset.keys()),
        "sample_count": max(len(values) for values in dataset.values()),
        "fit": selected_fit,
        "candidate_models": candidate_models,
        "parameter_intervals": selected.get("parameter_intervals", {}),
        "identifiability": selected.get("identifiability", {}),
    }


def simulate_mechanics_curve(
    *,
    model_type: str,
    x_values: ArrayLike,
    parameters: dict[str, float],
    applied_stress: float | None = None,
    applied_strain: float | None = None,
) -> dict[str, Any]:
    x_array = _as_1d_array("x_values", x_values)
    if model_type == "linear_elastic":
        y_array = linear_elastic_stress(x_array, float(parameters["modulus"]))
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "stress", "y_values": y_array.tolist()}
    if model_type == "power_law_elastic":
        y_array = power_law_stress(x_array, float(parameters["coefficient"]), float(parameters["exponent"]))
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "stress", "y_values": y_array.tolist()}
    if model_type in {"creep", "kelvin_voigt_creep"}:
        y_array = kelvin_voigt_creep_strain(x_array, float(applied_stress), float(parameters["elastic_modulus"]), float(parameters["viscosity"]))
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "strain", "y_values": y_array.tolist()}
    if model_type == "standard_linear_solid_creep":
        y_array = standard_linear_solid_creep_strain(
            x_array,
            float(applied_stress),
            float(parameters["instantaneous_modulus"]),
            float(parameters["equilibrium_modulus"]),
            float(parameters["relaxation_time"]),
        )
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "strain", "y_values": y_array.tolist()}
    if model_type == "burgers_creep":
        y_array = burgers_creep_strain(
            x_array,
            float(applied_stress),
            float(parameters["instantaneous_modulus"]),
            float(parameters["delayed_modulus"]),
            float(parameters["maxwell_viscosity"]),
            float(parameters["kelvin_viscosity"]),
        )
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "strain", "y_values": y_array.tolist()}
    if model_type in {"relaxation", "maxwell_relaxation"}:
        y_array = maxwell_stress_relaxation(x_array, float(applied_strain), float(parameters["elastic_modulus"]), float(parameters["viscosity"]))
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "stress", "y_values": y_array.tolist()}
    if model_type == "standard_linear_solid_relaxation":
        y_array = standard_linear_solid_stress_relaxation(
            x_array,
            float(applied_strain),
            float(parameters["instantaneous_modulus"]),
            float(parameters["equilibrium_modulus"]),
            float(parameters["relaxation_time"]),
        )
        return {"model_type": model_type, "x_values": x_array.tolist(), "y_label": "stress", "y_values": y_array.tolist()}
    if model_type == "generalized_maxwell_frequency_sweep":
        storage, loss = generalized_maxwell_frequency_response(
            x_array,
            float(parameters["equilibrium_modulus"]),
            float(parameters["dynamic_modulus"]),
            float(parameters["relaxation_time"]),
        )
        return {
            "model_type": model_type,
            "x_values": x_array.tolist(),
            "y_label": "storage_modulus",
            "y_values": storage.tolist(),
            "secondary_y_label": "loss_modulus",
            "secondary_y_values": loss.tolist(),
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_to_dict(fit: Any) -> dict[str, Any]:
    payload = asdict(fit)
    return {
        key: (float(value) if isinstance(value, (np.floating, float, int, np.integer)) else value)
        for key, value in payload.items()
    }


__all__ = [
    "BurgersFitResult",
    "CyclicLoopFitResult",
    "FrequencySweepFitResult",
    "KelvinVoigtFitResult",
    "LinearElasticFitResult",
    "MaxwellFitResult",
    "PowerLawElasticFitResult",
    "StandardLinearSolidFitResult",
    "analyze_cyclic_response",
    "burgers_creep_strain",
    "fit_burgers_creep_coarse",
    "fit_frequency_sweep_coarse",
    "fit_kelvin_voigt_coarse",
    "fit_linear_elastic_through_origin",
    "fit_mechanics_dataset",
    "fit_maxwell_coarse",
    "fit_power_law_elastic",
    "fit_sls_creep_coarse",
    "fit_sls_relaxation_coarse",
    "generalized_maxwell_frequency_response",
    "infer_experiment_type",
    "kelvin_voigt_creep_strain",
    "linear_elastic_stress",
    "load_mechanics_dataset",
    "maxwell_stress_relaxation",
    "power_law_stiffness",
    "power_law_stress",
    "simulate_mechanics_curve",
    "standard_linear_solid_creep_strain",
    "standard_linear_solid_stress_relaxation",
]
