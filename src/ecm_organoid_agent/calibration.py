from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import read_xlsx_workbook
from .fiber_network import default_design_search_space, design_ecm_candidates


@dataclass(frozen=True)
class SummaryMeasurement:
    dataset_id: str
    material_family: str
    sample_key: str
    source_file: str
    sheet_name: str
    metric_name: str
    value: float
    std: float | None
    units: str
    conditions: dict[str, str]


def extract_summary_measurements(
    *,
    project_dir: Path,
    dataset_id: str,
) -> list[SummaryMeasurement]:
    dataset_dir = project_dir / "datasets" / dataset_id / "extracted"
    files = sorted(path for path in dataset_dir.rglob("*.xlsx") if path.is_file())
    rows: list[SummaryMeasurement] = []
    for path in files:
        workbook = read_xlsx_workbook(path)
        for sheet_name, table in workbook.items():
            if sheet_name == "Viscosities":
                rows.extend(_extract_viscosity_curve_measurements(dataset_id, path, table))
            if sheet_name == "AcousticImpedance":
                rows.extend(_extract_acoustic_measurements(dataset_id, path, table))
            rows.extend(_extract_summary_measurements_from_sheet(dataset_id, path, sheet_name, table))
    return rows


def build_calibration_targets(
    measurements: list[SummaryMeasurement],
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for measurement in measurements:
        group = grouped.setdefault(
            measurement.sample_key,
            {
                "sample_key": measurement.sample_key,
                "dataset_id": measurement.dataset_id,
                "material_family": measurement.material_family,
                "conditions": dict(measurement.conditions),
                "measurements": {},
            },
        )
        group["measurements"][measurement.metric_name] = {
            "value": measurement.value,
            "std": measurement.std,
            "units": measurement.units,
            "source_file": measurement.source_file,
            "sheet_name": measurement.sheet_name,
        }

    concentration_lookup: dict[tuple[str, float, float | None], dict[str, Any]] = {}
    signature_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for group in grouped.values():
        condition_signature = _condition_signature_from_conditions(group["conditions"])
        concentration = _infer_concentration_fraction(group["conditions"])
        curing_seconds = _infer_curing_seconds(group["conditions"])
        if concentration is not None:
            concentration_lookup[(group["material_family"], concentration, curing_seconds)] = group["measurements"]
        if condition_signature:
            signature_lookup[(group["material_family"], _condition_signature_lookup_key(condition_signature))] = group["measurements"]

    targets = []
    for group in grouped.values():
        measurements_map = dict(group["measurements"])
        if not group["conditions"]:
            continue
        if "youngs_modulus_kpa" not in measurements_map:
            continue
        stiffness = float(measurements_map["youngs_modulus_kpa"]["value"])
        stiffness_metric = "youngs_modulus_kpa"

        condition_signature = _condition_signature_from_conditions(group["conditions"])
        concentration = _infer_concentration_fraction(group["conditions"])
        curing_seconds = _infer_curing_seconds(group["conditions"])
        aux_measurements = (
            signature_lookup.get((group["material_family"], _condition_signature_lookup_key(condition_signature)), {})
            if condition_signature
            else {}
        )
        if not aux_measurements and concentration is not None:
            aux_measurements = concentration_lookup.get((group["material_family"], concentration, curing_seconds), {})
        for metric_name in (
            "bulk_modulus_gpa",
            "density_g_ml",
            "speed_of_sound_m_s",
            "acoustic_impedance_mrayl",
            "viscosity_pas",
            "viscosity_low_shear_pas",
            "viscosity_high_shear_pas",
            "shear_thinning_ratio",
        ):
            if metric_name not in measurements_map and metric_name in aux_measurements:
                measurements_map[metric_name] = aux_measurements[metric_name]
        target = {
            "sample_key": group["sample_key"],
            "dataset_id": group["dataset_id"],
            "material_family": group["material_family"],
            "target_stiffness": stiffness,
            "target_stiffness_metric": stiffness_metric,
            "target_anisotropy": 0.15,
            "target_connectivity": 0.95,
            "target_stress_propagation": 0.5,
            "conditions": group["conditions"],
            "condition_signature": condition_signature,
            "measurement_bundle": measurements_map,
            "target_property_hints": _design_property_hints_from_measurement_bundle(measurements_map),
            "concentration_fraction": concentration,
            "curing_seconds": curing_seconds,
        }
        targets.append(target)

    targets.sort(key=lambda item: (item["material_family"], item["sample_key"]))
    if max_samples is not None:
        targets = _select_diverse_targets(targets, max(int(max_samples), 0))
    return targets


def calibrate_targets_to_ecm_priors(
    calibration_targets: list[dict[str, Any]],
    *,
    candidate_budget: int = 4,
    top_k: int = 1,
    monte_carlo_runs: int = 1,
) -> dict[str, Any]:
    cases = []
    for target in calibration_targets:
        constraints = {
            "max_anisotropy": 0.35,
            "min_connectivity": 0.9,
            "max_risk_index": 0.95,
        }
        result = design_ecm_candidates(
            {
                "stiffness": float(target["target_stiffness"]),
                "anisotropy": float(target["target_anisotropy"]),
                "connectivity": float(target["target_connectivity"]),
                "stress_propagation": float(target["target_stress_propagation"]),
                **{
                    str(key): float(value)
                    for key, value in (target.get("target_property_hints", {}) or {}).items()
                },
            },
            constraints=constraints,
            top_k=top_k,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
        )
        best_candidate = _select_best_calibrated_candidate(result, target)
        top_features = best_candidate["features"]
        error_breakdown = calibration_candidate_error(best_candidate, target)
        cases.append(
            {
                "sample_key": target["sample_key"],
                "material_family": target["material_family"],
                "target_stiffness": float(target["target_stiffness"]),
                "target_metric": target["target_stiffness_metric"],
                "conditions": target["conditions"],
                "condition_signature": target.get("condition_signature", {}),
                "target_property_hints": target.get("target_property_hints", {}),
                "concentration_fraction": target.get("concentration_fraction"),
                "curing_seconds": target.get("curing_seconds"),
                "best_candidate": best_candidate,
                "abs_error": float(error_breakdown["stiffness_abs_error"]),
                "rel_error": float(error_breakdown["stiffness_rel_error"]),
                "combined_error": float(error_breakdown["combined_error"]),
                "auxiliary_errors": error_breakdown["auxiliary_errors"],
            }
        )

    family_priors = _aggregate_family_priors(cases)
    condition_priors = _aggregate_condition_priors(cases)
    return {
        "target_count": len(calibration_targets),
        "cases": cases,
        "family_priors": family_priors,
        "condition_priors": condition_priors,
    }


def run_calibration_pipeline(
    *,
    project_dir: Path,
    dataset_id: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    measurements = extract_summary_measurements(project_dir=project_dir, dataset_id=dataset_id)
    calibration_targets = build_calibration_targets(measurements, max_samples=max_samples)
    calibration_results = calibrate_targets_to_ecm_priors(calibration_targets)
    impact_assessment = run_calibration_impact_assessment(
        calibration_targets,
        calibration_results.get("family_priors", []),
    )
    summary = {
        "dataset_id": dataset_id,
        "measurement_count": len(measurements),
        "target_count": len(calibration_targets),
        "family_count": len(calibration_results["family_priors"]),
        "metrics_covered": sorted({measurement.metric_name for measurement in measurements}),
        "mean_abs_error": float(
            np.mean([case["abs_error"] for case in calibration_results["cases"]]) if calibration_results["cases"] else 0.0
        ),
        "impact_available": bool(impact_assessment.get("summary", {}).get("available", False)),
        "impact_mean_abs_error_delta": float(impact_assessment.get("summary", {}).get("mean_abs_error_delta", 0.0)),
        "impact_mean_total_error_delta": float(impact_assessment.get("summary", {}).get("mean_total_error_delta", 0.0)),
    }
    return {
        "workflow": "calibration",
        "dataset_id": dataset_id,
        "measurements": [asdict(row) for row in measurements],
        "calibration_targets": calibration_targets,
        "calibration_results": calibration_results,
        "calibration_impact_assessment": impact_assessment,
        "summary": summary,
    }


def load_latest_family_priors(project_dir: Path) -> list[dict[str, Any]]:
    calibration_results = load_latest_calibration_results(project_dir)
    return calibration_results.get("family_priors", []) if isinstance(calibration_results, dict) else []


def load_latest_calibration_results(project_dir: Path) -> dict[str, Any]:
    runs_dir = project_dir / "runs"
    calibration_runs = sorted(runs_dir.glob("*calibration*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in calibration_runs:
        result_path = run_dir / "calibration_results.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            priors = payload.get("family_priors", [])
            if priors:
                return payload
    return {}


def family_prior_for_material(
    family_priors: list[dict[str, Any]],
    material_family: str,
) -> dict[str, Any] | None:
    normalized = material_family.strip().lower()
    for prior in family_priors:
        if str(prior.get("material_family", "")).strip().lower() == normalized:
            return prior
    return None


def condition_prior_for_material(
    condition_priors: list[dict[str, Any]],
    material_family: str,
    *,
    concentration_fraction: float | None = None,
    curing_seconds: float | None = None,
    condition_overrides: dict[str, Any] | None = None,
    target_stiffness: float | None = None,
    target_anisotropy: float | None = None,
    target_connectivity: float | None = None,
    target_stress_propagation: float | None = None,
    target_property_hints: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    ranked = condition_prior_rankings_for_material(
        condition_priors,
        material_family,
        concentration_fraction=concentration_fraction,
        curing_seconds=curing_seconds,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffness,
        target_anisotropy=target_anisotropy,
        target_connectivity=target_connectivity,
        target_stress_propagation=target_stress_propagation,
        target_property_hints=target_property_hints,
    )
    return ranked[0]["prior"] if ranked else None


def condition_prior_rankings_for_material(
    condition_priors: list[dict[str, Any]],
    material_family: str,
    *,
    concentration_fraction: float | None = None,
    curing_seconds: float | None = None,
    condition_overrides: dict[str, Any] | None = None,
    target_stiffness: float | None = None,
    target_anisotropy: float | None = None,
    target_connectivity: float | None = None,
    target_stress_propagation: float | None = None,
    target_property_hints: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    normalized = material_family.strip().lower()
    candidates = [
        prior
        for prior in condition_priors
        if str(prior.get("material_family", "")).strip().lower() == normalized
    ]
    if not candidates:
        return None
    requested_signature = _requested_condition_signature(
        concentration_fraction=concentration_fraction,
        curing_seconds=curing_seconds,
        condition_overrides=condition_overrides,
    )
    if (
        concentration_fraction is None
        and curing_seconds is None
        and not requested_signature
        and target_stiffness is None
        and target_anisotropy is None
        and target_connectivity is None
        and target_stress_propagation is None
        and not target_property_hints
    ):
        return []

    def score_components(prior: dict[str, Any]) -> dict[str, float]:
        concentration_penalty = 0.0
        curing_penalty = 0.0
        stiffness_penalty = 0.0
        anisotropy_penalty = 0.0
        connectivity_penalty = 0.0
        propagation_penalty = 0.0
        condition_signature_penalty = 0.0
        property_penalty = 0.0
        prior_concentration = prior.get("concentration_fraction")
        prior_curing = prior.get("curing_seconds")
        prior_stiffness = prior.get("target_stiffness_mean")
        achieved_features = prior.get("achieved_feature_summary", {})
        prior_signature = prior.get("condition_signature", {}) if isinstance(prior.get("condition_signature"), dict) else {}
        if concentration_fraction is not None:
            if prior_concentration is None:
                concentration_penalty = 10.0
            else:
                concentration_penalty = abs(float(prior_concentration) - float(concentration_fraction))
        if curing_seconds is not None:
            if prior_curing is None:
                curing_penalty = 10.0
            else:
                curing_penalty = abs(float(prior_curing) - float(curing_seconds)) / max(abs(float(curing_seconds)), 1.0)
        if target_stiffness is not None:
            if prior_stiffness is None:
                stiffness_penalty = 10.0
            else:
                stiffness_penalty = abs(float(prior_stiffness) - float(target_stiffness)) / max(abs(float(target_stiffness)), 1.0)
        if target_anisotropy is not None:
            prior_anisotropy = achieved_features.get("anisotropy_mean")
            if prior_anisotropy is None:
                anisotropy_penalty = 10.0
            else:
                anisotropy_penalty = abs(float(prior_anisotropy) - float(target_anisotropy)) / max(abs(float(target_anisotropy)), 0.05)
        if target_connectivity is not None:
            prior_connectivity = achieved_features.get("connectivity_mean")
            if prior_connectivity is None:
                connectivity_penalty = 10.0
            else:
                connectivity_penalty = abs(float(prior_connectivity) - float(target_connectivity)) / max(abs(float(target_connectivity)), 0.1)
        if target_stress_propagation is not None:
            prior_propagation = achieved_features.get("stress_propagation_mean")
            if prior_propagation is None:
                propagation_penalty = 10.0
            else:
                propagation_penalty = abs(float(prior_propagation) - float(target_stress_propagation)) / max(abs(float(target_stress_propagation)), 0.1)
        if requested_signature:
            condition_signature_penalty = _condition_signature_penalty(prior_signature, requested_signature)
        if target_property_hints:
            for key, target_value in target_property_hints.items():
                prior_key = f"{key}_mean"
                prior_value = achieved_features.get(prior_key)
                if prior_value is None:
                    property_penalty += 10.0
                else:
                    property_penalty += abs(float(prior_value) - float(target_value)) / max(abs(float(target_value)), 0.05)
        sample_bonus = 0.01 * float(prior.get("sample_count", 0))
        total = (
            concentration_penalty
            + curing_penalty
            + 0.5 * condition_signature_penalty
            + stiffness_penalty
            + 0.35 * anisotropy_penalty
            + 0.15 * connectivity_penalty
            + 0.35 * propagation_penalty
            + 0.2 * property_penalty
            - sample_bonus
        )
        return {
            "total_score": float(total),
            "concentration_penalty": float(concentration_penalty),
            "curing_penalty": float(curing_penalty),
            "condition_signature_penalty": float(condition_signature_penalty),
            "stiffness_penalty": float(stiffness_penalty),
            "anisotropy_penalty": float(anisotropy_penalty),
            "connectivity_penalty": float(connectivity_penalty),
            "propagation_penalty": float(propagation_penalty),
            "property_penalty": float(property_penalty),
            "sample_bonus": float(sample_bonus),
        }

    ranked = []
    for prior in candidates:
        breakdown = score_components(prior)
        ranked.append({"prior": prior, "score_breakdown": breakdown})
    ranked.sort(
        key=lambda item: (
            float(item["score_breakdown"]["total_score"]),
            float(item["score_breakdown"]["stiffness_penalty"]),
            float(item["score_breakdown"]["concentration_penalty"] + item["score_breakdown"]["curing_penalty"]),
            float(item["score_breakdown"]["anisotropy_penalty"] + item["score_breakdown"]["propagation_penalty"]),
            float(item["score_breakdown"]["connectivity_penalty"]),
            -float(item["prior"].get("sample_count", 0) or 0.0),
        )
    )
    return ranked


def calibrated_search_space_for_material(
    family_priors: list[dict[str, Any]],
    material_family: str,
) -> dict[str, dict[str, float]] | None:
    return calibrated_search_space_from_priors(family_priors, material_family)


def calibrated_search_space_from_calibration_results(
    calibration_results: dict[str, Any],
    *,
    material_family: str,
    concentration_fraction: float | None = None,
    curing_seconds: float | None = None,
    condition_overrides: dict[str, Any] | None = None,
    target_stiffness: float | None = None,
    target_anisotropy: float | None = None,
    target_connectivity: float | None = None,
    target_stress_propagation: float | None = None,
    target_property_hints: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, float]] | None, dict[str, Any] | None]:
    condition_priors = calibration_results.get("condition_priors", []) if isinstance(calibration_results, dict) else []
    family_priors = calibration_results.get("family_priors", []) if isinstance(calibration_results, dict) else []

    interpolated_condition = interpolated_condition_prior_for_material(
        condition_priors,
        material_family,
        concentration_fraction=concentration_fraction,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffness,
    )
    if interpolated_condition is not None:
        return search_space_from_family_prior(interpolated_condition), {
            "prior_level": "interpolated_condition",
            "material_family": material_family,
            "concentration_fraction": interpolated_condition.get("concentration_fraction"),
            "curing_seconds": interpolated_condition.get("curing_seconds"),
            "condition_signature": interpolated_condition.get("condition_signature", {}),
            "target_stiffness_mean": interpolated_condition.get("target_stiffness_mean"),
            "sample_count": interpolated_condition.get("sample_count"),
            "interpolation": interpolated_condition.get("interpolation"),
            "achieved_feature_summary": interpolated_condition.get("achieved_feature_summary"),
            "selection_reason": {
                "mode": "interpolated_by_target_stiffness",
                "target_stiffness": target_stiffness,
                "requested_condition_signature": _requested_condition_signature(
                    concentration_fraction=concentration_fraction,
                    curing_seconds=curing_seconds,
                    condition_overrides=condition_overrides,
                ),
            },
        }

    ranked_conditions = condition_prior_rankings_for_material(
        condition_priors,
        material_family,
        concentration_fraction=concentration_fraction,
        curing_seconds=curing_seconds,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffness,
        target_anisotropy=target_anisotropy,
        target_connectivity=target_connectivity,
        target_stress_propagation=target_stress_propagation,
        target_property_hints=target_property_hints,
    )
    matched_condition = ranked_conditions[0]["prior"] if ranked_conditions else None
    if matched_condition is not None:
        top_reason = ranked_conditions[0]["score_breakdown"] if ranked_conditions else {}
        alternatives = [
            {
                "concentration_fraction": item["prior"].get("concentration_fraction"),
                "curing_seconds": item["prior"].get("curing_seconds"),
                "target_stiffness_mean": item["prior"].get("target_stiffness_mean"),
                "score_breakdown": item["score_breakdown"],
            }
            for item in ranked_conditions[:3]
        ]
        return search_space_from_family_prior(matched_condition), {
            "prior_level": "condition",
            "material_family": material_family,
            "concentration_fraction": matched_condition.get("concentration_fraction"),
            "curing_seconds": matched_condition.get("curing_seconds"),
            "condition_signature": matched_condition.get("condition_signature", {}),
            "target_stiffness_mean": matched_condition.get("target_stiffness_mean"),
            "achieved_feature_summary": matched_condition.get("achieved_feature_summary"),
            "sample_count": matched_condition.get("sample_count"),
            "selection_reason": top_reason,
            "requested_condition_signature": _requested_condition_signature(
                concentration_fraction=concentration_fraction,
                curing_seconds=curing_seconds,
                condition_overrides=condition_overrides,
            ),
            "requested_property_hints": target_property_hints or {},
            "ranked_alternatives": alternatives,
        }

    matched_family = family_prior_for_material(family_priors, material_family)
    if matched_family is not None:
        return search_space_from_family_prior(matched_family), {
            "prior_level": "family",
            "material_family": material_family,
            "sample_count": matched_family.get("sample_count"),
            "requested_condition_signature": _requested_condition_signature(
                concentration_fraction=concentration_fraction,
                curing_seconds=curing_seconds,
                condition_overrides=condition_overrides,
            ),
            "requested_property_hints": target_property_hints or {},
            "selection_reason": {
                "fallback": "condition_prior_unavailable",
                "requested_concentration_fraction": concentration_fraction,
                "requested_curing_seconds": curing_seconds,
                "requested_target_stiffness": target_stiffness,
            },
        }

    return None, None


def interpolated_condition_prior_for_material(
    condition_priors: list[dict[str, Any]],
    material_family: str,
    *,
    concentration_fraction: float | None = None,
    condition_overrides: dict[str, Any] | None = None,
    target_stiffness: float | None = None,
) -> dict[str, Any] | None:
    if target_stiffness is None:
        return None
    normalized = material_family.strip().lower()
    candidates = [
        prior
        for prior in condition_priors
        if str(prior.get("material_family", "")).strip().lower() == normalized
    ]
    if concentration_fraction is not None:
        candidates = [
            prior
            for prior in candidates
            if prior.get("concentration_fraction") is not None
            and abs(float(prior.get("concentration_fraction")) - float(concentration_fraction)) <= 1e-9
        ]
    requested_signature = _requested_condition_signature(
        concentration_fraction=concentration_fraction,
        curing_seconds=None,
        condition_overrides=condition_overrides,
    )
    if requested_signature:
        candidates = [
            prior
            for prior in candidates
            if _condition_signature_penalty(
                prior.get("condition_signature", {}) if isinstance(prior.get("condition_signature"), dict) else {},
                requested_signature,
            )
            < 1e-9
        ]
    candidates = [prior for prior in candidates if prior.get("target_stiffness_mean") is not None]
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda prior: float(prior.get("target_stiffness_mean", 0.0)))

    lower = None
    upper = None
    for candidate in candidates:
        stiffness_mean = float(candidate.get("target_stiffness_mean", 0.0))
        if stiffness_mean <= float(target_stiffness):
            lower = candidate
        if stiffness_mean >= float(target_stiffness) and upper is None:
            upper = candidate
    if lower is None or upper is None or lower is upper:
        return None

    lower_stiffness = float(lower.get("target_stiffness_mean", 0.0))
    upper_stiffness = float(upper.get("target_stiffness_mean", 0.0))
    if upper_stiffness <= lower_stiffness:
        return None
    alpha = (float(target_stiffness) - lower_stiffness) / (upper_stiffness - lower_stiffness)
    alpha = min(max(alpha, 0.0), 1.0)

    return {
        "material_family": material_family,
        "concentration_fraction": concentration_fraction if concentration_fraction is not None else lower.get("concentration_fraction"),
        "curing_seconds": lower.get("curing_seconds"),
        "target_stiffness_mean": float(target_stiffness),
        "sample_count": int(lower.get("sample_count", 0)) + int(upper.get("sample_count", 0)),
        "mean_abs_error": _lerp(lower.get("mean_abs_error"), upper.get("mean_abs_error"), alpha),
        "parameter_priors": _lerp_stats_dict(
            lower.get("parameter_priors", {}),
            upper.get("parameter_priors", {}),
            alpha,
        ),
        "auxiliary_error_summary": _lerp_stats_dict(
            lower.get("auxiliary_error_summary", {}),
            upper.get("auxiliary_error_summary", {}),
            alpha,
        ),
        "achieved_feature_summary": _lerp_numeric_dict(
            lower.get("achieved_feature_summary", {}),
            upper.get("achieved_feature_summary", {}),
            alpha,
        ),
        "condition_signature": _merge_condition_signatures(
            lower.get("condition_signature", {}),
            upper.get("condition_signature", {}),
            concentration_fraction=concentration_fraction,
        ),
        "interpolation": {
            "alpha": float(alpha),
            "lower_condition": {
                "concentration_fraction": lower.get("concentration_fraction"),
                "curing_seconds": lower.get("curing_seconds"),
                "target_stiffness_mean": lower.get("target_stiffness_mean"),
            },
            "upper_condition": {
                "concentration_fraction": upper.get("concentration_fraction"),
                "curing_seconds": upper.get("curing_seconds"),
                "target_stiffness_mean": upper.get("target_stiffness_mean"),
            },
        },
    }


def run_calibration_impact_assessment(
    calibration_targets: list[dict[str, Any]],
    family_priors: list[dict[str, Any]],
    *,
    candidate_budget: int = 4,
    top_k: int = 1,
    monte_carlo_runs: int = 1,
) -> dict[str, Any]:
    if not calibration_targets or not family_priors:
        return {
            "cases": [],
            "summary": {
                "available": False,
                "eligible_case_count": 0,
                "improved_abs_case_count": 0,
                "improved_total_case_count": 0,
                "mean_abs_error_baseline": 0.0,
                "mean_abs_error_calibrated": 0.0,
                "mean_abs_error_delta": 0.0,
                "mean_total_error_baseline": 0.0,
                "mean_total_error_calibrated": 0.0,
                "mean_total_error_delta": 0.0,
                "baseline_feasible_count": 0,
                "calibrated_feasible_count": 0,
                "overall_pass": False,
            },
        }

    constraints = {
        "max_anisotropy": 0.35,
        "min_connectivity": 0.9,
        "max_risk_index": 0.95,
    }
    rows: list[dict[str, Any]] = []
    for target in calibration_targets:
        material_family = str(target.get("material_family", ""))
        holdout_targets = [
            other
            for other in calibration_targets
            if other is not target and str(other.get("material_family", "")) == material_family
        ]
        holdout_priors = (
            calibrate_targets_to_ecm_priors(
                holdout_targets,
                candidate_budget=candidate_budget,
                top_k=1,
                monte_carlo_runs=monte_carlo_runs,
            )
            if holdout_targets
            else {}
        )
        search_space, calibration_context = calibrated_search_space_from_calibration_results(
            holdout_priors,
            material_family=material_family,
            concentration_fraction=target.get("concentration_fraction"),
            curing_seconds=target.get("curing_seconds"),
            target_stiffness=float(target["target_stiffness"]),
            target_anisotropy=float(target["target_anisotropy"]),
            target_connectivity=float(target["target_connectivity"]),
            target_stress_propagation=float(target["target_stress_propagation"]),
            target_property_hints=target.get("target_property_hints", {}),
        )
        prior_source = "leave_one_out"
        if search_space is None:
            search_space, calibration_context = calibrated_search_space_from_calibration_results(
                {"family_priors": family_priors},
                material_family=material_family,
                concentration_fraction=target.get("concentration_fraction"),
                curing_seconds=target.get("curing_seconds"),
                target_stiffness=float(target["target_stiffness"]),
                target_anisotropy=float(target["target_anisotropy"]),
                target_connectivity=float(target["target_connectivity"]),
                target_stress_propagation=float(target["target_stress_propagation"]),
                target_property_hints=target.get("target_property_hints", {}),
            )
            prior_source = "global"
        if search_space is None:
            continue

        design_targets = {
            "stiffness": float(target["target_stiffness"]),
            "anisotropy": float(target["target_anisotropy"]),
            "connectivity": float(target["target_connectivity"]),
            "stress_propagation": float(target["target_stress_propagation"]),
            **{
                str(key): float(value)
                for key, value in (target.get("target_property_hints", {}) or {}).items()
            },
        }
        baseline = design_ecm_candidates(
            design_targets,
            constraints=constraints,
            top_k=top_k,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
        )
        calibrated = design_ecm_candidates(
            design_targets,
            search_space=search_space,
            constraints=constraints,
            top_k=top_k,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
        )

        baseline_best = _select_best_calibrated_candidate(baseline, target)
        calibrated_best = _select_best_calibrated_candidate(calibrated, target)
        baseline_summary = _candidate_error_summary(baseline_best, target)
        calibrated_summary = _candidate_error_summary(calibrated_best, target)
        rows.append(
            {
                "sample_key": target.get("sample_key", "NR"),
                "material_family": target.get("material_family", "NR"),
                "baseline": baseline_summary,
                "calibrated": calibrated_summary,
                "calibrated_search_space": search_space,
                "calibration_context": calibration_context,
                "prior_source": prior_source,
                "abs_error_delta": float(baseline_summary["abs_error"] - calibrated_summary["abs_error"]),
                "total_error_delta": float(baseline_summary["total_error"] - calibrated_summary["total_error"]),
                "improved_abs_error": bool(calibrated_summary["abs_error"] <= baseline_summary["abs_error"]),
                "improved_total_error": bool(calibrated_summary["total_error"] <= baseline_summary["total_error"]),
            }
        )

    if not rows:
        return {
            "cases": [],
            "summary": {
                "available": False,
                "eligible_case_count": 0,
                "improved_abs_case_count": 0,
                "improved_total_case_count": 0,
                "mean_abs_error_baseline": 0.0,
                "mean_abs_error_calibrated": 0.0,
                "mean_abs_error_delta": 0.0,
                "mean_total_error_baseline": 0.0,
                "mean_total_error_calibrated": 0.0,
                "mean_total_error_delta": 0.0,
                "baseline_feasible_count": 0,
                "calibrated_feasible_count": 0,
                "evaluation_mode": "leave_one_out_when_possible",
                "overall_pass": False,
            },
        }

    baseline_abs = np.asarray([row["baseline"]["abs_error"] for row in rows], dtype=float)
    calibrated_abs = np.asarray([row["calibrated"]["abs_error"] for row in rows], dtype=float)
    baseline_total = np.asarray([row["baseline"]["total_error"] for row in rows], dtype=float)
    calibrated_total = np.asarray([row["calibrated"]["total_error"] for row in rows], dtype=float)
    improved_abs = sum(1 for row in rows if row["improved_abs_error"])
    improved_total = sum(1 for row in rows if row["improved_total_error"])
    baseline_feasible = sum(1 for row in rows if row["baseline"]["feasible"])
    calibrated_feasible = sum(1 for row in rows if row["calibrated"]["feasible"])
    return {
        "cases": rows,
        "summary": {
            "available": True,
            "eligible_case_count": len(rows),
            "improved_abs_case_count": improved_abs,
            "improved_total_case_count": improved_total,
            "mean_abs_error_baseline": float(np.mean(baseline_abs)),
            "mean_abs_error_calibrated": float(np.mean(calibrated_abs)),
            "mean_abs_error_delta": float(np.mean(baseline_abs - calibrated_abs)),
            "mean_total_error_baseline": float(np.mean(baseline_total)),
            "mean_total_error_calibrated": float(np.mean(calibrated_total)),
            "mean_total_error_delta": float(np.mean(baseline_total - calibrated_total)),
            "baseline_feasible_count": baseline_feasible,
            "calibrated_feasible_count": calibrated_feasible,
            "evaluation_mode": "leave_one_out_when_possible",
            "overall_pass": bool(
                np.mean(calibrated_total) <= np.mean(baseline_total)
                and improved_total >= max(1, len(rows) // 2)
            ),
        },
    }


def search_space_from_family_prior(prior: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    parameter_priors = prior.get("parameter_priors", {}) if isinstance(prior, dict) else {}
    default_bounds = default_design_search_space()
    sample_count = int(prior.get("sample_count", 0)) if isinstance(prior, dict) else 0
    mean_abs_error = float(prior.get("mean_abs_error", 0.0)) if isinstance(prior, dict) else 0.0
    reliability_expansion = 1.0
    if sample_count and sample_count < 5:
        reliability_expansion += 0.25 * float(5 - sample_count)
    if mean_abs_error > 10.0:
        reliability_expansion += min((mean_abs_error - 10.0) / 25.0, 1.0)
    bounds: dict[str, dict[str, float]] = {}
    for name in ("fiber_density", "fiber_stiffness", "bending_stiffness", "crosslink_prob", "domain_size"):
        stat = parameter_priors.get(name)
        if not isinstance(stat, dict):
            continue
        mean = float(stat.get("mean", 0.0))
        std = float(stat.get("std", 0.0))
        default_span = float(default_bounds[name]["max"] - default_bounds[name]["min"])
        width_floor_fraction = _prior_width_floor_fraction(sample_count)
        width = max(std * 1.5, mean * 0.15, default_span * width_floor_fraction, 1e-6) * reliability_expansion
        lower = mean - width
        upper = mean + width
        if name == "fiber_density":
            lower, upper = max(lower, default_bounds[name]["min"], 0.05), min(upper, default_bounds[name]["max"], 1.0)
        elif name == "crosslink_prob":
            lower, upper = max(lower, default_bounds[name]["min"], 0.01), min(upper, default_bounds[name]["max"], 1.0)
        elif name == "bending_stiffness":
            lower = max(lower, default_bounds[name]["min"], 0.0)
            upper = min(upper, default_bounds[name]["max"])
        elif name == "domain_size":
            lower = max(lower, default_bounds[name]["min"], 0.1)
            upper = min(upper, default_bounds[name]["max"])
        else:
            lower = max(lower, default_bounds[name]["min"], 1e-3)
            upper = min(upper, default_bounds[name]["max"])
        bounds[name] = {"min": float(lower), "max": float(upper)}
    return bounds or None


def calibrated_search_space_from_priors(
    family_priors: list[dict[str, Any]],
    material_family: str,
) -> dict[str, dict[str, float]] | None:
    prior = family_prior_for_material(family_priors, material_family)
    if prior is None:
        return None
    return search_space_from_family_prior(prior)


def _extract_summary_measurements_from_sheet(
    dataset_id: str,
    path: Path,
    sheet_name: str,
    table: list[list[str]],
) -> list[SummaryMeasurement]:
    metric_info = _sheet_metric_info(sheet_name)
    if metric_info is None:
        return []
    data_start_idx, avg_col, std_col, units = _detect_summary_layout(sheet_name, table, metric_info["default_units"])
    if data_start_idx is None or avg_col is None:
        return []

    material_family = _material_from_path(path)
    carry = [""] * max(len(table[max(data_start_idx - 1, 0)]), avg_col + 1)
    rows: list[SummaryMeasurement] = []
    for row in table[data_start_idx:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(len(carry) - len(row), 0)
        for index in range(min(avg_col, len(padded))):
            if padded[index].strip():
                carry[index] = padded[index].strip()
            else:
                padded[index] = carry[index]
        if avg_col >= len(padded):
            continue
        avg_value = _safe_float(padded[avg_col])
        if avg_value is None:
            continue
        std_value = _safe_float(padded[std_col]) if std_col is not None and std_col < len(padded) else None
        conditions = _conditions_from_columns(sheet_name, padded[:avg_col])
        sample_key = _build_sample_key(material_family, conditions)
        rows.append(
            SummaryMeasurement(
                dataset_id=dataset_id,
                material_family=material_family,
                sample_key=sample_key,
                source_file=path.name,
                sheet_name=sheet_name,
                metric_name=metric_info["metric_name"],
                value=avg_value,
                std=std_value,
                units=units or metric_info["default_units"],
                conditions=conditions,
            )
        )
    return rows


def _extract_viscosity_curve_measurements(
    dataset_id: str,
    path: Path,
    table: list[list[str]],
) -> list[SummaryMeasurement]:
    pair_rows = _viscosity_pair_rows(table)
    if not pair_rows:
        return []

    material_family = _material_from_path(path)
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for header_row_index, value_col in pair_rows:
        filled_headers = _filled_header_rows(table[:header_row_index])
        conditions = _conditions_from_viscosity_headers(filled_headers, value_col)
        if not conditions:
            continue
        sample_key = _build_sample_key(material_family, conditions)
        shear_values: list[float] = []
        viscosity_values: list[float] = []
        for row in table[header_row_index + 2 :]:
            if value_col + 1 >= len(row):
                continue
            shear = _safe_float(row[value_col])
            viscosity = _safe_float(row[value_col + 1])
            if shear is None or viscosity is None:
                if shear_values:
                    break
                continue
            shear_values.append(float(shear))
            viscosity_values.append(float(viscosity))
        if len(viscosity_values) < 3:
            continue

        low_window = viscosity_values[: min(3, len(viscosity_values))]
        high_window = viscosity_values[-min(3, len(viscosity_values)) :]
        metrics = {
            "viscosity_low_shear_pas": float(np.mean(low_window)),
            "viscosity_high_shear_pas": float(np.mean(high_window)),
            "shear_thinning_ratio": float(np.mean(low_window) / max(np.mean(high_window), 1e-12)),
        }
        for metric_name, value in metrics.items():
            bucket = aggregates.setdefault(
                (sample_key, metric_name),
                {"conditions": conditions, "values": []},
            )
            bucket["values"].append(float(value))

    rows: list[SummaryMeasurement] = []
    for (sample_key, metric_name), payload in aggregates.items():
        values = np.asarray(payload["values"], dtype=float)
        rows.append(
            SummaryMeasurement(
                dataset_id=dataset_id,
                material_family=material_family,
                sample_key=sample_key,
                source_file=path.name,
                sheet_name="Viscosities",
                metric_name=metric_name,
                value=float(np.mean(values)),
                std=float(np.std(values)) if values.size > 1 else None,
                units="ratio" if metric_name == "shear_thinning_ratio" else "Pa.s",
                conditions=dict(payload["conditions"]),
            )
        )
    return rows


def _extract_acoustic_measurements(
    dataset_id: str,
    path: Path,
    table: list[list[str]],
) -> list[SummaryMeasurement]:
    if len(table) < 4:
        return []
    header = [cell.strip().lower() for cell in table[1]]
    units_row = [cell.strip() for cell in table[2]]
    material_family = _material_from_path(path)
    metric_columns = {
        "speed_of_sound_m_s": _find_column_index(header, "avg speed of sound"),
        "acoustic_impedance_mrayl": _find_column_index(header, "acoustic impedance"),
    }
    std_columns = {
        "speed_of_sound_m_s": _next_std_column(header, metric_columns["speed_of_sound_m_s"]),
        "acoustic_impedance_mrayl": _next_std_column(header, metric_columns["acoustic_impedance_mrayl"]),
    }

    rows: list[SummaryMeasurement] = []
    carry_concentration = ""
    for row in table[3:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(len(header) - len(row), 0)
        concentration = padded[0].strip() or carry_concentration
        if concentration:
            carry_concentration = concentration
        curing = padded[1].strip() if len(padded) > 1 else ""
        if not concentration:
            continue
        conditions = {"sample_label": concentration}
        if curing:
            conditions["curing"] = curing
        sample_key = _build_sample_key(material_family, conditions)
        for metric_name, value_col in metric_columns.items():
            if value_col is None or value_col >= len(padded):
                continue
            value = _safe_float(padded[value_col])
            if value is None:
                continue
            std_col = std_columns.get(metric_name)
            std = _safe_float(padded[std_col]) if std_col is not None and std_col < len(padded) else None
            units = units_row[value_col] if value_col < len(units_row) and units_row[value_col] else (
                "m/s" if metric_name == "speed_of_sound_m_s" else "10^6 kg/m^2s"
            )
            rows.append(
                SummaryMeasurement(
                    dataset_id=dataset_id,
                    material_family=material_family,
                    sample_key=sample_key,
                    source_file=path.name,
                    sheet_name="AcousticImpedance",
                    metric_name=metric_name,
                    value=float(value),
                    std=std,
                    units=units,
                    conditions=conditions,
                )
            )
    return rows


def _sheet_metric_info(sheet_name: str) -> dict[str, str] | None:
    mapping = {
        "YoungsModulus": {"metric_name": "youngs_modulus_kpa", "default_units": "kPa"},
        "BulkModulus": {"metric_name": "bulk_modulus_gpa", "default_units": "GPa"},
        "Density": {"metric_name": "density_g_ml", "default_units": "g/mL"},
    }
    return mapping.get(sheet_name)


def _viscosity_pair_rows(table: list[list[str]]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for row_index, row in enumerate(table[:8]):
        normalized = [cell.strip().lower() for cell in row]
        for column_index in range(len(normalized) - 1):
            if "shear rate" in normalized[column_index] and "viscosity" in normalized[column_index + 1]:
                rows.append((row_index, column_index))
    return rows


def _filled_header_rows(rows: list[list[str]]) -> list[list[str]]:
    filled_rows: list[list[str]] = []
    for row in rows:
        carry = ""
        filled: list[str] = []
        for cell in row:
            cleaned = cell.strip()
            if cleaned:
                carry = cleaned
                filled.append(cleaned)
            else:
                filled.append(carry)
        filled_rows.append(filled)
    return filled_rows


def _conditions_from_viscosity_headers(filled_headers: list[list[str]], value_col: int) -> dict[str, str]:
    import re

    tokens: list[str] = []
    for header in filled_headers:
        if value_col < len(header):
            token = header[value_col].strip()
            if token:
                tokens.append(token)

    generic_tokens = {"viscosities", "viscosity", "shear rate", "1/s", "pa.s"}
    conditions: dict[str, str] = {}
    for token in tokens:
        normalized = token.strip()
        lower = normalized.lower()
        if lower in generic_tokens:
            continue
        if "curing time" in lower:
            match = re.search(r"(\d+(?:\.\d+)?)\s*s", lower)
            if match:
                conditions["curing"] = f"{match.group(1)} s"
                prefix = normalized[: match.start()].strip(" -_")
                if prefix:
                    conditions.setdefault("sample_label", prefix)
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
            conditions.setdefault("concentration", normalized)
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", lower)
        if match:
            conditions["curing"] = f"{match.group(1)} s"
            prefix = normalized[: match.start()].strip(" -_")
            if prefix:
                conditions.setdefault("sample_label", prefix)
            continue
        if "%" in normalized:
            conditions.setdefault("sample_label", normalized)
            continue
        conditions.setdefault("sample_label", normalized)
    return conditions


def _detect_summary_layout(
    sheet_name: str,
    table: list[list[str]],
    default_units: str,
) -> tuple[int | None, int | None, int | None, str]:
    token_map = {
        "YoungsModulus": "avg",
        "BulkModulus": "bulk modulus",
        "Density": "avg density",
        "Viscosities": "viscosity",
    }
    token = token_map.get(sheet_name, "avg")
    best: tuple[int, int, int | None, str, int] | None = None
    for idx, row in enumerate(table[:8]):
        normalized = [cell.strip().lower() for cell in row]
        avg_candidates = [i for i, cell in enumerate(normalized) if token in cell]
        std_candidates = [i for i, cell in enumerate(normalized) if "standard" in cell]
        if avg_candidates:
            value_col = avg_candidates[-1]
            std_col = std_candidates[-1] if std_candidates else None
            units = default_units
            if idx + 1 < len(table) and value_col < len(table[idx + 1]):
                candidate = table[idx + 1][value_col].strip()
                if any(ch.isalpha() for ch in candidate):
                    units = candidate
            score = value_col + (50 if std_col is not None else 0)
            proposal = (idx + 2 if idx + 1 < len(table) else idx + 1, value_col, std_col, units, score)
            if best is None or proposal[-1] > best[-1]:
                best = proposal
    if best is not None:
        return best[0], best[1], best[2], best[3]
    return None, None, None, default_units


def _conditions_from_columns(sheet_name: str, values: list[str]) -> dict[str, str]:
    sheet_lower = sheet_name.lower()
    conditions: dict[str, str] = {}
    cleaned = [value.strip() for value in values if value.strip()]
    if "youngs" in sheet_lower and cleaned:
        if len(cleaned) >= 1:
            conditions["concentration"] = cleaned[0]
        if len(cleaned) >= 2:
            conditions["curing"] = cleaned[1]
    elif "bulk" in sheet_lower or "density" in sheet_lower:
        if cleaned:
            conditions["sample_label"] = cleaned[0]
        if len(cleaned) >= 2:
            second = cleaned[1]
            if second.lower().replace(" ", "").endswith("s"):
                conditions["curing"] = second
            elif not _safe_float(second):
                conditions["fluid"] = second
    else:
        for idx, value in enumerate(cleaned, start=1):
            conditions[f"condition_{idx}"] = value
    return conditions


def _find_column_index(header: list[str], token: str) -> int | None:
    for idx, value in enumerate(header):
        if token in value:
            return idx
    return None


def _next_std_column(header: list[str], start_idx: int | None) -> int | None:
    if start_idx is None:
        return None
    for idx in range(start_idx + 1, len(header)):
        if "standard deviation" in header[idx]:
            return idx
    return None


def _build_sample_key(material_family: str, conditions: dict[str, str]) -> str:
    tokens = [material_family]
    tokens.extend(_canonical_condition_tokens(conditions))
    return "_".join(_slugify_token(token) for token in tokens if token)


def _material_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "gelma" in lowered:
        return "GelMA"
    if "pegda" in lowered:
        return "PEGDA"
    return "unknown"


def _infer_concentration_fraction(conditions: dict[str, str]) -> float | None:
    for value in _canonical_condition_tokens(conditions):
        stripped = value.replace("%", "").strip()
        try:
            numeric = float(stripped)
            if 0 < numeric <= 100:
                return numeric / 100.0 if numeric > 1 else numeric
        except ValueError:
            continue
    return None


def _infer_curing_seconds(conditions: dict[str, str]) -> float | None:
    for value in _canonical_condition_tokens(conditions):
        token = value.lower().replace(" ", "")
        if token.endswith("s"):
            try:
                return float(token[:-1])
            except ValueError:
                continue
    return None


def _canonical_condition_tokens(conditions: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    if "sample_label" in conditions:
        tokens.extend(_tokens_from_sample_label(conditions["sample_label"]))
    if "concentration" in conditions:
        tokens.append(_normalize_concentration_token(conditions["concentration"]))
    if "curing" in conditions:
        tokens.append(conditions["curing"])
    for key, value in conditions.items():
        if key in {"sample_label", "concentration", "curing", "fluid"}:
            continue
        tokens.append(value)
    return tokens


def _condition_signature_from_conditions(conditions: dict[str, str]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    concentration_fraction = _infer_concentration_fraction(conditions)
    curing_seconds = _infer_curing_seconds(conditions)
    if concentration_fraction is not None:
        signature["concentration_fraction"] = float(concentration_fraction)
    if curing_seconds is not None:
        signature["curing_seconds"] = float(curing_seconds)
    signature.update(_parse_condition_signature_tokens(_canonical_condition_tokens(conditions)))
    for key, value in conditions.items():
        if key in {"sample_label", "concentration", "curing"}:
            continue
        if key not in signature and value:
            signature[key] = value
    return signature


def _requested_condition_signature(
    *,
    concentration_fraction: float | None,
    curing_seconds: float | None,
    condition_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = dict(condition_overrides or {})
    if concentration_fraction is not None:
        signature["concentration_fraction"] = float(concentration_fraction)
    if curing_seconds is not None:
        signature["curing_seconds"] = float(curing_seconds)
    return {
        str(key): _coerce_condition_signature_value(value)
        for key, value in signature.items()
        if value not in (None, "")
    }


def _parse_condition_signature_tokens(tokens: list[str]) -> dict[str, Any]:
    import re

    signature: dict[str, Any] = {}
    for token in tokens:
        lowered = token.lower().strip()
        temp_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°c|c)\b", lowered)
        if temp_match:
            signature.setdefault("temperature_c", float(temp_match.group(1)))
        if "lap" in lowered or "photoinitiator" in lowered or "initiator" in lowered:
            percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
            if percent_match:
                signature.setdefault("photoinitiator_fraction", float(percent_match.group(1)) / 100.0)
        mw_match = re.search(r"(\d+(?:\.\d+)?)\s*kda\b", lowered)
        if mw_match:
            signature.setdefault("polymer_mw_kda", float(mw_match.group(1)))
        ds_match = re.search(r"(?:dm|ds|methacrylation|substitution)[^\d]*(\d+(?:\.\d+)?)\s*%?", lowered)
        if ds_match:
            raw = float(ds_match.group(1))
            signature.setdefault("degree_substitution", raw / 100.0 if raw > 1.0 else raw)
    return signature


def _coerce_condition_signature_value(value: Any) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def _condition_signature_lookup_key(signature: dict[str, Any]) -> str:
    normalized = {
        str(key): _coerce_condition_signature_value(value)
        for key, value in signature.items()
        if value not in (None, "")
    }
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _condition_signature_penalty(prior_signature: dict[str, Any], requested_signature: dict[str, Any]) -> float:
    total = 0.0
    for key, requested_value in requested_signature.items():
        prior_value = prior_signature.get(key)
        if prior_value in (None, ""):
            total += 5.0
            continue
        if isinstance(requested_value, (int, float)) and isinstance(prior_value, (int, float)):
            total += abs(float(prior_value) - float(requested_value)) / max(abs(float(requested_value)), 1.0)
        else:
            total += 0.0 if str(prior_value).strip().lower() == str(requested_value).strip().lower() else 1.0
    return float(total)


def _merge_condition_signatures(
    lower: dict[str, Any],
    upper: dict[str, Any],
    *,
    concentration_fraction: float | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in sorted(set(lower) | set(upper)):
        if key in lower and key in upper and lower[key] == upper[key]:
            merged[key] = lower[key]
        elif key == "concentration_fraction" and concentration_fraction is not None:
            merged[key] = float(concentration_fraction)
    return merged


def _tokens_from_sample_label(sample_label: str) -> list[str]:
    import re

    label = sample_label.replace("+", " ").replace("_", " ")
    tokens: list[str] = []
    concentration_match = re.search(r"(\d+(?:\.\d+)?)%", label)
    if concentration_match:
        numeric = float(concentration_match.group(1))
        tokens.append(str(numeric / 100.0 if numeric > 1 else numeric))
        label = label.replace(concentration_match.group(0), " ")
    curing_match = re.search(r"(\d+(?:\.\d+)?)\s*s\b", label, re.IGNORECASE)
    if curing_match:
        tokens.append(f"{curing_match.group(1)} s")
        label = label.replace(curing_match.group(0), " ")
    for part in label.split():
        cleaned = part.strip()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _normalize_concentration_token(value: str) -> str:
    stripped = value.replace("%", "").strip()
    try:
        numeric = float(stripped)
        return str(numeric / 100.0 if numeric > 1 else numeric)
    except ValueError:
        return value


def _aggregate_family_priors(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["material_family"]].append(case)

    priors = []
    for family, family_cases in grouped.items():
        params = defaultdict(list)
        errors = []
        for case in family_cases:
            best = case["best_candidate"]
            for name, value in best["parameters"].items():
                params[name].append(float(value))
            errors.append(float(case["abs_error"]))
            if case.get("auxiliary_errors"):
                for key, value in case["auxiliary_errors"].items():
                    params[f"aux_{key}"].append(float(value))
        priors.append(
            {
                "material_family": family,
                "sample_count": len(family_cases),
                "parameter_priors": {
                    name: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                    for name, values in params.items()
                    if not name.startswith("aux_")
                },
                "auxiliary_error_summary": {
                    name.removeprefix("aux_"): {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                    for name, values in params.items()
                    if name.startswith("aux_")
                },
                "mean_abs_error": float(np.mean(errors)),
            }
        )
    priors.sort(key=lambda row: row["material_family"])
    return priors


def _aggregate_condition_priors(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        signature = case.get("condition_signature", {}) if isinstance(case.get("condition_signature"), dict) else {}
        key = (
            str(case.get("material_family", "")),
            _condition_signature_lookup_key(signature),
        )
        grouped[key].append(case)

    priors = []
    for (family, _signature_key), grouped_cases in grouped.items():
        signature = grouped_cases[0].get("condition_signature", {}) if grouped_cases else {}
        concentration_fraction = signature.get("concentration_fraction")
        curing_seconds = signature.get("curing_seconds")
        if not signature:
            continue
        params = defaultdict(list)
        errors = []
        aux_values = defaultdict(list)
        feature_values = defaultdict(list)
        for case in grouped_cases:
            best = case["best_candidate"]
            for name, value in best["parameters"].items():
                params[name].append(float(value))
            for name, value in best.get("features", {}).items():
                if isinstance(value, (int, float)):
                    feature_values[name].append(float(value))
            errors.append(float(case["abs_error"]))
            for key, value in case.get("auxiliary_errors", {}).items():
                aux_values[key].append(float(value))
        priors.append(
            {
                "material_family": family,
                "concentration_fraction": concentration_fraction,
                "curing_seconds": curing_seconds,
                "condition_signature": signature,
                "target_stiffness_mean": float(np.mean([float(case.get("target_stiffness", 0.0)) for case in grouped_cases])),
                "sample_count": len(grouped_cases),
                "parameter_priors": {
                    name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                    for name, values in params.items()
                },
                "achieved_feature_summary": {
                    f"{name}_mean": float(np.mean(values))
                    for name, values in feature_values.items()
                },
                "auxiliary_error_summary": {
                    name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                    for name, values in aux_values.items()
                },
                "mean_abs_error": float(np.mean(errors)),
            }
        )
    priors.sort(
        key=lambda row: (
            str(row.get("material_family", "")),
            float(row.get("concentration_fraction", 0.0) or 0.0),
            float(row.get("curing_seconds", 0.0) or 0.0),
        )
    )
    return priors


def _select_diverse_targets(targets: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return []
    if len(targets) <= max_samples:
        return list(targets)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        grouped[str(target.get("material_family", "unknown"))].append(target)

    family_names = sorted(grouped)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < max_samples and family_names:
        family = family_names[index % len(family_names)]
        bucket = grouped[family]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            family_names = [name for name in family_names if grouped[name]]
            index = 0
            continue
        index += 1
    return selected


def _select_best_calibrated_candidate(
    result: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(result.get("evaluated_candidates", []))
    if not candidates:
        return result["top_candidates"][0]

    def score(candidate: dict[str, Any]) -> tuple[float, float]:
        error_breakdown = calibration_candidate_error(candidate, target)
        candidate["auxiliary_errors"] = error_breakdown["auxiliary_errors"]
        feasibility_penalty = 0.0 if candidate.get("feasible") else 10.0
        return (feasibility_penalty + float(error_breakdown["combined_error"]), float(candidate["score"]))

    return min(candidates, key=score)


def _candidate_error_summary(candidate: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    features = candidate.get("features", {})
    target_stiffness = float(target["target_stiffness"])
    aux_errors = candidate.get("auxiliary_errors")
    if not isinstance(aux_errors, dict):
        aux_errors = _auxiliary_calibration_errors(candidate, target)
    rel_error = abs(float(features.get("stiffness_mean", 0.0)) - target_stiffness) / max(abs(target_stiffness), 1e-12)
    return {
        "candidate": candidate,
        "stiffness_mean": float(features.get("stiffness_mean", 0.0)),
        "abs_error": abs(float(features.get("stiffness_mean", 0.0)) - target_stiffness),
        "rel_error": rel_error,
        "auxiliary_errors": aux_errors,
        "total_error": float(rel_error + sum(float(value) for value in aux_errors.values())),
        "feasible": bool(candidate.get("feasible", False)),
    }


def _design_property_hints_from_measurement_bundle(bundle: dict[str, Any]) -> dict[str, float]:
    if not isinstance(bundle, dict):
        return {}

    def observed(metric_name: str) -> float | None:
        payload = bundle.get(metric_name)
        if isinstance(payload, dict) and payload.get("value") is not None:
            return float(payload["value"])
        return None

    density = observed("density_g_ml")
    bulk_modulus = observed("bulk_modulus_gpa")
    acoustic_impedance = observed("acoustic_impedance_mrayl")
    viscosity_low = observed("viscosity_low_shear_pas")
    viscosity_high = observed("viscosity_high_shear_pas")
    shear_thinning_ratio = observed("shear_thinning_ratio")

    hints: dict[str, float] = {}
    if density is not None or bulk_modulus is not None or acoustic_impedance is not None:
        density_term = density if density is not None else 1.0
        bulk_term = bulk_modulus if bulk_modulus is not None else 1.0
        acoustic_term = acoustic_impedance if acoustic_impedance is not None else 1.5
        compressibility = 1.0 / (1.0 + 0.5 * density_term + 5.0 * bulk_term + 0.3 * acoustic_term)
        permeability = 0.04 / (1.0 + 0.5 * density_term + 0.5 * acoustic_term)
        swelling_ratio = 1.1 + 0.35 / max(density_term, 0.5) + 0.25 / max(bulk_term * 5.0, 0.5)
        mesh_size = 0.35 + 0.2 / (1.0 + bulk_term * 3.0)
        hints["compressibility_proxy"] = float(np.clip(compressibility, 0.02, 0.25))
        hints["permeability_proxy"] = float(np.clip(permeability, 0.005, 0.08))
        hints["swelling_ratio_proxy"] = float(np.clip(swelling_ratio, 1.15, 2.2))
        hints["mesh_size_proxy"] = float(np.clip(mesh_size, 0.25, 0.9))
    if shear_thinning_ratio is not None or viscosity_low is not None or viscosity_high is not None:
        ratio_term = shear_thinning_ratio if shear_thinning_ratio is not None else (
            (viscosity_low or 1.0) / max((viscosity_high or 1.0), 1e-12)
        )
        low_viscosity_term = viscosity_low if viscosity_low is not None else viscosity_high if viscosity_high is not None else 1.0
        loss_tangent = 0.05 + 0.08 * np.log1p(max(ratio_term, 0.0)) + 0.015 * np.log1p(max(low_viscosity_term, 0.0))
        poroelastic_tau = 0.02 + 0.015 * np.log1p(max(low_viscosity_term, 0.0))
        strain_stiffening = 1.15 + 0.28 * np.log1p(max(ratio_term, 0.0))
        hints["loss_tangent_proxy"] = float(np.clip(loss_tangent, 0.05, 0.8))
        hints["poroelastic_time_constant_proxy"] = float(np.clip(poroelastic_tau, 0.01, 0.2))
        hints["strain_stiffening_proxy"] = float(np.clip(strain_stiffening, 1.0, 3.0))
    return hints


def calibration_candidate_error(candidate: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    stiffness_abs_error = abs(float(candidate["features"]["stiffness_mean"]) - float(target["target_stiffness"]))
    stiffness_rel_error = stiffness_abs_error / max(abs(float(target["target_stiffness"])), 1e-12)
    auxiliary_errors = _auxiliary_calibration_errors(candidate, target)
    combined_error = stiffness_rel_error + sum(auxiliary_errors.values())
    return {
        "stiffness_abs_error": float(stiffness_abs_error),
        "stiffness_rel_error": float(stiffness_rel_error),
        "auxiliary_errors": auxiliary_errors,
        "combined_error": float(combined_error),
    }


def _auxiliary_calibration_errors(candidate: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    bundle = target.get("measurement_bundle", {})
    params = candidate.get("parameters", {})
    errors: dict[str, float] = {}
    acoustic_predictions = _predicted_acoustic_observables(params)
    density_predicted = acoustic_predictions["density_g_ml"]
    sound_speed_predicted = acoustic_predictions["speed_of_sound_m_s"]
    acoustic_impedance_predicted = acoustic_predictions["acoustic_impedance_mrayl"]
    bulk_modulus_predicted = acoustic_predictions["bulk_modulus_gpa"]
    if "density_g_ml" in bundle:
        observed = float(bundle["density_g_ml"]["value"])
        errors["density"] = _weighted_relative_metric_error(
            "density",
            observed,
            density_predicted,
            bundle["density_g_ml"].get("std"),
        )
    if "bulk_modulus_gpa" in bundle:
        observed = float(bundle["bulk_modulus_gpa"]["value"])
        errors["bulk_modulus"] = _weighted_relative_metric_error(
            "bulk_modulus",
            observed,
            bulk_modulus_predicted,
            bundle["bulk_modulus_gpa"].get("std"),
        )
    if "speed_of_sound_m_s" in bundle:
        observed = float(bundle["speed_of_sound_m_s"]["value"])
        errors["speed_of_sound"] = _weighted_relative_metric_error(
            "speed_of_sound",
            observed,
            sound_speed_predicted,
            bundle["speed_of_sound_m_s"].get("std"),
        )
    if "acoustic_impedance_mrayl" in bundle:
        observed = float(bundle["acoustic_impedance_mrayl"]["value"])
        errors["acoustic_impedance"] = _weighted_relative_metric_error(
            "acoustic_impedance",
            observed,
            acoustic_impedance_predicted,
            bundle["acoustic_impedance_mrayl"].get("std"),
        )
    low_viscosity_predicted = 40.0 * (1.0 + float(params["crosslink_prob"])) * (1.0 + 4.0 * float(params["bending_stiffness"])) * (0.6 + float(params["fiber_density"]))
    thinning_ratio_predicted = (
        1.2
        + 4.5 * float(params["bending_stiffness"])
        + 0.9 * float(params["crosslink_prob"])
        + 1.8 * float(params["fiber_density"])
    )
    high_viscosity_predicted = low_viscosity_predicted / max(thinning_ratio_predicted, 1.0)
    if "viscosity_pas" in bundle:
        observed = float(bundle["viscosity_pas"]["value"])
        errors["viscosity"] = _weighted_relative_metric_error("viscosity", observed, low_viscosity_predicted, bundle["viscosity_pas"].get("std"))
    if "viscosity_low_shear_pas" in bundle:
        observed = float(bundle["viscosity_low_shear_pas"]["value"])
        errors["viscosity_low_shear"] = _weighted_relative_metric_error(
            "viscosity_low_shear",
            observed,
            low_viscosity_predicted,
            bundle["viscosity_low_shear_pas"].get("std"),
        )
    if "viscosity_high_shear_pas" in bundle:
        observed = float(bundle["viscosity_high_shear_pas"]["value"])
        errors["viscosity_high_shear"] = _weighted_relative_metric_error(
            "viscosity_high_shear",
            observed,
            high_viscosity_predicted,
            bundle["viscosity_high_shear_pas"].get("std"),
        )
    if "shear_thinning_ratio" in bundle:
        observed = float(bundle["shear_thinning_ratio"]["value"])
        errors["shear_thinning_ratio"] = _weighted_relative_metric_error(
            "shear_thinning_ratio",
            observed,
            thinning_ratio_predicted,
            bundle["shear_thinning_ratio"].get("std"),
        )
    return errors


def _predicted_acoustic_observables(params: dict[str, Any]) -> dict[str, float]:
    fiber_density = float(params["fiber_density"])
    fiber_stiffness = float(params["fiber_stiffness"])
    bending_stiffness = float(params["bending_stiffness"])
    crosslink_prob = float(params["crosslink_prob"])

    density_g_ml = 0.95 + 0.18 * fiber_density
    speed_of_sound_m_s = (
        1470.0
        + 42.0 * crosslink_prob
        + 28.0 * fiber_density
        + 1.25 * fiber_stiffness
        + 18.0 * bending_stiffness
    )
    acoustic_impedance_mrayl = density_g_ml * speed_of_sound_m_s / 1000.0
    bulk_modulus_gpa = density_g_ml * speed_of_sound_m_s * speed_of_sound_m_s / 1_000_000.0
    return {
        "density_g_ml": float(density_g_ml),
        "speed_of_sound_m_s": float(speed_of_sound_m_s),
        "acoustic_impedance_mrayl": float(acoustic_impedance_mrayl),
        "bulk_modulus_gpa": float(bulk_modulus_gpa),
    }


def predict_material_observables(params: dict[str, Any]) -> dict[str, float]:
    acoustic = _predicted_acoustic_observables(params)
    low_viscosity = 40.0 * (1.0 + float(params["crosslink_prob"])) * (1.0 + 4.0 * float(params["bending_stiffness"])) * (0.6 + float(params["fiber_density"]))
    shear_thinning_ratio = (
        1.2
        + 4.5 * float(params["bending_stiffness"])
        + 0.9 * float(params["crosslink_prob"])
        + 1.8 * float(params["fiber_density"])
    )
    high_viscosity = low_viscosity / max(shear_thinning_ratio, 1.0)
    return {
        **acoustic,
        "viscosity_low_shear_pas": float(low_viscosity),
        "viscosity_high_shear_pas": float(high_viscosity),
        "shear_thinning_ratio": float(shear_thinning_ratio),
    }


def _weighted_relative_metric_error(
    metric_name: str,
    observed: float,
    predicted: float,
    std: float | None,
) -> float:
    base_error = _metric_error_value(metric_name, observed, predicted)
    coefficient_of_variation = 0.0 if std is None else abs(float(std)) / max(abs(observed), 1e-12)
    uncertainty_discount = 1.0 / (1.0 + min(coefficient_of_variation, 1.0))
    weighted = base_error * _calibration_metric_weight(metric_name) * uncertainty_discount
    return float(min(weighted, _calibration_metric_cap(metric_name)))


def _metric_error_value(metric_name: str, observed: float, predicted: float) -> float:
    eps = 1e-12
    mode = _calibration_metric_mode(metric_name)
    if mode == "log_ratio":
        return float(abs(np.log(max(abs(predicted), eps) / max(abs(observed), eps))))
    return float(abs(predicted - observed) / max(abs(observed), eps))


def _calibration_metric_mode(metric_name: str) -> str:
    modes = {
        "viscosity": "log_ratio",
        "viscosity_low_shear": "log_ratio",
        "viscosity_high_shear": "log_ratio",
    }
    return modes.get(metric_name, "relative")


def _calibration_metric_weight(metric_name: str) -> float:
    weights = {
        "density": 0.7,
        "bulk_modulus": 0.7,
        "speed_of_sound": 0.45,
        "acoustic_impedance": 0.55,
        "viscosity": 0.15,
        "viscosity_low_shear": 0.2,
        "viscosity_high_shear": 0.08,
        "shear_thinning_ratio": 0.35,
    }
    return float(weights.get(metric_name, 1.0))


def _calibration_metric_cap(metric_name: str) -> float:
    caps = {
        "bulk_modulus": 1.5,
        "density": 0.8,
        "speed_of_sound": 0.6,
        "acoustic_impedance": 0.8,
        "viscosity": 1.0,
        "viscosity_low_shear": 1.0,
        "viscosity_high_shear": 1.0,
        "shear_thinning_ratio": 0.8,
    }
    return float(caps.get(metric_name, 2.0))


def _prior_width_floor_fraction(sample_count: int) -> float:
    if sample_count <= 2:
        return 0.25
    if sample_count <= 5:
        return 0.2
    return 0.175


def _lerp(lower: Any, upper: Any, alpha: float) -> float:
    lower_value = float(lower or 0.0)
    upper_value = float(upper or 0.0)
    return float((1.0 - alpha) * lower_value + alpha * upper_value)


def _lerp_numeric_dict(lower: dict[str, Any], upper: dict[str, Any], alpha: float) -> dict[str, float]:
    keys = set(lower.keys()) | set(upper.keys())
    payload: dict[str, float] = {}
    for key in keys:
        lower_value = lower.get(key, 0.0)
        upper_value = upper.get(key, 0.0)
        if isinstance(lower_value, (int, float)) or isinstance(upper_value, (int, float)):
            payload[str(key)] = _lerp(lower_value, upper_value, alpha)
    return payload


def _lerp_stats_dict(lower: dict[str, Any], upper: dict[str, Any], alpha: float) -> dict[str, dict[str, float]]:
    keys = set(lower.keys()) | set(upper.keys())
    payload: dict[str, dict[str, float]] = {}
    for key in keys:
        lower_stat = lower.get(key, {})
        upper_stat = upper.get(key, {})
        payload[str(key)] = {
            "mean": _lerp(
                lower_stat.get("mean", 0.0) if isinstance(lower_stat, dict) else 0.0,
                upper_stat.get("mean", 0.0) if isinstance(upper_stat, dict) else 0.0,
                alpha,
            ),
            "std": _lerp(
                lower_stat.get("std", 0.0) if isinstance(lower_stat, dict) else 0.0,
                upper_stat.get("std", 0.0) if isinstance(upper_stat, dict) else 0.0,
                alpha,
            ),
        }
    return payload


def _safe_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _slugify_token(value: str) -> str:
    import re

    return re.sub(r"[^\w.-]+", "_", value.strip().lower()).strip("._")
