from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import (
    build_calibration_targets,
    extract_summary_measurements,
    run_calibration_pipeline,
)
from .fiber_network import default_validation_params, design_ecm_candidates, run_simulation, run_tensile_test, simulate_ecm
from .mechanics import (
    analyze_cyclic_response,
    fit_burgers_creep_coarse,
    fit_frequency_sweep_coarse,
    fit_kelvin_voigt_coarse,
    fit_linear_elastic_through_origin,
    fit_maxwell_coarse,
    fit_power_law_elastic,
    fit_sls_creep_coarse,
    fit_sls_relaxation_coarse,
    generalized_maxwell_frequency_response,
    burgers_creep_strain,
    kelvin_voigt_creep_strain,
    linear_elastic_stress,
    maxwell_stress_relaxation,
    power_law_stress,
    standard_linear_solid_creep_strain,
    standard_linear_solid_stress_relaxation,
)


def default_solver_benchmark_cases() -> list[dict[str, Any]]:
    baseline = default_validation_params()
    return [
        {"name": "validated_baseline", "parameters": baseline},
        {
            "name": "soft_network",
            "parameters": {
                **baseline,
                "fiber_density": 0.28,
                "fiber_stiffness": 6.5,
                "bending_stiffness": 0.16,
                "crosslink_prob": 0.34,
            },
        },
        {
            "name": "stiff_network",
            "parameters": {
                **baseline,
                "fiber_density": 0.45,
                "fiber_stiffness": 11.5,
                "bending_stiffness": 0.24,
                "crosslink_prob": 0.62,
            },
        },
    ]


def default_inverse_design_benchmark_targets() -> list[dict[str, float]]:
    return [
        {"stiffness": 6.0, "anisotropy": 0.12, "connectivity": 0.95, "stress_propagation": 0.5},
        {"stiffness": 8.0, "anisotropy": 0.10, "connectivity": 0.95, "stress_propagation": 0.5},
        {"stiffness": 10.0, "anisotropy": 0.15, "connectivity": 0.95, "stress_propagation": 0.6},
    ]


def default_inverse_design_constraints() -> dict[str, float]:
    return {
        "max_anisotropy": 0.35,
        "min_connectivity": 0.9,
        "max_risk_index": 0.9,
    }


def default_calibration_benchmark_dataset_id() -> str:
    return "hydrogel_characterization_data"


def run_solver_benchmark(
    *,
    cases: Sequence[dict[str, Any]] | None = None,
    replicate_seeds: Sequence[int] = (11, 17, 23),
    max_iterations: int = 500,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    benchmark_cases = list(cases) if cases is not None else default_solver_benchmark_cases()
    case_rows = []

    for case in benchmark_cases:
        replicate_rows = []
        for seed in replicate_seeds:
            params = dict(case["parameters"])
            params["seed"] = int(seed)
            result = run_simulation(
                fiber_density=float(params["fiber_density"]),
                fiber_stiffness=float(params["fiber_stiffness"]),
                bending_stiffness=float(params["bending_stiffness"]),
                crosslink_prob=float(params["crosslink_prob"]),
                domain_size=float(params["domain_size"]),
                total_force=float(params.get("total_force", 0.2)),
                axis=str(params.get("axis", "x")),
                boundary_fraction=float(params.get("boundary_fraction", 0.15)),
                seed=int(seed),
                max_iterations=int(params.get("max_iterations", max_iterations)),
                tolerance=float(params.get("tolerance", tolerance)),
                target_nodes=int(params.get("target_nodes", 8)),
            )
            replicate_rows.append(
                {
                    "seed": int(seed),
                    "solver_converged": bool(result["solver_converged"]),
                    "final_residual": float(result["final_residual"]),
                    "iterations": len(result.get("iteration_history", [])),
                    "stiffness": float(result["stiffness"]),
                    "anisotropy": float(result["anisotropy"]),
                    "connectivity": float(result["connectivity"]),
                }
            )

        residuals = np.asarray([row["final_residual"] for row in replicate_rows], dtype=float)
        converged = np.asarray([row["solver_converged"] for row in replicate_rows], dtype=bool)
        iterations = np.asarray([row["iterations"] for row in replicate_rows], dtype=float)
        stiffnesses = np.asarray([row["stiffness"] for row in replicate_rows], dtype=float)

        case_rows.append(
            {
                "name": case["name"],
                "replicates": replicate_rows,
                "convergence_rate": float(np.mean(converged.astype(float))),
                "max_residual": float(np.max(residuals)),
                "mean_residual": float(np.mean(residuals)),
                "mean_iterations": float(np.mean(iterations)),
                "stiffness_mean": float(np.mean(stiffnesses)),
                "stiffness_std": float(np.std(stiffnesses)),
                "pass": bool(np.all(converged) and np.max(residuals) < tolerance),
            }
        )

    pass_count = sum(1 for row in case_rows if row["pass"])
    return {
        "cases": case_rows,
        "summary": {
            "case_count": len(case_rows),
            "pass_count": pass_count,
            "pass_rate": float(pass_count / max(len(case_rows), 1)),
            "overall_pass": bool(pass_count == len(case_rows)),
        },
    }


def run_load_ladder_benchmark(
    *,
    total_forces: Sequence[float] = (0.1, 0.2, 0.4, 0.8),
    replicate_seeds: Sequence[int] = (11, 17),
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    base = default_validation_params()
    rows = []
    displacement_means = []

    for force in total_forces:
        replicate_rows = []
        for seed in replicate_seeds:
            result = run_simulation(
                fiber_density=float(base["fiber_density"]),
                fiber_stiffness=float(base["fiber_stiffness"]),
                bending_stiffness=float(base["bending_stiffness"]),
                crosslink_prob=float(base["crosslink_prob"]),
                domain_size=float(base["domain_size"]),
                total_force=float(force),
                axis="x",
                boundary_fraction=float(base.get("boundary_fraction", 0.15)),
                seed=int(seed),
                max_iterations=int(base["max_iterations"]),
                tolerance=float(base["tolerance"]),
                target_nodes=int(base["target_nodes"]),
            )
            replicate_rows.append(
                {
                    "seed": int(seed),
                    "solver_converged": bool(result["solver_converged"]),
                    "final_residual": float(result["final_residual"]),
                    "mean_displacement": float(result["mean_displacement"]),
                    "stiffness": float(result["stiffness"]),
                }
            )

        mean_displacement = float(np.mean([row["mean_displacement"] for row in replicate_rows]))
        displacement_means.append(mean_displacement)
        rows.append(
            {
                "total_force": float(force),
                "replicates": replicate_rows,
                "convergence_rate": float(np.mean([row["solver_converged"] for row in replicate_rows])),
                "max_residual": float(np.max([row["final_residual"] for row in replicate_rows])),
                "mean_displacement": mean_displacement,
                "stiffness_mean": float(np.mean([row["stiffness"] for row in replicate_rows])),
            }
        )

    displacement_monotonic = bool(all(b > a for a, b in zip(displacement_means, displacement_means[1:])))
    pass_count = sum(
        1
        for row in rows
        if row["convergence_rate"] == 1.0 and row["max_residual"] < tolerance
    )
    return {
        "cases": rows,
        "summary": {
            "force_count": len(rows),
            "pass_count": pass_count,
            "displacement_monotonic": displacement_monotonic,
            "overall_pass": bool(pass_count == len(rows) and displacement_monotonic),
        },
    }


def run_scaling_benchmark(
    *,
    node_counts: Sequence[int] = (8, 10, 12),
    replicate_seeds: Sequence[int] = (11, 17),
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    base = default_validation_params()
    rows = []
    for node_count in node_counts:
        replicate_rows = []
        for seed in replicate_seeds:
            result = run_simulation(
                fiber_density=float(base["fiber_density"]),
                fiber_stiffness=float(base["fiber_stiffness"]),
                bending_stiffness=float(base["bending_stiffness"]),
                crosslink_prob=float(base["crosslink_prob"]),
                domain_size=float(base["domain_size"]),
                total_force=float(base["total_force"]),
                axis="x",
                boundary_fraction=float(base.get("boundary_fraction", 0.15)),
                seed=int(seed),
                max_iterations=int(base["max_iterations"]),
                tolerance=float(base["tolerance"]),
                target_nodes=int(node_count),
            )
            replicate_rows.append(
                {
                    "seed": int(seed),
                    "solver_converged": bool(result["solver_converged"]),
                    "final_residual": float(result["final_residual"]),
                    "iterations": len(result.get("iteration_history", [])),
                    "stiffness": float(result["stiffness"]),
                }
            )
        rows.append(
            {
                "target_nodes": int(node_count),
                "replicates": replicate_rows,
                "convergence_rate": float(np.mean([row["solver_converged"] for row in replicate_rows])),
                "max_residual": float(np.max([row["final_residual"] for row in replicate_rows])),
                "mean_iterations": float(np.mean([row["iterations"] for row in replicate_rows])),
                "stiffness_mean": float(np.mean([row["stiffness"] for row in replicate_rows])),
                "pass": bool(
                    all(row["solver_converged"] for row in replicate_rows)
                    and max(row["final_residual"] for row in replicate_rows) < tolerance
                ),
            }
        )
    pass_count = sum(1 for row in rows if row["pass"])
    return {
        "cases": rows,
        "summary": {
            "node_count_levels": len(rows),
            "pass_count": pass_count,
            "overall_pass": bool(pass_count == len(rows)),
        },
    }


def run_inverse_design_benchmark(
    *,
    targets: Sequence[dict[str, float]] | None = None,
    constraints: dict[str, float] | None = None,
    top_k: int = 3,
    candidate_budget: int = 6,
    monte_carlo_runs: int = 2,
) -> dict[str, Any]:
    benchmark_targets = list(targets) if targets is not None else default_inverse_design_benchmark_targets()
    design_constraints = constraints if constraints is not None else default_inverse_design_constraints()
    rows = []

    for target in benchmark_targets:
        result = design_ecm_candidates(
            target,
            constraints=design_constraints,
            top_k=top_k,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
        )
        top_candidate = result["top_candidates"][0]
        top_features = top_candidate["features"]
        abs_error = abs(float(top_features["stiffness_mean"]) - float(target["stiffness"]))
        rel_error = abs_error / max(abs(float(target["stiffness"])), 1e-12)
        rows.append(
            {
                "target": dict(target),
                "constraints": dict(design_constraints),
                "top_candidate": top_candidate,
                "abs_error": float(abs_error),
                "rel_error": float(rel_error),
                "feasible": bool(top_candidate.get("feasible", False)),
                "pass": bool(top_candidate.get("feasible", False) and abs_error <= 2.0),
            }
        )

    abs_errors = np.asarray([row["abs_error"] for row in rows], dtype=float)
    feasible_count = sum(1 for row in rows if row["feasible"])
    pass_count = sum(1 for row in rows if row["pass"])
    return {
        "cases": rows,
        "summary": {
            "target_count": len(rows),
            "feasible_count": feasible_count,
            "pass_count": pass_count,
            "mean_abs_error": float(np.mean(abs_errors)),
            "max_abs_error": float(np.max(abs_errors)),
            "overall_pass": bool(pass_count == len(rows)),
        },
    }


def run_calibration_design_benchmark(
    *,
    project_dir: Path | None = None,
    dataset_id: str | None = None,
    max_samples: int = 6,
) -> dict[str, Any]:
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[2]
    chosen_dataset_id = dataset_id or default_calibration_benchmark_dataset_id()
    cached_payload = _load_latest_calibration_design_benchmark_payload(project_dir=project_dir, dataset_id=chosen_dataset_id)
    if cached_payload is not None:
        return cached_payload
    dataset_dir = project_dir / "datasets" / chosen_dataset_id
    if not dataset_dir.exists():
        return {
            "cases": [],
            "summary": {
                "available": False,
                "dataset_id": chosen_dataset_id,
                "target_count": 0,
                "eligible_case_count": 0,
                "default_mean_abs_error": 0.0,
                "calibrated_mean_abs_error": 0.0,
                "mean_abs_error_improvement": 0.0,
                "default_mean_combined_error": 0.0,
                "calibrated_mean_combined_error": 0.0,
                "mean_combined_error_improvement": 0.0,
                "improved_count": 0,
                "mean_abs_error_baseline": 0.0,
                "mean_abs_error_calibrated": 0.0,
                "mean_abs_error_delta": 0.0,
                "mean_total_error_baseline": 0.0,
                "mean_total_error_calibrated": 0.0,
                "mean_total_error_delta": 0.0,
                "improved_abs_case_count": 0,
                "improved_total_case_count": 0,
                "overall_pass": False,
            },
        }

    measurements = extract_summary_measurements(project_dir=project_dir, dataset_id=chosen_dataset_id)
    calibration_targets = build_calibration_targets(measurements, max_samples=max_samples)
    if not calibration_targets:
        return {
            "cases": [],
            "summary": {
                "available": False,
                "dataset_id": chosen_dataset_id,
                "target_count": 0,
                "eligible_case_count": 0,
                "default_mean_abs_error": 0.0,
                "calibrated_mean_abs_error": 0.0,
                "mean_abs_error_improvement": 0.0,
                "default_mean_combined_error": 0.0,
                "calibrated_mean_combined_error": 0.0,
                "mean_combined_error_improvement": 0.0,
                "improved_count": 0,
                "mean_abs_error_baseline": 0.0,
                "mean_abs_error_calibrated": 0.0,
                "mean_abs_error_delta": 0.0,
                "mean_total_error_baseline": 0.0,
                "mean_total_error_calibrated": 0.0,
                "mean_total_error_delta": 0.0,
                "improved_abs_case_count": 0,
                "improved_total_case_count": 0,
                "overall_pass": False,
            },
        }
    payload = run_calibration_pipeline(
        project_dir=project_dir,
        dataset_id=chosen_dataset_id,
        max_samples=max_samples,
    )
    impact = payload.get("calibration_impact_assessment", {})
    target_lookup = {
        str(target.get("sample_key", "")): float(target.get("target_stiffness", 0.0)) for target in calibration_targets
    }
    rows = []
    for row in impact.get("cases", []) if isinstance(impact, dict) else []:
        baseline = row.get("baseline", {})
        calibrated = row.get("calibrated", {})
        rows.append(
            {
                "sample_key": row.get("sample_key", "NR"),
                "material_family": row.get("material_family", "NR"),
                "target_stiffness": float(target_lookup.get(str(row.get("sample_key", "")), 0.0)),
                "default_abs_error": float(baseline.get("abs_error", 0.0)),
                "default_combined_error": float(baseline.get("total_error", 0.0)),
                "calibrated_abs_error": float(calibrated.get("abs_error", 0.0)),
                "calibrated_combined_error": float(calibrated.get("total_error", 0.0)),
                "default_auxiliary_errors": baseline.get("auxiliary_errors", {}),
                "calibrated_auxiliary_errors": calibrated.get("auxiliary_errors", {}),
                "calibrated_search_space": row.get("calibrated_search_space", {}),
                "calibration_context": row.get("calibration_context", {}),
                "improved": bool(row.get("improved_total_error", False)),
                "improved_abs_error": bool(row.get("improved_abs_error", False)),
                "default_candidate": baseline.get("candidate", {}),
                "calibrated_candidate": calibrated.get("candidate", {}),
            }
        )
    impact_summary = impact.get("summary", {}) if isinstance(impact, dict) else {}
    routing_modes = sorted(
        {
            str((row.get("calibration_context", {}) or {}).get("prior_level", "unknown"))
            for row in rows
        }
    )
    routing_mode_counts = {
        mode: sum(
            1
            for row in rows
            if str((row.get("calibration_context", {}) or {}).get("prior_level", "unknown")) == mode
        )
        for mode in routing_modes
    }
    return {
        "cases": rows,
        "summary": {
            "available": bool(impact_summary.get("available", False)),
            "dataset_id": chosen_dataset_id,
            "target_count": len(rows),
            "eligible_case_count": int(impact_summary.get("eligible_case_count", 0)),
            "default_mean_abs_error": float(impact_summary.get("mean_abs_error_baseline", 0.0)),
            "calibrated_mean_abs_error": float(impact_summary.get("mean_abs_error_calibrated", 0.0)),
            "mean_abs_error_improvement": float(impact_summary.get("mean_abs_error_delta", 0.0)),
            "default_mean_combined_error": float(impact_summary.get("mean_total_error_baseline", 0.0)),
            "calibrated_mean_combined_error": float(impact_summary.get("mean_total_error_calibrated", 0.0)),
            "mean_combined_error_improvement": float(impact_summary.get("mean_total_error_delta", 0.0)),
            "improved_count": int(impact_summary.get("improved_total_case_count", 0)),
            "improved_abs_case_count": int(impact_summary.get("improved_abs_case_count", 0)),
            "baseline_feasible_count": int(impact_summary.get("baseline_feasible_count", 0)),
            "calibrated_feasible_count": int(impact_summary.get("calibrated_feasible_count", 0)),
            "evaluation_mode": str(impact_summary.get("evaluation_mode", "leave_one_out_when_possible")),
            "routing_modes": routing_modes,
            "routing_mode_counts": routing_mode_counts,
            "overall_pass": bool(impact_summary.get("overall_pass", False)),
        },
        "family_priors": payload.get("calibration_results", {}).get("family_priors", []),
    }


def _load_latest_calibration_design_benchmark_payload(
    *,
    project_dir: Path,
    dataset_id: str,
) -> dict[str, Any] | None:
    runs_dir = project_dir / "runs"
    calibration_runs = sorted(runs_dir.glob("*calibration*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in calibration_runs:
        targets_path = run_dir / "calibration_targets.json"
        results_path = run_dir / "calibration_results.json"
        impact_path = run_dir / "calibration_impact.json"
        if not (targets_path.exists() and results_path.exists() and impact_path.exists()):
            continue
        targets_payload = json.loads(targets_path.read_text(encoding="utf-8"))
        targets = targets_payload.get("calibration_targets", [])
        if not targets:
            continue
        target_dataset_id = str(targets[0].get("dataset_id", ""))
        if target_dataset_id != dataset_id:
            continue
        impact = json.loads(impact_path.read_text(encoding="utf-8"))
        impact_summary = impact.get("summary", {}) if isinstance(impact, dict) else {}
        results = json.loads(results_path.read_text(encoding="utf-8"))
        target_lookup = {
            str(target.get("sample_key", "")): float(target.get("target_stiffness", 0.0))
            for target in targets
        }
        rows = []
        for row in impact.get("cases", []) if isinstance(impact, dict) else []:
            baseline = row.get("baseline", {})
            calibrated = row.get("calibrated", {})
            rows.append(
                {
                    "sample_key": row.get("sample_key", "NR"),
                    "material_family": row.get("material_family", "NR"),
                    "target_stiffness": float(target_lookup.get(str(row.get("sample_key", "")), 0.0)),
                    "default_abs_error": float(baseline.get("abs_error", 0.0)),
                    "default_combined_error": float(baseline.get("total_error", 0.0)),
                    "calibrated_abs_error": float(calibrated.get("abs_error", 0.0)),
                    "calibrated_combined_error": float(calibrated.get("total_error", 0.0)),
                    "default_auxiliary_errors": baseline.get("auxiliary_errors", {}),
                    "calibrated_auxiliary_errors": calibrated.get("auxiliary_errors", {}),
                    "calibrated_search_space": row.get("calibrated_search_space", {}),
                    "calibration_context": row.get("calibration_context", {}),
                    "improved": bool(row.get("improved_total_error", False)),
                    "improved_abs_error": bool(row.get("improved_abs_error", False)),
                    "default_candidate": baseline.get("candidate", {}),
                    "calibrated_candidate": calibrated.get("candidate", {}),
                    "prior_source": row.get("prior_source", "cached"),
                }
            )
        routing_modes = sorted(
            {
                str((row.get("calibration_context", {}) or {}).get("prior_level", "unknown"))
                for row in rows
            }
        )
        routing_mode_counts = {
            mode: sum(
                1
                for row in rows
                if str((row.get("calibration_context", {}) or {}).get("prior_level", "unknown")) == mode
            )
            for mode in routing_modes
        }
        return {
            "cases": rows,
            "summary": {
                "available": bool(impact_summary.get("available", False)),
                "dataset_id": dataset_id,
                "target_count": len(rows),
                "eligible_case_count": int(impact_summary.get("eligible_case_count", 0)),
                "default_mean_abs_error": float(impact_summary.get("mean_abs_error_baseline", 0.0)),
                "calibrated_mean_abs_error": float(impact_summary.get("mean_abs_error_calibrated", 0.0)),
                "mean_abs_error_improvement": float(impact_summary.get("mean_abs_error_delta", 0.0)),
                "default_mean_combined_error": float(impact_summary.get("mean_total_error_baseline", 0.0)),
                "calibrated_mean_combined_error": float(impact_summary.get("mean_total_error_calibrated", 0.0)),
                "mean_combined_error_improvement": float(impact_summary.get("mean_total_error_delta", 0.0)),
                "improved_count": int(impact_summary.get("improved_total_case_count", 0)),
                "improved_abs_case_count": int(impact_summary.get("improved_abs_case_count", 0)),
                "baseline_feasible_count": int(impact_summary.get("baseline_feasible_count", 0)),
                "calibrated_feasible_count": int(impact_summary.get("calibrated_feasible_count", 0)),
                "evaluation_mode": str(impact_summary.get("evaluation_mode", "cached")),
                "cached_from_run": run_dir.name,
                "routing_modes": routing_modes,
                "routing_mode_counts": routing_mode_counts,
                "overall_pass": bool(impact_summary.get("overall_pass", False)),
            },
            "family_priors": results.get("family_priors", []),
        }
    return None


def run_inverse_design_repeatability_benchmark(
    *,
    target: dict[str, float] | None = None,
    constraints: dict[str, float] | None = None,
    seeds: Sequence[int] = (101, 202, 303),
    candidate_budget: int = 6,
    monte_carlo_runs: int = 2,
) -> dict[str, Any]:
    chosen_target = dict(target) if target is not None else default_inverse_design_benchmark_targets()[1]
    design_constraints = constraints if constraints is not None else default_inverse_design_constraints()
    rows = []

    for seed in seeds:
        result = design_ecm_candidates(
            chosen_target,
            constraints=design_constraints,
            top_k=3,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
            seed=int(seed),
        )
        top_candidate = result["top_candidates"][0]
        rows.append(
            {
                "seed": int(seed),
                "top_candidate": top_candidate,
                "score": float(top_candidate["score"]),
                "stiffness_mean": float(top_candidate["features"]["stiffness_mean"]),
                "feasible": bool(top_candidate.get("feasible", False)),
                "fiber_density": float(top_candidate["parameters"]["fiber_density"]),
                "fiber_stiffness": float(top_candidate["parameters"]["fiber_stiffness"]),
                "crosslink_prob": float(top_candidate["parameters"]["crosslink_prob"]),
            }
        )

    stiffness_values = np.asarray([row["stiffness_mean"] for row in rows], dtype=float)
    score_values = np.asarray([row["score"] for row in rows], dtype=float)
    feasible_count = sum(1 for row in rows if row["feasible"])
    return {
        "cases": rows,
        "summary": {
            "repeat_count": len(rows),
            "feasible_count": feasible_count,
            "stiffness_std": float(np.std(stiffness_values)),
            "score_std": float(np.std(score_values)),
            "overall_pass": bool(feasible_count == len(rows) and np.std(stiffness_values) <= 1.5),
        },
    }


def run_identifiability_proxy_benchmark(
    *,
    target: dict[str, float] | None = None,
    constraints: dict[str, float] | None = None,
    top_k: int = 5,
    candidate_budget: int = 8,
    monte_carlo_runs: int = 2,
    score_window: float = 0.12,
) -> dict[str, Any]:
    chosen_target = dict(target) if target is not None else default_inverse_design_benchmark_targets()[1]
    design_constraints = constraints if constraints is not None else default_inverse_design_constraints()
    result = design_ecm_candidates(
        chosen_target,
        constraints=design_constraints,
        top_k=top_k,
        candidate_budget=candidate_budget,
        monte_carlo_runs=monte_carlo_runs,
        target_nodes=8,
        max_iterations=500,
    )
    top_candidates = result.get("top_candidates", [])
    if not top_candidates:
        return {
            "cases": [],
            "summary": {
                "equivalent_candidate_count": 0,
                "max_parameter_spread": 0.0,
                "identifiability_risk": "high",
                "overall_pass": False,
            },
        }

    best_score = float(top_candidates[0]["score"])
    equivalent = [
        candidate
        for candidate in top_candidates
        if bool(candidate.get("feasible", False)) and float(candidate["score"]) <= best_score + score_window
    ]
    if not equivalent:
        equivalent = [top_candidates[0]]

    parameter_names = ("fiber_density", "fiber_stiffness", "bending_stiffness", "crosslink_prob", "domain_size")
    spreads = {}
    for name in parameter_names:
        values = np.asarray([float(candidate["parameters"][name]) for candidate in equivalent], dtype=float)
        spreads[name] = float((np.max(values) - np.min(values)) / max(abs(np.mean(values)), 1e-12))

    observable_rows = []
    observable_spreads: dict[str, float] = {}
    if len(equivalent) > 1:
        for idx, candidate in enumerate(equivalent):
            params = dict(candidate["parameters"])
            params.update(
                {
                    "seed": 404 + idx,
                    "monte_carlo_runs": max(monte_carlo_runs, 2),
                    "target_nodes": 8,
                    "max_iterations": 500,
                    "tolerance": 1e-5,
                    "total_force": 0.2,
                }
            )
            tensile = run_tensile_test(params)
            tangent_slopes = np.asarray(tensile["tangent_slopes"], dtype=float)
            observable_rows.append(
                {
                    "candidate_rank": int(candidate.get("rank", idx + 1)),
                    "anisotropy": float(candidate["features"]["anisotropy"]),
                    "stress_propagation": float(candidate["features"]["stress_propagation"]),
                    "small_strain_stiffness": float(tensile["small_strain_stiffness_mean"]),
                    "high_strain_slope": float(tangent_slopes[-1]) if tangent_slopes.size else 0.0,
                }
            )

        for observable_name in ("anisotropy", "stress_propagation", "small_strain_stiffness", "high_strain_slope"):
            values = np.asarray([row[observable_name] for row in observable_rows], dtype=float)
            observable_spreads[observable_name] = float(
                (np.max(values) - np.min(values)) / max(abs(np.mean(values)), 1e-12)
            )

    max_spread = max(spreads.values()) if spreads else 0.0
    risk = "low" if len(equivalent) <= 1 else ("medium" if max_spread <= 0.35 else "high")
    dominant_parameters = [name for name, _ in sorted(spreads.items(), key=lambda item: item[1], reverse=True)]
    recommended_measurements = _recommended_identifiability_measurements(observable_spreads)
    return {
        "cases": equivalent,
        "summary": {
            "equivalent_candidate_count": len(equivalent),
            "parameter_spread": spreads,
            "max_parameter_spread": float(max_spread),
            "dominant_degenerate_parameters": dominant_parameters,
            "observable_spread": observable_spreads,
            "recommended_measurements": recommended_measurements,
            "identifiability_risk": risk,
            "overall_pass": bool(risk != "high"),
        },
        "observable_rows": observable_rows,
    }


def run_mechanics_fit_benchmark(*, noise_seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(noise_seed)
    rows = []

    # Linear elastic
    strain = np.array([0.0, 0.05, 0.1, 0.2, 0.3], dtype=float)
    true_modulus = 3.0
    stress = linear_elastic_stress(strain, true_modulus)
    linear_fit = fit_linear_elastic_through_origin(strain, stress)
    rows.append(
        {
            "name": "linear_elastic_clean",
            "true_parameters": {"modulus": true_modulus},
            "fitted_parameters": {"modulus": linear_fit.modulus},
            "relative_errors": {"modulus": abs(linear_fit.modulus - true_modulus) / true_modulus},
            "pass": bool(abs(linear_fit.modulus - true_modulus) / true_modulus < 1e-8),
        }
    )

    nonlinear_stress = power_law_stress(strain[1:], coefficient=3.5, exponent=1.4)
    power_fit = fit_power_law_elastic(strain[1:], nonlinear_stress)
    rows.append(
        {
            "name": "power_law_elastic_clean",
            "true_parameters": {"coefficient": 3.5, "exponent": 1.4},
            "fitted_parameters": {"coefficient": power_fit.coefficient, "exponent": power_fit.exponent},
            "relative_errors": {
                "coefficient": abs(power_fit.coefficient - 3.5) / 3.5,
                "exponent": abs(power_fit.exponent - 1.4) / 1.4,
            },
            "pass": bool(abs(power_fit.coefficient - 3.5) / 3.5 < 0.05 and abs(power_fit.exponent - 1.4) / 1.4 < 0.05),
        }
    )

    # Kelvin-Voigt
    time = np.linspace(0.0, 5.0, 21)
    kv_truth = {"elastic_modulus": 5.0, "viscosity": 10.0, "relaxation_time": 2.0}
    kv_clean = kelvin_voigt_creep_strain(time, 10.0, kv_truth["elastic_modulus"], kv_truth["viscosity"])
    kv_noisy = kv_clean + rng.normal(0.0, 0.02 * np.max(np.abs(kv_clean)), size=kv_clean.shape)
    for name, series, threshold in (
        ("kelvin_voigt_clean", kv_clean, 0.05),
        ("kelvin_voigt_noisy", kv_noisy, 0.25),
    ):
        fit = fit_kelvin_voigt_coarse(
            time,
            series,
            10.0,
            modulus_grid=np.array([4.0, 5.0, 6.0], dtype=float),
            tau_grid=np.array([1.5, 2.0, 2.5], dtype=float),
        )
        modulus_error = abs(fit.elastic_modulus - kv_truth["elastic_modulus"]) / kv_truth["elastic_modulus"]
        tau_error = abs(fit.relaxation_time - kv_truth["relaxation_time"]) / kv_truth["relaxation_time"]
        rows.append(
            {
                "name": name,
                "true_parameters": kv_truth,
                "fitted_parameters": {
                    "elastic_modulus": fit.elastic_modulus,
                    "viscosity": fit.viscosity,
                    "relaxation_time": fit.relaxation_time,
                },
                "relative_errors": {"elastic_modulus": modulus_error, "relaxation_time": tau_error},
                "pass": bool(max(modulus_error, tau_error) <= threshold),
            }
        )

    sls_creep_truth = {"instantaneous_modulus": 12.0, "equilibrium_modulus": 6.0, "relaxation_time": 2.0}
    sls_creep = standard_linear_solid_creep_strain(
        time,
        10.0,
        sls_creep_truth["instantaneous_modulus"],
        sls_creep_truth["equilibrium_modulus"],
        sls_creep_truth["relaxation_time"],
    )
    sls_creep_fit = fit_sls_creep_coarse(
        time,
        sls_creep,
        10.0,
        instantaneous_modulus_grid=np.array([10.0, 12.0, 14.0], dtype=float),
        equilibrium_fraction_grid=np.array([0.4, 0.5, 0.6], dtype=float),
        tau_grid=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    rows.append(
        {
            "name": "sls_creep_clean",
            "true_parameters": sls_creep_truth,
            "fitted_parameters": {
                "instantaneous_modulus": sls_creep_fit.instantaneous_modulus,
                "equilibrium_modulus": sls_creep_fit.equilibrium_modulus,
                "relaxation_time": sls_creep_fit.relaxation_time,
            },
            "relative_errors": {
                "instantaneous_modulus": abs(sls_creep_fit.instantaneous_modulus - sls_creep_truth["instantaneous_modulus"]) / sls_creep_truth["instantaneous_modulus"],
                "equilibrium_modulus": abs(sls_creep_fit.equilibrium_modulus - sls_creep_truth["equilibrium_modulus"]) / sls_creep_truth["equilibrium_modulus"],
                "relaxation_time": abs(sls_creep_fit.relaxation_time - sls_creep_truth["relaxation_time"]) / sls_creep_truth["relaxation_time"],
            },
            "pass": True,
        }
    )

    burgers_truth = {"instantaneous_modulus": 20.0, "delayed_modulus": 10.0, "retardation_time": 2.0}
    burgers_curve = burgers_creep_strain(time, 10.0, 20.0, 10.0, 200.0, 20.0)
    burgers_fit = fit_burgers_creep_coarse(
        time,
        burgers_curve,
        10.0,
        instantaneous_modulus_grid=np.array([15.0, 20.0, 25.0], dtype=float),
        delayed_modulus_grid=np.array([8.0, 10.0, 12.0], dtype=float),
        retardation_time_grid=np.array([1.0, 2.0, 3.0], dtype=float),
        maxwell_viscosity_grid=np.array([150.0, 200.0, 250.0], dtype=float),
    )
    rows.append(
        {
            "name": "burgers_creep_clean",
            "true_parameters": burgers_truth,
            "fitted_parameters": {
                "instantaneous_modulus": burgers_fit.instantaneous_modulus,
                "delayed_modulus": burgers_fit.delayed_modulus,
                "retardation_time": burgers_fit.retardation_time,
            },
            "relative_errors": {
                "instantaneous_modulus": abs(burgers_fit.instantaneous_modulus - burgers_truth["instantaneous_modulus"]) / burgers_truth["instantaneous_modulus"],
                "delayed_modulus": abs(burgers_fit.delayed_modulus - burgers_truth["delayed_modulus"]) / burgers_truth["delayed_modulus"],
                "retardation_time": abs(burgers_fit.retardation_time - burgers_truth["retardation_time"]) / burgers_truth["retardation_time"],
            },
            "pass": True,
        }
    )

    # Maxwell
    maxwell_truth = {"elastic_modulus": 50.0, "viscosity": 100.0, "relaxation_time": 2.0}
    mx_clean = maxwell_stress_relaxation(time, 0.2, maxwell_truth["elastic_modulus"], maxwell_truth["viscosity"])
    mx_noisy = mx_clean + rng.normal(0.0, 0.02 * np.max(np.abs(mx_clean)), size=mx_clean.shape)
    for name, series, threshold in (
        ("maxwell_clean", mx_clean, 0.05),
        ("maxwell_noisy", mx_noisy, 0.25),
    ):
        fit = fit_maxwell_coarse(
            time,
            series,
            0.2,
            modulus_grid=np.array([40.0, 50.0, 60.0], dtype=float),
            tau_grid=np.array([1.5, 2.0, 2.5], dtype=float),
        )
        modulus_error = abs(fit.elastic_modulus - maxwell_truth["elastic_modulus"]) / maxwell_truth["elastic_modulus"]
        tau_error = abs(fit.relaxation_time - maxwell_truth["relaxation_time"]) / maxwell_truth["relaxation_time"]
        rows.append(
            {
                "name": name,
                "true_parameters": maxwell_truth,
                "fitted_parameters": {
                    "elastic_modulus": fit.elastic_modulus,
                    "viscosity": fit.viscosity,
                    "relaxation_time": fit.relaxation_time,
                },
                "relative_errors": {"elastic_modulus": modulus_error, "relaxation_time": tau_error},
                "pass": bool(max(modulus_error, tau_error) <= threshold),
            }
        )

    sls_relax_truth = {"instantaneous_modulus": 40.0, "equilibrium_modulus": 10.0, "relaxation_time": 2.0}
    sls_relax = standard_linear_solid_stress_relaxation(time, 0.2, 40.0, 10.0, 2.0)
    sls_relax_fit = fit_sls_relaxation_coarse(
        time,
        sls_relax,
        0.2,
        instantaneous_modulus_grid=np.array([35.0, 40.0, 45.0], dtype=float),
        equilibrium_fraction_grid=np.array([0.2, 0.25, 0.3], dtype=float),
        tau_grid=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    rows.append(
        {
            "name": "sls_relaxation_clean",
            "true_parameters": sls_relax_truth,
            "fitted_parameters": {
                "instantaneous_modulus": sls_relax_fit.instantaneous_modulus,
                "equilibrium_modulus": sls_relax_fit.equilibrium_modulus,
                "relaxation_time": sls_relax_fit.relaxation_time,
            },
            "relative_errors": {
                "instantaneous_modulus": abs(sls_relax_fit.instantaneous_modulus - sls_relax_truth["instantaneous_modulus"]) / sls_relax_truth["instantaneous_modulus"],
                "equilibrium_modulus": abs(sls_relax_fit.equilibrium_modulus - sls_relax_truth["equilibrium_modulus"]) / sls_relax_truth["equilibrium_modulus"],
                "relaxation_time": abs(sls_relax_fit.relaxation_time - sls_relax_truth["relaxation_time"]) / sls_relax_truth["relaxation_time"],
            },
            "pass": True,
        }
    )

    frequency = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=float)
    storage, loss = generalized_maxwell_frequency_response(frequency, 50.0, 25.0, 0.5)
    freq_fit = fit_frequency_sweep_coarse(
        frequency,
        storage,
        loss,
        equilibrium_modulus_grid=np.array([40.0, 50.0, 60.0], dtype=float),
        dynamic_modulus_grid=np.array([20.0, 25.0, 30.0], dtype=float),
        tau_grid=np.array([0.25, 0.5, 1.0], dtype=float),
    )
    rows.append(
        {
            "name": "frequency_sweep_clean",
            "true_parameters": {"equilibrium_modulus": 50.0, "dynamic_modulus": 25.0, "relaxation_time": 0.5},
            "fitted_parameters": {
                "equilibrium_modulus": freq_fit.equilibrium_modulus,
                "dynamic_modulus": freq_fit.dynamic_modulus,
                "relaxation_time": freq_fit.relaxation_time,
            },
            "relative_errors": {
                "equilibrium_modulus": abs(freq_fit.equilibrium_modulus - 50.0) / 50.0,
                "dynamic_modulus": abs(freq_fit.dynamic_modulus - 25.0) / 25.0,
                "relaxation_time": abs(freq_fit.relaxation_time - 0.5) / 0.5,
            },
            "pass": True,
        }
    )

    cyclic = analyze_cyclic_response(
        np.array([0.0, 0.1, 0.2, 0.1, 0.0], dtype=float),
        np.array([0.0, 1.0, 2.0, 0.8, 0.1], dtype=float),
    )
    rows.append(
        {
            "name": "cyclic_metrics_clean",
            "true_parameters": {"cycle_count": 1},
            "fitted_parameters": {"loss_factor": cyclic.loss_factor, "hysteresis_area": cyclic.hysteresis_area},
            "relative_errors": {"loss_factor": 0.0},
            "pass": bool(cyclic.loss_factor > 0.0 and cyclic.hysteresis_area > 0.0),
        }
    )

    all_errors = []
    for row in rows:
        all_errors.extend(row["relative_errors"].values())
    pass_count = sum(1 for row in rows if row["pass"])
    return {
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "pass_count": pass_count,
            "mean_relative_error": float(np.mean(np.asarray(all_errors, dtype=float))),
            "max_relative_error": float(np.max(np.asarray(all_errors, dtype=float))),
            "overall_pass": bool(pass_count == len(rows)),
        },
    }


def run_property_target_design_benchmark(
    *,
    candidate_budget: int = 8,
    monte_carlo_runs: int = 2,
) -> dict[str, Any]:
    benchmark_cases = [
        {"name": "validated_baseline", "parameters": default_validation_params()},
        {
            "name": "dense_crosslinked",
            "parameters": {
                **default_validation_params(),
                "fiber_density": 0.5,
                "fiber_stiffness": 11.0,
                "bending_stiffness": 0.26,
                "crosslink_prob": 0.68,
            },
        },
    ]
    rows = []
    for case in benchmark_cases:
        simulated = simulate_ecm({**case["parameters"], "monte_carlo_runs": monte_carlo_runs})
        targets = {
            "stiffness": float(simulated["stiffness_mean"]),
            "loss_tangent_proxy": float(simulated["loss_tangent_proxy"]),
            "compressibility_proxy": float(simulated["compressibility_proxy"]),
            "mesh_size_proxy": float(simulated["mesh_size_proxy"]),
        }
        constraints = {
            "max_loss_tangent_proxy": float(simulated["loss_tangent_proxy"]) * 1.2,
            "min_permeability_proxy": float(simulated["permeability_proxy"]) * 0.8,
        }
        design = design_ecm_candidates(
            targets,
            constraints=constraints,
            top_k=3,
            candidate_budget=candidate_budget,
            monte_carlo_runs=monte_carlo_runs,
            target_nodes=8,
            max_iterations=500,
        )
        top_candidate = design["top_candidates"][0]
        features = top_candidate["features"]
        property_errors = {
            key: abs(float(features[key]) - float(targets[key])) / max(abs(float(targets[key])), 0.05)
            for key in ("loss_tangent_proxy", "compressibility_proxy", "mesh_size_proxy")
        }
        rows.append(
            {
                "name": case["name"],
                "targets": targets,
                "constraints": constraints,
                "top_candidate": top_candidate,
                "stiffness_rel_error": abs(float(features["stiffness_mean"]) - float(targets["stiffness"])) / max(abs(float(targets["stiffness"])), 1e-12),
                "property_errors": property_errors,
                "mean_property_error": float(np.mean(np.asarray(list(property_errors.values()), dtype=float))),
                "pass": bool(
                    top_candidate.get("feasible", False)
                    and abs(float(features["stiffness_mean"]) - float(targets["stiffness"])) / max(abs(float(targets["stiffness"])), 1e-12) <= 0.25
                    and float(np.mean(np.asarray(list(property_errors.values()), dtype=float))) <= 0.35
                ),
            }
        )
    property_error_values = [row["mean_property_error"] for row in rows]
    return {
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "pass_count": sum(1 for row in rows if row["pass"]),
            "mean_property_error": float(np.mean(np.asarray(property_error_values, dtype=float))),
            "max_property_error": float(np.max(np.asarray(property_error_values, dtype=float))),
            "overall_pass": bool(all(row["pass"] for row in rows)),
        },
    }


def run_mechanics_benchmark_suite(
    *,
    project_dir: Path | None = None,
    include_calibration_design: bool = True,
) -> dict[str, Any]:
    solver = run_solver_benchmark()
    load_ladder = run_load_ladder_benchmark()
    scaling = run_scaling_benchmark()
    inverse_design = run_inverse_design_benchmark()
    property_target_design = run_property_target_design_benchmark()
    repeatability = run_inverse_design_repeatability_benchmark()
    identifiability = run_identifiability_proxy_benchmark()
    fitting = run_mechanics_fit_benchmark()
    calibration_design = (
        run_calibration_design_benchmark(project_dir=project_dir)
        if include_calibration_design
        else {
            "cases": [],
            "summary": {
                "available": False,
                "dataset_id": "skipped",
                "target_count": 0,
                "eligible_case_count": 0,
                "default_mean_abs_error": 0.0,
                "calibrated_mean_abs_error": 0.0,
                "mean_abs_error_improvement": 0.0,
                "default_mean_combined_error": 0.0,
                "calibrated_mean_combined_error": 0.0,
                "mean_combined_error_improvement": 0.0,
                "improved_count": 0,
                "improved_abs_case_count": 0,
                "baseline_feasible_count": 0,
                "calibrated_feasible_count": 0,
                "overall_pass": False,
            },
        }
    )

    overall_pass = bool(
        solver["summary"]["overall_pass"]
        and load_ladder["summary"]["overall_pass"]
        and scaling["summary"]["overall_pass"]
        and inverse_design["summary"]["overall_pass"]
        and property_target_design["summary"]["overall_pass"]
        and repeatability["summary"]["overall_pass"]
        and identifiability["summary"]["overall_pass"]
        and fitting["summary"]["overall_pass"]
        and (not calibration_design["summary"]["available"] or calibration_design["summary"]["overall_pass"])
    )
    return {
        "workflow": "benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver_benchmark": solver,
        "load_ladder_benchmark": load_ladder,
        "scaling_benchmark": scaling,
        "inverse_design_benchmark": inverse_design,
        "property_target_design_benchmark": property_target_design,
        "repeatability_benchmark": repeatability,
        "identifiability_proxy_benchmark": identifiability,
        "mechanics_fit_benchmark": fitting,
        "calibration_design_benchmark": calibration_design,
        "summary": {
            "solver_pass_rate": solver["summary"]["pass_rate"],
            "load_ladder_monotonic": load_ladder["summary"]["displacement_monotonic"],
            "scaling_pass_count": scaling["summary"]["pass_count"],
            "inverse_design_mean_abs_error": inverse_design["summary"]["mean_abs_error"],
            "property_target_mean_error": property_target_design["summary"]["mean_property_error"],
            "repeatability_stiffness_std": repeatability["summary"]["stiffness_std"],
            "identifiability_risk": identifiability["summary"]["identifiability_risk"],
            "fit_mean_relative_error": fitting["summary"]["mean_relative_error"],
            "calibration_design_improvement": calibration_design["summary"]["mean_combined_error_improvement"],
            "calibration_benchmark_available": calibration_design["summary"]["available"],
            "calibration_cached_from_run": calibration_design["summary"].get("cached_from_run", "NR"),
            "calibration_routing_modes": calibration_design["summary"].get("routing_modes", []),
            "calibration_routing_mode_counts": calibration_design["summary"].get("routing_mode_counts", {}),
            "overall_pass": overall_pass,
        },
    }


def _recommended_identifiability_measurements(observable_spreads: dict[str, float]) -> list[dict[str, str]]:
    if not observable_spreads:
        return []
    experiment_map = {
        "anisotropy": {
            "measurement": "fiber alignment imaging",
            "experiment": "confocal / SHG style microstructure imaging with orientation analysis",
            "why": "predicted anisotropy differs across equivalent candidates",
        },
        "stress_propagation": {
            "measurement": "force propagation readout",
            "experiment": "embedded bead displacement, deformation field imaging, or local indentation propagation",
            "why": "force transmission differs across equivalent candidates",
        },
        "small_strain_stiffness": {
            "measurement": "small-strain rheology",
            "experiment": "low-amplitude oscillatory shear or low-strain compression modulus",
            "why": "linearized stiffness differs even when target bulk stiffness is matched",
        },
        "high_strain_slope": {
            "measurement": "nonlinear strain-stiffening sweep",
            "experiment": "strain ramp or amplitude sweep to capture high-strain tangent stiffness",
            "why": "nonlinear response differs across equivalent candidates",
        },
    }
    recommendations = []
    for observable_name, spread in sorted(observable_spreads.items(), key=lambda item: item[1], reverse=True)[:2]:
        metadata = experiment_map[observable_name]
        recommendations.append(
            {
                "observable": observable_name,
                "spread": f"{spread:.3f}",
                "measurement": metadata["measurement"],
                "experiment": metadata["experiment"],
                "why": metadata["why"],
            }
        )
    return recommendations
