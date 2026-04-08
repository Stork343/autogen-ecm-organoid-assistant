from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import write_json
from .builder import BuildArtifacts, build_simulation_input
from .config import FEBioConfig
from .metrics import calculate_simulation_metrics
from .parser import parse_simulation_outputs
from .runner import run_febio_job
from .schemas import (
    BulkMechanicsRequest,
    OrganoidSpheroidRequest,
    SimulationRequest,
    SingleCellContractionRequest,
)


def _build_final_summary(
    *,
    request: SimulationRequest,
    result_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> str:
    lines = [
        "# FEBio Simulation Summary",
        f"- scenario: {request.scenario}",
        f"- status: {result_payload.get('status', 'unknown')}",
        f"- solver_converged: {metrics_payload.get('feasibility_flags', {}).get('solver_converged', False)}",
        f"- effective_stiffness: {metrics_payload.get('effective_stiffness', 'NR')}",
        f"- peak_stress: {metrics_payload.get('peak_stress', 'NR')}",
        f"- displacement_decay_length: {metrics_payload.get('displacement_decay_length', 'NR')}",
        f"- strain_heterogeneity: {metrics_payload.get('strain_heterogeneity', 'NR')}",
        f"- target_mismatch_score: {metrics_payload.get('target_mismatch_score', 'NR')}",
    ]
    warnings = result_payload.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])
    error_message = result_payload.get("error_message")
    if error_message:
        lines.extend(["", "## Error", f"- {error_message}"])
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- input_feb: {result_payload.get('raw_output_paths', {}).get('input_feb', 'NR')}",
            f"- simulation_result_json: {Path(result_payload.get('raw_output_paths', {}).get('input_feb', '')).with_name('simulation_result.json')}",
            f"- simulation_metrics_json: {Path(result_payload.get('raw_output_paths', {}).get('input_feb', '')).with_name('simulation_metrics.json')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_request(
    *,
    request: SimulationRequest,
    simulation_dir: Path,
    febio_config: FEBioConfig,
) -> dict[str, Any]:
    simulation_dir.mkdir(parents=True, exist_ok=True)
    input_request_path = simulation_dir / "input_request.json"
    write_json(input_request_path, request.to_dict())

    build_artifacts: BuildArtifacts = build_simulation_input(request, simulation_dir)
    runner_result = run_febio_job(
        febio_config=febio_config,
        simulation_dir=simulation_dir,
        input_path=build_artifacts.input_path,
    )
    result_payload = parse_simulation_outputs(
        build_artifacts=build_artifacts,
        runner_result=runner_result,
    )
    metrics_payload = calculate_simulation_metrics(result_payload, simulation_dir=simulation_dir)
    final_summary_text = _build_final_summary(
        request=request,
        result_payload=result_payload,
        metrics_payload=metrics_payload,
    )
    final_summary_path = simulation_dir / "final_summary.md"
    final_summary_path.write_text(final_summary_text, encoding="utf-8")
    return {
        "status": result_payload.get("status", "unknown"),
        "request": request.to_dict(),
        "runner": runner_result.to_dict(),
        "simulation_result": result_payload,
        "simulation_metrics": metrics_payload,
        "final_summary_path": str(final_summary_path),
        "final_summary": final_summary_text,
        "simulation_dir": str(simulation_dir),
        "febio": febio_config.to_metadata(),
    }


def run_bulk_mechanics_simulation(
    request: BulkMechanicsRequest,
    *,
    simulation_dir: Path,
    febio_config: FEBioConfig,
) -> dict[str, Any]:
    return _run_request(request=request, simulation_dir=simulation_dir, febio_config=febio_config)


def run_single_cell_contraction_simulation(
    request: SingleCellContractionRequest,
    *,
    simulation_dir: Path,
    febio_config: FEBioConfig,
) -> dict[str, Any]:
    return _run_request(request=request, simulation_dir=simulation_dir, febio_config=febio_config)


def run_organoid_spheroid_simulation(
    request: OrganoidSpheroidRequest,
    *,
    simulation_dir: Path,
    febio_config: FEBioConfig,
) -> dict[str, Any]:
    return _run_request(request=request, simulation_dir=simulation_dir, febio_config=febio_config)


def run_simulation_request(
    request: SimulationRequest,
    *,
    simulation_dir: Path,
    febio_config: FEBioConfig,
) -> dict[str, Any]:
    if isinstance(request, BulkMechanicsRequest):
        return run_bulk_mechanics_simulation(request, simulation_dir=simulation_dir, febio_config=febio_config)
    if isinstance(request, SingleCellContractionRequest):
        return run_single_cell_contraction_simulation(request, simulation_dir=simulation_dir, febio_config=febio_config)
    if isinstance(request, OrganoidSpheroidRequest):
        return run_organoid_spheroid_simulation(request, simulation_dir=simulation_dir, febio_config=febio_config)
    raise ValueError(f"Unsupported simulation request type: {type(request)!r}")
