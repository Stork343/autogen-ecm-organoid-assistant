from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..artifacts import read_json, write_json
from .builder import BuildArtifacts
from .runner import RunnerResult

_TIME_PATTERN = re.compile(r"time\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)


def _parse_numeric_table(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "series": [], "times": []}

    series: list[dict[str, Any]] = []
    current_time: float | None = None
    current_rows: list[list[float]] = []
    current_ids: list[int] = []

    def flush() -> None:
        nonlocal current_rows, current_ids, current_time
        if not current_rows:
            return
        series.append(
            {
                "time": current_time,
                "ids": list(current_ids),
                "values": [list(row) for row in current_rows],
            }
        )
        current_rows = []
        current_ids = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        time_match = _TIME_PATTERN.search(stripped)
        if time_match:
            flush()
            current_time = float(time_match.group(1))
            continue
        if stripped.startswith("*") or stripped.lower().startswith("data") or stripped.lower().startswith("step"):
            continue
        parts = [item for item in re.split(r"[,\s]+", stripped) if item]
        if len(parts) < 2:
            continue
        try:
            row_id = int(float(parts[0]))
            row_values = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        current_ids.append(row_id)
        current_rows.append(row_values)
    flush()

    final = series[-1] if series else {"time": current_time, "ids": [], "values": []}
    return {
        "available": True,
        "path": str(path),
        "series": series,
        "times": [row["time"] for row in series if row.get("time") is not None],
        "final": final,
    }


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _warnings_from_logs(runner_result: RunnerResult) -> list[str]:
    warnings: list[str] = []
    combined = "\n".join(
        [
            _read_text_if_exists(runner_result.stdout_path),
            _read_text_if_exists(runner_result.stderr_path),
            _read_text_if_exists(runner_result.log_path),
        ]
    ).lower()
    for token, message in [
        ("negative jacobian", "FEBio reported negative Jacobians."),
        ("inverted", "FEBio reported inverted or highly distorted elements."),
        ("error termination", "FEBio reported error termination."),
    ]:
        if token in combined and message not in warnings:
            warnings.append(message)
    if not runner_result.log_path.exists():
        warnings.append("FEBio logfile is missing.")
    if not runner_result.xplt_path.exists():
        warnings.append("FEBio plotfile is missing.")
    if not runner_result.dmp_path.exists():
        warnings.append("FEBio restart dump is missing.")
    return warnings


def parse_simulation_outputs(
    *,
    build_artifacts: BuildArtifacts,
    runner_result: RunnerResult,
) -> dict[str, Any]:
    """Parse FEBio log/data artifacts into a uniform JSON-ready payload."""

    metadata = read_json(build_artifacts.metadata_path)
    simulation_dir = build_artifacts.simulation_dir
    displacement = _parse_numeric_table(simulation_dir / "node_displacement.log")
    principal_stress = _parse_numeric_table(simulation_dir / "element_principal_stress.log")
    top_reaction = _parse_numeric_table(simulation_dir / "top_reaction.log")
    logfile_text = _read_text_if_exists(runner_result.log_path)
    normalized_logfile = re.sub(r"\s+", "", logfile_text.lower())
    solver_converged = "normaltermination" in normalized_logfile
    extracted_fields = {
        "node_displacement": displacement,
        "element_principal_stress": principal_stress,
        "top_reaction": top_reaction,
        "solver_converged": solver_converged,
        "logfile_exists": runner_result.log_path.exists(),
    }
    result = {
        "status": runner_result.status,
        "scenario": build_artifacts.request.scenario,
        "request": build_artifacts.request.to_dict(),
        "builder_metadata_path": str(build_artifacts.metadata_path),
        "raw_output_paths": {
            "input_feb": str(build_artifacts.input_path),
            "logfile": str(runner_result.log_path),
            "stdout": str(runner_result.stdout_path),
            "stderr": str(runner_result.stderr_path),
            "plotfile": str(runner_result.xplt_path),
            "restart_dump": str(runner_result.dmp_path),
        },
        "runner": runner_result.to_dict(),
        "mesh_metadata": metadata.get("mesh", {}),
        "node_sets": metadata.get("node_sets", {}),
        "element_sets": metadata.get("element_sets", {}),
        "extracted_fields": extracted_fields,
        "warnings": _warnings_from_logs(runner_result),
        "error_message": runner_result.error_message,
    }
    write_json(simulation_dir / "simulation_result.json", result)
    return result
