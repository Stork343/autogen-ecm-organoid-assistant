from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import SimulationRequest, simulation_request_from_dict

MAPPING_VERSION = "phase1-template-mapping-v2"


@dataclass(frozen=True)
class CandidateSimulationMappingOptions:
    scenario: str
    target_stiffness: float | None = None
    simulation_request_overrides: dict[str, Any] | None = None
    cell_contractility: float | None = None
    organoid_radius: float | None = None
    matrix_youngs_modulus: float | None = None
    matrix_poisson_ratio: float | None = None


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _base_metadata(candidate: Mapping[str, Any], *, mapping_notes: list[str]) -> dict[str, Any]:
    params = candidate.get("parameters", {}) if isinstance(candidate.get("parameters"), Mapping) else {}
    features = candidate.get("features", {}) if isinstance(candidate.get("features"), Mapping) else {}
    return {
        "mapping_version": MAPPING_VERSION,
        "candidate_rank": candidate.get("rank", "NR"),
        "candidate_score": candidate.get("score", "NR"),
        "candidate_parameters": dict(params),
        "candidate_features": dict(features),
        "mapping_notes": mapping_notes,
    }


def candidate_to_simulation_request(
    candidate: Mapping[str, Any],
    *,
    options: CandidateSimulationMappingOptions,
) -> SimulationRequest:
    """Map a design candidate into a constrained FEBio simulation request."""

    scenario = options.scenario.strip().lower()
    params = candidate.get("parameters", {}) if isinstance(candidate.get("parameters"), Mapping) else {}
    features = candidate.get("features", {}) if isinstance(candidate.get("features"), Mapping) else {}
    payload = dict(options.simulation_request_overrides or {})
    payload["scenario"] = scenario

    domain_size = _as_float(params.get("domain_size"), 1.0)
    matrix_extent = _clamp(domain_size if domain_size > 0 else 1.0, 0.6, 2.5)
    inferred_stiffness = (
        options.matrix_youngs_modulus
        if options.matrix_youngs_modulus is not None
        else _as_float(features.get("stiffness_mean"), _as_float(options.target_stiffness, 8.0))
    )
    stress_propagation = _clamp(_as_float(features.get("stress_propagation"), 0.5), 0.0, 2.0)
    anisotropy = _clamp(_as_float(features.get("anisotropy"), 0.1), 0.0, 1.0)
    connectivity = _clamp(_as_float(features.get("connectivity"), 0.95), 0.0, 1.0)

    mapping_notes = [
        f"matrix_extent mapped from candidate domain_size={domain_size:.4f}",
        f"matrix_youngs_modulus inferred from candidate stiffness_mean={_as_float(features.get('stiffness_mean'), inferred_stiffness):.4f}",
    ]
    payload.setdefault("title", f"Design candidate {candidate.get('rank', 'NR')} FEBio verification")
    payload.setdefault("matrix_extent", matrix_extent)
    payload.setdefault("matrix_youngs_modulus", inferred_stiffness)
    payload.setdefault("matrix_poisson_ratio", float(options.matrix_poisson_ratio if options.matrix_poisson_ratio is not None else 0.3))
    payload.setdefault("mesh_resolution", [4, 4, 4])
    if options.target_stiffness is not None:
        payload.setdefault("target_stiffness", float(options.target_stiffness))
    payload.setdefault("metadata", _base_metadata(candidate, mapping_notes=mapping_notes))

    if scenario == "bulk_mechanics":
        prescribed_strain = 0.05 + 0.03 * min(connectivity, 1.0)
        prescribed_displacement = -matrix_extent * prescribed_strain
        payload.setdefault("sample_dimensions", [matrix_extent, matrix_extent, matrix_extent])
        payload.setdefault("prescribed_displacement", prescribed_displacement)
        payload.setdefault("loading_mode", "uniaxial_compression")
        payload["metadata"]["mapping_notes"].append(
            f"bulk compression strain set to {prescribed_strain:.4f} from connectivity={connectivity:.4f}"
        )
    elif scenario == "single_cell_contraction":
        cell_radius = _clamp(0.14 * matrix_extent + 0.03 * anisotropy, 0.08, 0.24 * matrix_extent)
        contractility = (
            options.cell_contractility
            if options.cell_contractility is not None
            else _clamp(0.012 + 0.02 * min(stress_propagation, 1.5), 0.008, 0.05)
        )
        payload.setdefault("cell_radius", cell_radius)
        payload.setdefault("cell_contractility", contractility)
        payload.setdefault("cell_youngs_modulus", max(inferred_stiffness * 2.0, 12.0))
        payload.setdefault("loading_mode", "radial_contractility")
        payload.setdefault("target_stress_propagation_distance", 0.25 * matrix_extent + 0.1 * stress_propagation)
        payload.setdefault("target_strain_heterogeneity", 0.35)
        payload["metadata"]["mapping_notes"].extend(
            [
                f"cell_radius inferred from matrix_extent={matrix_extent:.4f} and anisotropy={anisotropy:.4f}",
                f"cell_contractility inferred from stress_propagation={stress_propagation:.4f}",
            ]
        )
    elif scenario == "organoid_spheroid":
        inferred_radius = _clamp(0.18 * matrix_extent + 0.04 * (1.0 - connectivity), 0.1, 0.28 * matrix_extent)
        radial_displacement = _clamp(0.02 + 0.015 * min(stress_propagation, 1.0), 0.01, 0.06)
        payload.setdefault("organoid_radius", float(options.organoid_radius if options.organoid_radius is not None else inferred_radius))
        payload.setdefault("organoid_radial_displacement", radial_displacement)
        payload.setdefault("organoid_youngs_modulus", max(inferred_stiffness * 1.5, 10.0))
        payload.setdefault("loading_mode", "radial_displacement")
        payload.setdefault("target_interface_deformation", radial_displacement)
        payload.setdefault("target_stress_propagation_distance", 0.20 * matrix_extent + 0.08 * stress_propagation)
        payload.setdefault("target_candidate_suitability", 0.65)
        payload["metadata"]["mapping_notes"].extend(
            [
                f"organoid_radius inferred from matrix_extent={matrix_extent:.4f} and connectivity={connectivity:.4f}",
                f"organoid_radial_displacement inferred from stress_propagation={stress_propagation:.4f}",
            ]
        )
    else:
        raise ValueError(f"Unsupported mapped simulation scenario: {scenario}")

    return simulation_request_from_dict(payload)


def design_payload_to_simulation_requests(
    design_payload: Mapping[str, Any],
    *,
    top_k: int,
    options: CandidateSimulationMappingOptions,
) -> list[SimulationRequest]:
    candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, Mapping) else []
    requests: list[SimulationRequest] = []
    for candidate in candidates[: max(1, int(top_k))]:
        if not isinstance(candidate, Mapping):
            continue
        requests.append(candidate_to_simulation_request(candidate, options=options))
    return requests


def candidate_requests_summary(requests: Sequence[SimulationRequest]) -> list[dict[str, Any]]:
    return [request.to_dict() for request in requests]
