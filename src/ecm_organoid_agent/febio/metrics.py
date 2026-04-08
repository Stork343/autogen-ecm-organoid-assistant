from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import write_json


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _final_table_rows(payload: Mapping[str, Any]) -> tuple[list[int], list[list[float]]]:
    final = payload.get("final", {}) if isinstance(payload, Mapping) else {}
    ids = final.get("ids", []) if isinstance(final, Mapping) else []
    values = final.get("values", []) if isinstance(final, Mapping) else []
    return (
        [int(item) for item in ids],
        [[float(number) for number in row] for row in values],
    )


def _nodes_map(result_payload: Mapping[str, Any]) -> dict[int, tuple[float, float, float]]:
    mesh = result_payload.get("mesh_metadata", {}) if isinstance(result_payload, Mapping) else {}
    nodes = mesh.get("nodes", {}) if isinstance(mesh, Mapping) else {}
    return {
        int(node_id): (float(coords[0]), float(coords[1]), float(coords[2]))
        for node_id, coords in nodes.items()
    }


def _elements_map(result_payload: Mapping[str, Any]) -> dict[int, tuple[int, int, int, int, int, int, int, int]]:
    mesh = result_payload.get("mesh_metadata", {}) if isinstance(result_payload, Mapping) else {}
    elements = mesh.get("elements", {}) if isinstance(mesh, Mapping) else {}
    return {
        int(element_id): tuple(int(node_id) for node_id in connectivity)
        for element_id, connectivity in elements.items()
    }


def _element_centroids_map(result_payload: Mapping[str, Any]) -> dict[int, tuple[float, float, float]]:
    mesh = result_payload.get("mesh_metadata", {}) if isinstance(result_payload, Mapping) else {}
    centroids = mesh.get("element_centroids", {}) if isinstance(mesh, Mapping) else {}
    return {
        int(element_id): (float(coords[0]), float(coords[1]), float(coords[2]))
        for element_id, coords in centroids.items()
    }


def _element_id_set(result_payload: Mapping[str, Any], set_name: str) -> set[int]:
    sets = result_payload.get("element_sets", {}) if isinstance(result_payload, Mapping) else {}
    raw_ids = sets.get(set_name, []) if isinstance(sets, Mapping) else []
    return {int(item) for item in raw_ids}


def _node_id_set(result_payload: Mapping[str, Any], set_name: str) -> set[int]:
    sets = result_payload.get("node_sets", {}) if isinstance(result_payload, Mapping) else {}
    raw_ids = sets.get(set_name, []) if isinstance(sets, Mapping) else []
    return {int(item) for item in raw_ids}


def _principal_stress_peak(
    result_payload: Mapping[str, Any],
    *,
    element_set_name: str | None = None,
) -> float | None:
    stress_payload = result_payload.get("extracted_fields", {}).get("element_principal_stress", {})
    element_ids, values = _final_table_rows(stress_payload if isinstance(stress_payload, Mapping) else {})
    if not values:
        return None
    allowed = _element_id_set(result_payload, element_set_name) if element_set_name else set(element_ids)
    maxima: list[float] = []
    for element_id, row in zip(element_ids, values):
        if element_set_name and element_id not in allowed:
            continue
        maxima.append(max(abs(item) for item in row))
    return max(maxima) if maxima else None


def _stress_distribution_summary(
    result_payload: Mapping[str, Any],
    *,
    element_set_name: str,
) -> dict[str, float] | None:
    stress_payload = result_payload.get("extracted_fields", {}).get("element_principal_stress", {})
    element_ids, values = _final_table_rows(stress_payload if isinstance(stress_payload, Mapping) else {})
    if not values:
        return None
    allowed = _element_id_set(result_payload, element_set_name) or set(element_ids)
    magnitudes = [max(abs(item) for item in row) for element_id, row in zip(element_ids, values) if element_id in allowed]
    if not magnitudes:
        return None
    sorted_values = sorted(magnitudes)
    index_95 = min(len(sorted_values) - 1, max(0, int(round(0.95 * (len(sorted_values) - 1)))))
    return {
        "mean": sum(magnitudes) / len(magnitudes),
        "median": sorted_values[len(sorted_values) // 2],
        "p95": sorted_values[index_95],
        "max": max(magnitudes),
        "min": min(magnitudes),
    }


def _final_displacement_map(result_payload: Mapping[str, Any]) -> dict[int, tuple[float, float, float]]:
    displacement_payload = result_payload.get("extracted_fields", {}).get("node_displacement", {})
    node_ids, rows = _final_table_rows(displacement_payload if isinstance(displacement_payload, Mapping) else {})
    return {
        node_id: (row[0], row[1], row[2])
        for node_id, row in zip(node_ids, rows)
        if len(row) >= 3
    }


def _engineering_stiffness(result_payload: Mapping[str, Any]) -> float | None:
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    if request.get("scenario") != "bulk_mechanics":
        return None
    sample_dimensions = request.get("sample_dimensions", [1.0, 1.0, 1.0])
    if not isinstance(sample_dimensions, list | tuple) or len(sample_dimensions) != 3:
        return None
    width, depth, height = [float(item) for item in sample_dimensions]
    prescribed_displacement = abs(float(request.get("prescribed_displacement", 0.0)))
    if prescribed_displacement <= 0 or height <= 0:
        return None
    reaction_payload = result_payload.get("extracted_fields", {}).get("top_reaction", {})
    _, rows = _final_table_rows(reaction_payload if isinstance(reaction_payload, Mapping) else {})
    if not rows:
        return None
    total_reaction_z = sum(abs(row[2]) for row in rows if len(row) >= 3)
    engineering_stress = total_reaction_z / max(width * depth, 1e-12)
    engineering_strain = prescribed_displacement / height
    if engineering_strain <= 0:
        return None
    return engineering_stress / engineering_strain


def _displacement_field_summary(result_payload: Mapping[str, Any]) -> dict[str, float] | None:
    displacement_map = _final_displacement_map(result_payload)
    if not displacement_map:
        return None
    magnitudes = [math.sqrt(sum(value * value for value in displacement)) for displacement in displacement_map.values()]
    sorted_values = sorted(magnitudes)
    index_95 = min(len(sorted_values) - 1, max(0, int(round(0.95 * (len(sorted_values) - 1)))))
    return {
        "peak": max(magnitudes),
        "mean": sum(magnitudes) / len(magnitudes),
        "median": sorted_values[len(sorted_values) // 2],
        "p95": sorted_values[index_95],
    }


def _first_value_above_threshold(distances: list[float], values: list[float], threshold: float) -> float | None:
    for distance, value in zip(distances, values):
        if value <= threshold:
            return distance
    return None


def _displacement_decay_length(result_payload: Mapping[str, Any]) -> float | None:
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    if request.get("scenario") not in {"single_cell_contraction", "organoid_spheroid"}:
        return None
    nodes = _nodes_map(result_payload)
    displacements = _final_displacement_map(result_payload)
    if not displacements:
        return None
    radius = (
        float(request.get("cell_radius"))
        if request.get("scenario") == "single_cell_contraction"
        else float(request.get("organoid_radius"))
    )
    outer_boundary = _node_id_set(result_payload, "outer_boundary_nodes")
    matrix_distances: list[float] = []
    matrix_values: list[float] = []
    for node_id, disp in displacements.items():
        if node_id in outer_boundary:
            continue
        coords = nodes.get(node_id)
        if coords is None:
            continue
        distance = math.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
        if distance <= radius:
            continue
        magnitude = math.sqrt(disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2)
        matrix_distances.append(distance - radius)
        matrix_values.append(magnitude)
    if not matrix_values:
        return None
    pairs = sorted(zip(matrix_distances, matrix_values), key=lambda item: item[0])
    sorted_distances = [item[0] for item in pairs]
    sorted_values = [item[1] for item in pairs]
    peak = max(sorted_values)
    if peak <= 0:
        return None
    threshold = peak / math.e
    threshold_distance = _first_value_above_threshold(sorted_distances, sorted_values, threshold)
    if threshold_distance is not None:
        return threshold_distance

    fit_x: list[float] = []
    fit_y: list[float] = []
    for distance, value in pairs:
        if value <= 0:
            continue
        fit_x.append(distance)
        fit_y.append(math.log(value / peak))
    if len(fit_x) < 2:
        return max(sorted_distances) if sorted_distances else None
    mean_x = sum(fit_x) / len(fit_x)
    mean_y = sum(fit_y) / len(fit_y)
    denominator = sum((x - mean_x) ** 2 for x in fit_x)
    if denominator <= 1e-12:
        return max(sorted_distances) if sorted_distances else None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(fit_x, fit_y)) / denominator
    if slope >= 0:
        return max(sorted_distances) if sorted_distances else None
    return -1.0 / slope


def _stress_propagation_distance(result_payload: Mapping[str, Any]) -> float | None:
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    if request.get("scenario") not in {"single_cell_contraction", "organoid_spheroid"}:
        return None
    radius = (
        float(request.get("cell_radius"))
        if request.get("scenario") == "single_cell_contraction"
        else float(request.get("organoid_radius"))
    )
    centroids = _element_centroids_map(result_payload)
    stress_payload = result_payload.get("extracted_fields", {}).get("element_principal_stress", {})
    element_ids, rows = _final_table_rows(stress_payload if isinstance(stress_payload, Mapping) else {})
    if not rows:
        return None
    matrix_ids = _element_id_set(result_payload, "matrix_domain") or set(element_ids)
    pairs: list[tuple[float, float]] = []
    for element_id, row in zip(element_ids, rows):
        if element_id not in matrix_ids:
            continue
        centroid = centroids.get(element_id)
        if centroid is None:
            continue
        distance = math.sqrt(centroid[0] ** 2 + centroid[1] ** 2 + centroid[2] ** 2) - radius
        if distance < 0:
            continue
        magnitude = max(abs(item) for item in row)
        pairs.append((distance, magnitude))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    distances = [item[0] for item in pairs]
    values = [item[1] for item in pairs]
    peak = max(values)
    if peak <= 0:
        return None
    threshold = peak / math.e
    threshold_distance = _first_value_above_threshold(distances, values, threshold)
    if threshold_distance is not None:
        return threshold_distance
    return max(distances)


def _strain_tensor_frobenius(
    *,
    coordinates: list[tuple[float, float, float]],
    displacements: list[tuple[float, float, float]],
) -> float | None:
    dnds = [
        (-0.125, -0.125, -0.125),
        (0.125, -0.125, -0.125),
        (0.125, 0.125, -0.125),
        (-0.125, 0.125, -0.125),
        (-0.125, -0.125, 0.125),
        (0.125, -0.125, 0.125),
        (0.125, 0.125, 0.125),
        (-0.125, 0.125, 0.125),
    ]
    jacobian = [[0.0, 0.0, 0.0] for _ in range(3)]
    for (dndxi, dndeta, dndzeta), coords in zip(dnds, coordinates):
        jacobian[0][0] += dndxi * coords[0]
        jacobian[0][1] += dndxi * coords[1]
        jacobian[0][2] += dndxi * coords[2]
        jacobian[1][0] += dndeta * coords[0]
        jacobian[1][1] += dndeta * coords[1]
        jacobian[1][2] += dndeta * coords[2]
        jacobian[2][0] += dndzeta * coords[0]
        jacobian[2][1] += dndzeta * coords[1]
        jacobian[2][2] += dndzeta * coords[2]

    det = (
        jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0])
    )
    if abs(det) <= 1e-12:
        return None

    inverse = [
        [
            (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) / det,
            (jacobian[0][2] * jacobian[2][1] - jacobian[0][1] * jacobian[2][2]) / det,
            (jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1]) / det,
        ],
        [
            (jacobian[1][2] * jacobian[2][0] - jacobian[1][0] * jacobian[2][2]) / det,
            (jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0]) / det,
            (jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2]) / det,
        ],
        [
            (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]) / det,
            (jacobian[0][1] * jacobian[2][0] - jacobian[0][0] * jacobian[2][1]) / det,
            (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]) / det,
        ],
    ]

    gradients = []
    for dndxi, dndeta, dndzeta in dnds:
        dndx = inverse[0][0] * dndxi + inverse[0][1] * dndeta + inverse[0][2] * dndzeta
        dndy = inverse[1][0] * dndxi + inverse[1][1] * dndeta + inverse[1][2] * dndzeta
        dndz = inverse[2][0] * dndxi + inverse[2][1] * dndeta + inverse[2][2] * dndzeta
        gradients.append((dndx, dndy, dndz))

    grad_u = [[0.0, 0.0, 0.0] for _ in range(3)]
    for displacement, gradients_row in zip(displacements, gradients):
        for i in range(3):
            grad_u[i][0] += displacement[i] * gradients_row[0]
            grad_u[i][1] += displacement[i] * gradients_row[1]
            grad_u[i][2] += displacement[i] * gradients_row[2]

    strain = [
        [
            0.5 * (grad_u[0][0] + grad_u[0][0]),
            0.5 * (grad_u[0][1] + grad_u[1][0]),
            0.5 * (grad_u[0][2] + grad_u[2][0]),
        ],
        [
            0.5 * (grad_u[1][0] + grad_u[0][1]),
            0.5 * (grad_u[1][1] + grad_u[1][1]),
            0.5 * (grad_u[1][2] + grad_u[2][1]),
        ],
        [
            0.5 * (grad_u[2][0] + grad_u[0][2]),
            0.5 * (grad_u[2][1] + grad_u[1][2]),
            0.5 * (grad_u[2][2] + grad_u[2][2]),
        ],
    ]
    return math.sqrt(sum(value * value for row in strain for value in row))


def _strain_heterogeneity(result_payload: Mapping[str, Any]) -> float | None:
    nodes = _nodes_map(result_payload)
    elements = _elements_map(result_payload)
    displacements = _final_displacement_map(result_payload)
    if not nodes or not elements or not displacements:
        return None
    matrix_ids = _element_id_set(result_payload, "matrix_domain") or set(elements.keys())
    magnitudes: list[float] = []
    for element_id, connectivity in elements.items():
        if element_id not in matrix_ids:
            continue
        coordinates = [nodes[node_id] for node_id in connectivity if node_id in nodes]
        element_displacements = [displacements.get(node_id, (0.0, 0.0, 0.0)) for node_id in connectivity]
        if len(coordinates) != 8:
            continue
        magnitude = _strain_tensor_frobenius(
            coordinates=coordinates,
            displacements=element_displacements,
        )
        if magnitude is not None:
            magnitudes.append(magnitude)
    if not magnitudes:
        return None
    mean_value = sum(magnitudes) / len(magnitudes)
    if abs(mean_value) <= 1e-12:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in magnitudes) / len(magnitudes)
    return math.sqrt(variance) / abs(mean_value)


def _target_mismatch_score(result_payload: Mapping[str, Any], effective_stiffness: float | None) -> float | None:
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    target = _as_float(request.get("target_stiffness"))
    if target is None or target <= 0:
        return None
    if effective_stiffness is not None:
        return abs(effective_stiffness - target) / target
    matrix_modulus = _as_float(request.get("matrix_youngs_modulus"))
    if matrix_modulus is None:
        return None
    return abs(matrix_modulus - target) / target


def _compression_tension_region_summary(
    compression_fraction: float | None,
    tension_fraction: float | None,
) -> dict[str, float] | None:
    if compression_fraction is None or tension_fraction is None:
        return None
    mixed_fraction = max(0.0, 1.0 - max(compression_fraction, tension_fraction))
    return {
        "compression_fraction": compression_fraction,
        "tension_fraction": tension_fraction,
        "mixed_fraction": mixed_fraction,
    }


def _organoid_suitability_components(
    *,
    result_payload: Mapping[str, Any],
    interface_deformation: float | None,
    peak_stress: float | None,
    stress_propagation_distance: float | None,
    strain_heterogeneity: float | None,
    target_mismatch_score: float | None,
    compression_fraction: float | None,
    tension_fraction: float | None,
) -> dict[str, float] | None:
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    if request.get("scenario") != "organoid_spheroid":
        return None
    target_interface = _as_float(request.get("target_interface_deformation")) or abs(_as_float(request.get("organoid_radial_displacement")) or 0.0)
    matrix_extent = _as_float(request.get("matrix_extent")) or 1.0
    interface_penalty = (
        abs((interface_deformation or 0.0) - target_interface) / max(target_interface, 1e-9)
        if target_interface > 0
        else 0.0
    )
    stress_penalty = (peak_stress or 0.0) / max(_as_float(request.get("matrix_youngs_modulus")) or 1.0, 1e-9)
    heterogeneity_penalty = strain_heterogeneity or 0.0
    mismatch_penalty = target_mismatch_score or 0.0
    balance_penalty = 0.0
    if compression_fraction is not None and tension_fraction is not None:
        balance_penalty = abs(compression_fraction - tension_fraction)
    propagation_bonus = min((stress_propagation_distance or 0.0) / max(matrix_extent, 1e-9), 1.0)
    suitability_score = max(
        0.0,
        1.0
        - 0.30 * interface_penalty
        - 0.20 * stress_penalty
        - 0.20 * heterogeneity_penalty
        - 0.20 * mismatch_penalty
        - 0.10 * balance_penalty
        + 0.10 * propagation_bonus,
    )
    return {
        "interface_penalty": interface_penalty,
        "stress_penalty": stress_penalty,
        "heterogeneity_penalty": heterogeneity_penalty,
        "mismatch_penalty": mismatch_penalty,
        "balance_penalty": balance_penalty,
        "propagation_bonus": propagation_bonus,
        "suitability_score": suitability_score,
    }


def compare_simulation_candidates_payloads(
    candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        metrics = candidate.get("simulation_metrics", candidate)
        status = str(candidate.get("status", metrics.get("status", "unknown")))
        feasibility = metrics.get("feasibility_flags", {}) if isinstance(metrics, Mapping) else {}
        feasible = bool(feasibility.get("solver_converged", False)) and not feasibility.get("negative_jacobian_warning", False)
        mismatch = _as_float(metrics.get("target_mismatch_score"))
        strain_heterogeneity = _as_float(metrics.get("strain_heterogeneity")) or 0.0
        peak_stress = abs(_as_float(metrics.get("peak_stress")) or 0.0)
        suitability_score = _as_float(
            metrics.get("candidate_suitability_score_components", {}).get("suitability_score")
            if isinstance(metrics.get("candidate_suitability_score_components"), Mapping)
            else None
        )
        score = (mismatch if mismatch is not None else 1.0) + 0.15 * strain_heterogeneity + 0.01 * peak_stress
        if suitability_score is not None:
            score -= 0.25 * suitability_score
        if status != "succeeded":
            score += 10.0
        if not feasible:
            score += 5.0
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id", f"candidate_{index:02d}"),
                "status": status,
                "comparison_score": round(score, 6),
                "feasible": feasible,
                "target_mismatch_score": mismatch,
                "peak_stress": _as_float(metrics.get("peak_stress")),
                "strain_heterogeneity": _as_float(metrics.get("strain_heterogeneity")),
                "stress_propagation_distance": _as_float(metrics.get("stress_propagation_distance")),
                "candidate_suitability_score": suitability_score,
                "simulation_metrics": metrics,
            }
        )
    rows.sort(key=lambda item: float(item["comparison_score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "candidate_count": len(rows),
        "ranking": rows,
        "best_candidate": rows[0] if rows else {},
    }


def calculate_simulation_metrics(result_payload: Mapping[str, Any], *, simulation_dir: Path) -> dict[str, Any]:
    """Derive decision-oriented metrics from parsed FEBio outputs."""

    effective_stiffness = _engineering_stiffness(result_payload)
    peak_stress = _principal_stress_peak(
        result_payload,
        element_set_name="matrix_domain" if result_payload.get("request", {}).get("scenario") != "bulk_mechanics" else None,
    )
    displacement_decay_length = _displacement_decay_length(result_payload)
    stress_propagation_distance = _stress_propagation_distance(result_payload)
    strain_heterogeneity = _strain_heterogeneity(result_payload)
    target_mismatch_score = _target_mismatch_score(result_payload, effective_stiffness)
    request = result_payload.get("request", {}) if isinstance(result_payload, Mapping) else {}
    solver_converged = bool(result_payload.get("extracted_fields", {}).get("solver_converged", False))
    warnings = result_payload.get("warnings", []) if isinstance(result_payload, Mapping) else []
    displacement_field_summary = _displacement_field_summary(result_payload)
    matrix_stress_distribution_summary = _stress_distribution_summary(result_payload, element_set_name="matrix_domain")
    feasibility_flags = {
        "solver_converged": solver_converged,
        "has_output_data": bool(_final_displacement_map(result_payload)),
        "negative_jacobian_warning": any("negative jacobian" in warning.lower() for warning in warnings),
        "inverted_element_warning": any("inverted" in warning.lower() for warning in warnings),
        "target_defined": request.get("target_stiffness") is not None,
    }

    stress_payload = result_payload.get("extracted_fields", {}).get("element_principal_stress", {})
    element_ids, rows = _final_table_rows(stress_payload if isinstance(stress_payload, Mapping) else {})
    compression_fraction = None
    tension_fraction = None
    if rows:
        matrix_ids = _element_id_set(result_payload, "matrix_domain") or set(element_ids)
        filtered = [row for element_id, row in zip(element_ids, rows) if element_id in matrix_ids]
        if filtered:
            compression_fraction = sum(1 for row in filtered if min(row) < 0) / len(filtered)
            tension_fraction = sum(1 for row in filtered if max(row) > 0) / len(filtered)
    compression_tension_region_summary = _compression_tension_region_summary(compression_fraction, tension_fraction)

    displacement_map = _final_displacement_map(result_payload)
    interface_ids = _node_id_set(result_payload, "interface_nodes")
    interface_deformation = None
    if interface_ids and displacement_map:
        magnitudes = [
            math.sqrt(sum(value * value for value in displacement_map[node_id]))
            for node_id in interface_ids
            if node_id in displacement_map
        ]
        if magnitudes:
            interface_deformation = sum(magnitudes) / len(magnitudes)

    candidate_suitability_score_components = _organoid_suitability_components(
        result_payload=result_payload,
        interface_deformation=interface_deformation,
        peak_stress=peak_stress,
        stress_propagation_distance=stress_propagation_distance,
        strain_heterogeneity=strain_heterogeneity,
        target_mismatch_score=target_mismatch_score,
        compression_fraction=compression_fraction,
        tension_fraction=tension_fraction,
    )

    metrics = {
        "scenario": request.get("scenario", "unknown"),
        "status": result_payload.get("status", "unknown"),
        "effective_stiffness": effective_stiffness,
        "peak_stress": peak_stress,
        "peak_matrix_stress": peak_stress,
        "displacement_decay_length": displacement_decay_length,
        "stress_propagation_distance": stress_propagation_distance,
        "strain_heterogeneity": strain_heterogeneity,
        "target_mismatch_score": target_mismatch_score,
        "feasibility_flags": feasibility_flags,
        "displacement_field_summary": displacement_field_summary,
        "matrix_stress_distribution_summary": matrix_stress_distribution_summary,
        "interface_deformation": interface_deformation,
        "compression_fraction": compression_fraction,
        "tension_fraction": tension_fraction,
        "compression_tension_region_summary": compression_tension_region_summary,
        "candidate_suitability_score_components": candidate_suitability_score_components,
    }
    write_json(simulation_dir / "simulation_metrics.json", metrics)
    return metrics
