from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _as_float(value: object, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field_name}` must be a number.") from exc


def _as_int(value: object, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field_name}` must be an integer.") from exc
    return parsed


def _coerce_tuple(
    value: object,
    *,
    field_name: str,
    length: int,
    cast: str,
) -> tuple[float, ...] | tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"`{field_name}` must be a sequence of length {length}.")
    if len(value) != length:
        raise ValueError(f"`{field_name}` must contain exactly {length} values.")
    if cast == "float":
        return tuple(_as_float(item, field_name=field_name) for item in value)
    if cast == "int":
        return tuple(_as_int(item, field_name=field_name) for item in value)
    raise ValueError(f"Unsupported cast mode for `{field_name}`: {cast}")


def _validate_positive(value: float, *, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"`{field_name}` must be > 0.")


def _validate_poisson_ratio(value: float, *, field_name: str) -> None:
    if value <= -0.99 or value >= 0.49:
        raise ValueError(f"`{field_name}` must be between -0.99 and 0.49 for this phase-1 solid setup.")


def _validate_mesh_resolution(value: tuple[int, int, int], *, field_name: str) -> None:
    if any(item < 1 or item > 12 for item in value):
        raise ValueError(f"`{field_name}` entries must be between 1 and 12.")


@dataclass(frozen=True)
class BaseSimulationRequest:
    """Base schema for safe FEBio-backed simulation requests."""

    title: str = "ECM FEBio simulation"
    matrix_youngs_modulus: float = 8.0
    matrix_poisson_ratio: float = 0.3
    matrix_extent: float = 1.0
    mesh_resolution: tuple[int, int, int] = (4, 4, 4)
    time_steps: int = 10
    step_size: float = 0.1
    target_stiffness: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario(self) -> str:
        raise NotImplementedError

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("`title` cannot be empty.")
        _validate_positive(self.matrix_youngs_modulus, field_name="matrix_youngs_modulus")
        _validate_poisson_ratio(self.matrix_poisson_ratio, field_name="matrix_poisson_ratio")
        _validate_positive(self.matrix_extent, field_name="matrix_extent")
        _validate_mesh_resolution(self.mesh_resolution, field_name="mesh_resolution")
        if self.time_steps < 1 or self.time_steps > 200:
            raise ValueError("`time_steps` must be between 1 and 200.")
        _validate_positive(self.step_size, field_name="step_size")
        if self.target_stiffness is not None:
            _validate_positive(self.target_stiffness, field_name="target_stiffness")
        if not isinstance(self.metadata, dict):
            raise ValueError("`metadata` must be a JSON-like object.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario"] = self.scenario
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class BulkMechanicsRequest(BaseSimulationRequest):
    """Schema for bulk matrix compression / prescribed displacement verification."""

    sample_dimensions: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prescribed_displacement: float = -0.1
    loading_mode: str = "uniaxial_compression"

    @property
    def scenario(self) -> str:
        return "bulk_mechanics"

    def __post_init__(self) -> None:
        self.validate()
        dims = self.sample_dimensions
        if len(dims) != 3:
            raise ValueError("`sample_dimensions` must contain 3 values.")
        for index, value in enumerate(dims):
            _validate_positive(float(value), field_name=f"sample_dimensions[{index}]")
        if self.prescribed_displacement == 0:
            raise ValueError("`prescribed_displacement` cannot be zero.")
        if self.loading_mode not in {"uniaxial_compression", "prescribed_displacement"}:
            raise ValueError("`loading_mode` must be `uniaxial_compression` or `prescribed_displacement`.")


@dataclass(frozen=True)
class SingleCellContractionRequest(BaseSimulationRequest):
    """Schema for a contracting inclusion inside an ECM block."""

    cell_radius: float = 0.12
    cell_contractility: float = 0.02
    cell_youngs_modulus: float = 20.0
    cell_poisson_ratio: float = 0.3
    loading_mode: str = "radial_contractility"
    target_stress_propagation_distance: float | None = None
    target_strain_heterogeneity: float | None = None

    @property
    def scenario(self) -> str:
        return "single_cell_contraction"

    def __post_init__(self) -> None:
        self.validate()
        _validate_positive(self.cell_radius, field_name="cell_radius")
        _validate_positive(self.cell_contractility, field_name="cell_contractility")
        _validate_positive(self.cell_youngs_modulus, field_name="cell_youngs_modulus")
        _validate_poisson_ratio(self.cell_poisson_ratio, field_name="cell_poisson_ratio")
        if self.loading_mode != "radial_contractility":
            raise ValueError("`loading_mode` must be `radial_contractility` for phase-1 single-cell contraction.")
        if self.target_stress_propagation_distance is not None:
            _validate_positive(self.target_stress_propagation_distance, field_name="target_stress_propagation_distance")
        if self.target_strain_heterogeneity is not None and self.target_strain_heterogeneity < 0:
            raise ValueError("`target_strain_heterogeneity` must be >= 0.")
        if self.cell_radius >= 0.45 * self.matrix_extent:
            raise ValueError("`cell_radius` must stay well inside the surrounding matrix block.")


@dataclass(frozen=True)
class OrganoidSpheroidRequest(BaseSimulationRequest):
    """Schema for a spheroid inclusion that expands or contracts inside ECM."""

    organoid_radius: float = 0.18
    organoid_radial_displacement: float = 0.03
    organoid_youngs_modulus: float = 15.0
    organoid_poisson_ratio: float = 0.35
    loading_mode: str = "radial_displacement"
    target_interface_deformation: float | None = None
    target_stress_propagation_distance: float | None = None
    target_candidate_suitability: float | None = None

    @property
    def scenario(self) -> str:
        return "organoid_spheroid"

    def __post_init__(self) -> None:
        self.validate()
        _validate_positive(self.organoid_radius, field_name="organoid_radius")
        if self.organoid_radial_displacement == 0:
            raise ValueError("`organoid_radial_displacement` cannot be zero.")
        _validate_positive(self.organoid_youngs_modulus, field_name="organoid_youngs_modulus")
        _validate_poisson_ratio(self.organoid_poisson_ratio, field_name="organoid_poisson_ratio")
        if self.loading_mode != "radial_displacement":
            raise ValueError("`loading_mode` must be `radial_displacement` for phase-1 organoid spheroid.")
        if self.target_interface_deformation is not None:
            _validate_positive(self.target_interface_deformation, field_name="target_interface_deformation")
        if self.target_stress_propagation_distance is not None:
            _validate_positive(self.target_stress_propagation_distance, field_name="target_stress_propagation_distance")
        if self.target_candidate_suitability is not None and not (0 <= self.target_candidate_suitability <= 1):
            raise ValueError("`target_candidate_suitability` must be between 0 and 1.")
        if self.organoid_radius >= 0.45 * self.matrix_extent:
            raise ValueError("`organoid_radius` must stay well inside the surrounding matrix block.")


SimulationRequest = BulkMechanicsRequest | SingleCellContractionRequest | OrganoidSpheroidRequest


def simulation_request_from_dict(payload: Mapping[str, Any]) -> SimulationRequest:
    scenario = str(payload.get("scenario", "")).strip().lower()
    if not scenario:
        raise ValueError("Simulation request must define `scenario`.")

    normalized = dict(payload)
    normalized.pop("scenario", None)
    if "mesh_resolution" in normalized:
        normalized["mesh_resolution"] = _coerce_tuple(
            normalized["mesh_resolution"],
            field_name="mesh_resolution",
            length=3,
            cast="int",
        )
    if "sample_dimensions" in normalized:
        normalized["sample_dimensions"] = _coerce_tuple(
            normalized["sample_dimensions"],
            field_name="sample_dimensions",
            length=3,
            cast="float",
        )

    if scenario == "bulk_mechanics":
        return BulkMechanicsRequest(**normalized)
    if scenario == "single_cell_contraction":
        return SingleCellContractionRequest(**normalized)
    if scenario == "organoid_spheroid":
        return OrganoidSpheroidRequest(**normalized)
    raise ValueError(
        "Unsupported simulation scenario. Expected one of "
        "`bulk_mechanics`, `single_cell_contraction`, or `organoid_spheroid`."
    )


def simulation_request_from_json(text: str) -> SimulationRequest:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Simulation request JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Simulation request JSON must decode to an object.")
    return simulation_request_from_dict(payload)
