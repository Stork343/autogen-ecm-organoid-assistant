from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..artifacts import write_json
from .schemas import (
    BaseSimulationRequest,
    BulkMechanicsRequest,
    OrganoidSpheroidRequest,
    SimulationRequest,
    SingleCellContractionRequest,
)
from .templates import (
    BoundaryConditionSpec,
    MaterialSpec,
    OutputSpec,
    TemplateContext,
    render_template,
)


@dataclass(frozen=True)
class HexMesh:
    nodes: dict[int, tuple[float, float, float]]
    elements: dict[int, tuple[int, int, int, int, int, int, int, int]]
    element_centroids: dict[int, tuple[float, float, float]]
    grid_shape: tuple[int, int, int]


@dataclass(frozen=True)
class BuildArtifacts:
    request: SimulationRequest
    simulation_dir: Path
    input_path: Path
    metadata_path: Path
    scenario_summary_path: Path
    metadata: dict[str, Any]


def _node_id(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i + 1


def _structured_hex_mesh(
    *,
    size_x: float,
    size_y: float,
    size_z: float,
    nx: int,
    ny: int,
    nz: int,
    centered: bool,
) -> HexMesh:
    x_origin = -size_x / 2.0 if centered else 0.0
    y_origin = -size_y / 2.0 if centered else 0.0
    z_origin = -size_z / 2.0 if centered else 0.0
    dx = size_x / nx
    dy = size_y / ny
    dz = size_z / nz

    nodes: dict[int, tuple[float, float, float]] = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id = _node_id(i, j, k, nx, ny)
                nodes[node_id] = (
                    x_origin + i * dx,
                    y_origin + j * dy,
                    z_origin + k * dz,
                )

    elements: dict[int, tuple[int, int, int, int, int, int, int, int]] = {}
    centroids: dict[int, tuple[float, float, float]] = {}
    element_id = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                connectivity = (
                    _node_id(i, j, k, nx, ny),
                    _node_id(i + 1, j, k, nx, ny),
                    _node_id(i + 1, j + 1, k, nx, ny),
                    _node_id(i, j + 1, k, nx, ny),
                    _node_id(i, j, k + 1, nx, ny),
                    _node_id(i + 1, j, k + 1, nx, ny),
                    _node_id(i + 1, j + 1, k + 1, nx, ny),
                    _node_id(i, j + 1, k + 1, nx, ny),
                )
                elements[element_id] = connectivity
                centroids[element_id] = (
                    x_origin + (i + 0.5) * dx,
                    y_origin + (j + 0.5) * dy,
                    z_origin + (k + 0.5) * dz,
                )
                element_id += 1

    return HexMesh(
        nodes=nodes,
        elements=elements,
        element_centroids=centroids,
        grid_shape=(nx, ny, nz),
    )


def _surface_nodes_by_coordinate(
    mesh: HexMesh,
    *,
    axis: int,
    coordinate: float,
    tolerance: float = 1e-9,
) -> list[int]:
    return sorted(
        node_id
        for node_id, coords in mesh.nodes.items()
        if abs(coords[axis] - coordinate) <= tolerance
    )


def _outer_boundary_nodes(mesh: HexMesh) -> list[int]:
    coordinates = list(mesh.nodes.values())
    x_values = [coords[0] for coords in coordinates]
    y_values = [coords[1] for coords in coordinates]
    z_values = [coords[2] for coords in coordinates]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)
    result: list[int] = []
    for node_id, (x, y, z) in mesh.nodes.items():
        if (
            abs(x - x_min) <= 1e-9
            or abs(x - x_max) <= 1e-9
            or abs(y - y_min) <= 1e-9
            or abs(y - y_max) <= 1e-9
            or abs(z - z_min) <= 1e-9
            or abs(z - z_max) <= 1e-9
        ):
            result.append(node_id)
    return sorted(result)


def _split_by_radius(mesh: HexMesh, *, radius: float) -> tuple[list[int], list[int]]:
    inner: list[int] = []
    outer: list[int] = []
    radial_distances: dict[int, float] = {}
    for element_id, centroid in mesh.element_centroids.items():
        radial_distance = (centroid[0] ** 2 + centroid[1] ** 2 + centroid[2] ** 2) ** 0.5
        radial_distances[element_id] = radial_distance
        if radial_distance <= radius:
            inner.append(element_id)
        else:
            outer.append(element_id)
    if not inner:
        min_distance = min(radial_distances.values())
        inner = sorted(
            element_id for element_id, radial_distance in radial_distances.items() if abs(radial_distance - min_distance) <= 1e-9
        )
        outer = sorted(element_id for element_id in mesh.elements if element_id not in set(inner))
    if not outer:
        raise ValueError("The inclusion radius is too large; at least one surrounding matrix element is required.")
    return sorted(inner), sorted(outer)


def _interface_node_ids(mesh: HexMesh, inner_element_ids: list[int], outer_element_ids: list[int]) -> list[int]:
    inner_nodes = {
        node_id
        for element_id in inner_element_ids
        for node_id in mesh.elements[element_id]
    }
    outer_nodes = {
        node_id
        for element_id in outer_element_ids
        for node_id in mesh.elements[element_id]
    }
    interface_nodes = sorted(inner_nodes.intersection(outer_nodes))
    if not interface_nodes:
        raise ValueError("Failed to identify interface nodes between the inclusion and the surrounding matrix.")
    return interface_nodes


def _radial_displacement_maps(
    mesh: HexMesh,
    *,
    node_ids: list[int],
    radial_displacement: float,
) -> tuple[list[float], list[float], list[float]]:
    disp_x: list[float] = []
    disp_y: list[float] = []
    disp_z: list[float] = []
    for node_id in node_ids:
        x, y, z = mesh.nodes[node_id]
        radius = (x * x + y * y + z * z) ** 0.5
        if radius <= 1e-12:
            disp_x.append(0.0)
            disp_y.append(0.0)
            disp_z.append(0.0)
            continue
        disp_x.append(radial_displacement * x / radius)
        disp_y.append(radial_displacement * y / radius)
        disp_z.append(radial_displacement * z / radius)
    return disp_x, disp_y, disp_z


def _element_subset(
    mesh: HexMesh,
    element_ids: list[int],
) -> dict[int, tuple[int, int, int, int, int, int, int, int]]:
    return {element_id: mesh.elements[element_id] for element_id in element_ids}


def _serialize_xml(root: ET.Element, output_path: Path) -> Path:
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _base_summary_lines(request: BaseSimulationRequest) -> list[str]:
    return [
        f"# FEBio Scenario Summary",
        f"- scenario: {request.scenario}",
        f"- title: {request.title}",
        f"- matrix_youngs_modulus: {request.matrix_youngs_modulus}",
        f"- matrix_poisson_ratio: {request.matrix_poisson_ratio}",
        f"- matrix_extent: {request.matrix_extent}",
        f"- mesh_resolution: {request.mesh_resolution}",
        f"- time_steps: {request.time_steps}",
        f"- step_size: {request.step_size}",
        f"- target_stiffness: {request.target_stiffness if request.target_stiffness is not None else 'NR'}",
    ]


def _metadata_from_context(
    request: SimulationRequest,
    mesh: HexMesh,
    *,
    node_sets: dict[str, list[int]],
    element_sets: dict[str, list[int]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request": request.to_dict(),
        "mesh": {
            "grid_shape": mesh.grid_shape,
            "node_count": len(mesh.nodes),
            "element_count": len(mesh.elements),
            "nodes": {str(node_id): list(coords) for node_id, coords in mesh.nodes.items()},
            "elements": {str(element_id): list(connectivity) for element_id, connectivity in mesh.elements.items()},
            "element_centroids": {
                str(element_id): list(centroid) for element_id, centroid in mesh.element_centroids.items()
            },
        },
        "node_sets": {name: list(node_ids) for name, node_ids in node_sets.items()},
        "element_sets": {name: list(element_ids) for name, element_ids in element_sets.items()},
        "scenario_summary": summary,
    }


def _build_bulk_context(request: BulkMechanicsRequest) -> tuple[TemplateContext, dict[str, Any], str]:
    nx, ny, nz = request.mesh_resolution
    size_x, size_y, size_z = request.sample_dimensions
    mesh = _structured_hex_mesh(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        nx=nx,
        ny=ny,
        nz=nz,
        centered=False,
    )
    bottom_nodes = _surface_nodes_by_coordinate(mesh, axis=2, coordinate=0.0)
    top_nodes = _surface_nodes_by_coordinate(mesh, axis=2, coordinate=size_z)
    element_ids = sorted(mesh.elements.keys())
    node_sets = {
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
    }
    element_sets = {"matrix_domain": element_ids}
    summary = {
        "scenario": request.scenario,
        "sample_dimensions": list(request.sample_dimensions),
        "prescribed_displacement": request.prescribed_displacement,
        "engineering_strain": abs(request.prescribed_displacement) / size_z,
        "top_surface_area": size_x * size_y,
    }
    context = TemplateContext(
        title=request.title,
        scenario=request.scenario,
        time_steps=request.time_steps,
        step_size=request.step_size,
        materials=[
            MaterialSpec(
                name="matrix_material",
                material_id=1,
                material_type="neo-Hookean",
                parameters={"E": request.matrix_youngs_modulus, "v": request.matrix_poisson_ratio},
            )
        ],
        nodes=mesh.nodes,
        element_sets={"matrix_domain": _element_subset(mesh, element_ids)},
        node_sets=node_sets,
        solid_domains={"matrix_domain": "matrix_material"},
        mesh_data={},
        boundaries=[
            BoundaryConditionSpec(
                name="bottom_fixed",
                boundary_type="zero displacement",
                node_set="bottom_nodes",
                x_dof=1,
                y_dof=1,
                z_dof=1,
            ),
            BoundaryConditionSpec(
                name="top_compression_z",
                boundary_type="prescribed displacement",
                node_set="top_nodes",
                dof="z",
                value=request.prescribed_displacement,
                loadcurve_id=1,
                relative=0,
            ),
        ],
        outputs=[
            OutputSpec(output_type="node_data", file_name="node_displacement.log", data="ux;uy;uz"),
            OutputSpec(
                output_type="node_data",
                file_name="top_reaction.log",
                data="Rx;Ry;Rz",
                item_ids=top_nodes,
            ),
            OutputSpec(output_type="element_data", file_name="element_principal_stress.log", data="s1;s2;s3"),
        ],
    )
    summary_lines = _base_summary_lines(request) + [
        f"- sample_dimensions: {request.sample_dimensions}",
        f"- prescribed_displacement: {request.prescribed_displacement}",
    ]
    metadata = _metadata_from_context(
        request,
        mesh,
        node_sets=node_sets,
        element_sets=element_sets,
        summary=summary,
    )
    return context, metadata, "\n".join(summary_lines) + "\n"


def _build_inclusion_context(
    request: SingleCellContractionRequest | OrganoidSpheroidRequest,
) -> tuple[TemplateContext, dict[str, Any], str]:
    nx, ny, nz = request.mesh_resolution
    mesh = _structured_hex_mesh(
        size_x=request.matrix_extent,
        size_y=request.matrix_extent,
        size_z=request.matrix_extent,
        nx=nx,
        ny=ny,
        nz=nz,
        centered=True,
    )

    if isinstance(request, SingleCellContractionRequest):
        radius = request.cell_radius
        inner_name = "cell_domain"
        inner_material = MaterialSpec(
            name="cell_material",
            material_id=2,
            material_type="neo-Hookean",
            parameters={"E": request.cell_youngs_modulus, "v": request.cell_poisson_ratio},
        )
        radial_displacement = -abs(request.cell_contractility)
        extra_summary = {
            "cell_radius": request.cell_radius,
            "cell_contractility": request.cell_contractility,
            "loading_mode": request.loading_mode,
            "target_stress_propagation_distance": request.target_stress_propagation_distance,
            "target_strain_heterogeneity": request.target_strain_heterogeneity,
        }
        extra_lines = [
            f"- cell_radius: {request.cell_radius}",
            f"- cell_contractility: {request.cell_contractility}",
            f"- loading_mode: {request.loading_mode}",
        ]
    else:
        radius = request.organoid_radius
        inner_name = "organoid_domain"
        inner_material = MaterialSpec(
            name="organoid_material",
            material_id=2,
            material_type="neo-Hookean",
            parameters={"E": request.organoid_youngs_modulus, "v": request.organoid_poisson_ratio},
        )
        radial_displacement = request.organoid_radial_displacement
        extra_summary = {
            "organoid_radius": request.organoid_radius,
            "organoid_radial_displacement": request.organoid_radial_displacement,
            "loading_mode": request.loading_mode,
            "target_interface_deformation": request.target_interface_deformation,
            "target_stress_propagation_distance": request.target_stress_propagation_distance,
            "target_candidate_suitability": request.target_candidate_suitability,
        }
        extra_lines = [
            f"- organoid_radius: {request.organoid_radius}",
            f"- organoid_radial_displacement: {request.organoid_radial_displacement}",
            f"- loading_mode: {request.loading_mode}",
        ]

    inner_element_ids, matrix_element_ids = _split_by_radius(mesh, radius=radius)
    interface_nodes = _interface_node_ids(mesh, inner_element_ids, matrix_element_ids)
    outer_boundary_nodes = _outer_boundary_nodes(mesh)
    disp_x, disp_y, disp_z = _radial_displacement_maps(
        mesh,
        node_ids=interface_nodes,
        radial_displacement=radial_displacement,
    )

    node_sets = {
        "outer_boundary_nodes": outer_boundary_nodes,
        "interface_nodes": interface_nodes,
    }
    element_sets = {
        "matrix_domain": matrix_element_ids,
        inner_name: inner_element_ids,
    }
    summary = {
        "scenario": request.scenario,
        "matrix_extent": request.matrix_extent,
        "inner_radius": radius,
        "interface_node_count": len(interface_nodes),
        **extra_summary,
    }
    context = TemplateContext(
        title=request.title,
        scenario=request.scenario,
        time_steps=request.time_steps,
        step_size=request.step_size,
        materials=[
            MaterialSpec(
                name="matrix_material",
                material_id=1,
                material_type="neo-Hookean",
                parameters={"E": request.matrix_youngs_modulus, "v": request.matrix_poisson_ratio},
            ),
            inner_material,
        ],
        nodes=mesh.nodes,
        element_sets={
            "matrix_domain": _element_subset(mesh, matrix_element_ids),
            inner_name: _element_subset(mesh, inner_element_ids),
        },
        node_sets=node_sets,
        solid_domains={
            "matrix_domain": "matrix_material",
            inner_name: inner_material.name,
        },
        mesh_data={
            "interface_disp_x": {"node_set": "interface_nodes", "values": disp_x},
            "interface_disp_y": {"node_set": "interface_nodes", "values": disp_y},
            "interface_disp_z": {"node_set": "interface_nodes", "values": disp_z},
        },
        boundaries=[
            BoundaryConditionSpec(
                name="outer_boundary_fixed",
                boundary_type="zero displacement",
                node_set="outer_boundary_nodes",
                x_dof=1,
                y_dof=1,
                z_dof=1,
            ),
            BoundaryConditionSpec(
                name="interface_prescribed_x",
                boundary_type="prescribed displacement",
                node_set="interface_nodes",
                dof="x",
                value="interface_disp_x",
                value_type="map",
                loadcurve_id=1,
                relative=0,
            ),
            BoundaryConditionSpec(
                name="interface_prescribed_y",
                boundary_type="prescribed displacement",
                node_set="interface_nodes",
                dof="y",
                value="interface_disp_y",
                value_type="map",
                loadcurve_id=1,
                relative=0,
            ),
            BoundaryConditionSpec(
                name="interface_prescribed_z",
                boundary_type="prescribed displacement",
                node_set="interface_nodes",
                dof="z",
                value="interface_disp_z",
                value_type="map",
                loadcurve_id=1,
                relative=0,
            ),
        ],
        outputs=[
            OutputSpec(output_type="node_data", file_name="node_displacement.log", data="ux;uy;uz"),
            OutputSpec(output_type="element_data", file_name="element_principal_stress.log", data="s1;s2;s3"),
        ],
    )
    summary_lines = _base_summary_lines(request) + extra_lines + [
        f"- interface_node_count: {len(interface_nodes)}",
    ]
    metadata = _metadata_from_context(
        request,
        mesh,
        node_sets=node_sets,
        element_sets=element_sets,
        summary=summary,
    )
    return context, metadata, "\n".join(summary_lines) + "\n"


def build_simulation_input(request: SimulationRequest, simulation_dir: Path) -> BuildArtifacts:
    """Validate a fixed scenario request and build the corresponding FEBio input bundle."""

    simulation_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(request, BulkMechanicsRequest):
        context, metadata, summary_text = _build_bulk_context(request)
    elif isinstance(request, (SingleCellContractionRequest, OrganoidSpheroidRequest)):
        context, metadata, summary_text = _build_inclusion_context(request)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported request type: {type(request)!r}")

    input_path = simulation_dir / "input.feb"
    metadata_path = simulation_dir / "metadata.json"
    scenario_summary_path = simulation_dir / "scenario_summary.md"
    root = render_template(context)
    _serialize_xml(root, input_path)
    write_json(metadata_path, metadata)
    scenario_summary_path.write_text(summary_text, encoding="utf-8")
    return BuildArtifacts(
        request=request,
        simulation_dir=simulation_dir,
        input_path=input_path,
        metadata_path=metadata_path,
        scenario_summary_path=scenario_summary_path,
        metadata=metadata,
    )
