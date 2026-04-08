from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    material_id: int
    material_type: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class BoundaryConditionSpec:
    name: str
    boundary_type: str
    node_set: str
    dof: str | None = None
    value: float | str | None = None
    value_type: str | None = None
    loadcurve_id: int | None = None
    relative: int | None = None
    x_dof: int | None = None
    y_dof: int | None = None
    z_dof: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    output_type: str
    file_name: str
    data: str
    item_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TemplateContext:
    title: str
    scenario: str
    time_steps: int
    step_size: float
    materials: list[MaterialSpec]
    nodes: dict[int, tuple[float, float, float]]
    element_sets: dict[str, dict[int, tuple[int, int, int, int, int, int, int, int]]]
    node_sets: dict[str, list[int]]
    solid_domains: dict[str, str]
    mesh_data: dict[str, dict[str, object]]
    boundaries: list[BoundaryConditionSpec]
    outputs: list[OutputSpec]
    logfile_name: str = "input.log"


def _text(parent: ET.Element, tag: str, value: object, **attrib: object) -> ET.Element:
    node = ET.SubElement(parent, tag, {key: str(item) for key, item in attrib.items() if item is not None})
    node.text = str(value)
    return node


def _csv(values: Iterable[object]) -> str:
    return ",".join(str(value) for value in values)


def _add_common_sections(root: ET.Element, context: TemplateContext) -> None:
    module = ET.SubElement(root, "Module", {"type": "solid"})
    module.tail = "\n"

    control = ET.SubElement(root, "Control")
    _text(control, "analysis", "STATIC")
    _text(control, "time_steps", context.time_steps)
    _text(control, "step_size", context.step_size)
    solver = ET.SubElement(control, "solver")
    _text(solver, "max_refs", 25)
    qn_method = ET.SubElement(solver, "qn_method")
    _text(qn_method, "max_ups", 0)
    _text(solver, "symmetric_stiffness", 0)
    time_stepper = ET.SubElement(control, "time_stepper")
    _text(time_stepper, "dtmin", max(context.step_size / 100.0, 1e-5))
    _text(time_stepper, "dtmax", context.step_size)
    _text(time_stepper, "max_retries", 5)
    _text(time_stepper, "opt_iter", 10)

    material_section = ET.SubElement(root, "Material")
    for material in context.materials:
        material_node = ET.SubElement(
            material_section,
            "material",
            {
                "name": material.name,
                "type": material.material_type,
                "id": str(material.material_id),
            },
        )
        for key, value in material.parameters.items():
            _text(material_node, key, value)

    mesh = ET.SubElement(root, "Mesh")
    nodes_node = ET.SubElement(mesh, "Nodes", {"name": "all_nodes"})
    for node_id, coords in context.nodes.items():
        _text(nodes_node, "node", _csv(coords), id=node_id)

    for element_set_name, elements in context.element_sets.items():
        elements_node = ET.SubElement(mesh, "Elements", {"type": "hex8", "name": element_set_name})
        for element_id, connectivity in elements.items():
            _text(elements_node, "elem", _csv(connectivity), id=element_id)

    for node_set_name, node_ids in context.node_sets.items():
        node_set_node = ET.SubElement(mesh, "NodeSet", {"name": node_set_name})
        node_set_node.text = _csv(node_ids)

    mesh_domains = ET.SubElement(root, "MeshDomains")
    for domain_name, material_name in context.solid_domains.items():
        ET.SubElement(mesh_domains, "SolidDomain", {"name": domain_name, "mat": material_name})

    if context.mesh_data:
        mesh_data = ET.SubElement(root, "MeshData")
        for data_name, payload in context.mesh_data.items():
            node_data = ET.SubElement(
                mesh_data,
                "NodeData",
                {
                    "name": data_name,
                    "node_set": str(payload["node_set"]),
                    "data_type": "scalar",
                },
            )
            node_values = payload["values"]
            for lid, value in enumerate(node_values, start=1):
                _text(node_data, "node", value, lid=lid)

    boundary = ET.SubElement(root, "Boundary")
    for item in context.boundaries:
        bc_node = ET.SubElement(
            boundary,
            "bc",
            {
                "name": item.name,
                "type": item.boundary_type,
                "node_set": item.node_set,
            },
        )
        if item.boundary_type == "zero displacement":
            _text(bc_node, "x_dof", item.x_dof if item.x_dof is not None else 0)
            _text(bc_node, "y_dof", item.y_dof if item.y_dof is not None else 0)
            _text(bc_node, "z_dof", item.z_dof if item.z_dof is not None else 0)
        else:
            _text(bc_node, "dof", item.dof or "z")
            value_attrs = {"lc": item.loadcurve_id}
            if item.value_type:
                value_attrs["type"] = item.value_type
            _text(bc_node, "value", item.value if item.value is not None else 0.0, **value_attrs)
            _text(bc_node, "relative", item.relative if item.relative is not None else 0)

    load_data = ET.SubElement(root, "LoadData")
    controller = ET.SubElement(
        load_data,
        "load_controller",
        {"id": "1", "name": "LC_1", "type": "loadcurve"},
    )
    _text(controller, "interpolate", "LINEAR")
    points = ET.SubElement(controller, "points")
    _text(points, "pt", "0,0")
    _text(points, "pt", "1,1")

    output = ET.SubElement(root, "Output")
    logfile = ET.SubElement(output, "logfile", {"file": context.logfile_name})
    for output_spec in context.outputs:
        output_node = ET.SubElement(
            logfile,
            output_spec.output_type,
            {"file": output_spec.file_name, "data": output_spec.data, "delim": ","},
        )
        if output_spec.item_ids:
            output_node.text = _csv(output_spec.item_ids)
    plotfile = ET.SubElement(output, "plotfile")
    _text(plotfile, "compression", 0)


def _build_root(context: TemplateContext) -> ET.Element:
    root = ET.Element("febio_spec", {"version": "4.0"})
    root.append(ET.Comment(f" {context.title} | {context.scenario} "))
    _add_common_sections(root, context)
    ET.indent(root, space="  ")
    return root


def render_bulk_mechanics_template(context: TemplateContext) -> ET.Element:
    return _build_root(context)


def render_single_cell_contraction_template(context: TemplateContext) -> ET.Element:
    return _build_root(context)


def render_organoid_spheroid_template(context: TemplateContext) -> ET.Element:
    return _build_root(context)


def render_template(context: TemplateContext) -> ET.Element:
    if context.scenario == "bulk_mechanics":
        return render_bulk_mechanics_template(context)
    if context.scenario == "single_cell_contraction":
        return render_single_cell_contraction_template(context)
    if context.scenario == "organoid_spheroid":
        return render_organoid_spheroid_template(context)
    raise ValueError(f"Unsupported FEBio template scenario: {context.scenario}")
