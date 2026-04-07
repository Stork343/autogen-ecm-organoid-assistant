from __future__ import annotations

from typing import Any


def recommend_formulations_from_design_payload(
    design_payload: dict[str, Any],
    *,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Translate abstract mechanics-optimal candidates into wet-lab starting formulations."""

    if not isinstance(design_payload, dict):
        return []
    recommendations = []
    for candidate in design_payload.get("top_candidates", [])[:max_candidates]:
        params = candidate.get("parameters", {})
        features = candidate.get("features", {})
        if not params or not features:
            continue
        recommendations.append(recommend_formulation(candidate))
    return recommendations


def recommend_formulation(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic formulation recommendation for one design candidate."""

    params = candidate.get("parameters", {})
    features = candidate.get("features", {})
    family = _select_material_family(params, features)
    primary = _primary_recipe_for_family(family, params, features)
    alternates = _alternate_templates_for_family(family, params, features)
    return {
        "candidate_rank": int(candidate.get("rank", 0)),
        "candidate_score": float(candidate.get("score", 0.0)),
        "feasible": bool(candidate.get("feasible", False)),
        "material_family": family["material_family"],
        "template_name": family["template_name"],
        "crosslinking_strategy": family["crosslinking_strategy"],
        "mapping_confidence": _mapping_confidence(candidate, family),
        "rationale": _formulation_rationale(params, features, family),
        "primary_recipe": primary,
        "alternate_templates": alternates,
        "experimental_checks": _experimental_checks(features, family),
        "caveats": _formulation_caveats(features, family),
    }


def recommend_campaign_formulations(campaign_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    for row in campaign_results:
        best_candidate = row.get("best_candidate", {})
        if not isinstance(best_candidate, dict) or not best_candidate:
            continue
        recommendation = recommend_formulation(best_candidate)
        recommendation["target_stiffness"] = float(row.get("target_stiffness", 0.0))
        recommendations.append(recommendation)
    return recommendations


def _select_material_family(params: dict[str, Any], features: dict[str, Any]) -> dict[str, str]:
    density = float(params.get("fiber_density", 0.35))
    stiffness = float(params.get("fiber_stiffness", 8.0))
    bending = float(params.get("bending_stiffness", 0.2))
    crosslink = float(params.get("crosslink_prob", 0.45))
    anisotropy = float(features.get("anisotropy", 0.2))

    if stiffness >= 10.0 or crosslink >= 0.62:
        return {
            "material_family": "PEG-4MAL",
            "template_name": "PEG-4MAL + laminin / MMP-linker",
            "crosslinking_strategy": "maleimide-thiol step-growth gelation",
        }
    if density >= 0.42 and bending >= 0.24:
        return {
            "material_family": "GelMA",
            "template_name": "GelMA + LAP photocrosslinking",
            "crosslinking_strategy": "visible-light photocrosslinking",
        }
    if crosslink <= 0.42 and anisotropy <= 0.24:
        return {
            "material_family": "NorHA",
            "template_name": "Norbornene-HA + thiol-ene peptide linker",
            "crosslinking_strategy": "thiol-ene photo-click gelation",
        }
    return {
        "material_family": "PEG-Laminin Hybrid",
        "template_name": "PEG hydrogel + laminin-rich adhesion supplement",
        "crosslinking_strategy": "covalent PEG network with laminin presentation",
    }


def _primary_recipe_for_family(
    family: dict[str, str],
    params: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    density = float(params.get("fiber_density", 0.35))
    stiffness = float(params.get("fiber_stiffness", 8.0))
    bending = float(params.get("bending_stiffness", 0.2))
    crosslink = float(params.get("crosslink_prob", 0.45))
    stress_prop = float(features.get("stress_propagation", 0.8))
    target_polymer = round(_polymer_wt_percent(family["material_family"], density, stiffness, crosslink), 2)
    ligand_level = _ligand_density_bucket(stress_prop, density)
    linker_ratio = round(0.78 + 0.45 * crosslink, 2)

    recipes = {
        "PEG-4MAL": {
            "polymer": "4-arm PEG-4MAL (20 kDa)",
            "polymer_wt_percent": f"{target_polymer:.2f} wt%",
            "crosslinker": "MMP-degradable dithiol peptide",
            "crosslink_ratio": f"{linker_ratio:.2f} thiol:maleimide",
            "adhesion_ligand": f"laminin-111 supplement or RGD at {ligand_level}",
            "buffer": "HEPES-buffered PBS, pH 7.2-7.4",
            "gelation": "mix precursor and crosslinker immediately before encapsulation",
            "tuning_priority": "raise PEG wt% or linker ratio to increase stiffness; raise laminin density for epithelial support",
        },
        "GelMA": {
            "polymer": "GelMA (medium methacrylation)",
            "polymer_wt_percent": f"{max(target_polymer + 1.5, 4.0):.2f} wt%",
            "crosslinker": "LAP photoinitiator",
            "crosslink_ratio": "0.03-0.07 wt% LAP",
            "adhesion_ligand": f"intrinsic gelatin motifs, optionally laminin at {ligand_level}",
            "buffer": "PBS or organoid basal medium",
            "gelation": "405 nm photocrosslinking, short exposure, low heating",
            "tuning_priority": "increase GelMA wt% or exposure dose for stiffer gels; reduce dose if viability drops",
        },
        "NorHA": {
            "polymer": "Norbornene-modified hyaluronic acid",
            "polymer_wt_percent": f"{max(target_polymer - 1.0, 1.0):.2f} wt%",
            "crosslinker": "dithiol peptide or DTT + LAP",
            "crosslink_ratio": f"{linker_ratio:.2f} thiol:norbornene",
            "adhesion_ligand": f"RGD peptide at {ligand_level}",
            "buffer": "PBS, pH 7.2-7.4",
            "gelation": "thiol-ene light-triggered gelation after cell mixing",
            "tuning_priority": "increase HA wt% or linker conversion to stiffen; keep photodose low for fragile organoids",
        },
        "PEG-Laminin Hybrid": {
            "polymer": "PEG macromer + laminin-rich adhesion supplement",
            "polymer_wt_percent": f"{target_polymer:.2f} wt%",
            "crosslinker": "PEG-compatible peptide linker",
            "crosslink_ratio": f"{linker_ratio:.2f} functional-group ratio",
            "adhesion_ligand": f"laminin / entactin-rich supplement at {ligand_level}",
            "buffer": "HEPES-buffered basal medium",
            "gelation": "pre-gel PEG network followed by laminin presentation during encapsulation",
            "tuning_priority": "use PEG network for mechanics and laminin density for organoid attachment/readout tuning",
        },
    }
    recipe = recipes[family["material_family"]]
    recipe["mechanics_anchor"] = {
        "fiber_density": round(density, 4),
        "fiber_stiffness": round(stiffness, 4),
        "bending_stiffness": round(bending, 4),
        "crosslink_prob": round(crosslink, 4),
    }
    return recipe


def _alternate_templates_for_family(
    family: dict[str, str],
    params: dict[str, Any],
    features: dict[str, Any],
) -> list[dict[str, str]]:
    density = float(params.get("fiber_density", 0.35))
    crosslink = float(params.get("crosslink_prob", 0.45))
    stress_prop = float(features.get("stress_propagation", 0.8))
    options = {
        "PEG-4MAL": [
            {
                "material_family": "NorHA",
                "why": "keeps synthetic control but offers softer click-chemistry processing",
                "starting_point": f"1.8-3.0 wt% NorHA, crosslink ratio {0.75 + 0.35 * crosslink:.2f}, RGD { _ligand_density_bucket(stress_prop, density) }",
            },
            {
                "material_family": "PEG-Laminin Hybrid",
                "why": "maintains mechanical control while boosting epithelial adhesion cues",
                "starting_point": f"PEG 3.5-5.5 wt% + laminin-rich supplement at {_ligand_density_bucket(stress_prop, density)}",
            },
        ],
        "GelMA": [
            {
                "material_family": "PEG-4MAL",
                "why": "reduces batch variability and increases stoichiometric control",
                "starting_point": "PEG-4MAL 4-6 wt% with peptide linker near stoichiometric balance",
            },
            {
                "material_family": "PEG-Laminin Hybrid",
                "why": "decouples mechanics from bioactivity more cleanly than pure GelMA",
                "starting_point": "PEG network with laminin supplement and low GelMA fraction",
            },
        ],
        "NorHA": [
            {
                "material_family": "Alginate-RGD",
                "why": "useful for softer or intermediate matrices with simple ionic tuning",
                "starting_point": "Alginate-RGD 0.8-1.5 wt% with CaCO3/GDL or covalent reinforcement",
            },
            {
                "material_family": "PEG-4MAL",
                "why": "preferred if you need tighter mechanical reproducibility",
                "starting_point": "PEG-4MAL 3.5-5.0 wt% plus low laminin dose",
            },
        ],
        "PEG-Laminin Hybrid": [
            {
                "material_family": "PEG-4MAL",
                "why": "best when mechanics must dominate over biological variability",
                "starting_point": "PEG-4MAL 4-6 wt% with controlled peptide linker and separate laminin titration",
            },
            {
                "material_family": "GelMA",
                "why": "best when native adhesion cues are more important than strict synthetic control",
                "starting_point": "GelMA 4-7 wt% with LAP 0.03-0.05 wt%",
            },
        ],
    }
    return options.get(family["material_family"], [])


def _mapping_confidence(candidate: dict[str, Any], family: dict[str, str]) -> str:
    score = float(candidate.get("score", 1.0))
    feasible = bool(candidate.get("feasible", False))
    if feasible and score <= 0.35:
        return "high"
    if feasible and score <= 0.6:
        return "medium"
    return "low"


def _formulation_rationale(
    params: dict[str, Any],
    features: dict[str, Any],
    family: dict[str, str],
) -> list[str]:
    density = float(params.get("fiber_density", 0.35))
    crosslink = float(params.get("crosslink_prob", 0.45))
    risk = float(features.get("risk_index", 0.5))
    rationale = [
        f"Selected `{family['material_family']}` because the candidate sits in a density/crosslink regime of {density:.3f}/{crosslink:.3f}.",
        f"`{family['crosslinking_strategy']}` matches the required control level for this mechanics target.",
    ]
    if risk > 0.65:
        rationale.append("The candidate has relatively high Monte Carlo variability, so start with a narrow formulation screen around the suggested base recipe.")
    else:
        rationale.append("The candidate has moderate-to-low variability, so the starting recipe is a reasonable first wet-lab anchor.")
    return rationale


def _experimental_checks(features: dict[str, Any], family: dict[str, str]) -> list[str]:
    anisotropy = float(features.get("anisotropy", 0.2))
    checks = [
        "Measure shear modulus or compression modulus after gelation to verify the design target was reached.",
        "Run a short-term organoid encapsulation pilot with viability and morphology readouts before scale-up.",
        "Track batch-to-batch gelation time and final stiffness for at least 3 repeats.",
    ]
    if anisotropy > 0.25:
        checks.append("Add imaging of fiber alignment or microstructure because predicted anisotropy is non-trivial.")
    if family["material_family"] in {"GelMA", "NorHA"}:
        checks.append("Check photoinitiator dose and exposure time against viability, especially for fragile epithelial organoids.")
    return checks


def _formulation_caveats(features: dict[str, Any], family: dict[str, str]) -> list[str]:
    risk = float(features.get("risk_index", 0.5))
    caveats = [
        "This translation is a mechanics-informed starting recipe, not a validated material-specific calibration.",
    ]
    if family["material_family"] == "GelMA":
        caveats.append("GelMA batch variability can decouple nominal wt% from actual modulus.")
    if family["material_family"] == "PEG-4MAL":
        caveats.append("Adhesion and organoid morphogenesis may remain ligand-limited even if bulk stiffness is on target.")
    if risk > 0.75:
        caveats.append("Predicted variability is high; prioritize a small local formulation matrix rather than a single-point formulation.")
    return caveats


def _polymer_wt_percent(material_family: str, density: float, stiffness: float, crosslink: float) -> float:
    base = 1.5 + 4.0 * density + 0.18 * stiffness + 1.6 * crosslink
    family_bounds = {
        "PEG-4MAL": (3.0, 8.0),
        "GelMA": (4.0, 10.0),
        "NorHA": (1.0, 4.0),
        "PEG-Laminin Hybrid": (3.0, 7.0),
    }
    lower, upper = family_bounds.get(material_family, (2.0, 8.0))
    return min(max(base, lower), upper)


def _ligand_density_bucket(stress_prop: float, density: float) -> str:
    signal = 0.6 * stress_prop + 0.4 * density
    if signal >= 0.75:
        return "high ligand presentation (e.g. 1.5-2.5 mM RGD or 100-200 ug/mL laminin)"
    if signal >= 0.55:
        return "medium ligand presentation (e.g. 0.8-1.5 mM RGD or 50-100 ug/mL laminin)"
    return "low-to-medium ligand presentation (e.g. 0.5-1.0 mM RGD or 25-75 ug/mL laminin)"
