from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .runner import run_research_agent
from .workspace import resolve_project_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a personal AutoGen research assistant for ECM and organoid synthesis."
    )
    parser.add_argument("--query", required=True, help="PubMed-style research question or topic.")
    parser.add_argument(
        "--report-name",
        default="ecm_organoid_report.md",
        help="Name of the Markdown report to save under reports/.",
    )
    parser.add_argument(
        "--workflow",
        default="team",
        choices=("team", "single", "mechanics", "hybrid", "simulation", "design", "design_campaign", "benchmark", "datasets", "calibration"),
        help="Execution mode. `team` runs the literature collaboration workflow, `single` keeps the original single-agent workflow, `mechanics` runs ECM mechanics modeling, `hybrid` chains literature + mechanics + simulation, `simulation` runs a fixed FEBio-backed scenario, `design` runs target-driven ECM inverse design, `design_campaign` scans multiple target windows, `benchmark` evaluates mechanics-core performance, `datasets` manages public mechanics datasets, and `calibration` turns experimental datasets into ECM calibration priors.",
    )
    parser.add_argument(
        "--project-dir",
        default=resolve_project_dir(),
        type=Path,
        help="Project root directory. Defaults to the starter project root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name. Falls back to OPENAI_MODEL or gpt-4.1-mini.",
    )
    parser.add_argument(
        "--library-dir",
        default=None,
        type=Path,
        help="Optional directory containing local PDFs, Markdown notes, or text files.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        type=Path,
        help="Optional output directory for saved Markdown reports.",
    )
    parser.add_argument(
        "--max-pubmed-results",
        default=None,
        type=int,
        help="Maximum number of PubMed records to inspect.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        type=Path,
        help="Path to a local mechanics dataset CSV/TSV file. Required for `--workflow mechanics`.",
    )
    parser.add_argument(
        "--experiment-type",
        default="auto",
        choices=("auto", "elastic", "creep", "relaxation", "frequency_sweep", "cyclic"),
        help="Mechanics experiment type for `--workflow mechanics`.",
    )
    parser.add_argument("--time-column", default="time", help="Time column name for mechanics datasets.")
    parser.add_argument("--stress-column", default="stress", help="Stress column name for mechanics datasets.")
    parser.add_argument("--strain-column", default="strain", help="Strain column name for mechanics datasets.")
    parser.add_argument(
        "--applied-stress",
        default=None,
        type=float,
        help="Applied step stress for creep fitting in `--workflow mechanics`.",
    )
    parser.add_argument(
        "--applied-strain",
        default=None,
        type=float,
        help="Applied step strain for relaxation fitting in `--workflow mechanics`.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV/TSV delimiter for mechanics datasets.",
    )
    parser.add_argument("--simulation-fiber-density", default=None, type=float, help="Fiber density for `--workflow hybrid`. If omitted, inferred from evidence and mechanics fit.")
    parser.add_argument("--simulation-fiber-stiffness", default=None, type=float, help="Fiber stiffness for `--workflow hybrid`. If omitted, inferred from mechanics fit.")
    parser.add_argument("--simulation-bending-stiffness", default=None, type=float, help="Bending stiffness for `--workflow hybrid`.")
    parser.add_argument("--simulation-crosslink-prob", default=None, type=float, help="Crosslink probability for `--workflow hybrid`.")
    parser.add_argument("--simulation-domain-size", default=None, type=float, help="Domain size for `--workflow hybrid`.")
    parser.add_argument("--target-stiffness", default=None, type=float, help="Target bulk stiffness for `--workflow design`.")
    parser.add_argument(
        "--simulation-scenario",
        default="bulk_mechanics",
        choices=("bulk_mechanics", "single_cell_contraction", "organoid_spheroid"),
        help="Fixed FEBio scenario for `--workflow simulation` or FEBio-backed design verification.",
    )
    parser.add_argument(
        "--simulation-request-json",
        default="",
        help="Optional JSON object with additional fixed-schema FEBio request fields. This does not allow arbitrary XML.",
    )
    parser.add_argument("--cell-contractility", default=None, type=float, help="Cell contraction magnitude for FEBio single-cell simulations.")
    parser.add_argument("--organoid-radius", default=None, type=float, help="Organoid radius for FEBio spheroid simulations.")
    parser.add_argument("--matrix-youngs-modulus", default=None, type=float, help="Matrix Young's modulus override for FEBio-backed simulations.")
    parser.add_argument("--matrix-poisson-ratio", default=None, type=float, help="Matrix Poisson ratio override for FEBio-backed simulations.")
    parser.add_argument("--target-anisotropy", default=0.1, type=float, help="Target anisotropy for `--workflow design`.")
    parser.add_argument("--target-connectivity", default=0.95, type=float, help="Target connectivity for `--workflow design`.")
    parser.add_argument("--target-stress-propagation", default=0.5, type=float, help="Target stress propagation for `--workflow design`.")
    parser.add_argument(
        "--design-extra-targets-json",
        default="",
        help="Optional JSON object for extra design targets such as mesh_size_proxy, permeability_proxy, compressibility_proxy, swelling_ratio_proxy, loss_tangent_proxy, poroelastic_time_constant_proxy, or strain_stiffening_proxy.",
    )
    parser.add_argument("--constraint-max-anisotropy", default=None, type=float, help="Optional hard upper bound on anisotropy for design workflows.")
    parser.add_argument("--constraint-min-connectivity", default=None, type=float, help="Optional hard lower bound on connectivity for design workflows.")
    parser.add_argument("--constraint-max-risk-index", default=None, type=float, help="Optional hard upper bound on risk index for design workflows.")
    parser.add_argument("--constraint-min-stress-propagation", default=None, type=float, help="Optional hard lower bound on stress propagation for design workflows.")
    parser.add_argument(
        "--design-extra-constraints-json",
        default="",
        help="Optional JSON object for extra design constraints, for example {\"max_loss_tangent_proxy\": 0.8, \"min_permeability_proxy\": 0.01}.",
    )
    parser.add_argument("--design-top-k", default=3, type=int, help="Number of top ECM candidates to report for `--workflow design`.")
    parser.add_argument("--design-candidate-budget", default=12, type=int, help="Number of deterministic candidate evaluations for `--workflow design`.")
    parser.add_argument("--design-monte-carlo-runs", default=4, type=int, help="Monte Carlo replicates per candidate for `--workflow design`.")
    parser.add_argument("--design-run-simulation", action="store_true", help="Run FEBio verification on top-k design candidates when FEBio is available.")
    parser.add_argument(
        "--design-simulation-scenario",
        default="bulk_mechanics",
        choices=("bulk_mechanics", "single_cell_contraction", "organoid_spheroid"),
        help="FEBio scenario used when `--design-run-simulation` is enabled.",
    )
    parser.add_argument("--design-simulation-top-k", default=2, type=int, help="How many top design candidates to verify with FEBio.")
    parser.add_argument("--condition-concentration-fraction", default=None, type=float, help="Optional concentration hint (fraction form, e.g. 0.15) for condition-aware calibrated design priors.")
    parser.add_argument("--condition-curing-seconds", default=None, type=float, help="Optional curing time hint in seconds for condition-aware calibrated design priors.")
    parser.add_argument(
        "--condition-overrides-json",
        default="",
        help="Optional JSON object for additional material-condition hints such as temperature_c, photoinitiator_fraction, polymer_mw_kda, or degree_substitution.",
    )
    parser.add_argument("--campaign-target-stiffnesses", default="", help="Comma-separated stiffness targets for `--workflow design_campaign`, e.g. `6,8,10`.")
    parser.add_argument("--dataset-id", default=None, help="Dataset identifier for `--workflow calibration`.")
    parser.add_argument("--calibration-max-samples", default=None, type=int, help="Optional maximum number of extracted samples to calibrate in `--workflow calibration`.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(
        run_research_agent(
            project_dir=args.project_dir,
            query=args.query,
            report_name=args.report_name,
            workflow=args.workflow,
            model=args.model,
            library_dir=args.library_dir,
            report_dir=args.report_dir,
            max_pubmed_results=args.max_pubmed_results,
            data_path=args.data_path,
            experiment_type=args.experiment_type,
            time_column=args.time_column,
            stress_column=args.stress_column,
            strain_column=args.strain_column,
            applied_stress=args.applied_stress,
            applied_strain=args.applied_strain,
            delimiter=args.delimiter,
            simulation_fiber_density=args.simulation_fiber_density,
            simulation_fiber_stiffness=args.simulation_fiber_stiffness,
            simulation_bending_stiffness=args.simulation_bending_stiffness,
            simulation_crosslink_prob=args.simulation_crosslink_prob,
            simulation_domain_size=args.simulation_domain_size,
            target_stiffness=args.target_stiffness,
            simulation_scenario=args.simulation_scenario,
            simulation_request_json=args.simulation_request_json,
            cell_contractility=args.cell_contractility,
            organoid_radius=args.organoid_radius,
            matrix_youngs_modulus=args.matrix_youngs_modulus,
            matrix_poisson_ratio=args.matrix_poisson_ratio,
            target_anisotropy=args.target_anisotropy,
            target_connectivity=args.target_connectivity,
            target_stress_propagation=args.target_stress_propagation,
            design_extra_targets_json=args.design_extra_targets_json,
            constraint_max_anisotropy=args.constraint_max_anisotropy,
            constraint_min_connectivity=args.constraint_min_connectivity,
            constraint_max_risk_index=args.constraint_max_risk_index,
            constraint_min_stress_propagation=args.constraint_min_stress_propagation,
            design_extra_constraints_json=args.design_extra_constraints_json,
            design_top_k=args.design_top_k,
            design_candidate_budget=args.design_candidate_budget,
            design_monte_carlo_runs=args.design_monte_carlo_runs,
            design_run_simulation=args.design_run_simulation,
            design_simulation_scenario=args.design_simulation_scenario,
            design_simulation_top_k=args.design_simulation_top_k,
            condition_concentration_fraction=args.condition_concentration_fraction,
            condition_curing_seconds=args.condition_curing_seconds,
            condition_overrides_json=args.condition_overrides_json,
            campaign_target_stiffnesses=args.campaign_target_stiffnesses,
            dataset_id=args.dataset_id,
            calibration_max_samples=args.calibration_max_samples,
        )
    )


if __name__ == "__main__":
    main()
