from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from textwrap import dedent
from typing import Any, Callable, Optional

from .runner import run_research_agent_sync
from .workspace import ensure_workspace, resolve_project_dir, source_project_dir


@dataclass(frozen=True)
class DemoStep:
    slug: str
    workflow: str
    title: str
    description: str
    query: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DemoStepResult:
    slug: str
    workflow: str
    title: str
    status: str
    report_path: Optional[Path]
    run_dir: Optional[Path]
    final_summary: str
    error: str = ""


@dataclass(frozen=True)
class DemoRunResult:
    demo_id: str
    include_ai_workflows: bool
    summary_path: Path
    manifest_path: Path
    step_results: list[DemoStepResult]

    @property
    def completed_steps(self) -> int:
        return sum(1 for step in self.step_results if step.status == "ok")

    @property
    def failed_steps(self) -> int:
        return sum(1 for step in self.step_results if step.status != "ok")


def sample_mechanics_dataset_path(project_dir: Path) -> Path:
    workspace_candidate = project_dir / "demo_assets" / "sample_mechanics_creep.csv"
    if workspace_candidate.exists():
        return workspace_candidate
    source_root = source_project_dir()
    for source_candidate in (
        source_root / "demo_assets" / "sample_mechanics_creep.csv",
        source_root / "runs" / "sample_mechanics_creep.csv",
    ):
        if source_candidate.exists():
            return source_candidate
    raise FileNotFoundError("Missing sample mechanics dataset for demo workflow.")


def prepare_demo_assets(project_dir: Path) -> dict[str, Path]:
    project_dir = ensure_workspace(project_dir)
    assets_dir = project_dir / "demo_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    source_root = source_project_dir()

    sample_src = sample_mechanics_dataset_path(source_root)
    sample_dst = assets_dir / "sample_mechanics_creep.csv"
    if sample_src.exists() and not sample_dst.exists():
        shutil.copy2(sample_src, sample_dst)

    dataset_src = source_root / "datasets" / "hydrogel_characterization_data"
    dataset_dst = project_dir / "datasets" / "hydrogel_characterization_data"
    if dataset_src.exists() and not dataset_dst.exists():
        dataset_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset_src, dataset_dst)

    note_path = project_dir / "library" / "demo_ecm_note.md"
    if not note_path.exists():
        note_path.write_text(
            dedent(
                """
                # Demo ECM Note

                This workspace includes a demo note about synthetic ECM design for intestinal organoid culture.

                Focus points:
                - PEG and GelMA families are common synthetic or semi-synthetic hydrogel anchors.
                - Mechanics tuning often trades off stiffness, degradability, and ligand presentation.
                - This note exists to help the local-library search path in the demo workflows.
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    return {
        "sample_mechanics_data": sample_dst if sample_dst.exists() else sample_mechanics_dataset_path(project_dir),
        "calibration_dataset_dir": dataset_dst,
        "library_note": note_path,
    }


def default_demo_steps(project_dir: Path, *, include_ai_workflows: bool = True) -> list[DemoStep]:
    assets = prepare_demo_assets(project_dir)
    mechanics_path = assets["sample_mechanics_data"]

    steps: list[DemoStep] = []
    if include_ai_workflows:
        steps.extend(
            [
                DemoStep(
                    slug="team",
                    workflow="team",
                    title="Multi-agent literature synthesis",
                    description="Run the collaborative literature workflow over a synthetic ECM query.",
                    query="synthetic ECM for intestinal organoid culture",
                    params={},
                ),
                DemoStep(
                    slug="single",
                    workflow="single",
                    title="Single-agent literature memo",
                    description="Run the single-agent literature workflow on the same topic for comparison.",
                    query="synthetic ECM for intestinal organoid culture",
                    params={},
                ),
            ]
        )

    steps.extend(
        [
            DemoStep(
                slug="datasets",
                workflow="datasets",
                title="Public dataset discovery",
                description="List and register hydrogel-related public datasets.",
                query="hydrogel rheology",
                params={},
            ),
            DemoStep(
                slug="calibration",
                workflow="calibration",
                title="Calibration priors from normalized data",
                description="Turn hydrogel characterization measurements into family priors.",
                query="Calibrate ECM mechanics priors from hydrogel characterization data",
                params={"dataset_id": "hydrogel_characterization_data", "calibration_max_samples": 4},
            ),
            DemoStep(
                slug="mechanics",
                workflow="mechanics",
                title="Mechanics fitting from sample creep data",
                description="Fit the bundled sample creep dataset with a deterministic constitutive model.",
                query="Model the creep behavior of a synthetic ECM sample",
                params={
                    "data_path": mechanics_path,
                    "experiment_type": "creep",
                    "time_column": "time",
                    "stress_column": "stress",
                    "strain_column": "strain",
                    "applied_stress": 1.0,
                    "delimiter": ",",
                },
            ),
        ]
    )

    if include_ai_workflows:
        steps.append(
            DemoStep(
                slug="hybrid",
                workflow="hybrid",
                title="Hybrid literature + mechanics + simulation",
                description="Combine evidence, mechanics fitting, and simulation in one report.",
                query="design and evaluate a mechanics-informed synthetic ECM for intestinal organoids",
                params={
                    "data_path": mechanics_path,
                    "experiment_type": "creep",
                    "time_column": "time",
                    "stress_column": "stress",
                    "strain_column": "strain",
                    "applied_stress": 1.0,
                    "delimiter": ",",
                },
            )
        )

    steps.extend(
        [
            DemoStep(
                slug="design",
                workflow="design",
                title="Single-window inverse design",
                description="Design one ECM candidate family near 8 Pa.",
                query="Design a GelMA-like ECM near stiffness 8 Pa",
                params={
                    "target_stiffness": 8.0,
                    "target_anisotropy": 0.1,
                    "target_connectivity": 0.95,
                    "target_stress_propagation": 0.5,
                    "constraint_max_anisotropy": 0.35,
                    "constraint_min_connectivity": 0.9,
                    "design_top_k": 3,
                    "design_candidate_budget": 6,
                    "design_monte_carlo_runs": 2,
                },
            ),
            DemoStep(
                slug="design_campaign",
                workflow="design_campaign",
                title="Multi-window design campaign",
                description="Compare whether one material family can span 6, 8, and 10 Pa windows.",
                query="Design a GelMA-like ECM family across 6, 8, and 10 Pa targets",
                params={
                    "campaign_target_stiffnesses": "6,8,10",
                    "target_anisotropy": 0.12,
                    "target_connectivity": 0.95,
                    "target_stress_propagation": 0.5,
                    "constraint_max_anisotropy": 0.35,
                    "constraint_min_connectivity": 0.9,
                    "design_top_k": 3,
                    "design_candidate_budget": 6,
                    "design_monte_carlo_runs": 2,
                },
            ),
            DemoStep(
                slug="benchmark",
                workflow="benchmark",
                title="Benchmark suite",
                description="Run the current mechanics and inverse-design benchmark suite.",
                query="Benchmark the current ECM mechanics core",
                params={},
            ),
        ]
    )
    return steps


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    return value


def build_demo_summary_markdown(
    *,
    demo_id: str,
    include_ai_workflows: bool,
    step_results: list[DemoStepResult],
) -> str:
    total = len(step_results)
    completed = sum(1 for step in step_results if step.status == "ok")
    failed = total - completed
    lines = [
        f"# Full Workflow Demo Summary",
        "",
        f"- Demo ID: `{demo_id}`",
        f"- Include AI Workflows: `{'yes' if include_ai_workflows else 'no'}`",
        f"- Completed Steps: `{completed}/{total}`",
        f"- Failed Steps: `{failed}`",
        "",
    ]
    for index, step in enumerate(step_results, start=1):
        lines.extend(
            [
                f"## {index}. {step.workflow}",
                f"- Title: {step.title}",
                f"- Status: `{step.status}`",
                f"- Report: `{step.report_path}`" if step.report_path else "- Report: `NR`",
                f"- Run Dir: `{step.run_dir}`" if step.run_dir else "- Run Dir: `NR`",
            ]
        )
        if step.error:
            lines.append(f"- Error: `{step.error}`")
        if step.final_summary:
            lines.append("")
            lines.append(step.final_summary.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_full_demo(
    *,
    project_dir: Path,
    include_ai_workflows: bool = True,
    continue_on_error: bool = True,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> DemoRunResult:
    project_dir = ensure_workspace(project_dir)
    prepare_demo_assets(project_dir)
    steps = default_demo_steps(project_dir, include_ai_workflows=include_ai_workflows)
    demo_id = datetime.now().strftime("demo_%Y%m%d_%H%M%S")
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    step_results: list[DemoStepResult] = []
    for index, step in enumerate(steps, start=1):
        if progress_callback is not None:
            progress_callback(
                step.workflow,
                f"[{index}/{len(steps)}] Running {step.workflow}: {step.title}",
            )
        report_name = f"{demo_id}_{step.slug}.md"
        try:
            result = run_research_agent_sync(
                project_dir=project_dir,
                query=step.query,
                workflow=step.workflow,
                report_name=report_name,
                **step.params,
            )
            step_results.append(
                DemoStepResult(
                    slug=step.slug,
                    workflow=step.workflow,
                    title=step.title,
                    status="ok",
                    report_path=result.report_path,
                    run_dir=result.run_dir,
                    final_summary=result.final_summary,
                )
            )
        except Exception as exc:
            step_results.append(
                DemoStepResult(
                    slug=step.slug,
                    workflow=step.workflow,
                    title=step.title,
                    status="error",
                    report_path=None,
                    run_dir=None,
                    final_summary="",
                    error=str(exc),
                )
            )
            if not continue_on_error:
                break

    summary_path = report_dir / f"{demo_id}_summary.md"
    manifest_path = report_dir / f"{demo_id}_manifest.json"

    summary_path.write_text(
        build_demo_summary_markdown(
            demo_id=demo_id,
            include_ai_workflows=include_ai_workflows,
            step_results=step_results,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "demo_id": demo_id,
                "include_ai_workflows": include_ai_workflows,
                "completed_steps": sum(1 for step in step_results if step.status == "ok"),
                "failed_steps": sum(1 for step in step_results if step.status != "ok"),
                "steps": [
                    _normalize_for_json(
                        {
                            **asdict(step_result),
                            "report_path": step_result.report_path,
                            "run_dir": step_result.run_dir,
                        }
                    )
                    for step_result in step_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if progress_callback is not None:
        progress_callback(
            "done",
            f"Demo finished. completed={sum(1 for step in step_results if step.status == 'ok')}/{len(step_results)}. Summary saved to {summary_path}",
        )
    return DemoRunResult(
        demo_id=demo_id,
        include_ai_workflows=include_ai_workflows,
        summary_path=summary_path,
        manifest_path=manifest_path,
        step_results=step_results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-shot demo across ECM workflows.")
    parser.add_argument(
        "--project-dir",
        default=resolve_project_dir(),
        type=Path,
        help="Workspace directory. Defaults to the current ECM project directory.",
    )
    parser.add_argument(
        "--skip-ai-workflows",
        action="store_true",
        help="Skip model-backed workflows such as team, single, and hybrid.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch immediately if any workflow fails.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_full_demo(
        project_dir=args.project_dir,
        include_ai_workflows=not args.skip_ai_workflows,
        continue_on_error=not args.stop_on_error,
        progress_callback=lambda stage, message: print(message, flush=True),
    )
    print(
        f"Demo complete. summary={result.summary_path} manifest={result.manifest_path} completed={result.completed_steps} failed={result.failed_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
