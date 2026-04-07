from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.demo import (  # noqa: E402
    DemoStepResult,
    build_demo_summary_markdown,
    default_demo_steps,
    prepare_demo_assets,
)
from ecm_organoid_agent.workspace import ensure_workspace  # noqa: E402


class DemoWorkflowTests(unittest.TestCase):
    def test_default_demo_steps_cover_all_workflows(self) -> None:
        steps = default_demo_steps(PROJECT_ROOT, include_ai_workflows=True)
        workflows = [step.workflow for step in steps]
        self.assertEqual(
            workflows,
            [
                "team",
                "single",
                "datasets",
                "calibration",
                "mechanics",
                "hybrid",
                "design",
                "design_campaign",
                "benchmark",
            ],
        )

        deterministic_steps = default_demo_steps(PROJECT_ROOT, include_ai_workflows=False)
        deterministic_workflows = [step.workflow for step in deterministic_steps]
        self.assertNotIn("team", deterministic_workflows)
        self.assertNotIn("single", deterministic_workflows)
        self.assertNotIn("hybrid", deterministic_workflows)
        self.assertIn("benchmark", deterministic_workflows)

    def test_prepare_demo_assets_bootstraps_workspace_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = ensure_workspace(Path(tmp_dir))
            assets = prepare_demo_assets(workspace)
            self.assertTrue(Path(assets["sample_mechanics_data"]).exists())
            self.assertTrue(Path(assets["calibration_dataset_dir"]).exists())
            self.assertTrue(Path(assets["library_note"]).exists())

    def test_build_demo_summary_markdown_lists_status_and_paths(self) -> None:
        markdown = build_demo_summary_markdown(
            demo_id="demo_20260401_120000",
            include_ai_workflows=True,
            step_results=[
                DemoStepResult(
                    slug="design",
                    workflow="design",
                    title="Single-window inverse design",
                    status="ok",
                    report_path=Path("/tmp/design.md"),
                    run_dir=Path("/tmp/run_design"),
                    final_summary="Design finished.",
                ),
                DemoStepResult(
                    slug="team",
                    workflow="team",
                    title="Multi-agent literature synthesis",
                    status="error",
                    report_path=None,
                    run_dir=None,
                    final_summary="",
                    error="API unavailable",
                ),
            ],
        )
        self.assertIn("Full Workflow Demo Summary", markdown)
        self.assertIn("design", markdown)
        self.assertIn("API unavailable", markdown)
        self.assertIn("/tmp/design.md", markdown)


if __name__ == "__main__":
    unittest.main()

