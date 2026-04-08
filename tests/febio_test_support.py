from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.febio.builder import BuildArtifacts, build_simulation_input
from ecm_organoid_agent.febio.runner import RunnerResult
from ecm_organoid_agent.febio.schemas import BulkMechanicsRequest


def make_workspace(tmp_dir: str) -> Path:
    project_dir = Path(tmp_dir)
    for dirname in ("memory", "library", "reports", "templates", "runs", ".cache"):
        (project_dir / dirname).mkdir()
    return project_dir


def write_fake_bulk_output_bundle(simulation_dir: Path) -> tuple[BuildArtifacts, RunnerResult]:
    request = BulkMechanicsRequest(
        matrix_youngs_modulus=8.0,
        matrix_poisson_ratio=0.3,
        sample_dimensions=(1.0, 1.0, 1.0),
        prescribed_displacement=-0.1,
        target_stiffness=8.0,
    )
    build = build_simulation_input(request, simulation_dir)
    (simulation_dir / "node_displacement.log").write_text(
        "\n".join(
            [
                "*Step  = 0",
                "*Time  = 0",
                "*Data  = ux;uy;uz",
                "1,0,0,0",
                "2,0,0,0",
                "*Step  = 1",
                "*Time  = 1",
                "*Data  = ux;uy;uz",
                "1,0,0,0",
                "2,0,0,-0.1",
            ]
        ),
        encoding="utf-8",
    )
    (simulation_dir / "top_reaction.log").write_text(
        "\n".join(
            [
                "*Step  = 1",
                "*Time  = 1",
                "*Data  = Rx;Ry;Rz",
                "2,0,0,-0.8",
            ]
        ),
        encoding="utf-8",
    )
    (simulation_dir / "element_principal_stress.log").write_text(
        "\n".join(
            [
                "*Step  = 1",
                "*Time  = 1",
                "*Data  = s1;s2;s3",
                "1,0.1,0.0,-0.4",
            ]
        ),
        encoding="utf-8",
    )
    (simulation_dir / "input.log").write_text(
        "N O R M A L   T E R M I N A T I O N\n",
        encoding="utf-8",
    )
    runner = RunnerResult(
        status="succeeded",
        returncode=0,
        command=["febio4", "-i", "input.feb", "-silent"],
        duration_seconds=0.1,
        stdout_path=simulation_dir / "febio_stdout.txt",
        stderr_path=simulation_dir / "febio_stderr.txt",
        log_path=simulation_dir / "input.log",
        xplt_path=simulation_dir / "input.xplt",
        dmp_path=simulation_dir / "input.dmp",
        metadata_path=simulation_dir / "runner_metadata.json",
    )
    runner.stdout_path.write_text("", encoding="utf-8")
    runner.stderr_path.write_text("", encoding="utf-8")
    return build, runner


def load_build_metadata(build: BuildArtifacts) -> dict:
    return json.loads(build.metadata_path.read_text(encoding="utf-8"))
