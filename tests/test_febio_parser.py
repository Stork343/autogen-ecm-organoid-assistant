from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.febio_test_support import PROJECT_ROOT, write_fake_bulk_output_bundle  # noqa: F401

from ecm_organoid_agent.febio.builder import build_simulation_input
from ecm_organoid_agent.febio.parser import parse_simulation_outputs
from ecm_organoid_agent.febio.runner import RunnerResult
from ecm_organoid_agent.febio.schemas import BulkMechanicsRequest


class FEBioParserTests(unittest.TestCase):
    def test_parser_extracts_last_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build, runner = write_fake_bulk_output_bundle(simulation_dir)
            result = parse_simulation_outputs(build_artifacts=build, runner_result=runner)
            self.assertTrue(result["extracted_fields"]["solver_converged"])
            self.assertEqual(result["extracted_fields"]["top_reaction"]["final"]["ids"], [2])

    def test_parser_reports_missing_files_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            build = build_simulation_input(
                BulkMechanicsRequest(
                    matrix_youngs_modulus=8.0,
                    matrix_poisson_ratio=0.3,
                    sample_dimensions=(1.0, 1.0, 1.0),
                    prescribed_displacement=-0.1,
                ),
                simulation_dir,
            )
            runner = RunnerResult(
                status="failed",
                returncode=1,
                command=["febio4", "-i", "input.feb", "-silent"],
                duration_seconds=0.1,
                stdout_path=simulation_dir / "febio_stdout.txt",
                stderr_path=simulation_dir / "febio_stderr.txt",
                log_path=simulation_dir / "input.log",
                xplt_path=simulation_dir / "input.xplt",
                dmp_path=simulation_dir / "input.dmp",
                metadata_path=simulation_dir / "runner_metadata.json",
                error_message="runner failed",
            )
            runner.stdout_path.write_text("negative jacobian", encoding="utf-8")
            runner.stderr_path.write_text("error termination", encoding="utf-8")
            result = parse_simulation_outputs(build_artifacts=build, runner_result=runner)
            self.assertIn("FEBio logfile is missing.", result["warnings"])
            self.assertIn("FEBio plotfile is missing.", result["warnings"])
            self.assertIn("FEBio restart dump is missing.", result["warnings"])
            self.assertIn("FEBio reported negative Jacobians.", result["warnings"])
            self.assertEqual(result["error_message"], "runner failed")


if __name__ == "__main__":
    unittest.main()
