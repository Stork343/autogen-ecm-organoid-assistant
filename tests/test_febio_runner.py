from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.febio_test_support import PROJECT_ROOT  # noqa: F401

from ecm_organoid_agent.febio.config import FEBioConfig
from ecm_organoid_agent.febio.runner import build_febio_command, run_febio_job


class FEBioRunnerTests(unittest.TestCase):
    def test_build_febio_command_uses_fixed_safe_arguments(self) -> None:
        command = build_febio_command("/path/to/febio4", Path("/tmp/input.feb"))
        self.assertEqual(command, ["/path/to/febio4", "-i", "input.feb", "-silent"])

    def test_run_febio_job_handles_unavailable_runtime_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            input_path = simulation_dir / "input.feb"
            input_path.write_text("<febio_spec version='4.0'/>", encoding="utf-8")
            config = FEBioConfig(
                enabled=True,
                executable=None,
                timeout_seconds=30,
                default_tmp_dir=simulation_dir,
                available=False,
                status_message="FEBio missing for test.",
            )
            result = run_febio_job(
                febio_config=config,
                simulation_dir=simulation_dir,
                input_path=input_path,
            )
            self.assertEqual(result.status, "unavailable")
            self.assertTrue((simulation_dir / "runner_metadata.json").exists())

    def test_run_febio_job_constructs_expected_subprocess_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            input_path = simulation_dir / "input.feb"
            input_path.write_text("<febio_spec version='4.0'/>", encoding="utf-8")
            config = FEBioConfig(
                enabled=True,
                executable="/path/to/febio4",
                timeout_seconds=45,
                default_tmp_dir=simulation_dir,
                available=True,
                status_message="ok",
            )
            completed = type("Completed", (), {"stdout": "", "stderr": "", "returncode": 0})()
            with patch("ecm_organoid_agent.febio.runner.subprocess.run", return_value=completed) as mock_run:
                result = run_febio_job(
                    febio_config=config,
                    simulation_dir=simulation_dir,
                    input_path=input_path,
                )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(mock_run.call_args.kwargs["cwd"], simulation_dir)
            self.assertEqual(mock_run.call_args.kwargs["timeout"], 45)
            self.assertEqual(mock_run.call_args.args[0], ["/path/to/febio4", "-i", "input.feb", "-silent"])

    def test_run_febio_job_handles_timeout_and_writes_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_dir = Path(tmp_dir)
            input_path = simulation_dir / "input.feb"
            input_path.write_text("<febio_spec version='4.0'/>", encoding="utf-8")
            config = FEBioConfig(
                enabled=True,
                executable="/path/to/febio4",
                timeout_seconds=2,
                default_tmp_dir=simulation_dir,
                available=True,
                status_message="ok",
            )
            from subprocess import TimeoutExpired

            with patch(
                "ecm_organoid_agent.febio.runner.subprocess.run",
                side_effect=TimeoutExpired(cmd=["/path/to/febio4"], timeout=2, output="out", stderr="err"),
            ):
                result = run_febio_job(
                    febio_config=config,
                    simulation_dir=simulation_dir,
                    input_path=input_path,
                )
            self.assertEqual(result.status, "timed_out")
            self.assertTrue(result.stdout_path.exists())
            self.assertTrue(result.stderr_path.exists())


if __name__ == "__main__":
    unittest.main()
