from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import write_json
from .config import FEBioConfig


@dataclass(frozen=True)
class RunnerResult:
    status: str
    returncode: int | None
    command: list[str]
    duration_seconds: float
    stdout_path: Path
    stderr_path: Path
    log_path: Path
    xplt_path: Path
    dmp_path: Path
    metadata_path: Path
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "log_path": str(self.log_path),
            "xplt_path": str(self.xplt_path),
            "dmp_path": str(self.dmp_path),
            "runner_metadata_path": str(self.metadata_path),
            "error_message": self.error_message,
        }


def build_febio_command(executable: str, input_path: Path) -> list[str]:
    return [executable, "-i", input_path.name, "-silent"]


def run_febio_job(
    *,
    febio_config: FEBioConfig,
    simulation_dir: Path,
    input_path: Path,
) -> RunnerResult:
    """Execute FEBio in a controlled directory and persist stdout/stderr/metadata."""

    stdout_path = simulation_dir / "febio_stdout.txt"
    stderr_path = simulation_dir / "febio_stderr.txt"
    log_path = simulation_dir / "input.log"
    xplt_path = simulation_dir / "input.xplt"
    dmp_path = simulation_dir / "input.dmp"
    metadata_path = simulation_dir / "runner_metadata.json"

    if not febio_config.available or not febio_config.executable:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        result = RunnerResult(
            status="unavailable",
            returncode=None,
            command=[],
            duration_seconds=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=log_path,
            xplt_path=xplt_path,
            dmp_path=dmp_path,
            metadata_path=metadata_path,
            error_message=febio_config.status_message,
        )
        write_json(metadata_path, result.to_dict())
        return result

    command = build_febio_command(febio_config.executable, input_path)
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=simulation_dir,
            capture_output=True,
            text=True,
            timeout=febio_config.timeout_seconds,
            check=False,
        )
        duration = time.time() - started
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        status = "succeeded" if completed.returncode == 0 else "failed"
        error_message = None if status == "succeeded" else f"FEBio exited with return code {completed.returncode}."
        result = RunnerResult(
            status=status,
            returncode=completed.returncode,
            command=command,
            duration_seconds=duration,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=log_path,
            xplt_path=xplt_path,
            dmp_path=dmp_path,
            metadata_path=metadata_path,
            error_message=error_message,
        )
        write_json(metadata_path, result.to_dict())
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        result = RunnerResult(
            status="timed_out",
            returncode=None,
            command=command,
            duration_seconds=duration,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=log_path,
            xplt_path=xplt_path,
            dmp_path=dmp_path,
            metadata_path=metadata_path,
            error_message=f"FEBio timed out after {febio_config.timeout_seconds} seconds.",
        )
        write_json(metadata_path, result.to_dict())
        return result
