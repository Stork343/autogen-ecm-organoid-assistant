from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FEBioConfig:
    """Runtime configuration for local FEBio execution."""

    enabled: bool
    executable: str | None
    timeout_seconds: int
    default_tmp_dir: Path
    available: bool
    status_message: str

    @classmethod
    def from_env(cls, project_dir: Path | None = None) -> "FEBioConfig":
        enabled = _env_flag("FEBIO_ENABLED", default=True)
        explicit_executable = os.getenv("FEBIO_EXECUTABLE")
        executable = explicit_executable or shutil.which("febio4") or shutil.which("febio")
        timeout_seconds = int(os.getenv("FEBIO_TIMEOUT_SECONDS", "300"))
        tmp_dir_env = os.getenv("FEBIO_DEFAULT_TMP_DIR")
        default_tmp_dir = (
            Path(tmp_dir_env).expanduser()
            if tmp_dir_env
            else ((project_dir / ".cache" / "febio_tmp") if project_dir else Path(tempfile.gettempdir()) / "ecm_organoid_agent_febio")
        )

        if not enabled:
            return cls(
                enabled=False,
                executable=executable,
                timeout_seconds=max(1, timeout_seconds),
                default_tmp_dir=default_tmp_dir,
                available=False,
                status_message="FEBio support was disabled by `FEBIO_ENABLED`.",
            )
        if executable is None:
            return cls(
                enabled=True,
                executable=None,
                timeout_seconds=max(1, timeout_seconds),
                default_tmp_dir=default_tmp_dir,
                available=False,
                status_message=(
                    "FEBio executable was not found. Set `FEBIO_EXECUTABLE` or install FEBio to enable FE-backed evaluation."
                ),
            )
        return cls(
            enabled=True,
            executable=str(Path(executable).expanduser()),
            timeout_seconds=max(1, timeout_seconds),
            default_tmp_dir=default_tmp_dir,
            available=True,
            status_message=f"FEBio available at {Path(executable).expanduser()}",
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "executable": self.executable,
            "timeout_seconds": self.timeout_seconds,
            "default_tmp_dir": str(self.default_tmp_dir),
            "available": self.available,
            "status_message": self.status_message,
        }
