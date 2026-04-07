from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_WORKSPACE_NAME = "ECM-Organoid-Research-Desk"
DEFAULT_RESOURCE_FILES = [
    ".env.example",
    "README.md",
    "memory/research_profile.md",
    "memory/focus_questions.md",
    "templates/weekly_report_template.md",
    "library/README.md",
]


def source_project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def bundled_resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return source_project_dir()


def default_user_workspace() -> Path:
    return Path.home() / DEFAULT_WORKSPACE_NAME


def resolve_project_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if os.getenv("ECM_ORGANOID_PROJECT_DIR"):
        return Path(os.environ["ECM_ORGANOID_PROJECT_DIR"]).expanduser().resolve()
    return source_project_dir()


def ensure_workspace(path: Path | None = None) -> Path:
    workspace = resolve_project_dir(path or _workspace_candidate())
    workspace.mkdir(parents=True, exist_ok=True)
    for subdir in ("memory", "library", "reports", "templates"):
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    resource_dir = bundled_resource_dir()
    for relative in DEFAULT_RESOURCE_FILES:
        src = resource_dir / relative
        dst = workspace / relative
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return workspace


def _workspace_candidate() -> Path | None:
    env_value = os.getenv("ECM_ORGANOID_PROJECT_DIR")
    if env_value:
        return Path(env_value)
    if getattr(sys, "frozen", False):
        return default_user_workspace()
    return None
