from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_for_path(value: str, *, max_length: int = 48) -> str:
    normalized = re.sub(r"\s+", "_", value.strip().lower())
    normalized = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE).strip("._")
    if not normalized:
        return "research_run"
    return normalized[:max_length].rstrip("._")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def tool_log_path(self) -> Path:
        return self.run_dir / "tool_calls.jsonl"


def create_run_artifacts(
    *,
    runs_dir: Path,
    query: str,
    workflow: str,
) -> RunArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{workflow}_{slugify_for_path(query, max_length=36)}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_id=run_id, run_dir=run_dir)


def write_stage_text(run_dir: Path, stage_name: str, content: str) -> Path:
    path = run_dir / f"{stage_name}.md"
    path.write_text(content.strip() + ("\n" if content.strip() else ""), encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
