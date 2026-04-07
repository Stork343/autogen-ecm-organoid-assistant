from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.artifacts import create_run_artifacts, slugify_for_path, write_stage_text


class ArtifactTests(unittest.TestCase):
    def test_slugify_for_path_keeps_readable_token(self) -> None:
        slug = slugify_for_path("Synthetic ECM for Intestinal Organoid Culture")
        self.assertIn("synthetic_ecm", slug)
        self.assertNotIn(" ", slug)

    def test_create_run_artifacts_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = create_run_artifacts(
                runs_dir=Path(tmp_dir),
                query="synthetic ECM for intestinal organoid culture",
                workflow="team",
            )
            self.assertTrue(artifacts.run_dir.exists())
            self.assertIn("_team_", artifacts.run_id)

    def test_write_stage_text_persists_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_stage_text(Path(tmp_dir), "search_agent", "## Search Coverage")
            self.assertTrue(path.exists())
            self.assertIn("Search Coverage", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
