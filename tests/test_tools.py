from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.config import AppConfig
from ecm_organoid_agent.tools import (
    build_tools,
    extract_abstract_excerpt,
    guess_title_from_text,
    normalize_crossref_item,
    save_markdown_report,
    search_library,
)


class SearchLibraryTests(unittest.TestCase):
    def test_search_library_finds_markdown_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            library_dir = Path(tmp_dir)
            (library_dir / "note.md").write_text(
                (
                    "Synthetic ECM for Intestinal Organoid Culture\n\n"
                    "Abstract\n"
                    "Synthetic extracellular matrix improves intestinal organoid morphology.\n"
                    "Introduction\nMore text."
                ),
                encoding="utf-8",
            )
            (library_dir / "other.txt").write_text(
                "This note discusses hydrogel stiffness and organoid formation.",
                encoding="utf-8",
            )
            (library_dir / "ignore.csv").write_text("not indexed", encoding="utf-8")

            result = search_library("extracellular matrix organoid", library_dir, max_results=5)

            matches = result["matches"]
            self.assertEqual(len(matches), 2)
            self.assertEqual(matches[0]["file_name"], "note.md")
            self.assertEqual(matches[0]["title_guess"], "Synthetic ECM for Intestinal Organoid Culture")
            self.assertIn("Synthetic extracellular matrix", matches[0]["abstract_excerpt"])

    def test_search_library_reports_missing_directory(self) -> None:
        result = search_library("organoid", Path("/tmp/definitely_missing_for_test"), max_results=5)
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["notes"])


class SaveReportTests(unittest.TestCase):
    def test_save_report_enforces_markdown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_markdown_report(Path(tmp_dir), "weekly review", "# Title")
            self.assertTrue(path.name.endswith(".md"))
            self.assertEqual(path.read_text(encoding="utf-8"), "# Title")

    def test_search_local_library_logs_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "memory").mkdir()
            (project_dir / "library").mkdir()
            (project_dir / "reports").mkdir()
            (project_dir / "templates").mkdir()
            (project_dir / "runs").mkdir()
            (project_dir / ".cache").mkdir()
            run_dir = project_dir / "runs" / "example_run"
            run_dir.mkdir(parents=True)
            (project_dir / "library" / "note.md").write_text(
                "Synthetic extracellular matrix improves intestinal organoid morphology.",
                encoding="utf-8",
            )

            config = AppConfig(
                project_dir=project_dir,
                memory_dir=project_dir / "memory",
                library_dir=project_dir / "library",
                report_dir=project_dir / "reports",
                runs_dir=project_dir / "runs",
                cache_dir=project_dir / ".cache",
                template_dir=project_dir / "templates",
                model_provider="deepseek",
                model="deepseek-chat",
                model_api_key="placeholder",
                model_base_url="https://api.deepseek.com/v1",
                active_run_dir=run_dir,
                cache_ttl_hours=168,
                max_pubmed_results=5,
                ncbi_email=None,
                ncbi_api_key=None,
                crossref_mailto=None,
                frontend_require_login=False,
                frontend_username=None,
                frontend_password=None,
                frontend_password_sha256=None,
                frontend_public_host="0.0.0.0",
                frontend_public_port=8525,
            )
            tools = {tool.__name__: tool for tool in build_tools(config)}

            result = tools["search_local_library"]("synthetic extracellular matrix", 3)

            self.assertIn("note.md", result)
            log_path = run_dir / "tool_calls.jsonl"
            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(lines[-1])
            self.assertEqual(payload["tool_name"], "search_local_library")
            self.assertEqual(payload["status"], "ok")

    def test_design_fiber_network_candidates_returns_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            (project_dir / "memory").mkdir()
            (project_dir / "library").mkdir()
            (project_dir / "reports").mkdir()
            (project_dir / "templates").mkdir()
            (project_dir / "runs").mkdir()
            (project_dir / ".cache").mkdir()
            run_dir = project_dir / "runs" / "design_run"
            run_dir.mkdir(parents=True)

            config = AppConfig(
                project_dir=project_dir,
                memory_dir=project_dir / "memory",
                library_dir=project_dir / "library",
                report_dir=project_dir / "reports",
                runs_dir=project_dir / "runs",
                cache_dir=project_dir / ".cache",
                template_dir=project_dir / "templates",
                model_provider="deepseek",
                model="deepseek-chat",
                model_api_key="placeholder",
                model_base_url="https://api.deepseek.com/v1",
                active_run_dir=run_dir,
                cache_ttl_hours=168,
                max_pubmed_results=5,
                ncbi_email=None,
                ncbi_api_key=None,
                crossref_mailto=None,
                frontend_require_login=False,
                frontend_username=None,
                frontend_password=None,
                frontend_password_sha256=None,
                frontend_public_host="0.0.0.0",
                frontend_public_port=8525,
            )
            tools = {tool.__name__: tool for tool in build_tools(config)}

            result = tools["design_fiber_network_candidates"](
                target_stiffness=8.0,
                target_anisotropy=0.1,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                constraint_max_anisotropy=0.3,
                constraint_min_connectivity=0.9,
                top_k=3,
                candidate_budget=6,
                monte_carlo_runs=2,
                total_force=0.2,
                axis="x",
                boundary_fraction=0.15,
                seed=1234,
                max_iterations=200,
                tolerance=1e-5,
                target_nodes=8,
                search_space_json="",
            )

            payload = json.loads(result)
            self.assertEqual(payload["top_k"], 3)
            self.assertEqual(len(payload["top_candidates"]), 3)
            self.assertIn("constraints", payload)
            log_path = run_dir / "tool_calls.jsonl"
            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(json.loads(lines[-1])["tool_name"], "design_fiber_network_candidates")


class ParsingHelperTests(unittest.TestCase):
    def test_extract_abstract_excerpt_stops_before_introduction(self) -> None:
        text = (
            "Synthetic Matrices for Organoid Engineering\n"
            "Abstract\n"
            "Synthetic hydrogels can improve reproducibility in intestinal organoid culture.\n"
            "They also enable tunable stiffness and ligand density.\n"
            "Introduction\n"
            "This is background text that should not be included."
        )
        excerpt = extract_abstract_excerpt(text)
        self.assertIn("Synthetic hydrogels can improve reproducibility", excerpt)
        self.assertNotIn("background text", excerpt)

    def test_guess_title_from_text_uses_first_meaningful_line(self) -> None:
        text = "\n\nSynthetic Biomaterials for Liver Organoids\nAbstract\nA short abstract."
        self.assertEqual(
            guess_title_from_text(text, "fallback"),
            "Synthetic Biomaterials for Liver Organoids",
        )

    def test_normalize_crossref_item_cleans_abstract_and_date(self) -> None:
        item = {
            "title": ["Defined matrices for organoid culture"],
            "container-title": ["Nature Materials"],
            "published-online": {"date-parts": [[2026, 3, 1]]},
            "type": "journal-article",
            "DOI": "10.1000/example",
            "URL": "https://doi.org/10.1000/example",
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "abstract": "<jats:p>Defined matrices improve <b>organoid</b> reproducibility.</jats:p>",
        }
        record = normalize_crossref_item(item)
        self.assertEqual(record["published"], "2026-3-1")
        self.assertEqual(record["authors"], ["Ada Lovelace"])
        self.assertIn("improve organoid reproducibility", record["abstract_excerpt"])


if __name__ == "__main__":
    unittest.main()
