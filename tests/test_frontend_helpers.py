from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.frontend import (
    available_stage_files,
    auth_cookie_name,
    build_auth_cookie_value,
    benchmark_demo_rows,
    browser_entry_rows,
    build_dashboard_summary,
    campaign_result_rows,
    calibration_run_options,
    calibration_run_snapshot,
    collaboration_coverage_rows,
    demo_calibration_rows,
    demo_dataset_rows,
    demo_highlight_rows,
    credentials_valid,
    design_comparison_rows,
    design_best_candidate,
    design_campaign_run_options,
    design_campaign_snapshot,
    design_candidate_rows,
    formulation_recommendation_rows,
    design_run_options,
    design_run_snapshot,
    design_run_summary,
    design_sensitivity_rows,
    design_stage_files,
    formulation_family_distribution,
    guide_markdown,
    latest_campaign_overview,
    latest_calibration_overview,
    latest_design_summary,
    latest_tool_result,
    load_json_file,
    load_jsonl_file,
    mapping_rows,
    markdown_section_excerpt,
    mechanics_demo_snapshot,
    find_markdown_section,
    parse_markdown_sections,
    password_matches,
    project_flow_dot,
    report_excerpt,
    recent_report_rows,
    render_stage_lines,
    system_layer_rows,
    validate_auth_cookie_value,
    workflow_category_rows,
    workflow_flow_family_specs,
    workflow_artifact_labels,
    workflow_catalog_rows,
    workflow_examples,
)
from ecm_organoid_agent.config import AppConfig
from ecm_organoid_agent.workspace import ensure_workspace
from ecm_organoid_agent.runner import ResearchRunResult


class FrontendHelperTests(unittest.TestCase):
    def test_render_stage_lines_includes_all_messages(self) -> None:
        html = render_stage_lines(
            [
                ("search", "[1/4] SearchAgent collecting evidence..."),
                ("writer", "[4/4] WriterAgent drafting and saving report..."),
            ]
        )
        self.assertIn("Search", html)
        self.assertIn("Writer", html)
        self.assertIn("collecting evidence", html)

    def test_result_dataclass_stores_report_path(self) -> None:
        result = ResearchRunResult(
            workflow="team",
            report_path=Path("/tmp/example.md"),
            final_summary="done",
        )
        self.assertEqual(result.report_path.name, "example.md")

    def test_ensure_workspace_creates_expected_dirs(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_workspace"
        if temp_root.exists():
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temp_root.rmdir()
        workspace = ensure_workspace(temp_root)
        self.assertTrue((workspace / "memory").exists())
        self.assertTrue((workspace / "library").exists())
        self.assertTrue((workspace / "reports").exists())
        self.assertTrue((workspace / "templates").exists())
        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_load_json_helpers_read_metadata_and_tool_log(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_dashboard"
        ensure_workspace(temp_root)
        metadata_path = temp_root / "runs" / "sample" / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({"workflow": "team"}), encoding="utf-8")
        tool_log = metadata_path.parent / "tool_calls.jsonl"
        tool_log.write_text(json.dumps({"tool_name": "search_pubmed"}) + "\n", encoding="utf-8")

        self.assertEqual(load_json_file(metadata_path)["workflow"], "team")
        self.assertEqual(load_jsonl_file(tool_log)[0]["tool_name"], "search_pubmed")

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_build_dashboard_summary_counts_runs_reports_and_cache(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_summary"
        ensure_workspace(temp_root)
        (temp_root / "reports" / "a.md").write_text("# x", encoding="utf-8")
        (temp_root / ".cache" / "pubmed").mkdir(parents=True, exist_ok=True)
        (temp_root / ".cache" / "pubmed" / "one.json").write_text("{}", encoding="utf-8")
        (temp_root / "runs" / "r1").mkdir(parents=True, exist_ok=True)

        config = AppConfig(
            project_dir=temp_root,
            memory_dir=temp_root / "memory",
            library_dir=temp_root / "library",
            report_dir=temp_root / "reports",
            runs_dir=temp_root / "runs",
            cache_dir=temp_root / ".cache",
            template_dir=temp_root / "templates",
            model_provider="deepseek",
            model="deepseek-chat",
            model_api_key="x",
            model_base_url="https://api.deepseek.com/v1",
            active_run_dir=None,
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
        summary = build_dashboard_summary(config)
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["report_count"], 1)
        self.assertEqual(summary["cache_count"], 1)

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_guide_markdown_reads_docs_file(self) -> None:
        guide = guide_markdown()
        self.assertIn("ECM Research Desk Guide", guide)

    def test_workflow_catalog_and_layers_cover_full_product_surface(self) -> None:
        catalog = workflow_catalog_rows()
        workflow_names = {row["workflow"] for row in catalog}
        self.assertEqual(len(catalog), 9)
        self.assertIn("benchmark", workflow_names)
        self.assertIn("datasets", workflow_names)
        self.assertIn("calibration", workflow_names)
        self.assertIn("benchmark_summary", workflow_artifact_labels("benchmark"))
        self.assertIn("dataset_manifest_snapshot", workflow_artifact_labels("datasets"))

        layers = system_layer_rows()
        self.assertEqual(len(layers), 4)
        self.assertEqual(layers[0]["layer"], "Experience Layer")
        self.assertIn("runner.py", layers[1]["modules"])
        categories = workflow_category_rows()
        self.assertEqual(len(categories), 4)
        self.assertEqual(categories[0]["category"], "Evidence")
        self.assertIn("design_campaign", categories[2]["workflows"])
        flow_families = workflow_flow_family_specs()
        self.assertEqual(len(flow_families), 4)
        self.assertIn("hybrid", flow_families[1]["workflows"])
        self.assertIn("benchmark", flow_families[3]["stages"])
        dot = project_flow_dot()
        self.assertIn("run_research_agent", dot)
        self.assertIn("Evidence", dot)
        self.assertIn("Outputs", dot)

    def test_mapping_rows_formats_scalars_lists_and_dicts(self) -> None:
        rows = mapping_rows(
            {
                "physics_valid": True,
                "stiffness_mean": 8.123456,
                "families": ["GelMA", "NorHA"],
                "nested": {"a": 1},
            }
        )
        values = {row["field"]: row["value"] for row in rows}
        self.assertEqual(values["physics valid"], "yes")
        self.assertEqual(values["stiffness mean"], "8.1235")
        self.assertEqual(values["families"], "GelMA, NorHA")
        self.assertIn('"a": 1', values["nested"])

    def test_browser_rows_and_examples_cover_expected_paths(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_browser_rows"
        ensure_workspace(temp_root)
        (temp_root / "folder_a").mkdir(exist_ok=True)
        (temp_root / "file_a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        rows = browser_entry_rows(temp_root, selection_mode="file")
        labels = {row["label"] for row in rows}
        self.assertIn("[DIR] folder_a", labels)
        self.assertIn("[FILE] file_a.csv", labels)

        examples = workflow_examples()
        self.assertIn("design_campaign", examples)
        self.assertIn("outputs", examples["mechanics"])

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_demo_snapshot_helpers_extract_dataset_calibration_mechanics_and_benchmark(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_demo_snapshots"
        ensure_workspace(temp_root)
        run_dir = temp_root / "runs" / "demo_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "dataset_manifest_snapshot.json").write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "dataset_id": "demo_set",
                            "title": "Demo Dataset",
                            "source": "Mendeley",
                            "status": "manual_download_required",
                            "file_count": 0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "calibration_results.json").write_text(
            json.dumps(
                {
                    "family_priors": [
                        {
                            "material_family": "GelMA",
                            "sample_count": 4,
                            "mean_abs_error": 12.3,
                            "parameter_priors": {
                                "fiber_density": {"mean": 0.4},
                                "fiber_stiffness": {"mean": 9.8},
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "overall_pass": False,
                        "solver_pass_rate": 1.0,
                        "load_ladder_monotonic": True,
                        "scaling_pass_count": 3,
                        "inverse_design_mean_abs_error": 0.56,
                        "identifiability_risk": "high",
                        "fit_mean_relative_error": 0.01,
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "tool_calls.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "tool_name": "fit_mechanics_model",
                            "status": "ok",
                            "result": json.dumps(
                                {
                                    "experiment_type": "creep",
                                    "sample_count": 6,
                                    "fit": {
                                        "elastic_modulus": 0.54,
                                        "viscosity": 0.82,
                                        "relaxation_time": 1.5,
                                        "mse": 0.004,
                                    },
                                }
                            ),
                        }
                    ),
                    json.dumps(
                        {
                            "tool_name": "simulate_mechanics_model",
                            "status": "ok",
                            "result": json.dumps({"x_values": [0, 1], "y_values": [0, 1]}),
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = run_dir / "report.md"
        report_path.write_text("# Demo\n\nline1\nline2\n", encoding="utf-8")

        self.assertEqual(demo_dataset_rows(run_dir)[0]["dataset_id"], "demo_set")
        self.assertEqual(demo_calibration_rows(run_dir)[0]["material_family"], "GelMA")
        mechanics = mechanics_demo_snapshot(run_dir)
        self.assertEqual(mechanics["fit"]["fit"]["elastic_modulus"], 0.54)
        benchmark_rows = benchmark_demo_rows(run_dir)
        self.assertEqual(benchmark_rows[0]["metric"], "overall_pass")
        self.assertIn("# Demo", report_excerpt(report_path))

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_markdown_section_helpers_and_demo_highlights_cover_ai_reports(self) -> None:
        sample_report = (
            "# Demo Report\n\n"
            "## 1. 本周研究问题\n- synthetic ECM for intestinal organoid culture\n\n"
            "## 6. 研究判断与模式总结\n1. PEG is the leading candidate.\n2. Mechanics tuning matters.\n\n"
            "## 9. 下周行动清单\n1. Validate stiffness window.\n"
        )
        sections = parse_markdown_sections(sample_report)
        self.assertEqual(len(sections), 3)
        judgement = find_markdown_section(sample_report, ["研究判断与模式总结"])
        self.assertIn("PEG is the leading candidate", judgement["content"])
        excerpt = markdown_section_excerpt(judgement, max_lines=2)
        self.assertIn("PEG is the leading candidate", excerpt)

        temp_root = PROJECT_ROOT / ".tmp_test_ai_highlights"
        ensure_workspace(temp_root)
        report_path = temp_root / "reports" / "team_demo.md"
        report_path.write_text(sample_report, encoding="utf-8")
        rows = demo_highlight_rows(
            [
                {
                    "workflow": "team",
                    "report_path": str(report_path),
                    "run_dir": "",
                    "status": "ok",
                }
            ]
        )
        self.assertEqual(rows[0]["workflow"], "team")
        self.assertIn("PEG is the leading candidate", rows[0]["highlight"])

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_console_helpers_detect_stage_files(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_console"
        ensure_workspace(temp_root)
        run_dir = temp_root / "runs" / "r2"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "planner_agent.md").write_text("plan", encoding="utf-8")
        (run_dir / "critic_agent.md").write_text("critic", encoding="utf-8")
        (run_dir / "design_agent.md").write_text("design", encoding="utf-8")
        (run_dir / "formulation_mapping.md").write_text("formulation", encoding="utf-8")
        (run_dir / "design_summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "calibration_targets.md").write_text("targets", encoding="utf-8")
        (run_dir / "calibration_results.json").write_text("{}", encoding="utf-8")
        (run_dir / "calibration_impact.json").write_text("{}", encoding="utf-8")
        (run_dir / "dataset_manifest_snapshot.json").write_text("{}", encoding="utf-8")
        (run_dir / "benchmark_summary.json").write_text("{}", encoding="utf-8")

        stage_map = available_stage_files(run_dir)
        self.assertIn("planner_agent", stage_map)
        self.assertIn("design_agent", stage_map)
        self.assertIn("formulation_mapping", stage_map)
        self.assertIn("design_summary", stage_map)
        self.assertIn("calibration_targets", stage_map)
        self.assertIn("calibration_results", stage_map)
        self.assertIn("calibration_impact", stage_map)
        self.assertIn("dataset_manifest_snapshot", stage_map)
        self.assertIn("benchmark_summary", stage_map)
        coverage = collaboration_coverage_rows(run_dir)
        present = {row["stage"]: row["present"] for row in coverage}
        self.assertEqual(present["planner_agent"], "yes")
        self.assertEqual(present["search_agent"], "no")
        self.assertEqual(present["design_agent"], "yes")
        self.assertEqual(present["formulation_mapping"], "yes")
        self.assertEqual(present["design_summary"], "yes")
        self.assertEqual(present["calibration_targets"], "yes")
        self.assertEqual(present["calibration_results"], "yes")
        self.assertEqual(present["calibration_impact"], "yes")
        self.assertEqual(present["dataset_manifest_snapshot"], "yes")
        self.assertEqual(present["benchmark_summary"], "yes")
        self.assertIn("design_agent", design_stage_files(run_dir))

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_latest_tool_result_extracts_latest_json_payload(self) -> None:
        rows = [
            {"tool_name": "run_fiber_network_simulation", "status": "ok", "result": "{\"stiffness_mean\": 1.0}"},
            {"tool_name": "run_fiber_network_parameter_scan", "status": "ok", "result": "{\"sensitivity_ranking\": []}"},
            {"tool_name": "run_fiber_network_simulation", "status": "ok", "result": "{\"stiffness_mean\": 2.0}"},
        ]
        payload = latest_tool_result(rows, "run_fiber_network_simulation")
        self.assertEqual(payload["stiffness_mean"], 2.0)

    def test_design_helpers_extract_snapshot_and_rows(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_design_helpers"
        ensure_workspace(temp_root)
        run_dir = temp_root / "runs" / "design_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(json.dumps({"workflow": "design"}), encoding="utf-8")
        design_summary = {
            "workflow": "design",
            "targets": {"stiffness": 8.0, "anisotropy": 0.1, "connectivity": 0.95},
            "constraints": {"max_anisotropy": 0.3},
            "validation_payload": {"physics_valid": True, "solver_converged": True, "monotonicity_valid": True},
            "design_assessment": {
                "status": "ready_for_screening",
                "recommended_for_screening": True,
                "metrics": {"stiffness_rel_error": 0.25},
            },
            "design_payload": {
                "top_candidates": [
                    {
                        "rank": 1,
                        "score": 0.1,
                        "feasible": True,
                        "parameters": {
                            "fiber_density": 0.35,
                            "fiber_stiffness": 8.0,
                            "bending_stiffness": 0.2,
                            "crosslink_prob": 0.45,
                            "domain_size": 1.0,
                        },
                        "features": {
                            "stiffness_mean": 6.0,
                            "anisotropy": 0.2,
                            "connectivity": 1.0,
                            "stress_propagation": 0.9,
                            "risk_index": 0.5,
                        },
                    }
                ],
                "evaluated_candidates": [{}],
            },
            "sensitivity_payload": {
                "sensitivity_ranking": [
                    {"parameter": "crosslink_prob", "normalized_stiffness_span": 0.8}
                ]
            },
            "formulation_recommendations": [
                {
                    "candidate_rank": 1,
                    "material_family": "PEG-4MAL",
                    "template_name": "PEG-4MAL + laminin / MMP-linker",
                    "crosslinking_strategy": "maleimide-thiol step-growth gelation",
                    "primary_recipe": {"polymer": "PEG-4MAL", "polymer_wt_percent": "4.50 wt%"},
                }
            ],
        }
        (run_dir / "design_summary.json").write_text(json.dumps(design_summary), encoding="utf-8")
        (run_dir / "tool_calls.jsonl").write_text("", encoding="utf-8")

        snapshot = design_run_snapshot(run_dir)
        self.assertEqual(snapshot["targets"]["stiffness"], 8.0)
        self.assertTrue(snapshot["validation_payload"]["physics_valid"])
        rows = design_candidate_rows(snapshot["design_payload"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["fiber_density"], 0.35)
        sensitivity = design_sensitivity_rows(snapshot["scan_payload"])
        self.assertEqual(sensitivity[0]["parameter"], "crosslink_prob")
        best = design_best_candidate(snapshot["design_payload"])
        self.assertEqual(best["rank"], 1)
        formulations = formulation_recommendation_rows(snapshot["formulation_recommendations"])
        self.assertEqual(formulations[0]["material_family"], "PEG-4MAL")
        summary = design_run_summary(run_dir)
        self.assertTrue(summary["physics_valid"])
        self.assertEqual(summary["top_candidate_score"], 0.1)
        self.assertEqual(summary["best_fiber_density"], 0.35)
        self.assertEqual(summary["best_material_family"], "PEG-4MAL")
        self.assertTrue(summary["recommended_for_screening"])
        self.assertEqual(summary["screening_status"], "ready_for_screening")
        self.assertEqual(summary["stiffness_rel_error"], 0.25)

        run_dir_2 = temp_root / "runs" / "design_run_b"
        run_dir_2.mkdir(parents=True, exist_ok=True)
        (run_dir_2 / "metadata.json").write_text(json.dumps({"workflow": "design"}), encoding="utf-8")
        design_summary_2 = {
            "workflow": "design",
            "targets": {"stiffness": 9.0, "anisotropy": 0.12, "connectivity": 0.9},
            "constraints": {"max_anisotropy": 0.35},
            "validation_payload": {"physics_valid": True, "solver_converged": True, "monotonicity_valid": True},
            "design_assessment": {
                "status": "caution",
                "recommended_for_screening": False,
                "metrics": {"stiffness_rel_error": 0.1667},
            },
            "design_payload": {
                "top_candidates": [
                    {
                        "rank": 1,
                        "score": 0.2,
                        "feasible": True,
                        "parameters": {
                            "fiber_density": 0.42,
                            "fiber_stiffness": 9.0,
                            "bending_stiffness": 0.25,
                            "crosslink_prob": 0.5,
                            "domain_size": 1.1,
                        },
                        "features": {
                            "stiffness_mean": 7.5,
                            "anisotropy": 0.15,
                            "connectivity": 1.0,
                            "stress_propagation": 0.85,
                            "risk_index": 0.4,
                        },
                    }
                ],
                "evaluated_candidates": [{}],
            },
            "sensitivity_payload": {},
            "formulation_recommendations": [
                {
                    "candidate_rank": 1,
                    "material_family": "NorHA",
                    "template_name": "Norbornene-HA + thiol-ene peptide linker",
                    "crosslinking_strategy": "thiol-ene photo-click gelation",
                    "primary_recipe": {"polymer": "NorHA", "polymer_wt_percent": "2.20 wt%"},
                }
            ],
        }
        (run_dir_2 / "design_summary.json").write_text(json.dumps(design_summary_2), encoding="utf-8")
        (run_dir_2 / "tool_calls.jsonl").write_text("", encoding="utf-8")
        comparison = design_comparison_rows([run_dir, run_dir_2])
        self.assertEqual(len(comparison), 2)
        self.assertEqual(comparison[0]["run"], "design_run")
        self.assertEqual(comparison[1]["target_stiffness"], 9.0)
        self.assertEqual(comparison[1]["best_crosslink_prob"], 0.5)
        self.assertEqual(comparison[1]["best_material_family"], "NorHA")

        config = AppConfig(
            project_dir=temp_root,
            memory_dir=temp_root / "memory",
            library_dir=temp_root / "library",
            report_dir=temp_root / "reports",
            runs_dir=temp_root / "runs",
            cache_dir=temp_root / ".cache",
            template_dir=temp_root / "templates",
            model_provider="deepseek",
            model="deepseek-chat",
            model_api_key="x",
            model_base_url="https://api.deepseek.com/v1",
            active_run_dir=None,
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
        options = design_run_options(config)
        self.assertEqual(len(options), 2)
        self.assertTrue(all("design" in label for label, _ in options))
        latest = latest_design_summary(config)
        self.assertIn(latest["best_material_family"], {"PEG-4MAL", "NorHA"})
        family_rows = formulation_family_distribution(config)
        self.assertEqual({row["material_family"] for row in family_rows}, {"PEG-4MAL", "NorHA"})

        campaign_run = temp_root / "runs" / "campaign_run"
        campaign_run.mkdir(parents=True, exist_ok=True)
        (campaign_run / "metadata.json").write_text(json.dumps({"workflow": "design_campaign"}), encoding="utf-8")
        campaign_summary = {
            "workflow": "design_campaign",
            "campaign_assessment": {"status": "ready_for_screening", "recommended_for_screening": True},
            "campaign_results": [
                {
                    "target_stiffness": 6.0,
                    "design_assessment": {"status": "ready_for_screening", "recommended_for_screening": True},
                    "best_candidate": {
                        "feasible": True,
                        "score": 0.25,
                        "parameters": {"fiber_density": 0.33, "crosslink_prob": 0.42},
                        "features": {"stiffness_mean": 5.9, "anisotropy": 0.18, "risk_index": 0.4},
                    },
                }
            ],
            "formulation_recommendations": [
                {
                    "target_stiffness": 6.0,
                    "material_family": "NorHA",
                    "template_name": "Norbornene-HA + thiol-ene peptide linker",
                }
            ],
        }
        (campaign_run / "campaign_summary.json").write_text(json.dumps(campaign_summary), encoding="utf-8")
        campaign_options = design_campaign_run_options(config)
        self.assertEqual(len(campaign_options), 1)
        campaign_snapshot = design_campaign_snapshot(campaign_run)
        self.assertEqual(campaign_snapshot["workflow"], "design_campaign")
        campaign_rows = campaign_result_rows(campaign_snapshot)
        self.assertEqual(campaign_rows[0]["material_family"], "NorHA")
        campaign_overview = latest_campaign_overview(config)
        self.assertEqual(campaign_overview["target_count"], 1)

        calibration_run = temp_root / "runs" / "calibration_run"
        calibration_run.mkdir(parents=True, exist_ok=True)
        (calibration_run / "metadata.json").write_text(json.dumps({"workflow": "calibration"}), encoding="utf-8")
        (calibration_run / "calibration_targets.json").write_text(
            json.dumps({"calibration_targets": [{"sample_key": "gelma_0.1_30_s"}]}),
            encoding="utf-8",
        )
        (calibration_run / "calibration_results.json").write_text(
            json.dumps(
                {
                    "target_count": 1,
                    "family_priors": [{"material_family": "GelMA", "mean_abs_error": 12.0}],
                }
            ),
            encoding="utf-8",
        )
        (calibration_run / "calibration_impact.json").write_text(
            json.dumps({"summary": {"available": True, "mean_total_error_delta": 0.42}}),
            encoding="utf-8",
        )
        cal_options = calibration_run_options(config)
        self.assertEqual(len(cal_options), 1)
        cal_snapshot = calibration_run_snapshot(calibration_run)
        self.assertIn("calibration_targets", cal_snapshot["targets_payload"])
        self.assertIn("summary", cal_snapshot["impact_payload"])
        cal_overview = latest_calibration_overview(config)
        self.assertEqual(cal_overview["family_count"], 1)
        self.assertEqual(cal_overview["families"], "GelMA")
        self.assertEqual(cal_overview["impact_delta"], 0.42)

        (temp_root / "reports" / "sample.md").write_text("# Report", encoding="utf-8")
        report_rows = recent_report_rows(config)
        self.assertEqual(report_rows[0]["report"], "sample.md")

        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temp_root.rmdir()

    def test_login_helpers_validate_plain_and_hashed_passwords(self) -> None:
        config_plain = AppConfig(
            project_dir=PROJECT_ROOT,
            memory_dir=PROJECT_ROOT / "memory",
            library_dir=PROJECT_ROOT / "library",
            report_dir=PROJECT_ROOT / "reports",
            runs_dir=PROJECT_ROOT / "runs",
            cache_dir=PROJECT_ROOT / ".cache",
            template_dir=PROJECT_ROOT / "templates",
            model_provider="deepseek",
            model="deepseek-chat",
            model_api_key="x",
            model_base_url="https://api.deepseek.com/v1",
            active_run_dir=None,
            cache_ttl_hours=168,
            max_pubmed_results=5,
            ncbi_email=None,
            ncbi_api_key=None,
            crossref_mailto=None,
            frontend_require_login=True,
            frontend_username="researcher",
            frontend_password="secret",
            frontend_password_sha256=None,
            frontend_public_host="0.0.0.0",
            frontend_public_port=8525,
        )
        self.assertTrue(password_matches(config_plain, "secret"))
        self.assertTrue(credentials_valid(config_plain, "researcher", "secret"))
        self.assertFalse(credentials_valid(config_plain, "wrong", "secret"))

        config_hashed = AppConfig(
            project_dir=PROJECT_ROOT,
            memory_dir=PROJECT_ROOT / "memory",
            library_dir=PROJECT_ROOT / "library",
            report_dir=PROJECT_ROOT / "reports",
            runs_dir=PROJECT_ROOT / "runs",
            cache_dir=PROJECT_ROOT / ".cache",
            template_dir=PROJECT_ROOT / "templates",
            model_provider="deepseek",
            model="deepseek-chat",
            model_api_key="x",
            model_base_url="https://api.deepseek.com/v1",
            active_run_dir=None,
            cache_ttl_hours=168,
            max_pubmed_results=5,
            ncbi_email=None,
            ncbi_api_key=None,
            crossref_mailto=None,
            frontend_require_login=True,
            frontend_username="researcher",
            frontend_password=None,
            frontend_password_sha256=hashlib.sha256(b"secret").hexdigest(),
            frontend_public_host="0.0.0.0",
            frontend_public_port=8525,
        )
        self.assertTrue(password_matches(config_hashed, "secret"))
        self.assertTrue(credentials_valid(config_hashed, "researcher", "secret"))

    def test_auth_cookie_helpers_validate_signed_expiring_token(self) -> None:
        config = AppConfig(
            project_dir=PROJECT_ROOT,
            memory_dir=PROJECT_ROOT / "memory",
            library_dir=PROJECT_ROOT / "library",
            report_dir=PROJECT_ROOT / "reports",
            runs_dir=PROJECT_ROOT / "runs",
            cache_dir=PROJECT_ROOT / ".cache",
            template_dir=PROJECT_ROOT / "templates",
            model_provider="deepseek",
            model="deepseek-chat",
            model_api_key="x",
            model_base_url="https://api.deepseek.com/v1",
            active_run_dir=None,
            cache_ttl_hours=168,
            max_pubmed_results=5,
            ncbi_email=None,
            ncbi_api_key=None,
            crossref_mailto=None,
            frontend_require_login=True,
            frontend_username="researcher",
            frontend_password=None,
            frontend_password_sha256=hashlib.sha256(b"secret").hexdigest(),
            frontend_public_host="0.0.0.0",
            frontend_public_port=8525,
        )
        self.assertEqual(auth_cookie_name(), "ecm_organoid_auth")
        token = build_auth_cookie_value(config, username="researcher", remember_days=7, issued_at=100)
        payload = validate_auth_cookie_value(config, token, now=101)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["u"], "researcher")
        self.assertEqual(payload["exp"], 100 + 7 * 86400)
        self.assertIsNone(validate_auth_cookie_value(config, token, now=100 + 7 * 86400 + 1))

        payload_part, signature = token.rsplit(".", 1)
        tampered = payload_part + "." + ("0" * len(signature))
        self.assertIsNone(validate_auth_cookie_value(config, tampered, now=101))


if __name__ == "__main__":
    unittest.main()
