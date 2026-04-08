from __future__ import annotations

from collections import Counter
import base64
import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit.components.v1 import html as components_html

try:
    from .artifacts import read_json, read_jsonl
    from .config import AppConfig
    from .demo import DemoRunResult, default_demo_steps, run_full_demo
    from .fiber_network import default_validation_params, run_tensile_test, run_validation, simulate_ecm
    from .runner import ResearchRunResult, run_research_agent_sync
    from .workspace import ensure_workspace, resolve_project_dir
except ImportError:  # pragma: no cover
    from ecm_organoid_agent.artifacts import read_json, read_jsonl
    from ecm_organoid_agent.config import AppConfig
    from ecm_organoid_agent.demo import DemoRunResult, default_demo_steps, run_full_demo
    from ecm_organoid_agent.fiber_network import default_validation_params, run_tensile_test, run_validation, simulate_ecm
    from ecm_organoid_agent.runner import ResearchRunResult, run_research_agent_sync
    from ecm_organoid_agent.workspace import ensure_workspace, resolve_project_dir

PROJECT_DIR = ensure_workspace(resolve_project_dir())
GUIDE_PATH = PROJECT_DIR / "docs" / "frontend_guide.md"


def launch() -> None:
    from streamlit.web import cli as stcli
    import sys

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


def page_css() -> str:
    return """
    <style>
    :root {
        --paper: #f2f3f8;
        --paper-2: #eaeaea;
        --ink: #212529;
        --ink-soft: #495057;
        --teal: #10137d;
        --forest: #006db4;
        --copper: #10137d;
        --gold: #96c9fd;
        --line: rgba(16, 19, 125, 0.1);
        --line-strong: rgba(16, 19, 125, 0.2);
        --card: rgba(255, 255, 255, 0.92);
        --card-alt: rgba(247, 247, 247, 0.96);
    }
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(150, 201, 253, 0.22), transparent 26%),
            radial-gradient(circle at 100% 0%, rgba(0, 109, 180, 0.14), transparent 28%),
            linear-gradient(180deg, var(--paper) 0%, #ffffff 46%, #f7f7f7 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 0.9rem;
        padding-bottom: 2.2rem;
    }
    html, body, [class*="css"]  {
        font-family: "Avenir Next", "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
    }
    h1, h2, h3 {
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        letter-spacing: -0.02em;
        color: var(--ink);
    }
    .hero-card {
        background:
            linear-gradient(140deg, rgba(255, 255, 255, 0.98), rgba(242, 243, 248, 0.94));
        border: 1px solid var(--line-strong);
        border-radius: 28px;
        padding: 1.05rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 22px 48px rgba(33, 37, 41, 0.06);
        position: relative;
        overflow: hidden;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2.28rem;
        line-height: 1.02;
        max-width: 11ch;
    }
    .hero-card p {
        margin: 0.38rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.55;
        max-width: 66ch;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 1rem;
        align-items: end;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.73rem;
        color: var(--forest);
        font-weight: 700;
    }
    .hero-note {
        border-left: 2px solid rgba(0, 109, 180, 0.25);
        padding-left: 0.9rem;
    }
    .hero-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }
    .hero-stat {
        background: rgba(255,255,255,0.64);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem;
    }
    .hero-stat-label {
        color: var(--ink-soft);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }
    .hero-stat-value {
        font-size: 1.28rem;
        font-weight: 700;
        color: var(--ink);
    }
    .hero-bar {
        margin-top: 0.72rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .panel-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 18px 36px rgba(27, 34, 39, 0.04);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        overflow: hidden;
    }
    .metric-chip {
        border-radius: 999px;
        padding: 0.3rem 0.76rem;
        display: inline-block;
        background: rgba(150, 201, 253, 0.22);
        color: var(--teal);
        border: 1px solid rgba(16, 19, 125, 0.12);
        font-size: 0.86rem;
        font-weight: 600;
    }
    .stage-line {
        padding: 0.75rem 0.9rem;
        border-left: 3px solid var(--teal);
        background: rgba(150, 201, 253, 0.16);
        margin-bottom: 0.55rem;
        border-radius: 10px;
        color: var(--ink);
    }
    .report-box {
        background: var(--card-alt);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
    }
    .mini-card {
        background: rgba(255, 255, 255, 0.58);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.75rem;
    }
    .mini-card h4 {
        margin: 0 0 0.35rem 0;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        color: var(--ink);
        font-size: 1rem;
    }
    .mini-card p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.55;
    }
    .mini-card strong {
        color: var(--ink);
    }
    .section-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: var(--forest);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .guide-callout {
        background: rgba(150, 201, 253, 0.18);
        border: 1px solid rgba(16, 19, 125, 0.14);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: var(--ink);
    }
    .guide-callout strong {
        color: var(--copper);
    }
    .layer-grid,
    .atlas-grid {
        display: grid;
        gap: 0.8rem;
        margin: 0.7rem 0 1rem 0;
    }
    .layer-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .atlas-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .layer-card,
    .atlas-card,
    .workflow-focus {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 16px 34px rgba(27, 34, 39, 0.04);
    }
    .atlas-card.is-active,
    .workflow-focus {
        background:
            linear-gradient(160deg, rgba(255, 255, 255, 0.94), rgba(150, 201, 253, 0.18));
        border-color: rgba(16, 19, 125, 0.18);
    }
    .card-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.69rem;
        color: var(--forest);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .layer-card h4,
    .atlas-card h4,
    .workflow-focus h4 {
        margin: 0;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        color: var(--ink);
        font-size: 1.02rem;
    }
    .layer-card p,
    .atlas-card p,
    .workflow-focus p {
        margin: 0.35rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.55;
    }
    .card-meta {
        margin-top: 0.7rem;
        color: var(--ink-soft);
        font-size: 0.88rem;
        line-height: 1.48;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-top: 0.75rem;
    }
    .artifact-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.26rem 0.66rem;
        border: 1px solid rgba(16, 19, 125, 0.12);
        background: rgba(150, 201, 253, 0.18);
        color: var(--teal);
        font-size: 0.79rem;
        font-weight: 600;
    }
    .flow-board {
        display: grid;
        grid-template-columns: 0.9fr 0.18fr 1.05fr 0.18fr 1.6fr 0.18fr 0.95fr;
        gap: 0.75rem;
        align-items: start;
        margin: 0.8rem 0 1.1rem 0;
    }
    .flow-column {
        display: grid;
        gap: 0.7rem;
    }
    .flow-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 0.95rem 1rem;
        box-shadow: 0 14px 28px rgba(27, 34, 39, 0.04);
    }
    .flow-card.is-primary {
        background:
            linear-gradient(155deg, rgba(255, 255, 255, 0.96), rgba(150, 201, 253, 0.18));
        border-color: rgba(16, 19, 125, 0.16);
    }
    .flow-card h4 {
        margin: 0;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        color: var(--ink);
        font-size: 1.02rem;
    }
    .flow-card p {
        margin: 0.3rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.5;
        font-size: 0.92rem;
    }
    .flow-label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.69rem;
        color: var(--forest);
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100%;
        color: var(--teal);
        font-size: 1.6rem;
        font-weight: 700;
        opacity: 0.6;
    }
    .flow-mini-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.36rem;
        margin-top: 0.65rem;
    }
    .flow-mini {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.6rem;
        border-radius: 999px;
        background: rgba(150, 201, 253, 0.18);
        border: 1px solid rgba(16, 19, 125, 0.12);
        color: var(--teal);
        font-size: 0.76rem;
        font-weight: 600;
    }
    .flow-family-stack {
        display: grid;
        gap: 0.65rem;
    }
    .flow-family {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
    }
    .flow-family-title {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.22rem;
    }
    .flow-family-title strong {
        color: var(--ink);
        font-size: 0.98rem;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
    }
    .flow-family-title span {
        color: var(--ink-soft);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .flow-family p {
        margin: 0.12rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.48;
        font-size: 0.9rem;
    }
    .flow-stage-line {
        margin-top: 0.5rem;
        color: var(--ink);
        font-size: 0.86rem;
        line-height: 1.52;
    }
    .dashboard-grid,
    .snapshot-grid,
    .category-grid {
        display: grid;
        gap: 0.8rem;
        margin: 0.75rem 0 1rem 0;
    }
    .dashboard-grid {
        grid-template-columns: 1.05fr 0.95fr;
    }
    .snapshot-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .category-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .dashboard-card,
    .snapshot-card,
    .category-card {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 14px 30px rgba(27, 34, 39, 0.04);
    }
    .snapshot-card {
        background:
            linear-gradient(155deg, rgba(255, 255, 255, 0.95), rgba(150, 201, 253, 0.16));
    }
    .dashboard-card h4,
    .snapshot-card h4,
    .category-card h4 {
        margin: 0;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        color: var(--ink);
        font-size: 1.02rem;
    }
    .dashboard-card p,
    .snapshot-card p,
    .category-card p {
        margin: 0.35rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.55;
    }
    .snapshot-value {
        margin-top: 0.55rem;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.05;
    }
    .snapshot-meta {
        margin-top: 0.65rem;
        color: var(--ink-soft);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .home-note {
        background: rgba(150, 201, 253, 0.14);
        border: 1px solid rgba(16, 19, 125, 0.1);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: var(--ink-soft);
        line-height: 1.58;
        margin-top: 0.75rem;
    }
    .workflow-focus {
        margin-bottom: 0.95rem;
    }
    .workflow-focus ul {
        margin: 0.55rem 0 0 1.1rem;
        color: var(--ink-soft);
    }
    .help-panel {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }
    .help-panel h4 {
        margin: 0 0 0.3rem 0;
        font-family: "Iowan Old Style", "Charter", "Georgia", serif;
        color: var(--ink);
        font-size: 1rem;
    }
    .help-panel p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.52;
    }
    .compact-list {
        margin: 0.45rem 0 0 0;
        padding-left: 1rem;
        color: var(--ink-soft);
        line-height: 1.55;
    }
    .data-caption {
        color: var(--ink-soft);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.8rem;
    }
    .insight-tile {
        background: rgba(255,255,255,0.62);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.85rem 0.9rem;
    }
    .insight-tile-label {
        color: var(--ink-soft);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.2rem;
    }
    .insight-tile-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--ink);
    }
    .method-note {
        border-top: 1px solid var(--line);
        margin-top: 0.85rem;
        padding-top: 0.75rem;
        color: var(--ink-soft);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    button[data-baseweb="tab"] {
        background: rgba(255,255,255,0.55);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(150, 201, 253, 0.28);
        border-color: rgba(16, 19, 125, 0.18);
    }
    div[data-baseweb="tab-panel"] {
        padding-top: 0.85rem;
    }
    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        min-height: 2.65rem;
        white-space: normal;
    }
    .stCodeBlock pre,
    .stMarkdown,
    .stCaption,
    p,
    li {
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .small-note {
        color: var(--ink-soft);
        font-size: 0.92rem;
    }
    @media (max-width: 900px) {
        .hero-grid,
        .insight-grid,
        .hero-stat-grid,
        .dashboard-grid,
        .snapshot-grid,
        .category-grid,
        .flow-board,
        .layer-grid,
        .atlas-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """


def main() -> None:
    st.set_page_config(
        page_title="ECM Organoid Research Desk",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(page_css(), unsafe_allow_html=True)
    config = AppConfig.from_project_dir(PROJECT_DIR)

    ensure_session_defaults()
    render_auth_cookie_bridge()
    if not require_frontend_auth(config):
        return
    render_help_sidebar(config)
    hero(config)

    tab_dashboard, tab_design, tab_simulation, tab_console, tab_run, tab_demo, tab_reports, tab_library, tab_guide, tab_settings = st.tabs(
        ["Dashboard", "Design Board", "Simulation", "Agent Console", "Research Run", "Demo", "Reports", "Library", "Guide", "Settings"]
    )
    with tab_dashboard:
        render_dashboard_tab(config)
    with tab_design:
        render_design_board_tab(config)
    with tab_simulation:
        render_simulation_tab(config)
    with tab_console:
        render_console_tab(config)
    with tab_run:
        render_run_tab(config)
    with tab_demo:
        render_demo_tab(config)
    with tab_reports:
        render_reports_tab(config)
    with tab_library:
        render_library_tab(config)
    with tab_guide:
        render_guide_tab(config)
    with tab_settings:
        render_settings_tab(config)


def render_help_sidebar(config: AppConfig) -> None:
    active_workflow = str(st.session_state.get("run_workflow_select", "team"))
    spec = workflow_specs().get(active_workflow, workflow_specs()["team"])
    with st.sidebar:
        st.markdown("### Help")
        st.markdown(
            f"""
            <div class="help-panel">
                <h4>Current Workflow</h4>
                <p><strong>{active_workflow}</strong>: {spec["headline"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="help-panel">
                <h4>Path Browser</h4>
                <p>带有 <code>Browse</code> 的输入框会展开应用内路径浏览器。适合选力学数据文件、library 目录和报告保存位置。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="help-panel">
                <h4>Readable Outputs</h4>
                <p>页面默认优先显示摘要表和渲染后的 Markdown。需要排查细节时，再打开 Raw 视图或 Console 看原始工件。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Key Paths**")
        st.code(
            "\n".join(
                [
                    f"reports: {config.report_dir}",
                    f"runs: {config.runs_dir}",
                    f"library: {config.library_dir}",
                ]
            ),
            language="text",
        )
        with st.expander("Workflow Hint", expanded=False):
            st.markdown(f"**Use when**: {spec['when_to_use']}")
            st.markdown(f"**Needs**: {spec['requires']}")
            st.markdown(f"**Artifacts**: {', '.join(workflow_artifact_labels(active_workflow))}")
        with st.expander("Guide", expanded=False):
            st.markdown(
                """
                1. 文献问题优先 `team`。
                2. 数据拟合先 `mechanics` 或 `calibration`。
                3. 材料窗口比较先 `design_campaign`。
                4. 想看完整批量演示，直接去 `Demo` tab。
                5. 想看全过程证据链再去 `Agent Console`。
                """
            )
        with st.expander("End-to-End Example", expanded=False):
            render_example_walkthrough(default_workflow="design_campaign")


def ensure_session_defaults() -> None:
    st.session_state.setdefault("run_events", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("demo_events", [])
    st.session_state.setdefault("last_demo_result", None)
    st.session_state.setdefault("physics_snapshot", None)
    st.session_state.setdefault("auth_ok", False)
    st.session_state.setdefault("auth_error", "")
    st.session_state.setdefault("auth_cookie_action", None)
    st.session_state.setdefault("auth_user", "")
    st.session_state.setdefault("ui_notice", "")


def ensure_run_input_defaults(config: AppConfig) -> None:
    defaults = {
        "run_workflow_select": "team",
        "run_query_input": "synthetic ECM for intestinal organoid culture",
        "run_report_name_input": default_report_name(),
        "run_report_dir_input": str(config.report_dir),
        "run_library_dir_input": str(config.library_dir),
        "run_max_pubmed_results": min(max(config.max_pubmed_results, 3), 12),
        "run_data_path_input": "",
        "run_experiment_type_input": "auto",
        "run_applied_stress_input": 0.0,
        "run_applied_strain_input": 0.0,
        "run_time_column_input": "time",
        "run_stress_column_input": "stress",
        "run_strain_column_input": "strain",
        "run_delimiter_input": ",",
        "run_simulation_fiber_density_input": 0.0,
        "run_simulation_fiber_stiffness_input": 0.0,
        "run_simulation_bending_stiffness_input": 0.0,
        "run_simulation_crosslink_prob_input": 0.0,
        "run_simulation_domain_size_input": 0.0,
        "run_simulation_scenario_input": "bulk_mechanics",
        "run_simulation_request_json_input": "",
        "run_cell_contractility_input": 0.02,
        "run_organoid_radius_input": 0.18,
        "run_matrix_youngs_modulus_input": 0.0,
        "run_matrix_poisson_ratio_input": 0.3,
        "run_target_stiffness_input": 8.0,
        "run_campaign_target_stiffnesses_input": "6,8,10",
        "run_target_anisotropy_input": 0.1,
        "run_target_connectivity_input": 0.95,
        "run_target_stress_propagation_input": 0.5,
        "run_design_extra_targets_json_input": "",
        "run_design_top_k_input": 3,
        "run_design_candidate_budget_input": 12,
        "run_design_monte_carlo_runs_input": 4,
        "run_design_run_simulation_input": False,
        "run_design_simulation_scenario_input": "bulk_mechanics",
        "run_design_simulation_top_k_input": 2,
        "run_condition_concentration_input": 0.0,
        "run_condition_curing_input": 0.0,
        "run_condition_overrides_json_input": "",
        "run_constraint_max_anisotropy_input": 0.0,
        "run_constraint_min_connectivity_input": 0.0,
        "run_constraint_max_risk_index_input": 0.0,
        "run_constraint_min_stress_propagation_input": 0.0,
        "run_design_extra_constraints_json_input": "",
        "run_dataset_id_input": "hydrogel_characterization_data",
        "run_calibration_max_samples_input": 6,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_ui_notice(message: str) -> None:
    st.session_state["ui_notice"] = message


def pop_ui_notice() -> str:
    message = str(st.session_state.get("ui_notice", "")).strip()
    st.session_state["ui_notice"] = ""
    return message


def _initial_dialog_dir(path_hint: str | None) -> str:
    if not path_hint:
        return str(PROJECT_DIR)
    expanded = Path(path_hint).expanduser()
    if expanded.is_dir():
        return str(expanded)
    if expanded.parent.exists():
        return str(expanded.parent)
    return str(PROJECT_DIR)


def _browser_dir_for_value(value: str, *, selection_mode: str) -> Path:
    candidate = Path(value).expanduser() if value else PROJECT_DIR
    if candidate.is_dir():
        return candidate
    if selection_mode == "file" and candidate.is_file():
        return candidate.parent
    if candidate.parent.exists():
        return candidate.parent
    return PROJECT_DIR


def browser_entry_rows(current_dir: Path, *, selection_mode: str) -> List[Dict[str, str]]:
    if not current_dir.exists() or not current_dir.is_dir():
        return []
    rows: List[Dict[str, str]] = []
    children = sorted(current_dir.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
    for path in children:
        if path.name.startswith("."):
            continue
        if selection_mode == "file" and not path.is_dir() and not path.is_file():
            continue
        rows.append(
            {
                "label": f"[DIR] {path.name}" if path.is_dir() else f"[FILE] {path.name}",
                "path": str(path),
                "type": "dir" if path.is_dir() else "file",
            }
        )
    return rows


def render_path_browser(
    *,
    key: str,
    selection_mode: str,
    title: str,
) -> None:
    open_key = f"{key}_browser_open"
    dir_key = f"{key}_browser_dir"
    choice_key = f"{key}_browser_choice"

    current_value = str(st.session_state.get(key, "")).strip()
    if dir_key not in st.session_state:
        st.session_state[dir_key] = str(_browser_dir_for_value(current_value, selection_mode=selection_mode))
    current_dir = Path(str(st.session_state.get(dir_key, PROJECT_DIR))).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = _browser_dir_for_value(current_value, selection_mode=selection_mode)
        st.session_state[dir_key] = str(current_dir)

    st.markdown(f"**{title}**")
    nav1, nav2, nav3, nav4 = st.columns(4, gap="small")
    if nav1.button("Home", key=f"{key}_browser_home"):
        st.session_state[dir_key] = str(Path.home())
        st.rerun()
    if nav2.button("Project", key=f"{key}_browser_project"):
        st.session_state[dir_key] = str(PROJECT_DIR)
        st.rerun()
    if nav3.button("Up", key=f"{key}_browser_up"):
        st.session_state[dir_key] = str(current_dir.parent if current_dir.parent.exists() else current_dir)
        st.rerun()
    if selection_mode == "directory" and nav4.button("Use This", key=f"{key}_browser_use_current"):
        st.session_state[key] = str(current_dir)
        st.session_state[open_key] = False
        set_ui_notice(f"Selected folder: {current_dir}")
        st.rerun()
    st.caption(str(current_dir))

    rows = browser_entry_rows(current_dir, selection_mode=selection_mode)
    if not rows:
        st.info("Current folder is empty or unavailable.")
        return

    label_to_row = {row["label"]: row for row in rows}
    selected_label = st.selectbox(
        "Items",
        [row["label"] for row in rows],
        key=choice_key,
        label_visibility="collapsed",
    )
    selected_row = label_to_row[selected_label]
    selected_path = Path(selected_row["path"])

    action1, action2 = st.columns(2, gap="small")
    if selected_path.is_dir():
        if action1.button("Open Folder", key=f"{key}_browser_open_dir"):
            st.session_state[dir_key] = str(selected_path)
            st.rerun()
        if selection_mode == "directory" and action2.button("Use Folder", key=f"{key}_browser_use_dir"):
            st.session_state[key] = str(selected_path)
            st.session_state[open_key] = False
            set_ui_notice(f"Selected folder: {selected_path}")
            st.rerun()
    else:
        st.caption(str(selected_path))
        if action1.button("Use File", key=f"{key}_browser_use_file"):
            st.session_state[key] = str(selected_path)
            st.session_state[open_key] = False
            set_ui_notice(f"Selected file: {selected_path}")
            st.rerun()
        if action2.button("Close", key=f"{key}_browser_close_file"):
            st.session_state[open_key] = False
            st.rerun()


def render_path_input_row(
    *,
    label: str,
    key: str,
    dialog_mode: str,
    dialog_title: str,
    help_text: str | None = None,
    save_target: bool = False,
) -> str:
    input_col, button_col = st.columns([0.78, 0.22], gap="small")
    with input_col:
        st.text_input(label, key=key, help=help_text)
    with button_col:
        if st.button("Browse", key=f"{key}_browse"):
            st.session_state[f"{key}_browser_open"] = not bool(st.session_state.get(f"{key}_browser_open", False))
            st.session_state[f"{key}_browser_dir"] = str(
                _browser_dir_for_value(str(st.session_state.get(key, "")), selection_mode=dialog_mode)
            )
            st.rerun()
    if st.session_state.get(f"{key}_browser_open"):
        render_path_browser(
            key=key,
            selection_mode=dialog_mode,
            title=dialog_title,
        )
    return str(st.session_state.get(key, ""))


def render_mapping_table(
    mapping: Dict[str, Any],
    *,
    label_map: Dict[str, str] | None = None,
    order: List[str] | None = None,
    hide_index: bool = True,
) -> None:
    rows = mapping_rows(mapping, label_map=label_map, order=order)
    if not rows:
        st.info("No structured fields available.")
        return
    st.dataframe(rows, width="stretch", hide_index=hide_index)


def mapping_rows(
    mapping: Dict[str, Any],
    *,
    label_map: Dict[str, str] | None = None,
    order: List[str] | None = None,
) -> List[Dict[str, str]]:
    if not isinstance(mapping, dict):
        return []
    ordered_keys = order or list(mapping.keys())
    rows: List[Dict[str, str]] = []
    for key in ordered_keys:
        if key not in mapping:
            continue
        value = mapping[key]
        rows.append(
            {
                "field": label_map.get(key, key.replace("_", " ")) if label_map else key.replace("_", " "),
                "value": format_display_value(value),
            }
        )
    return rows


def format_display_value(value: Any) -> str:
    if value is None:
        return "NR"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "NR"
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return ", ".join(str(item) for item in value)
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, dict):
        if not value:
            return "NR"
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def render_markdown_or_code(content: str, *, key: str) -> None:
    preview_tab, raw_tab = st.tabs(["Preview", "Raw"])
    with preview_tab:
        st.markdown(content)
    with raw_tab:
        st.code(content, language="markdown")


def render_text_output_expander(label: str, content: str, *, key: str) -> None:
    with st.expander(label):
        if not content:
            st.caption("No captured output.")
            return
        render_markdown_or_code(content, key=key)


def render_artifact_file(path: Path, *, key: str) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        payload = load_json_file(path)
        if payload:
            render_mapping_table(payload)
        with st.expander("Raw JSON"):
            st.code(text, language="json")
        return
    render_markdown_or_code(text, key=key)


def best_candidate_sections(candidate: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "summary": {
            "rank": candidate.get("rank", "NR"),
            "feasible": candidate.get("feasible", False),
            "score": candidate.get("score", "NR"),
        },
        "parameters": candidate.get("parameters", {}) if isinstance(candidate.get("parameters"), dict) else {},
        "features": candidate.get("features", {}) if isinstance(candidate.get("features"), dict) else {},
    }


def auth_cookie_name() -> str:
    return "ecm_organoid_auth"


def auth_cookie_secret(config: AppConfig) -> bytes:
    seed = (
        config.frontend_password_sha256
        or hashlib.sha256(str(config.frontend_password or "").encode("utf-8")).hexdigest()
        or ""
    )
    material = f"{config.frontend_username or ''}|{seed}|{config.project_dir}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def build_auth_cookie_value(
    config: AppConfig,
    *,
    username: str,
    remember_days: int,
    issued_at: Optional[int] = None,
) -> str:
    now = issued_at if issued_at is not None else int(time.time())
    payload = {
        "u": username,
        "iat": now,
        "exp": now + remember_days * 86400,
        "v": 1,
    }
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_token = _b64url_encode(payload_text.encode("utf-8"))
    signature = hmac.new(auth_cookie_secret(config), payload_token.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_token}.{signature}"


def validate_auth_cookie_value(
    config: AppConfig,
    token: str,
    *,
    now: Optional[int] = None,
) -> Dict[str, Any] | None:
    if not token or "." not in token:
        return None
    payload_token, signature = token.rsplit(".", 1)
    expected_signature = hmac.new(
        auth_cookie_secret(config),
        payload_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_token).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    username = str(payload.get("u", "")).strip()
    expires_at = int(payload.get("exp", 0))
    current_time = now if now is not None else int(time.time())
    if not username or expires_at <= current_time:
        return None
    if config.frontend_username and not hmac.compare_digest(username, config.frontend_username):
        return None
    return payload


def queue_auth_cookie_set(config: AppConfig, *, username: str, remember_days: int) -> None:
    st.session_state["auth_cookie_action"] = {
        "mode": "set",
        "name": auth_cookie_name(),
        "value": build_auth_cookie_value(config, username=username, remember_days=remember_days),
        "remember_days": remember_days,
    }


def queue_auth_cookie_clear() -> None:
    st.session_state["auth_cookie_action"] = {
        "mode": "clear",
        "name": auth_cookie_name(),
    }


def render_auth_cookie_bridge() -> None:
    action = st.session_state.get("auth_cookie_action")
    if not isinstance(action, dict):
        return
    cookie_name = json.dumps(str(action.get("name", auth_cookie_name())))
    if action.get("mode") == "set":
        cookie_value = json.dumps(str(action.get("value", "")))
        max_age_seconds = int(action.get("remember_days", 0)) * 86400
        script = f"""
        <script>
        (function() {{
          const name = {cookie_name};
          const value = {cookie_value};
          const maxAge = {max_age_seconds};
          let cookie = `${{name}}=${{value}}; path=/; max-age=${{maxAge}}; SameSite=Lax`;
          if (window.location.protocol === "https:") {{
            cookie += "; Secure";
          }}
          document.cookie = cookie;
        }})();
        </script>
        """
    else:
        script = f"""
        <script>
        (function() {{
          const name = {cookie_name};
          document.cookie = `${{name}}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax`;
        }})();
        </script>
        """
    components_html(script, height=0, width=0)
    st.session_state["auth_cookie_action"] = None


def password_matches(config: AppConfig, candidate_password: str) -> bool:
    if config.frontend_password_sha256:
        digest = hashlib.sha256(candidate_password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, config.frontend_password_sha256.strip().lower())
    if config.frontend_password is None:
        return False
    return hmac.compare_digest(candidate_password, config.frontend_password)


def credentials_valid(config: AppConfig, username: str, password: str) -> bool:
    if config.frontend_username:
        if not hmac.compare_digest(username, config.frontend_username):
            return False
    return password_matches(config, password)


def require_frontend_auth(config: AppConfig) -> bool:
    if not config.frontend_require_login:
        return True
    if st.session_state.get("auth_ok"):
        return True
    cookie_value = st.context.cookies.get(auth_cookie_name()) if hasattr(st, "context") else None
    if cookie_value:
        payload = validate_auth_cookie_value(config, str(cookie_value))
        if payload is not None:
            st.session_state["auth_ok"] = True
            st.session_state["auth_error"] = ""
            st.session_state["auth_user"] = str(payload.get("u", ""))
            return True
        queue_auth_cookie_clear()
    render_login_gate(config)
    return False


def render_login_gate(config: AppConfig) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Protected Research Console</div>
            <h1>Sign In Required</h1>
            <p>This ECM research workspace is protected. Authentication is required before any local files, reports, or design tools are exposed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Frontend Access Control")
    st.caption("Configure credentials in `.env`. Keep login enabled before exposing the UI beyond localhost.")
    with st.form("frontend_login_form", clear_on_submit=False):
        username = st.text_input("Username", value="", autocomplete="username")
        password = st.text_input("Password", value="", type="password", autocomplete="current-password")
        remember_option = st.radio(
            "Remember this browser",
            options=("No", "7 days", "15 days"),
            horizontal=True,
        )
        submitted = st.form_submit_button("Sign In", width="stretch")
    if submitted:
        if credentials_valid(config, username.strip(), password):
            st.session_state["auth_ok"] = True
            st.session_state["auth_error"] = ""
            st.session_state["auth_user"] = username.strip()
            remember_days = 0
            if remember_option == "7 days":
                remember_days = 7
            elif remember_option == "15 days":
                remember_days = 15
            if remember_days > 0:
                queue_auth_cookie_set(config, username=username.strip(), remember_days=remember_days)
            else:
                queue_auth_cookie_clear()
            st.rerun()
        else:
            st.session_state["auth_error"] = "Invalid username or password."
    if st.session_state.get("auth_error"):
        st.error(st.session_state["auth_error"])
    st.info(f"Suggested public bind address: http://{config.frontend_public_host}:{config.frontend_public_port}")
    st.stop()


def hero(config: AppConfig) -> None:
    library_count = len(list(config.library_dir.rglob("*"))) if config.library_dir.exists() else 0
    report_count = len(list(config.report_dir.glob("*.md"))) if config.report_dir.exists() else 0
    run_count = len(list(config.runs_dir.glob("*"))) if config.runs_dir.exists() else 0
    workflow_count = len(workflow_catalog_rows())
    latest_design = latest_design_summary(config)
    latest_campaign = latest_campaign_overview(config)
    hero_note = latest_campaign.get(
        "note",
        "Evidence retrieval, mechanics fitting, fiber-network design, dataset calibration, and report artifacts in one research desk.",
    )
    best_stiffness = latest_design.get("best_stiffness_mean", "NR")
    best_stiffness_text = f"{best_stiffness:.2f} Pa" if isinstance(best_stiffness, (int, float)) else "NR"
    stiffness_error = latest_design.get("stiffness_error", "NR")
    stiffness_error_text = f"{stiffness_error:.2f}" if isinstance(stiffness_error, (int, float)) else "NR"
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">ECM Research Control Surface</div>
                    <h1>ECM Organoid Research Desk</h1>
                    <p>一个围绕 ECM 与类器官研究闭环搭建的工作台。重点不是聊天，而是把文献证据、力学拟合、网络模拟、逆向设计、数据校准和 runs 工件放到同一张桌面上。</p>
                    <div class="hero-bar">
                        <span class="metric-chip">Provider: {config.model_provider}</span>
                        <span class="metric-chip">Model: {config.model}</span>
                        <span class="metric-chip">Workflows: {workflow_count}</span>
                        <span class="metric-chip">Library Files: {library_count}</span>
                        <span class="metric-chip">Reports: {report_count}</span>
                        <span class="metric-chip">Runs: {run_count}</span>
                    </div>
                </div>
                <div class="hero-note">
                    <div class="hero-stat-grid">
                        <div class="hero-stat">
                            <div class="hero-stat-label">Latest Design</div>
                            <div class="hero-stat-value">{best_stiffness_text}</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Design Error</div>
                            <div class="hero-stat-value">{stiffness_error_text}</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Best Material</div>
                            <div class="hero-stat-value">{latest_design.get("best_material_family", "NR")}</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Feasible Windows</div>
                            <div class="hero-stat-value">{latest_campaign.get("feasible_count", 0)}/{latest_campaign.get("target_count", 0)}</div>
                        </div>
                    </div>
                    <div class="method-note">{hero_note}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Home")
    st.caption("首页先给完整流程图，再给最近结果和下一步入口。")

    summary = build_dashboard_summary(config)
    latest_design = latest_design_summary(config)
    latest_campaign = latest_campaign_overview(config)
    latest_calibration = latest_calibration_overview(config)
    family_rows = formulation_family_distribution(config)
    latest_reports = recent_report_rows(config)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", summary["agent_count"])
    c2.metric("Runs", summary["run_count"])
    c3.metric("Cached Queries", summary["cache_count"])
    c4.metric("Reports", summary["report_count"])

    intro_left, intro_right = st.columns([1.06, 0.94], gap="large")
    with intro_left:
        st.markdown("### ECM Agent End-to-End Flow")
        render_project_flow_diagram()
    with intro_right:
        st.markdown("### What This Desk Is")
        render_workflow_category_cards()
        st.markdown(
            """
            <div class="home-note">
                这不是聊天首页，而是研究控制台首页。先理解完整工作流，再看最近结果，最后决定下一步该去 Research Run、Design Board、Demo 还是 Console。
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Recent Outcomes")
    dashboard_snapshot_cards(latest_design, latest_campaign, latest_calibration)

    lower_left, lower_right = st.columns([1.02, 0.98], gap="large")
    with lower_left:
        st.markdown("### Where To Go Next")
        st.markdown(quickstart_markdown())
        if family_rows:
            st.markdown("### Recent Material Winners")
            render_family_cards(family_rows)
    with lower_right:
        st.markdown("### Recent Reports")
        render_recent_report_cards(latest_reports)

    with st.expander("Latest Structured Results", expanded=False):
        result_left, result_right = st.columns([0.98, 1.02], gap="large")
        with result_left:
            st.markdown("### Latest Design Snapshot")
            st.dataframe([latest_design], width="stretch", hide_index=True)
            if latest_campaign:
                st.markdown("### Latest Campaign Snapshot")
                st.dataframe([latest_campaign], width="stretch", hide_index=True)
            if latest_calibration:
                st.markdown("### Latest Calibration Snapshot")
                st.dataframe([latest_calibration], width="stretch", hide_index=True)
        with result_right:
            st.markdown("### Inspect Latest Run")
            run_options = recent_run_options(config)
            if not run_options:
                st.info("还没有 `runs/` 工件。先执行一次研究流程。")
            else:
                labels = [label for label, _ in run_options]
                choice = st.selectbox("Choose a run", labels, index=0, key="dashboard_run_inspector")
                run_dir = dict(run_options)[choice]
                render_run_inspector(run_dir)

    with st.expander("System Details", expanded=False):
        st.markdown("### System Layers")
        render_system_layer_cards()
        with st.expander("Physics Validation", expanded=False):
            render_physics_validation_card()
        with st.expander("Workflow Atlas", expanded=False):
            render_workflow_atlas()
        ops_left, ops_right = st.columns(2, gap="large")
        with ops_left:
            st.markdown("### Workspace Health")
            st.dataframe(workspace_health_rows(config), width="stretch", hide_index=True)
            st.markdown("### Cache Snapshot")
            st.dataframe(cache_rows(config), width="stretch", hide_index=True)
        with ops_right:
            st.markdown("### Capability Board")
            st.dataframe(agent_capability_rows(), width="stretch", hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_design_board_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Design Board")
    st.caption("聚焦目标驱动 ECM 逆向设计：查看物理验证门槛、候选方案排名、最佳候选敏感性和设计工件。")

    run_options = design_run_options(config)
    if not run_options:
        st.info("还没有 design workflow 的 run。先在 Research Run 里选择 `design` workflow 执行一次。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    labels = [label for label, _ in run_options]
    run_lookup = dict(run_options)
    campaign_options = design_campaign_run_options(config)
    calibration = latest_calibration_overview(config)

    choice = st.selectbox("Select Design Run", labels, index=0, key="design_run")
    run_dir = run_lookup[choice]
    snapshot = design_run_snapshot(run_dir)
    selected_summary = design_run_summary(run_dir)

    top_cards = st.columns(3, gap="large")
    top_cards[0].metric("Selected Target", selected_summary.get("target_stiffness", "NR"))
    top_cards[1].metric("Best Family", selected_summary.get("best_material_family", "NR"))
    top_cards[2].metric("Physics", "PASS" if selected_summary.get("physics_valid") else "FAIL")

    overview_tab, compare_tab, inspect_tab = st.tabs(["Overview", "Compare", "Inspect"])

    with overview_tab:
        overview_left, overview_right = st.columns([1.02, 0.98], gap="large")
        with overview_left:
            st.markdown("### Selected Design Snapshot")
            render_mapping_table(
                selected_summary,
                order=[
                    "target_stiffness",
                    "target_anisotropy",
                    "target_connectivity",
                    "target_stress_propagation",
                    "feasible",
                    "febio_status",
                    "febio_scenario",
                    "febio_best_candidate",
                    "febio_best_score",
                    "calibration_prior_level",
                    "calibration_concentration",
                    "calibration_curing",
                    "calibration_target_stiffness_mean",
                    "calibration_selection_score",
                    "best_material_family",
                    "best_stiffness_mean",
                    "stiffness_error",
                    "best_risk_index",
                    "top_candidate_score",
                ],
            )
        with overview_right:
            if calibration:
                st.markdown("### Calibration Context")
                st.dataframe([calibration], width="stretch", hide_index=True)
            if campaign_options:
                st.markdown("### Latest Campaign Overview")
                campaign_labels = [label for label, _ in campaign_options]
                campaign_choice = st.selectbox("Select Campaign", campaign_labels, index=0, key="design_campaign_run")
                campaign_run_dir = dict(campaign_options)[campaign_choice]
                campaign_snapshot = design_campaign_snapshot(campaign_run_dir)
                campaign_rows = campaign_result_rows(campaign_snapshot)
                if campaign_rows:
                    st.dataframe(campaign_rows, width="stretch", hide_index=True)

    with compare_tab:
        st.markdown("### Compare Design Runs")
        default_labels = labels[: min(2, len(labels))]
        compare_labels = st.multiselect(
            "Select 2-4 design runs to compare",
            labels,
            default=default_labels,
            key="design_compare_runs",
        )
        if len(compare_labels) < 2:
            st.info("请选择至少 2 个 design run 进行横向比较。")
        else:
            if len(compare_labels) > 4:
                st.warning("只显示前 4 个 design run。")
                compare_labels = compare_labels[:4]
            compare_rows = design_comparison_rows([run_lookup[label] for label in compare_labels])
            st.dataframe(compare_rows, width="stretch", hide_index=True)

    with inspect_tab:
        left, right = st.columns([0.58, 0.42], gap="large")
        with left:
            st.markdown("### Target Mechanics")
            if snapshot["targets"]:
                render_mapping_table(
                    snapshot["targets"],
                    order=["stiffness", "anisotropy", "connectivity", "stress_propagation"],
                )
            else:
                st.info("当前 run 没有解析到 design targets。")
            if snapshot.get("constraints"):
                st.markdown("### Hard Constraints")
                render_mapping_table(snapshot["constraints"])

            st.markdown("### Top Candidate Ranking")
            candidate_rows = design_candidate_rows(snapshot["design_payload"])
            if candidate_rows:
                st.dataframe(candidate_rows, width="stretch", hide_index=True)
            else:
                st.info("当前 run 没有解析到 top candidates。")

            formulation_rows = formulation_recommendation_rows(snapshot.get("formulation_recommendations", []))
            if formulation_rows:
                st.markdown("### Formulation Translation")
                st.dataframe(formulation_rows, width="stretch", hide_index=True)

            sensitivity_rows = design_sensitivity_rows(snapshot["scan_payload"])
            if sensitivity_rows:
                st.markdown("### Best Candidate Sensitivity")
                st.dataframe(sensitivity_rows, width="stretch", hide_index=True)

        with right:
            st.markdown("### Physics Gate")
            validation = snapshot["validation_payload"]
            if validation:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Solver", "PASS" if validation.get("solver_converged") else "FAIL")
                c2.metric("Monotonicity", "PASS" if validation.get("monotonicity_valid") else "FAIL")
                c3.metric("Nonlinearity", "PASS" if validation.get("nonlinearity_valid") else "FAIL")
                c4.metric("Physics", "PASS" if validation.get("physics_valid") else "FAIL")
            else:
                st.info("当前 run 没有 validation payload。")

            calibration_context = snapshot.get("calibration_context", {}) if isinstance(snapshot.get("calibration_context"), dict) else {}
            if calibration_context:
                st.markdown("### Routing Context")
                render_mapping_table(
                    calibration_context,
                    order=[
                        "prior_level",
                        "material_family",
                        "concentration_fraction",
                        "curing_seconds",
                        "target_stiffness_mean",
                        "sample_count",
                    ],
                )
                selection_reason = calibration_context.get("selection_reason", {})
                if isinstance(selection_reason, dict) and selection_reason:
                    st.markdown("#### Routing Score")
                    render_mapping_table(selection_reason)

            best_candidate = design_best_candidate(snapshot["design_payload"])
            if best_candidate:
                st.markdown("### Best Candidate")
                sections = best_candidate_sections(best_candidate)
                st.markdown("#### Summary")
                render_mapping_table(sections["summary"], order=["rank", "feasible", "score"])
                if sections["parameters"]:
                    st.markdown("#### Parameters")
                    render_mapping_table(sections["parameters"])
                if sections["features"]:
                    st.markdown("#### Features")
                    render_mapping_table(sections["features"])
                with st.expander("Raw Candidate Payload", expanded=False):
                    st.code(json.dumps(best_candidate, ensure_ascii=False, indent=2), language="json")

            design_simulation = snapshot.get("design_simulation", {}) if isinstance(snapshot.get("design_simulation"), dict) else {}
            if design_simulation:
                st.markdown("### FEBio Verification")
                render_mapping_table(
                    {
                        "status": design_simulation.get("status", "NR"),
                        "scenario": design_simulation.get("scenario", "NR"),
                        "reason": design_simulation.get("reason", "NR"),
                    },
                    order=["status", "scenario", "reason"],
                )
                ranking = design_simulation.get("comparison", {}).get("ranking", []) if isinstance(design_simulation.get("comparison"), dict) else []
                if ranking:
                    st.markdown("#### FEBio Ranking")
                    st.dataframe(ranking[:3], width="stretch", hide_index=True)

            stage_map = design_stage_files(run_dir)
            if stage_map:
                st.markdown("### Design Stage Viewer")
                stage_name = st.selectbox(
                    "Artifact",
                    list(stage_map.keys()),
                    index=0,
                    key=f"design_stage_{run_dir.name}",
                )
                stage_path = stage_map[stage_name]
                st.caption(str(stage_path))
                render_artifact_file(stage_path, key=f"design_artifact_{run_dir.name}_{stage_name}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_simulation_tab(config: AppConfig) -> None:
    ensure_run_input_defaults(config)
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Simulation")
    st.caption("这个页面专门用于运行固定场景 FEBio simulation，并检查结构化指标、原始工件和失败信息。")

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.markdown("### Run Scenario")
        st.selectbox(
            "Scenario",
            ["bulk_mechanics", "single_cell_contraction", "organoid_spheroid"],
            key="run_simulation_scenario_input",
        )
        st.number_input("Target stiffness", min_value=0.0, key="run_target_stiffness_input")
        st.number_input("Matrix Young's modulus", min_value=0.0, key="run_matrix_youngs_modulus_input")
        st.number_input("Matrix Poisson ratio", min_value=0.0, max_value=0.49, key="run_matrix_poisson_ratio_input")
        st.number_input("Cell contractility", min_value=0.0, key="run_cell_contractility_input")
        st.number_input("Organoid radius", min_value=0.0, key="run_organoid_radius_input")
        st.text_area(
            "Scenario request JSON",
            key="run_simulation_request_json_input",
            help="只接受固定 schema 可识别的字段，例如 sample_dimensions、cell_radius、organoid_radial_displacement 等。",
        )
        st.caption(f"FEBio status: {config.febio_status_message}")
        if st.button("Run Simulation", use_container_width=True, key="run_simulation_tab_button"):
            run_workflow(
                config=config,
                query=f"Simulation | {st.session_state.get('run_simulation_scenario_input', 'bulk_mechanics')}",
                workflow="simulation",
                report_name=str(st.session_state.get("run_report_name_input", "simulation_report.md")) or "simulation_report.md",
                report_dir=Path(str(st.session_state.get("run_report_dir_input", config.report_dir))).expanduser(),
                library_dir=config.library_dir,
                max_pubmed_results=int(st.session_state.get("run_max_pubmed_results", config.max_pubmed_results)),
                data_path=None,
                experiment_type="auto",
                time_column="time",
                stress_column="stress",
                strain_column="strain",
                applied_stress=None,
                applied_strain=None,
                delimiter=",",
                simulation_fiber_density=None,
                simulation_fiber_stiffness=None,
                simulation_bending_stiffness=None,
                simulation_crosslink_prob=None,
                simulation_domain_size=None,
                target_stiffness=float(st.session_state.get("run_target_stiffness_input", 0.0)) or None,
                simulation_scenario=str(st.session_state.get("run_simulation_scenario_input", "bulk_mechanics")),
                simulation_request_json=str(st.session_state.get("run_simulation_request_json_input", "")),
                cell_contractility=float(st.session_state.get("run_cell_contractility_input", 0.0)) or None,
                organoid_radius=float(st.session_state.get("run_organoid_radius_input", 0.0)) or None,
                matrix_youngs_modulus=float(st.session_state.get("run_matrix_youngs_modulus_input", 0.0)) or None,
                matrix_poisson_ratio=float(st.session_state.get("run_matrix_poisson_ratio_input", 0.3)),
                target_anisotropy=0.1,
                target_connectivity=0.95,
                target_stress_propagation=0.5,
                design_extra_targets_json="",
                constraint_max_anisotropy=None,
                constraint_min_connectivity=None,
                constraint_max_risk_index=None,
                constraint_min_stress_propagation=None,
                design_extra_constraints_json="",
                design_top_k=3,
                design_candidate_budget=12,
                design_monte_carlo_runs=4,
                design_run_simulation=False,
                design_simulation_scenario="bulk_mechanics",
                design_simulation_top_k=1,
                condition_concentration_fraction=None,
                condition_curing_seconds=None,
                condition_overrides_json="",
                campaign_target_stiffnesses="",
                dataset_id=None,
                calibration_max_samples=6,
            )

        st.markdown("### Run Progress")
        events: List[Tuple[str, str]] = st.session_state.get("run_events", [])
        if events:
            st.markdown(render_stage_lines(events), unsafe_allow_html=True)
        else:
            st.caption("点击运行后，这里会显示构建 request、运行 FEBio 和写报告的进度。")

    with right:
        st.markdown("### Latest Simulation Runs")
        options = simulation_run_options(config)
        if not options:
            st.info("还没有 simulation workflow 的 run。")
        else:
            labels = [label for label, _ in options]
            choice = st.selectbox("Select Simulation Run", labels, index=0, key="simulation_run_select")
            run_dir = dict(options)[choice]
            summary = simulation_run_summary(run_dir)
            snapshot = simulation_run_snapshot(run_dir)
            render_mapping_table(
                summary,
                order=[
                    "scenario",
                    "status",
                    "effective_stiffness",
                    "peak_stress",
                    "stress_propagation_distance",
                    "strain_heterogeneity",
                    "interface_deformation",
                    "candidate_suitability_score",
                ],
            )
            metrics_payload = snapshot.get("metrics_payload", {}) if isinstance(snapshot.get("metrics_payload"), dict) else {}
            if metrics_payload:
                st.markdown("#### Structured Metrics")
                render_mapping_table(metrics_payload)
            final_summary_path = Path(str(snapshot.get("final_summary_path", ""))) if snapshot.get("final_summary_path") else None
            if final_summary_path and final_summary_path.exists():
                with st.expander("Final Summary", expanded=False):
                    render_artifact_file(final_summary_path, key=f"simulation_summary_{run_dir.name}")
            simulation_dir = Path(str(snapshot.get("simulation_dir", ""))) if snapshot.get("simulation_dir") else None
            if simulation_dir and simulation_dir.exists():
                artifact_map = {
                    path.name: path
                    for path in sorted(simulation_dir.iterdir())
                    if path.is_file()
                }
                if artifact_map:
                    st.markdown("#### Simulation Artifacts")
                    artifact_name = st.selectbox(
                        "Artifact",
                        list(artifact_map.keys()),
                        index=0,
                        key=f"simulation_artifact_{run_dir.name}",
                    )
                    render_artifact_file(artifact_map[artifact_name], key=f"simulation_artifact_view_{run_dir.name}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_console_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Agent Console")
    st.caption("这个控制台用于逐个检视多智能体阶段产物和协作痕迹。当前版本偏重检查与审阅，不直接修改运行结果。")

    run_options = recent_run_options(config)
    if not run_options:
        st.info("还没有可供检视的 run。先执行一次研究流程。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    labels = [label for label, _ in run_options]
    choice = st.selectbox("Select Run Context", labels, index=0, key="console_run")
    run_dir = dict(run_options)[choice]
    metadata = read_json(run_dir / "metadata.json")

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        st.markdown("### Run Context")
        if metadata:
            render_mapping_table(
                {
                    "run_id": metadata.get("run_id"),
                    "workflow": metadata.get("workflow"),
                    "query": metadata.get("query"),
                    "report_path": metadata.get("report_path"),
                },
                order=["run_id", "workflow", "query", "report_path"],
            )

        tool_rows = read_jsonl(run_dir / "tool_calls.jsonl")
        st.markdown("### Tool Timeline")
        if tool_rows:
            compact_rows = [
                {
                    "time": row.get("timestamp", ""),
                    "tool": row.get("tool_name", ""),
                    "cached": row.get("cached", ""),
                    "status": row.get("status", ""),
                }
                for row in tool_rows
            ]
            st.dataframe(compact_rows, use_container_width=True, hide_index=True)
        else:
            st.info("这个 run 没有工具日志。")
        render_run_simulation_summary(run_dir)

    with right:
        st.markdown("### Agent Stage Viewer")
        stage_map = available_stage_files(run_dir)
        if not stage_map:
            st.info("这个 run 没有阶段工件。")
        else:
            agent_stage = st.selectbox(
                "Stage Artifact",
                list(stage_map.keys()),
                index=0,
                key=f"console_stage_{run_dir.name}",
            )
            stage_path = stage_map[agent_stage]
            st.caption(str(stage_path))
            render_artifact_file(stage_path, key=f"console_artifact_{run_dir.name}_{agent_stage}")

    st.markdown("### Collaboration Coverage")
    st.dataframe(collaboration_coverage_rows(run_dir), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_run_tab(config: AppConfig) -> None:
    left, right = st.columns([0.95, 1.15], gap="large")
    ensure_run_input_defaults(config)
    workflow_options = list(workflow_specs().keys())
    literature_workflows = {"team", "single", "hybrid"}
    mechanics_workflows = {"mechanics", "hybrid"}
    febio_workflows = {"simulation"}
    design_workflows = {"design", "design_campaign"}
    data_workflows = {"datasets", "calibration"}

    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Run Configuration")
        workflow = st.selectbox(
            "Workflow",
            workflow_options,
            index=workflow_options.index(str(st.session_state.get("run_workflow_select", "team"))),
            key="run_workflow_select",
            help="先选 workflow，再填写该流程真正需要的输入。这里的选项与后端 `runner.py` 保持一致。",
        )
        render_workflow_focus(workflow)

        notice = pop_ui_notice()
        if notice:
            st.info(notice)

        if workflow in design_workflows:
            query_help = "这里的 query 会写入 metadata，并作为材料家族 hint 参与 design / campaign 报告叙事。"
        elif workflow in data_workflows:
            query_help = "写成数据目标，例如材料类型、实验范围或你想补齐的数据空白。"
        elif workflow == "benchmark":
            query_help = "写成 benchmark 目的，例如 solver stability、scaling 或 inverse design regression。"
        else:
            query_help = "建议写成研究问题或 PubMed 风格检索主题。"

        st.text_area(
            "Goal / Query",
            key="run_query_input",
            height=130,
            help=query_help,
        )

        st.markdown("### Output Location")
        c1, c2 = st.columns([0.52, 0.48], gap="large")
        with c1:
            st.text_input(
                "Report Filename",
                key="run_report_name_input",
                help="默认保存为 Markdown 报告文件名。",
            )
        with c2:
            render_path_input_row(
                label="Report Folder",
                key="run_report_dir_input",
                dialog_mode="directory",
                dialog_title="Browse report folder",
                help_text="报告默认写入这个目录。",
            )
        st.caption(
            f"Save target: {Path(str(st.session_state.get('run_report_dir_input', config.report_dir))).expanduser() / str(st.session_state.get('run_report_name_input', 'research_report.md'))}"
        )

        if workflow in literature_workflows:
            st.markdown("### Literature Inputs")
            render_path_input_row(
                label="Library Folder",
                key="run_library_dir_input",
                dialog_mode="directory",
                dialog_title="Browse library folder",
                help_text="用于本地 PDF / Markdown / text 检索。",
            )
            st.slider(
                "PubMed Count",
                min_value=3,
                max_value=12,
                key="run_max_pubmed_results",
            )
        else:
            st.caption("这个 workflow 默认不读取本地 `library/`。")

        if workflow in mechanics_workflows:
            st.markdown("### Mechanics Dataset")
            d1, d2 = st.columns(2, gap="large")
            with d1:
                render_path_input_row(
                    label="Mechanics File",
                    key="run_data_path_input",
                    dialog_mode="file",
                    dialog_title="Browse mechanics dataset file",
                    help_text="支持本地 CSV / TSV / 其它文本数据文件。",
                )
                st.selectbox(
                    "Experiment",
                    ["auto", "elastic", "creep", "relaxation", "frequency_sweep", "cyclic"],
                    key="run_experiment_type_input",
                )
                st.number_input("Applied stress", key="run_applied_stress_input")
                st.number_input("Applied strain", key="run_applied_strain_input")
            with d2:
                st.text_input("Time col", key="run_time_column_input")
                st.text_input("Stress col", key="run_stress_column_input")
                st.text_input("Strain col", key="run_strain_column_input")
                st.text_input("Delimiter", key="run_delimiter_input")

        if workflow == "hybrid":
            st.markdown("### Simulation Overrides")
            s1, s2 = st.columns(2, gap="large")
            with s1:
                st.number_input("Fiber Density Override", key="run_simulation_fiber_density_input")
                st.number_input("Fiber Stiffness Override", key="run_simulation_fiber_stiffness_input")
                st.number_input("Bending Stiffness Override", key="run_simulation_bending_stiffness_input")
            with s2:
                st.number_input("Crosslink Prob Override", key="run_simulation_crosslink_prob_input")
                st.number_input("Domain Size Override", key="run_simulation_domain_size_input")

        if workflow in febio_workflows:
            st.markdown("### FEBio Simulation Request")
            s1, s2 = st.columns(2, gap="large")
            with s1:
                st.selectbox(
                    "Simulation scenario",
                    ["bulk_mechanics", "single_cell_contraction", "organoid_spheroid"],
                    key="run_simulation_scenario_input",
                )
                st.number_input("Matrix Young's modulus", min_value=0.0, key="run_matrix_youngs_modulus_input")
                st.number_input("Matrix Poisson ratio", min_value=0.0, max_value=0.49, key="run_matrix_poisson_ratio_input")
                st.number_input("Target stiffness", min_value=0.0, key="run_target_stiffness_input")
            with s2:
                st.number_input("Cell contractility", min_value=0.0, key="run_cell_contractility_input")
                st.number_input("Organoid radius", min_value=0.0, key="run_organoid_radius_input")
                st.caption(
                    f"FEBio status: {'configured' if config.febio_executable else 'unavailable'} | {config.febio_status_message}"
                )
            st.text_area(
                "Scenario request JSON",
                key="run_simulation_request_json_input",
                help="可选 JSON，用于补充固定 schema 字段，例如 bulk 的 sample_dimensions 或 spheroid 的 organoid_radial_displacement。",
            )

        if workflow in design_workflows:
            st.markdown("### Design Targets")
            g1, g2 = st.columns(2, gap="large")
            with g1:
                if workflow == "design":
                    st.number_input("Target stiffness", min_value=0.0, key="run_target_stiffness_input")
                else:
                    st.text_input("Campaign targets (Pa)", key="run_campaign_target_stiffnesses_input")
                st.number_input("Anisotropy", min_value=0.0, key="run_target_anisotropy_input")
                st.number_input("Connectivity", min_value=0.0, max_value=1.0, key="run_target_connectivity_input")
            with g2:
                st.number_input("Stress propagation", min_value=0.0, key="run_target_stress_propagation_input")
                st.number_input("Top K", min_value=1, step=1, key="run_design_top_k_input")
                st.number_input("Candidate budget", min_value=3, step=1, key="run_design_candidate_budget_input")
                st.number_input("MC runs", min_value=1, step=1, key="run_design_monte_carlo_runs_input")
            st.text_area(
                "Extra design targets (JSON)",
                key="run_design_extra_targets_json_input",
                help="可选 JSON 对象，例如 {\"loss_tangent_proxy\": 0.3, \"mesh_size_proxy\": 0.8}。",
            )

            st.markdown("### Condition-Aware Calibration Hints")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.number_input(
                    "Concentration hint (fraction)",
                    min_value=0.0,
                    key="run_condition_concentration_input",
                    help="例如 0.15 表示 15%。留空或 0 表示仅依赖 query 自动推断。",
                )
            with c2:
                st.number_input(
                    "Curing hint (seconds)",
                    min_value=0.0,
                    key="run_condition_curing_input",
                    help="例如 120。留空或 0 表示仅依赖 query 自动推断。",
                )
            st.text_area(
                "Additional condition hints (JSON)",
                key="run_condition_overrides_json_input",
                help="可选 JSON 对象，例如 {\"temperature_c\": 37, \"photoinitiator_fraction\": 0.0005, \"polymer_mw_kda\": 20, \"degree_substitution\": 0.6}。",
            )

            st.markdown("### Design Constraints")
            h1, h2 = st.columns(2, gap="large")
            with h1:
                st.number_input("Max anisotropy", min_value=0.0, key="run_constraint_max_anisotropy_input")
                st.number_input("Min connectivity", min_value=0.0, max_value=1.0, key="run_constraint_min_connectivity_input")
            with h2:
                st.number_input("Max risk index", min_value=0.0, key="run_constraint_max_risk_index_input")
                st.number_input("Min stress prop", min_value=0.0, key="run_constraint_min_stress_propagation_input")
            st.text_area(
                "Extra design constraints (JSON)",
                key="run_design_extra_constraints_json_input",
                help="可选 JSON 对象，例如 {\"max_loss_tangent_proxy\": 0.8, \"min_permeability_proxy\": 0.01}。",
            )

            st.markdown("### FEBio Verification")
            st.checkbox(
                "Run FEBio on top candidates",
                key="run_design_run_simulation_input",
                help="启用后会对 top-k design candidates 运行固定 FEBio 场景验证；若未安装 FEBio，会自动降级为仅 mechanics-informed 设计。",
            )
            sim1, sim2 = st.columns(2, gap="large")
            with sim1:
                st.selectbox(
                    "Design simulation scenario",
                    ["bulk_mechanics", "single_cell_contraction", "organoid_spheroid"],
                    key="run_design_simulation_scenario_input",
                )
                st.number_input("Design simulation top-k", min_value=1, step=1, key="run_design_simulation_top_k_input")
            with sim2:
                st.number_input("Cell contractility", min_value=0.0, key="run_cell_contractility_input")
                st.number_input("Organoid radius", min_value=0.0, key="run_organoid_radius_input")

        if workflow in data_workflows:
            st.markdown("### Data Pipeline Inputs")
            if workflow == "calibration":
                st.text_input("Dataset ID", key="run_dataset_id_input")
                st.number_input("Calibration samples", min_value=1, step=1, key="run_calibration_max_samples_input")
            else:
                st.caption("`datasets` 会根据 query 去发现、下载和规范化匹配的数据集。")

        if workflow == "benchmark":
            st.info("`benchmark` 不需要额外输入文件。它会直接运行 solver、scaling、inverse design 和 mechanics fit 的基准测试。")

        run_clicked = st.button(f"Run `{workflow}` Workflow", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_clicked:
            query = str(st.session_state.get("run_query_input", "")).strip()
            report_name = str(st.session_state.get("run_report_name_input", "")).strip()
            report_dir_value = str(st.session_state.get("run_report_dir_input", "")).strip()
            library_dir_value = str(st.session_state.get("run_library_dir_input", str(config.library_dir))).strip()
            data_path_value = str(st.session_state.get("run_data_path_input", "")).strip()
            dataset_id_value = str(st.session_state.get("run_dataset_id_input", "")).strip()
            validation_error = ""
            if not query:
                validation_error = "Goal / Query 不能为空。"
            elif not report_name:
                validation_error = "Report Filename 不能为空。"
            elif workflow in mechanics_workflows and not data_path_value:
                validation_error = "`mechanics` 和 `hybrid` 需要提供本地数据文件。"
            elif workflow == "calibration" and not dataset_id_value:
                validation_error = "`calibration` 需要提供 dataset_id。"

            if validation_error:
                st.error(validation_error)
            else:
                report_dir = Path(report_dir_value).expanduser() if report_dir_value else config.report_dir
                library_dir = Path(library_dir_value).expanduser() if library_dir_value else config.library_dir
                data_path = Path(data_path_value).expanduser() if data_path_value else None
                run_workflow(
                    config=config,
                    query=query,
                    workflow=workflow,
                    report_name=report_name,
                    report_dir=report_dir,
                    library_dir=library_dir,
                    max_pubmed_results=int(st.session_state.get("run_max_pubmed_results", config.max_pubmed_results)),
                    data_path=data_path,
                    experiment_type=str(st.session_state.get("run_experiment_type_input", "auto")),
                    time_column=str(st.session_state.get("run_time_column_input", "time")),
                    stress_column=str(st.session_state.get("run_stress_column_input", "stress")),
                    strain_column=str(st.session_state.get("run_strain_column_input", "strain")),
                    applied_stress=float(st.session_state.get("run_applied_stress_input", 0.0)) or None,
                    applied_strain=float(st.session_state.get("run_applied_strain_input", 0.0)) or None,
                    delimiter=str(st.session_state.get("run_delimiter_input", ",")),
                    simulation_fiber_density=float(st.session_state.get("run_simulation_fiber_density_input", 0.0)) or None,
                    simulation_fiber_stiffness=float(st.session_state.get("run_simulation_fiber_stiffness_input", 0.0)) or None,
                    simulation_bending_stiffness=float(st.session_state.get("run_simulation_bending_stiffness_input", 0.0)) or None,
                    simulation_crosslink_prob=float(st.session_state.get("run_simulation_crosslink_prob_input", 0.0)) or None,
                    simulation_domain_size=float(st.session_state.get("run_simulation_domain_size_input", 0.0)) or None,
                    target_stiffness=float(st.session_state.get("run_target_stiffness_input", 0.0)) or None,
                    simulation_scenario=str(st.session_state.get("run_simulation_scenario_input", "bulk_mechanics")),
                    simulation_request_json=str(st.session_state.get("run_simulation_request_json_input", "")),
                    cell_contractility=float(st.session_state.get("run_cell_contractility_input", 0.0)) or None,
                    organoid_radius=float(st.session_state.get("run_organoid_radius_input", 0.0)) or None,
                    matrix_youngs_modulus=float(st.session_state.get("run_matrix_youngs_modulus_input", 0.0)) or None,
                    matrix_poisson_ratio=float(st.session_state.get("run_matrix_poisson_ratio_input", 0.3)),
                    target_anisotropy=float(st.session_state.get("run_target_anisotropy_input", 0.1)),
                    target_connectivity=float(st.session_state.get("run_target_connectivity_input", 0.95)),
                    target_stress_propagation=float(st.session_state.get("run_target_stress_propagation_input", 0.5)),
                    design_extra_targets_json=str(st.session_state.get("run_design_extra_targets_json_input", "")),
                    constraint_max_anisotropy=float(st.session_state.get("run_constraint_max_anisotropy_input", 0.0)) or None,
                    constraint_min_connectivity=float(st.session_state.get("run_constraint_min_connectivity_input", 0.0)) or None,
                    constraint_max_risk_index=float(st.session_state.get("run_constraint_max_risk_index_input", 0.0)) or None,
                    constraint_min_stress_propagation=float(st.session_state.get("run_constraint_min_stress_propagation_input", 0.0)) or None,
                    design_extra_constraints_json=str(st.session_state.get("run_design_extra_constraints_json_input", "")),
                    design_top_k=int(st.session_state.get("run_design_top_k_input", 3)),
                    design_candidate_budget=int(st.session_state.get("run_design_candidate_budget_input", 12)),
                    design_monte_carlo_runs=int(st.session_state.get("run_design_monte_carlo_runs_input", 4)),
                    design_run_simulation=bool(st.session_state.get("run_design_run_simulation_input", False)),
                    design_simulation_scenario=str(st.session_state.get("run_design_simulation_scenario_input", "bulk_mechanics")),
                    design_simulation_top_k=int(st.session_state.get("run_design_simulation_top_k_input", 2)),
                    condition_concentration_fraction=float(st.session_state.get("run_condition_concentration_input", 0.0)) or None,
                    condition_curing_seconds=float(st.session_state.get("run_condition_curing_input", 0.0)) or None,
                    condition_overrides_json=str(st.session_state.get("run_condition_overrides_json_input", "")),
                    campaign_target_stiffnesses=str(st.session_state.get("run_campaign_target_stiffnesses_input", "6,8,10")),
                    dataset_id=dataset_id_value or None,
                    calibration_max_samples=int(st.session_state.get("run_calibration_max_samples_input", 6)),
                )

    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Workflow Brief")
        render_workflow_focus(workflow)
        st.caption("上面这张卡片对应的是后端真实 workflow，不是前端自造的页面概念。")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Example Walkthrough")
        render_example_walkthrough(default_workflow=workflow if workflow in workflow_examples() else "design_campaign")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Run Progress")
        stage_placeholder = st.empty()
        summary_placeholder = st.empty()

        events: List[Tuple[str, str]] = st.session_state.get("run_events", [])
        if events:
            stage_placeholder.markdown(render_stage_lines(events), unsafe_allow_html=True)
        else:
            stage_placeholder.markdown(
                '<p class="small-note">还没有运行记录。点击左侧按钮后，这里会显示多智能体各阶段进度。</p>',
                unsafe_allow_html=True,
            )

        result = st.session_state.get("last_result")
        if result:
            render_result_block(summary_placeholder, result)
        st.markdown("</div>", unsafe_allow_html=True)


def run_workflow(
    *,
    config: AppConfig,
    query: str,
    workflow: str,
    report_name: str,
    report_dir: Path,
    library_dir: Path,
    max_pubmed_results: int,
    data_path: Path | None,
    experiment_type: str,
    time_column: str,
    stress_column: str,
    strain_column: str,
    applied_stress: float | None,
    applied_strain: float | None,
    delimiter: str,
    simulation_fiber_density: float | None,
    simulation_fiber_stiffness: float | None,
    simulation_bending_stiffness: float | None,
    simulation_crosslink_prob: float | None,
    simulation_domain_size: float | None,
    target_stiffness: float | None,
    simulation_scenario: str,
    simulation_request_json: str,
    cell_contractility: float | None,
    organoid_radius: float | None,
    matrix_youngs_modulus: float | None,
    matrix_poisson_ratio: float | None,
    target_anisotropy: float,
    target_connectivity: float,
    target_stress_propagation: float,
    design_extra_targets_json: str,
    constraint_max_anisotropy: float | None,
    constraint_min_connectivity: float | None,
    constraint_max_risk_index: float | None,
    constraint_min_stress_propagation: float | None,
    design_extra_constraints_json: str,
    design_top_k: int,
    design_candidate_budget: int,
    design_monte_carlo_runs: int,
    design_run_simulation: bool,
    design_simulation_scenario: str,
    design_simulation_top_k: int,
    condition_concentration_fraction: float | None,
    condition_curing_seconds: float | None,
    condition_overrides_json: str,
    campaign_target_stiffnesses: str,
    dataset_id: str | None,
    calibration_max_samples: int,
) -> None:
    st.session_state["run_events"] = []
    progress_box = st.empty()

    def on_progress(stage: str, message: str) -> None:
        events = list(st.session_state.get("run_events", []))
        events.append((stage, message))
        st.session_state["run_events"] = events
        progress_box.markdown(render_stage_lines(events), unsafe_allow_html=True)

    with st.spinner("Agents are working through the research pipeline..."):
        result = run_research_agent_sync(
            project_dir=config.project_dir,
            query=query,
            workflow=workflow,
            report_name=report_name,
            report_dir=report_dir,
            library_dir=library_dir,
            max_pubmed_results=max_pubmed_results,
            data_path=data_path,
            experiment_type=experiment_type,
            time_column=time_column,
            stress_column=stress_column,
            strain_column=strain_column,
            applied_stress=applied_stress,
            applied_strain=applied_strain,
            delimiter=delimiter,
            simulation_fiber_density=simulation_fiber_density,
            simulation_fiber_stiffness=simulation_fiber_stiffness,
            simulation_bending_stiffness=simulation_bending_stiffness,
            simulation_crosslink_prob=simulation_crosslink_prob,
            simulation_domain_size=simulation_domain_size,
            target_stiffness=target_stiffness,
            simulation_scenario=simulation_scenario,
            simulation_request_json=simulation_request_json,
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
            target_anisotropy=target_anisotropy,
            target_connectivity=target_connectivity,
            target_stress_propagation=target_stress_propagation,
            design_extra_targets_json=design_extra_targets_json,
            constraint_max_anisotropy=constraint_max_anisotropy,
            constraint_min_connectivity=constraint_min_connectivity,
            constraint_max_risk_index=constraint_max_risk_index,
            constraint_min_stress_propagation=constraint_min_stress_propagation,
            design_extra_constraints_json=design_extra_constraints_json,
            design_top_k=design_top_k,
            design_candidate_budget=design_candidate_budget,
            design_monte_carlo_runs=design_monte_carlo_runs,
            design_run_simulation=design_run_simulation,
            design_simulation_scenario=design_simulation_scenario,
            design_simulation_top_k=design_simulation_top_k,
            condition_concentration_fraction=condition_concentration_fraction,
            condition_curing_seconds=condition_curing_seconds,
            condition_overrides_json=condition_overrides_json,
            campaign_target_stiffnesses=campaign_target_stiffnesses,
            dataset_id=dataset_id,
            calibration_max_samples=calibration_max_samples,
            progress_callback=on_progress,
        )

    st.session_state["last_result"] = {
        "workflow": result.workflow,
        "summary": result.final_summary,
        "report_path": str(result.report_path) if result.report_path else "",
        "run_dir": str(result.run_dir) if result.run_dir else "",
        "metadata_path": str(result.metadata_path) if result.metadata_path else "",
        "planner_output": result.planner_output,
        "search_output": result.search_output,
        "evidence_output": result.evidence_output,
        "hypothesis_output": result.hypothesis_output,
        "mechanics_output": result.mechanics_output,
        "simulation_output": result.simulation_output,
        "design_output": result.design_output,
        "formulation_output": result.formulation_output,
        "benchmark_output": result.benchmark_output,
        "dataset_output": result.dataset_output,
        "calibration_output": result.calibration_output,
        "critic_output": result.critic_output,
        "query": query,
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.rerun()


def demo_step_rows(project_dir: Path, *, include_ai_workflows: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for step in default_demo_steps(project_dir, include_ai_workflows=include_ai_workflows):
        rows.append(
            {
                "workflow": step.workflow,
                "title": step.title,
                "query": step.query,
                "description": step.description,
            }
        )
    return rows


def run_demo_workflow(*, config: AppConfig, include_ai_workflows: bool) -> None:
    st.session_state["demo_events"] = []
    progress_box = st.empty()

    def on_progress(stage: str, message: str) -> None:
        events = list(st.session_state.get("demo_events", []))
        events.append((stage, message))
        st.session_state["demo_events"] = events
        progress_box.markdown(render_stage_lines(events), unsafe_allow_html=True)

    with st.spinner("Running full ECM workflow demo..."):
        result = run_full_demo(
            project_dir=config.project_dir,
            include_ai_workflows=include_ai_workflows,
            progress_callback=on_progress,
        )

    st.session_state["last_demo_result"] = {
        "demo_id": result.demo_id,
        "include_ai_workflows": result.include_ai_workflows,
        "summary_path": str(result.summary_path),
        "manifest_path": str(result.manifest_path),
        "completed_steps": result.completed_steps,
        "failed_steps": result.failed_steps,
        "step_results": [
            {
                "slug": step.slug,
                "workflow": step.workflow,
                "title": step.title,
                "status": step.status,
                "report_path": str(step.report_path) if step.report_path else "",
                "run_dir": str(step.run_dir) if step.run_dir else "",
                "final_summary": step.final_summary,
                "error": step.error,
            }
            for step in result.step_results
        ],
    }
    st.rerun()


def report_excerpt(path: Path, *, max_lines: int = 32) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    excerpt = "\n".join(lines[:max_lines]).strip()
    if len(lines) > max_lines:
        excerpt += "\n\n..."
    return excerpt


def parse_markdown_sections(text: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_lines).strip(),
                    }
                )
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})
    return sections


def normalize_heading_token(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", normalized)
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def find_markdown_section(text: str, keywords: List[str]) -> Dict[str, str]:
    sections = parse_markdown_sections(text)
    normalized_keywords = [normalize_heading_token(keyword) for keyword in keywords]
    for section in sections:
        normalized_title = normalize_heading_token(section["title"])
        if any(keyword in normalized_title for keyword in normalized_keywords):
            return section
    return {}


def markdown_section_excerpt(section: Dict[str, str], *, max_lines: int = 10) -> str:
    content = str(section.get("content", "")).strip()
    if not content:
        return ""
    lines = [line for line in content.splitlines() if line.strip()]
    excerpt = "\n".join(lines[:max_lines]).strip()
    if len(lines) > max_lines:
        excerpt += "\n\n..."
    return excerpt


def render_demo_section_block(title: str, content: str, *, key: str) -> None:
    if not content.strip():
        return
    st.markdown(f"#### {title}")
    render_markdown_or_code(content, key=key)


def demo_step_report_path(step_payload: Dict[str, Any]) -> Optional[Path]:
    raw_value = str(step_payload.get("report_path", "")).strip()
    return Path(raw_value) if raw_value else None


def demo_step_run_dir(step_payload: Dict[str, Any]) -> Optional[Path]:
    raw_value = str(step_payload.get("run_dir", "")).strip()
    return Path(raw_value) if raw_value else None


def demo_dataset_rows(run_dir: Path) -> List[Dict[str, Any]]:
    payload = load_json_file(run_dir / "dataset_manifest_snapshot.json")
    rows: List[Dict[str, Any]] = []
    for item in payload.get("datasets", []):
        rows.append(
            {
                "dataset_id": item.get("dataset_id", "NR"),
                "title": item.get("title", "NR"),
                "source": item.get("source", "NR"),
                "status": item.get("status", "NR"),
                "file_count": item.get("file_count", "NR"),
            }
        )
    return rows


def demo_calibration_rows(run_dir: Path) -> List[Dict[str, Any]]:
    payload = load_json_file(run_dir / "calibration_results.json")
    rows: List[Dict[str, Any]] = []
    for item in payload.get("family_priors", []):
        parameter_priors = item.get("parameter_priors", {})
        rows.append(
            {
                "material_family": item.get("material_family", "NR"),
                "sample_count": item.get("sample_count", "NR"),
                "mean_abs_error": round(float(item.get("mean_abs_error", 0.0)), 4),
                "fiber_density_mean": round(float(parameter_priors.get("fiber_density", {}).get("mean", 0.0)), 4),
                "fiber_stiffness_mean": round(float(parameter_priors.get("fiber_stiffness", {}).get("mean", 0.0)), 4),
            }
        )
    return rows


def mechanics_demo_snapshot(run_dir: Path) -> Dict[str, Any]:
    tool_rows = load_jsonl_file(run_dir / "tool_calls.jsonl")
    fit_payload = latest_tool_result(tool_rows, "fit_mechanics_model")
    simulation_payload = latest_tool_result(tool_rows, "simulate_mechanics_model")
    return {"fit": fit_payload, "simulation": simulation_payload}


def benchmark_demo_rows(run_dir: Path) -> List[Dict[str, Any]]:
    payload = load_json_file(run_dir / "benchmark_summary.json")
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return [
        {"metric": "overall_pass", "value": format_display_value(summary.get("overall_pass"))},
        {"metric": "solver_pass_rate", "value": format_display_value(summary.get("solver_pass_rate"))},
        {"metric": "load_ladder_monotonic", "value": format_display_value(summary.get("load_ladder_monotonic"))},
        {"metric": "scaling_pass_count", "value": format_display_value(summary.get("scaling_pass_count"))},
        {"metric": "inverse_design_mean_abs_error", "value": format_display_value(summary.get("inverse_design_mean_abs_error"))},
        {"metric": "identifiability_risk", "value": format_display_value(summary.get("identifiability_risk"))},
        {"metric": "fit_mean_relative_error", "value": format_display_value(summary.get("fit_mean_relative_error"))},
        {"metric": "simulation_smoke_status", "value": format_display_value(summary.get("simulation_smoke_status"))},
        {"metric": "simulation_smoke_pass", "value": format_display_value(summary.get("simulation_smoke_pass"))},
        {"metric": "simulation_smoke_effective_stiffness", "value": format_display_value(summary.get("simulation_smoke_effective_stiffness"))},
        {"metric": "simulation_smoke_target_mismatch_score", "value": format_display_value(summary.get("simulation_smoke_target_mismatch_score"))},
        {"metric": "calibration_design_improvement", "value": format_display_value(summary.get("calibration_design_improvement"))},
        {"metric": "calibration_cached_from_run", "value": format_display_value(summary.get("calibration_cached_from_run"))},
        {"metric": "calibration_routing_modes", "value": format_display_value(summary.get("calibration_routing_modes"))},
        {"metric": "calibration_routing_mode_counts", "value": format_display_value(summary.get("calibration_routing_mode_counts"))},
    ]


def demo_highlight_rows(step_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for step in step_rows:
        workflow = str(step.get("workflow", ""))
        run_dir = demo_step_run_dir(step)
        report_path = demo_step_report_path(step)
        report_text = (
            report_path.read_text(encoding="utf-8", errors="ignore")
            if report_path is not None and report_path.exists()
            else ""
        )
        if workflow in {"team", "single"} and report_text:
            judgement = markdown_section_excerpt(
                find_markdown_section(report_text, ["研究判断与模式总结", "judgement", "patterns"]),
                max_lines=3,
            )
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": judgement.replace("\n", " ")[:180] if judgement else "Literature synthesis completed",
                }
            )
            continue
        if workflow == "hybrid" and report_text:
            decision = markdown_section_excerpt(
                find_markdown_section(report_text, ["design decision", "next experimental step"]),
                max_lines=3,
            )
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": decision.replace("\n", " ")[:180] if decision else "Hybrid report completed",
                }
            )
            continue
        if workflow == "calibration" and run_dir is not None:
            payload = load_json_file(run_dir / "calibration_results.json")
            family_count = len(payload.get("family_priors", [])) if isinstance(payload, dict) else 0
            rows.append({"workflow": workflow, "highlight": f"{family_count} calibrated family prior(s)"})
        elif workflow == "mechanics" and run_dir is not None:
            fit_payload = mechanics_demo_snapshot(run_dir).get("fit", {})
            fit = fit_payload.get("fit", {}) if isinstance(fit_payload, dict) else {}
            selected_model = fit_payload.get("selected_model", fit_payload.get("model_type", "NR")) if isinstance(fit_payload, dict) else "NR"
            primary_value = (
                fit.get("elastic_modulus")
                or fit.get("instantaneous_modulus")
                or fit.get("equilibrium_modulus")
                or fit.get("secant_modulus")
                or fit.get("coefficient")
            )
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": f"model={selected_model}, primary={format_display_value(primary_value)}, identifiability={format_display_value(fit_payload.get('identifiability', {}).get('risk') if isinstance(fit_payload.get('identifiability'), dict) else 'NR')}",
                }
            )
        elif workflow == "design" and run_dir is not None:
            summary = design_run_summary(run_dir)
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": (
                        f"best stiffness={summary.get('best_stiffness_mean', 'NR')}, "
                        f"family={summary.get('best_material_family', 'NR')}, "
                        f"prior={summary.get('calibration_prior_level', 'NR')}"
                    ),
                }
            )
        elif workflow == "design_campaign" and run_dir is not None:
            campaign = design_campaign_snapshot(run_dir)
            result_rows = campaign_result_rows(campaign)
            feasible_count = sum(1 for row in result_rows if row.get("feasible"))
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": (
                        f"{feasible_count}/{len(result_rows)} feasible target window(s), "
                        f"prior={campaign.get('calibration_context', {}).get('prior_level', 'NR') if isinstance(campaign, dict) else 'NR'}"
                    ),
                }
            )
        elif workflow == "benchmark" and run_dir is not None:
            payload = load_json_file(run_dir / "benchmark_summary.json")
            summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
            rows.append(
                {
                    "workflow": workflow,
                    "highlight": f"overall_pass={format_display_value(summary.get('overall_pass'))}, identifiability={format_display_value(summary.get('identifiability_risk'))}",
                }
            )
    return rows


def render_demo_step_outcome(step_payload: Dict[str, Any]) -> None:
    workflow = str(step_payload.get("workflow", ""))
    title = str(step_payload.get("title", ""))
    status = str(step_payload.get("status", ""))
    summary = str(step_payload.get("final_summary", "")).strip()
    error = str(step_payload.get("error", "")).strip()
    report_path = demo_step_report_path(step_payload)
    run_dir = demo_step_run_dir(step_payload)

    status_cols = st.columns(4)
    status_cols[0].metric("Workflow", workflow)
    status_cols[1].metric("Status", status)
    status_cols[2].metric("Has report", "yes" if report_path else "no")
    status_cols[3].metric("Has run dir", "yes" if run_dir else "no")

    st.markdown(f"**{title}**")
    if error:
        st.error(error)
    elif summary:
        st.markdown(summary)

    if run_dir is not None and workflow == "datasets":
        rows = demo_dataset_rows(run_dir)
        if rows:
            st.markdown("Dataset Snapshot")
            st.dataframe(rows, use_container_width=True, hide_index=True)

    if run_dir is not None and workflow == "calibration":
        rows = demo_calibration_rows(run_dir)
        if rows:
            st.markdown("Calibration Snapshot")
            st.dataframe(rows, use_container_width=True, hide_index=True)

    if run_dir is not None and workflow == "mechanics":
        snapshot = mechanics_demo_snapshot(run_dir)
        fit_payload = snapshot.get("fit", {})
        fit = fit_payload.get("fit", {}) if isinstance(fit_payload, dict) else {}
        if fit:
            st.markdown("Mechanics Fit")
            fit_rows = {
                "experiment_type": fit_payload.get("experiment_type"),
                "selected_model": fit_payload.get("selected_model", fit_payload.get("model_type")),
                "sample_count": fit_payload.get("sample_count"),
                "identifiability_risk": fit_payload.get("identifiability", {}).get("risk") if isinstance(fit_payload.get("identifiability"), dict) else "NR",
                "modulus": fit.get("modulus"),
                "elastic_modulus": fit.get("elastic_modulus"),
                "instantaneous_modulus": fit.get("instantaneous_modulus"),
                "equilibrium_modulus": fit.get("equilibrium_modulus"),
                "delayed_modulus": fit.get("delayed_modulus"),
                "dynamic_modulus": fit.get("dynamic_modulus"),
                "viscosity": fit.get("viscosity"),
                "branch_viscosity": fit.get("branch_viscosity"),
                "maxwell_viscosity": fit.get("maxwell_viscosity"),
                "kelvin_viscosity": fit.get("kelvin_viscosity"),
                "relaxation_time": fit.get("relaxation_time"),
                "retardation_time": fit.get("retardation_time"),
                "coefficient": fit.get("coefficient"),
                "exponent": fit.get("exponent"),
                "loss_factor": fit.get("loss_factor"),
                "hysteresis_area": fit.get("hysteresis_area"),
                "mse": fit.get("mse"),
            }
            render_mapping_table(
                fit_rows,
                order=[
                    "experiment_type",
                    "selected_model",
                    "sample_count",
                    "identifiability_risk",
                    "modulus",
                    "elastic_modulus",
                    "instantaneous_modulus",
                    "equilibrium_modulus",
                    "delayed_modulus",
                    "dynamic_modulus",
                    "viscosity",
                    "branch_viscosity",
                    "maxwell_viscosity",
                    "kelvin_viscosity",
                    "relaxation_time",
                    "retardation_time",
                    "coefficient",
                    "exponent",
                    "loss_factor",
                    "hysteresis_area",
                    "mse",
                ],
            )
        simulation_payload = snapshot.get("simulation", {})
        x_values = simulation_payload.get("x_values", []) if isinstance(simulation_payload, dict) else []
        y_values = simulation_payload.get("y_values", []) if isinstance(simulation_payload, dict) else []
        if x_values and y_values and len(x_values) == len(y_values):
            st.markdown("Simulated Curve")
            st.line_chart(
                [{"x": x, "y": y} for x, y in zip(x_values, y_values)],
                x="x",
                y="y",
                height=220,
            )
        secondary_y = simulation_payload.get("secondary_y_values", []) if isinstance(simulation_payload, dict) else []
        if x_values and secondary_y and len(x_values) == len(secondary_y):
            st.markdown("Secondary Simulated Curve")
            st.line_chart(
                [{"x": x, "y": y} for x, y in zip(x_values, secondary_y)],
                x="x",
                y="y",
                height=220,
            )

    if run_dir is not None and workflow == "design":
        summary_row = design_run_summary(run_dir)
        st.markdown("Design Snapshot")
        render_mapping_table(
            summary_row,
            order=[
                "target_stiffness",
                "physics_valid",
                "feasible",
                "best_material_family",
                "best_stiffness_mean",
                "stiffness_error",
                "best_crosslinking_strategy",
                "top_candidate_score",
            ],
        )
        snapshot = design_run_snapshot(run_dir)
        candidate_rows = design_candidate_rows(snapshot.get("design_payload", {}))
        if candidate_rows:
            st.markdown("Top Candidates")
            st.dataframe(candidate_rows[:3], use_container_width=True, hide_index=True)
        formulation_rows = formulation_recommendation_rows(snapshot.get("formulation_recommendations", []))
        if formulation_rows:
            st.markdown("Formulation Translation")
            st.dataframe(formulation_rows[:3], use_container_width=True, hide_index=True)

    if run_dir is not None and workflow == "design_campaign":
        snapshot = design_campaign_snapshot(run_dir)
        rows = campaign_result_rows(snapshot)
        if rows:
            st.markdown("Campaign Comparison")
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.line_chart(
                [{"target_stiffness": row["target_stiffness"], "stiffness_mean": row["stiffness_mean"]} for row in rows],
                x="target_stiffness",
                y="stiffness_mean",
                height=240,
            )

    if run_dir is not None and workflow == "benchmark":
        rows = benchmark_demo_rows(run_dir)
        st.markdown("Benchmark Summary")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if report_path is not None and report_path.exists():
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        if workflow in {"team", "single"}:
            st.markdown("Research Snapshot")
            overview_tab, evidence_tab, decision_tab, report_tab = st.tabs(["Question", "Evidence", "Decisions", "Report"])
            with overview_tab:
                render_demo_section_block(
                    "Research Question",
                    markdown_section_excerpt(find_markdown_section(report_text, ["本周研究问题", "research question"]), max_lines=8),
                    key=f"demo_{workflow}_question",
                )
                render_demo_section_block(
                    "Search Strategy",
                    markdown_section_excerpt(find_markdown_section(report_text, ["检索策略", "search strategy"]), max_lines=10),
                    key=f"demo_{workflow}_search_strategy",
                )
            with evidence_tab:
                render_demo_section_block(
                    "Key Evidence",
                    markdown_section_excerpt(find_markdown_section(report_text, ["关键证据", "literature evidence"]), max_lines=14),
                    key=f"demo_{workflow}_evidence",
                )
            with decision_tab:
                render_demo_section_block(
                    "Research Judgement",
                    markdown_section_excerpt(find_markdown_section(report_text, ["研究判断与模式总结", "patterns", "judgement"]), max_lines=12),
                    key=f"demo_{workflow}_judgement",
                )
                render_demo_section_block(
                    "Gaps And Risks",
                    markdown_section_excerpt(find_markdown_section(report_text, ["证据空白与风险", "limitations", "risk"]), max_lines=12),
                    key=f"demo_{workflow}_gaps",
                )
                render_demo_section_block(
                    "Next Steps",
                    markdown_section_excerpt(find_markdown_section(report_text, ["下周行动清单", "next step", "next experimental step"]), max_lines=12),
                    key=f"demo_{workflow}_next_steps",
                )
            with report_tab:
                preview_text = report_excerpt(report_path)
                render_markdown_or_code(preview_text, key=f"demo_report_preview_{workflow}")
        elif workflow == "hybrid":
            st.markdown("Hybrid Research Snapshot")
            goal_tab, evidence_tab, mechanics_tab, decision_tab, report_tab = st.tabs(
                ["Goal", "Evidence", "Mechanics", "Decision", "Report"]
            )
            with goal_tab:
                render_demo_section_block(
                    "Research And Engineering Goal",
                    markdown_section_excerpt(find_markdown_section(report_text, ["research and engineering goal", "modeling goal", "本周研究问题"]), max_lines=10),
                    key="demo_hybrid_goal",
                )
                render_demo_section_block(
                    "Material Strategy",
                    markdown_section_excerpt(find_markdown_section(report_text, ["suggested ecm material strategy", "material strategy"]), max_lines=12),
                    key="demo_hybrid_strategy",
                )
            with evidence_tab:
                render_demo_section_block(
                    "Literature Evidence",
                    markdown_section_excerpt(find_markdown_section(report_text, ["literature evidence", "关键证据"]), max_lines=14),
                    key="demo_hybrid_evidence",
                )
            with mechanics_tab:
                render_demo_section_block(
                    "Mechanics Parameters",
                    markdown_section_excerpt(find_markdown_section(report_text, ["mechanics modeling parameters", "estimated parameters"]), max_lines=14),
                    key="demo_hybrid_mechanics",
                )
                render_demo_section_block(
                    "Simulation Outputs",
                    markdown_section_excerpt(find_markdown_section(report_text, ["simulation outputs", "fiber-network simulation setup"]), max_lines=14),
                    key="demo_hybrid_simulation",
                )
            with decision_tab:
                render_demo_section_block(
                    "Design Decision",
                    markdown_section_excerpt(find_markdown_section(report_text, ["design decision", "研究判断与模式总结"]), max_lines=12),
                    key="demo_hybrid_decision",
                )
                render_demo_section_block(
                    "Limitations",
                    markdown_section_excerpt(find_markdown_section(report_text, ["model and evidence limitations", "limitations"]), max_lines=12),
                    key="demo_hybrid_limits",
                )
                render_demo_section_block(
                    "Next Experimental Step",
                    markdown_section_excerpt(find_markdown_section(report_text, ["next experimental step", "下周行动清单"]), max_lines=12),
                    key="demo_hybrid_next",
                )
            with report_tab:
                preview_text = report_excerpt(report_path)
                render_markdown_or_code(preview_text, key=f"demo_report_preview_{workflow}")
        else:
            st.markdown("Report Preview")
            preview_text = report_excerpt(report_path)
            render_markdown_or_code(preview_text, key=f"demo_report_preview_{workflow}")

    if run_dir is not None:
        with st.expander("Run Metadata", expanded=False):
            metadata = load_json_file(run_dir / "metadata.json")
            if metadata:
                render_mapping_table(metadata)


def render_demo_result(payload: Dict[str, Any]) -> None:
    summary_path = Path(str(payload.get("summary_path", ""))) if payload.get("summary_path") else None
    manifest_path = Path(str(payload.get("manifest_path", ""))) if payload.get("manifest_path") else None

    st.markdown("### Demo Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Demo ID", str(payload.get("demo_id", "NR")))
    summary_cols[1].metric("AI Workflows", "yes" if payload.get("include_ai_workflows") else "no")
    summary_cols[2].metric("Completed", int(payload.get("completed_steps", 0)))
    summary_cols[3].metric("Failed", int(payload.get("failed_steps", 0)))

    step_rows = list(payload.get("step_results", []))
    if step_rows:
        st.markdown("### Outcome Highlights")
        highlight_rows = demo_highlight_rows(step_rows)
        if highlight_rows:
            st.dataframe(highlight_rows, width="stretch", hide_index=True)

        compact_rows = [
            {
                "workflow": row.get("workflow", ""),
                "title": row.get("title", ""),
                "status": row.get("status", ""),
                "report_path": row.get("report_path", "") or "NR",
            }
            for row in step_rows
        ]
        st.markdown("### Workflow Status")
        st.dataframe(compact_rows, width="stretch", hide_index=True)

        st.markdown("### Workflow Results")
        step_tabs = st.tabs([f"{row.get('workflow', 'workflow')} | {row.get('status', 'unknown')}" for row in step_rows])
        for tab, row in zip(step_tabs, step_rows):
            with tab:
                render_demo_step_outcome(row)

    if summary_path and summary_path.exists():
        with st.expander("Demo Summary Report", expanded=False):
            st.markdown(f"**Summary File**: `{summary_path}`")
            summary_text = summary_path.read_text(encoding="utf-8", errors="ignore")
            st.download_button(
                "Download Demo Summary",
                summary_text,
                file_name=summary_path.name,
                mime="text/markdown",
                width="stretch",
                key="demo_summary_download",
            )
            render_markdown_or_code(summary_text, key="demo_summary_preview")

    if manifest_path and manifest_path.exists():
        with st.expander("Demo Manifest", expanded=False):
            render_artifact_file(manifest_path, key="demo_manifest_preview")


def render_demo_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Full Workflow Demo")
    st.caption("这个 demo 会顺序跑完整的 ECM workflow 栈，并输出汇总 summary 与 manifest。")

    include_ai_workflows = st.checkbox(
        "Include model-backed workflows (`team`, `single`, `hybrid`)",
        value=True,
        key="demo_include_ai_workflows",
        help="关闭后会只跑确定性工作流，速度更快，也不调用模型 API。",
    )
    st.info(
        "建议第一次先完整跑一遍，让使用者看到 literature -> datasets -> calibration -> mechanics -> hybrid -> design -> campaign -> benchmark 的完整链路。"
    )
    st.markdown("### Demo Plan")
    st.dataframe(demo_step_rows(config.project_dir, include_ai_workflows=include_ai_workflows), width="stretch", hide_index=True)

    if st.button("Run Full Demo", width="stretch", key="run_full_demo_button"):
        run_demo_workflow(config=config, include_ai_workflows=include_ai_workflows)

    st.markdown("### Demo Progress")
    demo_events: List[Tuple[str, str]] = st.session_state.get("demo_events", [])
    if demo_events:
        st.markdown(render_stage_lines(demo_events), unsafe_allow_html=True)
    else:
        st.caption("点击上方按钮后，这里会显示整套 demo 的批量运行进度。")

    demo_payload = st.session_state.get("last_demo_result")
    if demo_payload:
        render_demo_result(demo_payload)

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_block(container, result: Dict[str, str]) -> None:
    report_path = Path(result["report_path"]) if result.get("report_path") else None
    container.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(f"**Last Run**: `{result['ran_at']}`")
    st.markdown(f"**Workflow**: `{result['workflow']}`")
    st.markdown(f"**Query**: {result['query']}")
    st.markdown("**Summary**")
    st.markdown(result["summary"])
    if result.get("run_dir"):
        st.markdown(f"**Run Artifacts**: `{result['run_dir']}`")

    if report_path and report_path.exists():
        st.markdown(f"**Saved Report**: `{report_path}`")
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        st.download_button(
            "Download Report Markdown",
            report_text,
            file_name=report_path.name,
            mime="text/markdown",
            width="stretch",
        )
        with st.expander("Preview Markdown Report", expanded=True):
            st.markdown(report_text)

    render_text_output_expander("PlannerAgent Output", result.get("planner_output") or "", key="planner_output")
    render_text_output_expander("SearchAgent Output", result.get("search_output") or "", key="search_output")
    render_text_output_expander("EvidenceAgent Output", result.get("evidence_output") or "", key="evidence_output")
    render_text_output_expander("HypothesisAgent Output", result.get("hypothesis_output") or "", key="hypothesis_output")
    render_text_output_expander("MechanicsAgent Output", result.get("mechanics_output") or "", key="mechanics_output")
    render_text_output_expander("SimulationAgent Output", result.get("simulation_output") or "", key="simulation_output")
    render_text_output_expander("DesignAgent Output", result.get("design_output") or "", key="design_output")
    render_text_output_expander("Formulation Mapping", result.get("formulation_output") or "", key="formulation_output")
    render_text_output_expander("Benchmark Output", result.get("benchmark_output") or "", key="benchmark_output")
    render_text_output_expander("Dataset Output", result.get("dataset_output") or "", key="dataset_output")
    render_text_output_expander("Calibration Output", result.get("calibration_output") or "", key="calibration_output")
    render_text_output_expander("CriticAgent Output", result.get("critic_output") or "", key="critic_output")
    container.markdown("</div>", unsafe_allow_html=True)


def render_reports_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Reports Archive")
    report_files = sorted(config.report_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not report_files:
        st.info("`reports/` 里还没有 Markdown 报告。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    labels = [f"{path.name}  |  {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}" for path in report_files]
    choice = st.selectbox("Choose a report", labels, index=0)
    selected = report_files[labels.index(choice)]
    report_text = selected.read_text(encoding="utf-8", errors="ignore")

    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        st.markdown(report_text)
    with c2:
        st.caption("Report Metadata")
        st.code(str(selected))
        st.caption(f"Size: {selected.stat().st_size} bytes")
        st.download_button(
            "Download Markdown",
            report_text,
            file_name=selected.name,
            mime="text/markdown",
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_library_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Local Library Status")
    files = sorted([path for path in config.library_dir.rglob("*") if path.is_file()])
    st.caption(f"Directory: `{config.library_dir}`")
    if not files:
        st.warning("`library/` 目前没有可检索文件。你可以放入 PDF、Markdown 或文本笔记。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    for path in files[:80]:
        rows.append(
            {
                "name": path.name,
                "type": path.suffix.lower() or "file",
                "size_kb": round(path.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_guide_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Guide")
    st.caption("把这套系统当成 ECM mechanics design cockpit 来用，而不是普通聊天界面。")

    left, right = st.columns([0.66, 0.34], gap="large")
    with left:
        st.markdown(guide_markdown())
    with right:
        st.markdown(
            """
            <div class="guide-callout">
                <strong>Recommended flow</strong><br>
                先跑 <code>design_campaign</code> 选目标窗口，再用 <code>design</code> 深挖单个窗口，最后把配方模板带去 wet-lab screening。
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="mini-card"><h4>Where To Look First</h4><p>如果你只看一个地方，先看 <code>Design Board</code>。那里现在会同时显示单次 design、campaign、最佳材料家族和配方模板。</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="mini-card"><h4>What Not To Overread</h4><p>配方翻译是 mechanics-informed starting recipe，不是最终实验 SOP。真正的校准仍然要靠你们的 rheology / viability / morphology 数据。</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="mini-card"><h4>Artifact Logic</h4><p><code>design_summary.json</code> 和 <code>campaign_summary.json</code> 是结构化真源；Markdown 报告负责叙事说明。</p></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_settings_tab(config: AppConfig) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Runtime Settings")
    env_status = [
        {"item": "Provider", "value": config.model_provider},
        {"item": "Model", "value": config.model},
        {"item": "API Key", "value": "Configured" if config.model_api_key else "Missing"},
        {"item": "Base URL", "value": config.model_base_url or "Default"},
        {"item": "Project Dir", "value": str(config.project_dir)},
        {"item": "Report Dir", "value": str(config.report_dir)},
        {"item": "Runs Dir", "value": str(config.runs_dir)},
        {"item": "Cache Dir", "value": str(config.cache_dir)},
        {"item": "Library Dir", "value": str(config.library_dir)},
        {"item": "FEBio", "value": config.febio_executable or "Unavailable"},
        {"item": "FEBio Status", "value": config.febio_status_message},
        {"item": "Login Protection", "value": "Enabled" if config.frontend_require_login else "Disabled"},
        {"item": "Current User", "value": st.session_state.get("auth_user") or "Anonymous"},
        {"item": "Public Host", "value": config.frontend_public_host},
        {"item": "Public Port", "value": str(config.frontend_public_port)},
    ]
    st.dataframe(env_status, width="stretch", hide_index=True)
    st.caption("当前前端不直接编辑密钥，继续沿用 `.env`。这样更稳，也更容易控制安全边界。")
    if config.frontend_require_login and st.session_state.get("auth_ok"):
        if st.button("Log Out", width="stretch"):
            queue_auth_cookie_clear()
            st.session_state["auth_ok"] = False
            st.session_state["auth_user"] = ""
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_stage_lines(events: List[Tuple[str, str]]) -> str:
    pieces = []
    for stage, message in events:
        label = stage.capitalize()
        pieces.append(f'<div class="stage-line"><strong>{label}</strong><br>{message}</div>')
    return "".join(pieces)


def default_report_name() -> str:
    return f"research_report_{datetime.now():%Y%m%d_%H%M}.md"


def system_layer_rows() -> List[Dict[str, str]]:
    return [
        {
            "layer": "Experience Layer",
            "role": "Streamlit 工作台和桌面壳，负责启动、导航、报告预览和运行控制。",
            "modules": "frontend.py, desktop.py",
        },
        {
            "layer": "Workflow Orchestration",
            "role": "统一 CLI / UI 参数，路由 9 条 workflow，写入 runs/ artifacts 和 metadata。",
            "modules": "__main__.py, runner.py, artifacts.py",
        },
        {
            "layer": "Scientific Engines",
            "role": "力学拟合、纤维网络模拟、FEBio 场景仿真、逆向设计、配方映射和实验校准等确定性核心。",
            "modules": "mechanics.py, fiber_network.py, febio/, formulation.py, calibration.py",
        },
        {
            "layer": "Evidence And Data",
            "role": "PubMed/Crossref/本地 library/公共 dataset 接入，以及 cache 与工作区状态管理。",
            "modules": "tools.py, datasets.py, workspace.py, config.py",
        },
    ]


def workflow_specs() -> Dict[str, Dict[str, object]]:
    return {
        "team": {
            "category": "Evidence Synthesis",
            "headline": "多智能体文献协作链",
            "when_to_use": "需要周报、证据表、研究判断和下一步实验假设时。",
            "requires": "研究问题；可选本地 library。",
            "writes": ["planner_agent", "search_agent", "evidence_agent", "hypothesis_agent", "critic_agent", "final_summary"],
        },
        "single": {
            "category": "Evidence Synthesis",
            "headline": "单智能体快速研究备忘录",
            "when_to_use": "想快速得到一份简化版检索与报告，不需要完整协作轨迹时。",
            "requires": "研究问题；可选本地 library。",
            "writes": ["single_agent_prompt", "final_summary"],
        },
        "mechanics": {
            "category": "Mechanics Fit",
            "headline": "实验力学数据拟合",
            "when_to_use": "已有 creep / relaxation / elastic / frequency sweep / cyclic 数据，希望得到本构参数与属性摘要时。",
            "requires": "数据路径、实验类型、列名和载荷条件。",
            "writes": ["mechanics_planner", "mechanics_agent", "mechanics_critic", "final_summary"],
        },
        "hybrid": {
            "category": "Integrated Analysis",
            "headline": "文献 + 力学 + 网络模拟联合报告",
            "when_to_use": "希望把证据、拟合参数和模拟解释串成一份可读报告时。",
            "requires": "研究问题、力学数据，以及可选 simulation override。",
            "writes": ["planner_agent", "search_agent", "evidence_agent", "hypothesis_agent", "mechanics_agent", "simulation_agent", "critic_agent", "final_summary"],
        },
        "simulation": {
            "category": "Biomechanics Simulation",
            "headline": "固定场景 FEBio 仿真评估",
            "when_to_use": "需要把候选 ECM 参数送入固定 FEBio 模板，得到结构化仿真指标时。",
            "requires": "simulation scenario，以及矩阵模量/收缩/球体半径等结构化参数。",
            "writes": ["simulation_agent", "simulation/final_summary", "simulation/simulation_result", "simulation/simulation_metrics", "final_summary"],
        },
        "design": {
            "category": "Inverse Design",
            "headline": "单目标 ECM 逆向设计",
            "when_to_use": "已经知道目标 stiffness window，想拿到 top-k 候选、配方模板，并可选叠加 FEBio 验证时。",
            "requires": "目标 stiffness / anisotropy / connectivity / stress propagation 与约束；可选 FEBio scenario。",
            "writes": ["design_validation", "design_agent", "design_sensitivity", "design_simulation", "formulation_mapping", "design_summary", "final_summary"],
        },
        "design_campaign": {
            "category": "Inverse Design",
            "headline": "多窗口 campaign 比较",
            "when_to_use": "需要跨多个 stiffness target 比较同一材料家族的可达性时。",
            "requires": "多个 target stiffness、目标结构指标与约束。",
            "writes": ["campaign_validation", "campaign_agent", "formulation_mapping", "campaign_summary", "final_summary"],
        },
        "benchmark": {
            "category": "Core Validation",
            "headline": "数值与设计核心基准测试",
            "when_to_use": "要检验 solver、scaling、inverse design 和 fit 核心表现时。",
            "requires": "benchmark 主题描述即可。",
            "writes": ["benchmark_solver", "benchmark_load_ladder", "benchmark_scaling", "benchmark_design", "benchmark_repeatability", "benchmark_identifiability", "benchmark_fit", "benchmark_simulation_smoke", "benchmark_calibration_design", "benchmark_summary", "final_summary"],
        },
        "datasets": {
            "category": "Data Pipeline",
            "headline": "公共数据集发现、下载与规范化",
            "when_to_use": "要接入 ECM / hydrogel 力学公开数据并整理到本地工作区时。",
            "requires": "与数据相关的 query；无需实验文件。",
            "writes": ["dataset_search", "dataset_register", "dataset_download", "dataset_normalize", "dataset_manifest_snapshot", "final_summary"],
        },
        "calibration": {
            "category": "Data Pipeline",
            "headline": "实验数据到 family priors 的校准",
            "when_to_use": "已经有 normalized dataset，希望把实验测量转成 design search priors 时。",
            "requires": "dataset_id 与可选 calibration sample 上限。",
            "writes": ["calibration_targets", "calibration_results", "calibration_impact", "final_summary"],
        },
    }


def workflow_catalog_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for workflow, spec in workflow_specs().items():
        rows.append(
            {
                "workflow": workflow,
                "category": str(spec["category"]),
                "headline": str(spec["headline"]),
                "when_to_use": str(spec["when_to_use"]),
                "requires": str(spec["requires"]),
                "writes": ", ".join(str(name) for name in spec["writes"]),
            }
        )
    return rows


def workflow_artifact_labels(workflow: str) -> List[str]:
    spec = workflow_specs().get(workflow, {})
    writes = spec.get("writes", [])
    return [str(item) for item in writes] if isinstance(writes, list) else []


def render_system_layer_cards() -> None:
    columns = st.columns(2, gap="medium")
    for index, row in enumerate(system_layer_rows()):
        with columns[index % 2]:
            with st.container(border=True):
                st.caption(row["layer"])
                st.markdown(f"**{row['modules']}**")
                st.write(row["role"])


def render_workflow_atlas(*, workflows: List[str] | None = None, active_workflow: str | None = None) -> None:
    ordered = workflows or list(workflow_specs().keys())
    pieces = []
    for workflow in ordered:
        spec = workflow_specs().get(workflow)
        if spec is None:
            continue
        active_class = " is-active" if workflow == active_workflow else ""
        artifact_pills = "".join(
            f'<span class="artifact-pill">{artifact}</span>'
            for artifact in workflow_artifact_labels(workflow)[:6]
        )
        pieces.append(
            f"""
            <div class="atlas-card{active_class}">
                <div class="card-eyebrow">{spec["category"]}</div>
                <h4>{workflow}</h4>
                <p>{spec["headline"]}</p>
                <div class="card-meta"><strong>Use when:</strong> {spec["when_to_use"]}</div>
                <div class="card-meta"><strong>Needs:</strong> {spec["requires"]}</div>
                <div class="pill-row">{artifact_pills}</div>
            </div>
            """
        )
    st.markdown(f'<div class="atlas-grid">{"".join(pieces)}</div>', unsafe_allow_html=True)


def render_workflow_focus(workflow: str) -> None:
    spec = workflow_specs().get(workflow)
    if spec is None:
        return
    artifact_pills = "".join(
        f'<span class="artifact-pill">{artifact}</span>' for artifact in workflow_artifact_labels(workflow)
    )
    st.markdown(
        f"""
        <div class="workflow-focus">
            <div class="card-eyebrow">{spec["category"]}</div>
            <h4>{workflow}: {spec["headline"]}</h4>
            <p>{spec["when_to_use"]}</p>
            <div class="card-meta"><strong>Required context:</strong> {spec["requires"]}</div>
            <div class="pill-row">{artifact_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_category_rows() -> List[Dict[str, str]]:
    return [
        {
            "category": "Evidence",
            "workflows": "team, single",
            "description": "先做文献和研究判断，建立问题边界、证据表和下一步假设。",
        },
        {
            "category": "Modeling",
            "workflows": "mechanics, hybrid, simulation",
            "description": "把实验力学、网络模拟、FEBio 场景仿真和工程解释串起来，形成机制层判断。",
        },
        {
            "category": "Design",
            "workflows": "design, design_campaign",
            "description": "围绕目标 stiffness window 做单点反推或多窗口 campaign 比较。",
        },
        {
            "category": "Data & Core",
            "workflows": "datasets, calibration, benchmark",
            "description": "维护公开数据接入、family priors 和数值核心的可靠性。",
        },
    ]


def workflow_flow_family_specs() -> List[Dict[str, str]]:
    return [
        {
            "family": "Evidence",
            "workflows": "team / single",
            "summary": "把研究问题拆解成文献检索、证据抽取、假设生成和写作。",
            "stages": "team: planner -> search -> evidence -> hypothesis -> critic -> writer | single: assistant -> writer",
        },
        {
            "family": "Modeling",
            "workflows": "mechanics / hybrid / simulation",
            "summary": "把实验力学拟合、网络模拟、FEBio 固定场景验证和工程解释串成机制层报告。",
            "stages": "mechanics: planner -> fit -> critic -> writer | hybrid: planner -> search -> evidence -> hypothesis -> mechanics -> simulation -> critic -> writer | simulation: request -> FEBio run -> parse -> metrics -> report",
        },
        {
            "family": "Design",
            "workflows": "design / design_campaign",
            "summary": "围绕目标 stiffness 与约束做单点反推或多窗口 campaign 比较。",
            "stages": "design: validation -> search -> sensitivity -> formulation -> report | campaign: validation -> multi-target search -> formulation -> report",
        },
        {
            "family": "Data & Core",
            "workflows": "datasets / calibration / benchmark",
            "summary": "维护数据接入、family priors 和数值核心质量，为 design 与 demo 提供底座。",
            "stages": "datasets: search -> register -> download -> normalize | calibration: extract -> calibrate -> impact -> report | benchmark: solver -> scaling -> inverse-design -> fit -> summary",
        },
    ]


def render_workflow_category_cards() -> None:
    columns = st.columns(2, gap="medium")
    for index, row in enumerate(workflow_category_rows()):
        with columns[index % 2]:
            with st.container(border=True):
                st.caption(row["category"])
                st.markdown(f"**{row['workflows']}**")
                st.write(row["description"])


def project_flow_dot() -> str:
    return """
digraph ECMFlow {
  graph [rankdir=TB, splines=ortho, nodesep=0.35, ranksep=0.7, pad=0.2, bgcolor="transparent"];
  node [shape=box, style="rounded,filled", margin="0.18,0.12", fontname="Helvetica", fontsize=12, color="#d8e3f5", fillcolor="#ffffff"];
  edge [color="#6f88b8", penwidth=1.5, arrowsize=0.75];

  entry [label="Entry Surfaces\\nResearch Run | Demo | CLI | Desktop", fillcolor="#eef6ff"];
  inputs [label="Research Inputs\\nquery | mechanics file | target window\\ndataset_id | library | memory", fillcolor="#f7fbff"];
  runner [label="Workflow Router\\nrun_research_agent\\ncreate run_dir + metadata + progress", fillcolor="#eef6ff"];
  engines [label="Tools + Engines\\nsearch | datasets | calibration\\nmechanics | fiber network | FEBio | formulation", fillcolor="#f7fbff"];

  evidence [label="Evidence\\nteam / single\\nsearch -> evidence -> hypothesis -> writer", fillcolor="#ffffff"];
  modeling [label="Modeling\\nmechanics / hybrid / simulation\\nfit -> simulation -> interpretation", fillcolor="#ffffff"];
  design [label="Design\\ndesign / design_campaign\\nvalidation -> search -> formulation", fillcolor="#ffffff"];
  datacore [label="Data & Core\\ndatasets / calibration / benchmark\\nnormalize -> priors -> QA", fillcolor="#ffffff"];

  outputs [label="Outputs\\nreports/ + runs/\\nsummary json + stage artifacts", fillcolor="#eef6ff"];
  boards [label="UI Surfaces\\nHome | Design Board | Demo | Console", fillcolor="#f7fbff"];

  entry -> inputs -> runner -> engines;
  engines -> evidence;
  engines -> modeling;
  engines -> design;
  engines -> datacore;
  runner -> evidence;
  runner -> modeling;
  runner -> design;
  runner -> datacore;
  evidence -> outputs;
  modeling -> outputs;
  design -> outputs;
  datacore -> outputs;
  outputs -> boards;
  datacore -> design [label="priors", fontsize=10];
}
""".strip()


def render_project_flow_diagram() -> None:
    st.graphviz_chart(project_flow_dot(), width="stretch")
    cols = st.columns(2, gap="medium")
    for index, item in enumerate(workflow_flow_family_specs()):
        with cols[index % 2]:
            with st.container(border=True):
                st.caption(item["family"])
                st.markdown(f"**{item['workflows']}**")
                st.write(item["summary"])
                st.code(item["stages"], language="text")


def dashboard_snapshot_cards(
    latest_design: Dict[str, object],
    latest_campaign: Dict[str, object],
    latest_calibration: Dict[str, object],
) -> None:
    best_stiffness = latest_design.get("best_stiffness_mean", "NR")
    best_stiffness_text = f"{best_stiffness:.2f} Pa" if isinstance(best_stiffness, (int, float)) else "NR"
    design_meta = (
        f"family={latest_design.get('best_material_family', 'NR')} | error={latest_design.get('stiffness_error', 'NR')}"
    )

    feasible_count = latest_campaign.get("feasible_count", "NR")
    target_count = latest_campaign.get("target_count", "NR")
    campaign_value = f"{feasible_count}/{target_count}" if latest_campaign else "NR"
    campaign_meta = latest_campaign.get("note", "Run a design campaign to compare multiple target windows.")

    calibration_family_count = latest_calibration.get("family_count", "NR")
    calibration_value = str(calibration_family_count) if latest_calibration else "NR"
    calibration_meta = (
        f"mean abs error={latest_calibration.get('mean_abs_error', 'NR')} | families={latest_calibration.get('families', 'NR')}"
        if latest_calibration
        else "No calibration priors detected yet."
    )

    cols = st.columns(3, gap="medium")
    with cols[0]:
        with st.container(border=True):
            st.caption("Latest Design")
            st.metric("Best Candidate", best_stiffness_text)
            st.write(design_meta)
    with cols[1]:
        with st.container(border=True):
            st.caption("Latest Campaign")
            st.metric("Feasible Windows", campaign_value)
            st.write(campaign_meta)
    with cols[2]:
        with st.container(border=True):
            st.caption("Latest Calibration")
            st.metric("Family Priors", calibration_value)
            st.write(calibration_meta)


def render_recent_report_cards(rows: List[Dict[str, object]]) -> None:
    if not rows:
        st.info("还没有可归档的研究报告。")
        return
    cols = st.columns(2, gap="medium")
    for index, row in enumerate(rows[:4]):
        with cols[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{row.get('report', 'NR')}**")
                st.caption(f"Modified: {row.get('modified', 'NR')}")
                st.caption(f"Size: {row.get('size_kb', 'NR')} KB")


def render_family_cards(rows: List[Dict[str, object]]) -> None:
    if not rows:
        st.info("最近的 design run 里还没有稳定的材料家族分布。")
        return
    cols = st.columns(2, gap="medium")
    for index, row in enumerate(rows[:6]):
        with cols[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{row.get('material_family', 'NR')}**")
                st.caption(f"Recent wins: {row.get('count', 'NR')}")


def quickstart_markdown() -> str:
    return (
        "1. 文献起步先用 `team`，只想快速扫一遍再用 `single`。\n"
        "2. 有实验数据先跑 `mechanics` 或 `calibration`，把参数和 priors 建起来。\n"
        "3. 要选材料窗口时先用 `design_campaign`，再回到 `design` 深挖单点。\n"
        "4. 想让别人快速理解整套系统时，直接去 `Demo` tab 跑全流程。"
    )


def workflow_examples() -> Dict[str, Dict[str, object]]:
    return {
        "simulation": {
            "title": "FEBio bulk mechanics verification",
            "inputs": [
                {"field": "workflow", "value": "simulation"},
                {"field": "scenario", "value": "bulk_mechanics"},
                {"field": "matrix modulus", "value": "8 Pa"},
                {"field": "target stiffness", "value": "8 Pa"},
            ],
            "steps": [
                "先把 CLI / UI 输入约束成固定 schema request。",
                "用模板 + 参数注入生成 `input.feb`，禁止任意 XML。",
                "运行 FEBio，并解析位移/反力/主应力输出。",
                "写出 `simulation_result.json`、`simulation_metrics.json` 和最终 Markdown 报告。",
            ],
            "outputs": [
                "simulation/input_request.json",
                "simulation/input.feb",
                "simulation/simulation_result.json",
                "simulation/simulation_metrics.json",
                "reports/febio_simulation_report.md",
            ],
        },
        "design_campaign": {
            "title": "GelMA-like campaign across 6 / 8 / 10 Pa",
            "inputs": [
                {"field": "workflow", "value": "design_campaign"},
                {"field": "query", "value": "Design a GelMA-like ECM family across 6, 8, and 10 Pa targets"},
                {"field": "campaign targets", "value": "6,8,10"},
                {"field": "constraints", "value": "max anisotropy=0.35, min connectivity=0.90"},
            ],
            "steps": [
                "先验证 fiber-network physics core 是否通过。",
                "对每个 stiffness window 运行 design search，筛 top candidate。",
                "把抽象参数翻译成材料家族和 recipe template。",
                "写出 `campaign_summary.json` 和最终 Markdown 报告。",
            ],
            "outputs": [
                "campaign_validation.md",
                "campaign_agent.md",
                "formulation_mapping.md",
                "campaign_summary.json",
                "reports/calibrated_design_campaign_report.md",
            ],
        },
        "mechanics": {
            "title": "Creep dataset fitting",
            "inputs": [
                {"field": "workflow", "value": "mechanics"},
                {"field": "file", "value": "runs/sample_mechanics_creep.csv"},
                {"field": "experiment", "value": "creep"},
                {"field": "columns", "value": "time / stress / strain"},
            ],
            "steps": [
                "PlannerAgent 先识别模型目标和参数。",
                "MechanicsAgent 拟合 Kelvin-Voigt 或其它支持的简单模型。",
                "CriticAgent 检查单位、一致性和模型失配风险。",
                "WriterAgent 写出参数解释和实验含义。",
            ],
            "outputs": [
                "mechanics_planner.md",
                "mechanics_agent.md",
                "mechanics_critic.md",
                "reports/mechanics_report.md",
            ],
        },
        "team": {
            "title": "Literature synthesis for synthetic ECM",
            "inputs": [
                {"field": "workflow", "value": "team"},
                {"field": "query", "value": "synthetic ECM for intestinal organoid culture"},
                {"field": "library", "value": "library/"},
                {"field": "pubmed count", "value": "6"},
            ],
            "steps": [
                "PlannerAgent 拆成可执行子问题。",
                "SearchAgent 调 PubMed、Crossref、本地 library。",
                "Evidence / Hypothesis / Critic 连续处理证据、假设和质控。",
                "WriterAgent 生成周报并保存。",
            ],
            "outputs": [
                "planner_agent.md",
                "search_agent.md",
                "evidence_agent.md",
                "hypothesis_agent.md",
                "critic_agent.md",
                "reports/research_report_*.md",
            ],
        },
    }


def render_example_walkthrough(*, default_workflow: str = "design_campaign") -> None:
    examples = workflow_examples()
    example_keys = list(examples.keys())
    selected = st.selectbox(
        "Example Flow",
        example_keys,
        index=example_keys.index(default_workflow) if default_workflow in example_keys else 0,
        key=f"example_flow_{default_workflow}",
    )
    example = examples[selected]
    st.markdown(f"**{example['title']}**")
    st.markdown("Inputs")
    st.dataframe(example["inputs"], use_container_width=True, hide_index=True)
    st.markdown("What the system does")
    for idx, step in enumerate(example["steps"], start=1):
        st.markdown(f"{idx}. {step}")
    st.markdown("Outputs")
    st.code("\n".join(str(item) for item in example["outputs"]), language="text")


def guide_markdown() -> str:
    if GUIDE_PATH.exists():
        return GUIDE_PATH.read_text(encoding="utf-8")
    return (
        "# Guide\n\n"
        "Use `design` for one target mechanics window and `design_campaign` for multiple windows.\n"
        "Treat formulation translation as a starting recipe template, not a final validated SOP."
    )


def latest_design_summary(config: AppConfig) -> Dict[str, object]:
    options = design_run_options(config)
    if not options:
        return {
            "run": "NR",
            "best_stiffness_mean": "NR",
            "stiffness_error": "NR",
            "best_material_family": "NR",
            "best_crosslinking_strategy": "NR",
            "physics_valid": False,
        }
    return design_run_summary(options[0][1])


def latest_campaign_overview(config: AppConfig) -> Dict[str, object]:
    options = design_campaign_run_options(config)
    if not options:
        return {}
    payload = design_campaign_snapshot(options[0][1])
    results = payload.get("campaign_results", []) if isinstance(payload, dict) else []
    feasible_count = sum(
        1
        for row in results
        if isinstance(row.get("best_candidate"), dict) and row["best_candidate"].get("feasible")
    )
    return {
        "run": options[0][1].name,
        "target_count": len(results),
        "feasible_count": feasible_count,
        "constraints": payload.get("constraints", {}),
        "calibration_prior_level": payload.get("calibration_context", {}).get("prior_level", "NR")
        if isinstance(payload, dict)
        else "NR",
        "calibration_context_count": len(payload.get("calibration_context", {}).get("contexts", []))
        if isinstance(payload, dict) and isinstance(payload.get("calibration_context", {}), dict)
        else "NR",
        "note": (
            f"Latest campaign compares {len(results)} target windows with {feasible_count} feasible best-candidate(s); "
            f"prior routing={payload.get('calibration_context', {}).get('prior_level', 'NR') if isinstance(payload, dict) else 'NR'}."
        ),
    }


def calibration_run_options(config: AppConfig) -> List[Tuple[str, Path]]:
    options = []
    for label, run_dir in recent_run_options(config):
        metadata = load_json_file(run_dir / "metadata.json")
        if metadata.get("workflow") == "calibration":
            options.append((label, run_dir))
    return options


def calibration_run_snapshot(run_dir: Path) -> Dict[str, object]:
    targets_path = run_dir / "calibration_targets.json"
    results_path = run_dir / "calibration_results.json"
    impact_path = run_dir / "calibration_impact.json"
    return {
        "targets_payload": load_json_file(targets_path),
        "results_payload": load_json_file(results_path),
        "impact_payload": load_json_file(impact_path),
    }


def latest_calibration_overview(config: AppConfig) -> Dict[str, object]:
    options = calibration_run_options(config)
    if not options:
        return {}
    run_dir = options[0][1]
    snapshot = calibration_run_snapshot(run_dir)
    results = snapshot.get("results_payload", {})
    impact = snapshot.get("impact_payload", {})
    impact_summary = impact.get("summary", {}) if isinstance(impact, dict) else {}
    priors = results.get("family_priors", []) if isinstance(results, dict) else []
    condition_priors = results.get("condition_priors", []) if isinstance(results, dict) else []
    summary_rows = {
        "run": run_dir.name,
        "target_count": results.get("target_count", "NR") if isinstance(results, dict) else "NR",
        "family_count": len(priors),
        "condition_prior_count": len(condition_priors),
        "mean_abs_error": round(
            float(sum(float(row.get("mean_abs_error", 0.0)) for row in priors) / len(priors)),
            4,
        )
        if priors
        else "NR",
        "impact_delta": round(float(impact_summary.get("mean_total_error_delta", 0.0)), 4)
        if impact_summary.get("available")
        else "NR",
        "evaluation_mode": impact_summary.get("evaluation_mode", "NR") if impact_summary else "NR",
        "families": ", ".join(str(row.get("material_family", "NR")) for row in priors) if priors else "NR",
    }
    return summary_rows


def formulation_family_distribution(config: AppConfig, *, limit: int = 10) -> List[Dict[str, object]]:
    counts: Counter[str] = Counter()
    for _, run_dir in design_run_options(config)[:limit]:
        summary = design_run_snapshot(run_dir)
        formulations = summary.get("formulation_recommendations", [])
        if formulations:
            counts[str(formulations[0].get("material_family", "NR"))] += 1
    return [{"material_family": family, "count": count} for family, count in counts.most_common()]


def recent_report_rows(config: AppConfig, *, limit: int = 6) -> List[Dict[str, object]]:
    rows = []
    report_files = sorted(config.report_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for path in report_files:
        rows.append(
            {
                "report": path.name,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return rows


def build_dashboard_summary(config: AppConfig) -> Dict[str, int]:
    return {
        "agent_count": len(agent_capability_rows()),
        "run_count": len([path for path in config.runs_dir.glob("*") if path.is_dir()]) if config.runs_dir.exists() else 0,
        "cache_count": len(list(config.cache_dir.rglob("*.json"))) if config.cache_dir.exists() else 0,
        "report_count": len(list(config.report_dir.glob("*.md"))) if config.report_dir.exists() else 0,
    }


def agent_capability_rows() -> List[Dict[str, str]]:
    return [
        {
            "agent": "PlannerAgent",
            "role": "任务拆解",
            "inputs": "用户研究问题、周报模板",
            "outputs": "subquestions / extraction targets / stop conditions",
            "tools": "无",
        },
        {
            "agent": "SearchAgent",
            "role": "证据收集",
            "inputs": "研究问题、Planner 输出",
            "outputs": "candidate studies / patterns / evidence gaps",
            "tools": "search_pubmed, search_crossref, search_local_library",
        },
        {
            "agent": "EvidenceAgent",
            "role": "证据结构化",
            "inputs": "SearchAgent 输出、Planner 输出",
            "outputs": "evidence table / direct evidence / inference",
            "tools": "无",
        },
        {
            "agent": "HypothesisAgent",
            "role": "假设生成",
            "inputs": "Planner 输出、Evidence 输出",
            "outputs": "supported hypotheses / experiments / decision points",
            "tools": "无",
        },
        {
            "agent": "CriticAgent",
            "role": "质控与门控",
            "inputs": "planner + evidence + hypothesis",
            "outputs": "PASS_TO_WRITER / REVISION_NEEDED + issues",
            "tools": "无",
        },
        {
            "agent": "MechanicsAgent",
            "role": "参数拟合",
            "inputs": "实验数据, elastic/creep/relaxation/frequency-sweep/cyclic 条件",
            "outputs": "selected model / modulus-viscosity-time constants / damping or sweep metrics / fit quality",
            "tools": "fit_mechanics_model, simulate_mechanics_model",
        },
        {
            "agent": "SimulationAgent",
            "role": "网络模拟与 FEBio 仿真",
            "inputs": "材料假设 + mechanics 结果 + simulation params 或固定 FEBio scenario request",
            "outputs": "fiber-network features 或 FEBio metrics / warnings / structured artifacts",
            "tools": "run_fiber_network_simulation, run_fiber_network_parameter_scan, build_febio_simulation_request, run_febio_simulation",
        },
        {
            "agent": "DesignAgent",
            "role": "逆向设计",
            "inputs": "target stiffness / anisotropy / connectivity / stress propagation",
            "outputs": "top-k ECM candidates / score / robustness ranking",
            "tools": "design_fiber_network_candidates, run_fiber_network_validation",
        },
        {
            "agent": "FormulationMapper",
            "role": "配方翻译",
            "inputs": "abstract design parameters + candidate features",
            "outputs": "material family / recipe template / wet-lab checks",
            "tools": "deterministic formulation mapping",
        },
        {
            "agent": "CalibrationAgent",
            "role": "实验校准",
            "inputs": "normalized experimental measurements",
            "outputs": "calibration targets / calibration results / family priors",
            "tools": "dataset parsing + calibration pipeline",
        },
        {
            "agent": "WriterAgent",
            "role": "报告写作",
            "inputs": "all stage outputs + critic feedback",
            "outputs": "Markdown report + REPORT_COMPLETE",
            "tools": "save_report",
        },
    ]


def workspace_health_rows(config: AppConfig) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for label, path in [
        ("memory", config.memory_dir),
        ("library", config.library_dir),
        ("reports", config.report_dir),
        ("runs", config.runs_dir),
        ("cache", config.cache_dir),
        ("templates", config.template_dir),
    ]:
        file_count = len([item for item in path.rglob("*") if item.is_file()]) if path.exists() else 0
        rows.append(
            {
                "resource": label,
                "path": str(path),
                "exists": "yes" if path.exists() else "no",
                "files": str(file_count),
            }
        )
    return rows


def recent_run_options(config: AppConfig) -> List[Tuple[str, Path]]:
    if not config.runs_dir.exists():
        return []
    runs = sorted(
        [path for path in config.runs_dir.iterdir() if path.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    options: List[Tuple[str, Path]] = []
    for run_dir in runs[:15]:
        metadata_path = run_dir / "metadata.json"
        summary = ""
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                summary = payload.get("workflow", "")
            except json.JSONDecodeError:
                summary = ""
        label = f"{run_dir.name}  |  {summary or 'unknown workflow'}"
        options.append((label, run_dir))
    return options


def design_run_options(config: AppConfig) -> List[Tuple[str, Path]]:
    options = []
    for label, run_dir in recent_run_options(config):
        metadata = load_json_file(run_dir / "metadata.json")
        if metadata.get("workflow") == "design":
            options.append((label, run_dir))
    return options


def design_campaign_run_options(config: AppConfig) -> List[Tuple[str, Path]]:
    options = []
    for label, run_dir in recent_run_options(config):
        metadata = load_json_file(run_dir / "metadata.json")
        if metadata.get("workflow") == "design_campaign":
            options.append((label, run_dir))
    return options


def simulation_run_options(config: AppConfig) -> List[Tuple[str, Path]]:
    options = []
    for label, run_dir in recent_run_options(config):
        metadata = load_json_file(run_dir / "metadata.json")
        if metadata.get("workflow") == "simulation":
            options.append((label, run_dir))
    return options


def render_run_inspector(run_dir: Path) -> None:
    metadata = load_json_file(run_dir / "metadata.json")
    if metadata:
        st.caption("Metadata")
        render_mapping_table(metadata)

    tool_log_rows = load_jsonl_file(run_dir / "tool_calls.jsonl")
    if tool_log_rows:
        st.caption("Tool Calls")
        compact_rows = [
            {
                "time": row.get("timestamp", ""),
                "tool": row.get("tool_name", ""),
                "cached": row.get("cached", ""),
                "status": row.get("status", ""),
            }
            for row in tool_log_rows
        ]
        st.dataframe(compact_rows, use_container_width=True, hide_index=True)

    stage_files = sorted(
        [path for path in run_dir.glob("*.md") if path.is_file()],
        key=lambda p: p.name,
    )
    if stage_files:
        labels = [path.name for path in stage_files]
        choice = st.selectbox("Stage artifact", labels, index=0, key=f"stage_{run_dir.name}")
        selected = stage_files[labels.index(choice)]
        st.caption(str(selected))
        render_artifact_file(selected, key=f"run_inspector_{run_dir.name}_{selected.name}")


def design_run_snapshot(run_dir: Path) -> Dict[str, Dict]:
    summary_path = run_dir / "design_summary.json"
    if summary_path.exists():
        summary_payload = load_json_file(summary_path)
        return {
            "design_payload": summary_payload.get("design_payload", {}) if isinstance(summary_payload, dict) else {},
            "design_assessment": summary_payload.get("design_assessment", {}) if isinstance(summary_payload, dict) else {},
            "validation_payload": summary_payload.get("validation_payload", {}) if isinstance(summary_payload, dict) else {},
            "scan_payload": summary_payload.get("sensitivity_payload", {}) if isinstance(summary_payload, dict) else {},
            "design_simulation": summary_payload.get("design_simulation", {}) if isinstance(summary_payload, dict) else {},
            "targets": summary_payload.get("targets", {}) if isinstance(summary_payload, dict) else {},
            "constraints": summary_payload.get("constraints", {}) if isinstance(summary_payload, dict) else {},
            "calibration_context": summary_payload.get("calibration_context", {}) if isinstance(summary_payload, dict) else {},
            "predicted_material_observables": summary_payload.get("predicted_material_observables", {}) if isinstance(summary_payload, dict) else {},
            "requested_condition_overrides": summary_payload.get("requested_condition_overrides", {}) if isinstance(summary_payload, dict) else {},
            "formulation_recommendations": summary_payload.get("formulation_recommendations", []) if isinstance(summary_payload, dict) else [],
        }
    tool_rows = read_jsonl(run_dir / "tool_calls.jsonl")
    design_payload = latest_tool_result(tool_rows, "design_fiber_network_candidates")
    validation_payload = latest_tool_result(tool_rows, "run_fiber_network_validation")
    scan_payload = latest_tool_result(tool_rows, "run_fiber_network_parameter_scan")
    return {
        "design_payload": design_payload,
        "design_assessment": {},
        "validation_payload": validation_payload,
        "scan_payload": scan_payload,
        "design_simulation": {},
        "targets": design_payload.get("targets", {}) if isinstance(design_payload, dict) else {},
        "constraints": design_payload.get("constraints", {}) if isinstance(design_payload, dict) else {},
        "calibration_context": {},
        "predicted_material_observables": {},
        "requested_condition_overrides": {},
        "formulation_recommendations": [],
    }


def simulation_run_snapshot(run_dir: Path) -> Dict[str, object]:
    simulation_dir = run_dir / "simulation"
    return {
        "request_payload": load_json_file(simulation_dir / "input_request.json"),
        "result_payload": load_json_file(simulation_dir / "simulation_result.json"),
        "metrics_payload": load_json_file(simulation_dir / "simulation_metrics.json"),
        "runner_payload": load_json_file(simulation_dir / "runner_metadata.json"),
        "final_summary_path": str(simulation_dir / "final_summary.md") if (simulation_dir / "final_summary.md").exists() else "",
        "simulation_dir": str(simulation_dir) if simulation_dir.exists() else "",
    }


def design_campaign_snapshot(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "campaign_summary.json"
    if summary_path.exists():
        return load_json_file(summary_path)
    return {}


def simulation_run_summary(run_dir: Path) -> Dict[str, object]:
    snapshot = simulation_run_snapshot(run_dir)
    request = snapshot.get("request_payload", {}) if isinstance(snapshot.get("request_payload"), dict) else {}
    result = snapshot.get("result_payload", {}) if isinstance(snapshot.get("result_payload"), dict) else {}
    metrics = snapshot.get("metrics_payload", {}) if isinstance(snapshot.get("metrics_payload"), dict) else {}
    return {
        "run": run_dir.name,
        "scenario": request.get("scenario", "NR"),
        "status": metrics.get("status", result.get("status", "NR")),
        "effective_stiffness": metrics.get("effective_stiffness", "NR"),
        "peak_stress": metrics.get("peak_stress", "NR"),
        "peak_matrix_stress": metrics.get("peak_matrix_stress", "NR"),
        "stress_propagation_distance": metrics.get("stress_propagation_distance", "NR"),
        "strain_heterogeneity": metrics.get("strain_heterogeneity", "NR"),
        "interface_deformation": metrics.get("interface_deformation", "NR"),
        "candidate_suitability_score": (
            metrics.get("candidate_suitability_score_components", {}).get("suitability_score", "NR")
            if isinstance(metrics.get("candidate_suitability_score_components"), dict)
            else "NR"
        ),
    }


def design_run_summary(run_dir: Path) -> Dict[str, object]:
    metadata = load_json_file(run_dir / "metadata.json")
    snapshot = design_run_snapshot(run_dir)
    validation = snapshot["validation_payload"] if isinstance(snapshot["validation_payload"], dict) else {}
    design_payload = snapshot["design_payload"] if isinstance(snapshot["design_payload"], dict) else {}
    best_candidate = design_best_candidate(design_payload)
    best_params = best_candidate.get("parameters", {}) if isinstance(best_candidate, dict) else {}
    best_features = best_candidate.get("features", {}) if isinstance(best_candidate, dict) else {}
    targets = snapshot["targets"] if isinstance(snapshot["targets"], dict) else {}
    constraints = snapshot["constraints"] if isinstance(snapshot["constraints"], dict) else {}
    formulations = snapshot["formulation_recommendations"] if isinstance(snapshot.get("formulation_recommendations"), list) else []
    primary_formulation = formulations[0] if formulations else {}
    calibration_context = snapshot.get("calibration_context", {}) if isinstance(snapshot.get("calibration_context"), dict) else {}
    selection_reason = calibration_context.get("selection_reason", {}) if calibration_context else {}
    assessment = snapshot.get("design_assessment", {}) if isinstance(snapshot.get("design_assessment"), dict) else {}
    assessment_metrics = assessment.get("metrics", {}) if isinstance(assessment.get("metrics"), dict) else {}
    requested_condition_overrides = snapshot.get("requested_condition_overrides", {}) if isinstance(snapshot.get("requested_condition_overrides"), dict) else {}
    predicted_material_observables = snapshot.get("predicted_material_observables", {}) if isinstance(snapshot.get("predicted_material_observables"), dict) else {}
    design_simulation = snapshot.get("design_simulation", {}) if isinstance(snapshot.get("design_simulation"), dict) else {}
    simulation_comparison = design_simulation.get("comparison", {}) if isinstance(design_simulation.get("comparison"), dict) else {}
    simulation_best = simulation_comparison.get("best_candidate", {}) if isinstance(simulation_comparison, dict) else {}

    stiffness_mean = float(best_features.get("stiffness_mean", 0.0))
    target_stiffness = float(targets.get("stiffness", 0.0)) if "stiffness" in targets else 0.0
    stiffness_error = abs(stiffness_mean - target_stiffness) if target_stiffness else 0.0

    return {
        "run": run_dir.name,
        "workflow": metadata.get("workflow", "NR"),
        "created_at": metadata.get("created_at", "NR"),
        "target_stiffness": round(float(targets.get("stiffness", 0.0)), 4) if "stiffness" in targets else "NR",
        "target_anisotropy": round(float(targets.get("anisotropy", 0.0)), 4) if "anisotropy" in targets else "NR",
        "target_connectivity": round(float(targets.get("connectivity", 0.0)), 4) if "connectivity" in targets else "NR",
        "target_stress_propagation": round(float(targets.get("stress_propagation", 0.0)), 4)
        if "stress_propagation" in targets
        else "NR",
        "solver_converged": bool(validation.get("solver_converged", False)),
        "monotonicity_valid": bool(validation.get("monotonicity_valid", False)),
        "nonlinearity_valid": bool(validation.get("nonlinearity_valid", False)),
        "physics_valid": bool(validation.get("physics_valid", False)),
        "feasible": bool(best_candidate.get("feasible", False)) if best_candidate else False,
        "screening_status": assessment.get("status", "NR"),
        "recommended_for_screening": bool(assessment.get("recommended_for_screening", False)),
        "constraint_max_anisotropy": round(float(constraints.get("max_anisotropy", 0.0)), 4) if "max_anisotropy" in constraints else "NR",
        "constraint_min_connectivity": round(float(constraints.get("min_connectivity", 0.0)), 4) if "min_connectivity" in constraints else "NR",
        "constraint_max_risk_index": round(float(constraints.get("max_risk_index", 0.0)), 4) if "max_risk_index" in constraints else "NR",
        "top_candidate_score": round(float(best_candidate.get("score", 0.0)), 4) if best_candidate else "NR",
        "febio_status": design_simulation.get("status", "NR") if design_simulation else "NR",
        "febio_scenario": design_simulation.get("scenario", "NR") if design_simulation else "NR",
        "febio_best_candidate": simulation_best.get("candidate_id", "NR") if simulation_best else "NR",
        "febio_best_score": round(float(simulation_best.get("comparison_score", 0.0)), 4) if simulation_best else "NR",
        "best_material_family": primary_formulation.get("material_family", "NR") if primary_formulation else "NR",
        "best_crosslinking_strategy": primary_formulation.get("crosslinking_strategy", "NR") if primary_formulation else "NR",
        "best_fiber_density": round(float(best_params.get("fiber_density", 0.0)), 4) if best_params else "NR",
        "best_fiber_stiffness": round(float(best_params.get("fiber_stiffness", 0.0)), 4) if best_params else "NR",
        "best_bending_stiffness": round(float(best_params.get("bending_stiffness", 0.0)), 4) if best_params else "NR",
        "best_crosslink_prob": round(float(best_params.get("crosslink_prob", 0.0)), 4) if best_params else "NR",
        "best_domain_size": round(float(best_params.get("domain_size", 0.0)), 4) if best_params else "NR",
        "best_stiffness_mean": round(stiffness_mean, 4) if best_candidate else "NR",
        "best_anisotropy": round(float(best_features.get("anisotropy", 0.0)), 4) if best_candidate else "NR",
        "best_connectivity": round(float(best_features.get("connectivity", 0.0)), 4) if best_candidate else "NR",
        "best_stress_propagation": round(float(best_features.get("stress_propagation", 0.0)), 4)
        if best_candidate
        else "NR",
        "best_risk_index": round(float(best_features.get("risk_index", 0.0)), 4) if best_candidate else "NR",
        "stiffness_error": round(stiffness_error, 4) if best_candidate else "NR",
        "stiffness_rel_error": round(float(assessment_metrics.get("stiffness_rel_error", 0.0)), 4)
        if "stiffness_rel_error" in assessment_metrics and assessment_metrics.get("stiffness_rel_error") is not None
        else "NR",
        "best_density_g_ml": round(float(predicted_material_observables.get("density_g_ml", 0.0)), 4)
        if predicted_material_observables
        else "NR",
        "best_acoustic_impedance_mrayl": round(float(predicted_material_observables.get("acoustic_impedance_mrayl", 0.0)), 4)
        if predicted_material_observables
        else "NR",
        "best_viscosity_low_shear_pas": round(float(predicted_material_observables.get("viscosity_low_shear_pas", 0.0)), 4)
        if predicted_material_observables
        else "NR",
        "best_shear_thinning_ratio": round(float(predicted_material_observables.get("shear_thinning_ratio", 0.0)), 4)
        if predicted_material_observables
        else "NR",
        "candidate_count": len(design_payload.get("evaluated_candidates", [])) if isinstance(design_payload, dict) else 0,
        "calibration_prior_level": calibration_context.get("prior_level", "NR") if calibration_context else "NR",
        "calibration_concentration": calibration_context.get("concentration_fraction", "NR") if calibration_context else "NR",
        "calibration_curing": calibration_context.get("curing_seconds", "NR") if calibration_context else "NR",
        "requested_condition_overrides": json.dumps(requested_condition_overrides, ensure_ascii=False) if requested_condition_overrides else "NR",
        "calibration_target_stiffness_mean": calibration_context.get("target_stiffness_mean", "NR") if calibration_context else "NR",
        "calibration_selection_score": round(float(selection_reason.get("total_score", 0.0)), 4)
        if selection_reason and "total_score" in selection_reason
        else "NR",
    }


def design_comparison_rows(run_dirs: List[Path]) -> List[Dict[str, object]]:
    return [design_run_summary(run_dir) for run_dir in run_dirs]


def campaign_result_rows(campaign_payload: Dict) -> List[Dict[str, object]]:
    if not isinstance(campaign_payload, dict):
        return []
    formulation_map = {
        float(item.get("target_stiffness", 0.0)): item
        for item in campaign_payload.get("formulation_recommendations", [])
        if isinstance(item, dict)
    }
    rows: List[Dict[str, object]] = []
    for row in campaign_payload.get("campaign_results", []):
        best = row.get("best_candidate", {})
        features = best.get("features", {}) if isinstance(best, dict) else {}
        params = best.get("parameters", {}) if isinstance(best, dict) else {}
        target_stiffness = float(row.get("target_stiffness", 0.0))
        formulation = formulation_map.get(target_stiffness, {})
        calibration_context = row.get("calibration_context", {}) if isinstance(row.get("calibration_context"), dict) else {}
        selection_reason = calibration_context.get("selection_reason", {}) if calibration_context else {}
        assessment = row.get("design_assessment", {}) if isinstance(row.get("design_assessment"), dict) else {}
        observables = row.get("predicted_material_observables", {}) if isinstance(row.get("predicted_material_observables"), dict) else {}
        rows.append(
            {
                "target_stiffness": round(target_stiffness, 4),
                "feasible": bool(best.get("feasible", False)) if isinstance(best, dict) else False,
                "recommended_for_screening": bool(assessment.get("recommended_for_screening", False)),
                "screening_status": assessment.get("status", "NR"),
                "top_score": round(float(best.get("score", 0.0)), 4) if isinstance(best, dict) else "NR",
                "stiffness_mean": round(float(features.get("stiffness_mean", 0.0)), 4),
                "anisotropy": round(float(features.get("anisotropy", 0.0)), 4),
                "risk_index": round(float(features.get("risk_index", 0.0)), 4),
                "fiber_density": round(float(params.get("fiber_density", 0.0)), 4),
                "crosslink_prob": round(float(params.get("crosslink_prob", 0.0)), 4),
                "density_g_ml": round(float(observables.get("density_g_ml", 0.0)), 4) if observables else "NR",
                "viscosity_low_shear_pas": round(float(observables.get("viscosity_low_shear_pas", 0.0)), 4) if observables else "NR",
                "material_family": formulation.get("material_family", "NR"),
                "template_name": formulation.get("template_name", "NR"),
                "calibration_prior_level": calibration_context.get("prior_level", "NR") if calibration_context else "NR",
                "calibration_concentration": calibration_context.get("concentration_fraction", "NR") if calibration_context else "NR",
                "calibration_curing": calibration_context.get("curing_seconds", "NR") if calibration_context else "NR",
                "calibration_target_stiffness_mean": calibration_context.get("target_stiffness_mean", "NR") if calibration_context else "NR",
                "calibration_selection_score": round(float(selection_reason.get("total_score", 0.0)), 4)
                if selection_reason and "total_score" in selection_reason
                else "NR",
            }
        )
    return rows


def design_candidate_rows(design_payload: Dict) -> List[Dict[str, object]]:
    if not isinstance(design_payload, dict):
        return []
    rows: List[Dict[str, object]] = []
    for candidate in design_payload.get("top_candidates", []):
        params = candidate.get("parameters", {})
        features = candidate.get("features", {})
        rows.append(
            {
                "rank": candidate.get("rank"),
                "feasible": bool(candidate.get("feasible", False)),
                "score": round(float(candidate.get("score", 0.0)), 4),
                "fiber_density": round(float(params.get("fiber_density", 0.0)), 4),
                "fiber_stiffness": round(float(params.get("fiber_stiffness", 0.0)), 4),
                "bending_stiffness": round(float(params.get("bending_stiffness", 0.0)), 4),
                "crosslink_prob": round(float(params.get("crosslink_prob", 0.0)), 4),
                "domain_size": round(float(params.get("domain_size", 0.0)), 4),
                "stiffness_mean": round(float(features.get("stiffness_mean", 0.0)), 4),
                "anisotropy": round(float(features.get("anisotropy", 0.0)), 4),
                "connectivity": round(float(features.get("connectivity", 0.0)), 4),
                "risk_index": round(float(features.get("risk_index", 0.0)), 4),
            }
        )
    return rows


def formulation_recommendation_rows(recommendations: List[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in recommendations:
        recipe = item.get("primary_recipe", {})
        rows.append(
            {
                "rank": item.get("candidate_rank", "NR"),
                "material_family": item.get("material_family", "NR"),
                "template_name": item.get("template_name", "NR"),
                "crosslinking_strategy": item.get("crosslinking_strategy", "NR"),
                "mapping_confidence": item.get("mapping_confidence", "NR"),
                "polymer": recipe.get("polymer", "NR"),
                "polymer_wt_percent": recipe.get("polymer_wt_percent", "NR"),
            }
        )
    return rows


def design_sensitivity_rows(scan_payload: Dict) -> List[Dict[str, object]]:
    if not isinstance(scan_payload, dict):
        return []
    rows = []
    for row in scan_payload.get("sensitivity_ranking", []):
        rows.append(
            {
                "parameter": row.get("parameter"),
                "normalized_stiffness_span": round(float(row.get("normalized_stiffness_span", 0.0)), 6),
            }
        )
    return rows


def design_best_candidate(design_payload: Dict) -> Dict:
    if not isinstance(design_payload, dict):
        return {}
    top_candidates = design_payload.get("top_candidates", [])
    if not top_candidates:
        return {}
    return top_candidates[0]


def cache_rows(config: AppConfig) -> List[Dict[str, str]]:
    if not config.cache_dir.exists():
        return [{"namespace": "none", "entries": "0"}]
    namespaces = sorted([path for path in config.cache_dir.iterdir() if path.is_dir()])
    if not namespaces:
        return [{"namespace": "none", "entries": "0"}]
    rows = []
    for namespace in namespaces:
        rows.append(
            {
                "namespace": namespace.name,
                "entries": str(len(list(namespace.glob("*.json")))),
            }
        )
    return rows


def load_json_file(path: Path) -> Dict:
    return read_json(path)


def load_jsonl_file(path: Path) -> List[Dict]:
    return read_jsonl(path)


def available_stage_files(run_dir: Path) -> Dict[str, Path]:
    ordered_names = [
        "single_agent_prompt.md",
        "planner_agent.md",
        "search_agent.md",
        "evidence_agent.md",
        "hypothesis_agent.md",
        "mechanics_planner.md",
        "mechanics_agent.md",
        "simulation_agent.md",
        "mechanics_critic.md",
        "critic_agent.md",
        "evidence_agent_revision.md",
        "hypothesis_agent_revision.md",
        "critic_agent_revision.md",
        "design_validation.md",
        "design_agent.md",
        "design_sensitivity.md",
        "formulation_mapping.md",
        "design_summary.json",
        "calibration_targets.md",
        "calibration_results.md",
        "calibration_impact.md",
        "calibration_targets.json",
        "calibration_results.json",
        "calibration_impact.json",
        "campaign_validation.md",
        "campaign_agent.md",
        "campaign_summary.json",
        "dataset_search.md",
        "dataset_register.md",
        "dataset_download.md",
        "dataset_normalize.md",
        "dataset_manifest_snapshot.json",
        "benchmark_solver.md",
        "benchmark_load_ladder.md",
        "benchmark_scaling.md",
        "benchmark_design.md",
        "benchmark_repeatability.md",
        "benchmark_identifiability.md",
        "benchmark_fit.md",
        "benchmark_calibration_design.md",
        "benchmark_summary.json",
        "final_summary.md",
    ]
    mapping: Dict[str, Path] = {}
    for name in ordered_names:
        path = run_dir / name
        if path.exists():
            label = name.removesuffix(".md").removesuffix(".json")
            mapping[label] = path
    return mapping


def collaboration_coverage_rows(run_dir: Path) -> List[Dict[str, str]]:
    rows = []
    for label in [
        "single_agent_prompt",
        "planner_agent",
        "search_agent",
        "evidence_agent",
        "hypothesis_agent",
        "mechanics_planner",
        "mechanics_agent",
        "simulation_agent",
        "mechanics_critic",
        "critic_agent",
        "evidence_agent_revision",
        "hypothesis_agent_revision",
        "critic_agent_revision",
        "design_validation",
        "design_agent",
        "design_sensitivity",
        "formulation_mapping",
        "design_summary",
        "calibration_targets",
        "calibration_results",
        "calibration_impact",
        "campaign_validation",
        "campaign_agent",
        "campaign_summary",
        "dataset_search",
        "dataset_register",
        "dataset_download",
        "dataset_normalize",
        "dataset_manifest_snapshot",
        "benchmark_solver",
        "benchmark_load_ladder",
        "benchmark_scaling",
        "benchmark_design",
        "benchmark_repeatability",
        "benchmark_identifiability",
        "benchmark_fit",
        "benchmark_calibration_design",
        "benchmark_summary",
        "final_summary",
    ]:
        path = run_dir / f"{label}.md"
        if not path.exists():
            json_path = run_dir / f"{label}.json"
            if json_path.exists():
                path = json_path
        rows.append(
            {
                "stage": label,
                "present": "yes" if path.exists() else "no",
                "path": str(path) if path.exists() else "",
            }
        )
    return rows


def design_stage_files(run_dir: Path) -> Dict[str, Path]:
    names = ["design_validation.md", "design_agent.md", "design_sensitivity.md", "formulation_mapping.md", "design_summary.json", "final_summary.md"]
    mapping: Dict[str, Path] = {}
    for name in names:
        path = run_dir / name
        if path.exists():
            mapping[name.removesuffix(".md").removesuffix(".json")] = path
    return mapping


def render_physics_validation_card() -> None:
    st.markdown("### Physics Validation")
    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        if st.button("Run Validation Snapshot", use_container_width=True):
            with st.spinner("Running fiber-network validation snapshot..."):
                params = default_validation_params()
                validation = run_validation()
                simulation = simulate_ecm(params)
                tensile = run_tensile_test(params)
                st.session_state["physics_snapshot"] = {
                    "validation": validation,
                    "simulation": simulation,
                    "tensile": tensile,
                }
        snapshot = st.session_state.get("physics_snapshot")
        if snapshot:
            validation = snapshot["validation"]
            sim = snapshot["simulation"]
            cva, cvb, cvc, cvd = st.columns(4)
            cva.metric("Solver", "PASS" if validation["solver_converged"] else "FAIL")
            cvb.metric("Monotonicity", "PASS" if validation["monotonicity_valid"] else "FAIL")
            cvc.metric("Nonlinearity", "PASS" if validation["nonlinearity_valid"] else "FAIL")
            cvd.metric("Physics", "PASS" if validation["physics_valid"] else "FAIL")
            st.caption(
                f"stiffness_mean={sim['stiffness_mean']:.3f}, std={sim['stiffness_std']:.3f}, risk_index={sim['risk_index']:.3f}"
            )
        else:
            st.caption("点击按钮后，这里会显示当前科研级物理核的验证结果。")
    with c2:
        snapshot = st.session_state.get("physics_snapshot")
        if snapshot:
            tensile = snapshot["tensile"]
            curve = tensile["stress_strain_curve"]
            chart_rows = [
                {"strain": point["strain"], "stress": point["stress"]}
                for point in curve
            ]
            st.line_chart(chart_rows, x="strain", y="stress", height=220)
        else:
            st.caption("未生成 stress-strain 验证曲线。")


def render_run_simulation_summary(run_dir: Path) -> None:
    tool_rows = read_jsonl(run_dir / "tool_calls.jsonl")
    baseline = latest_tool_result(tool_rows, "run_fiber_network_simulation")
    scan = latest_tool_result(tool_rows, "run_fiber_network_parameter_scan")
    febio = latest_tool_result(tool_rows, "run_febio_simulation")
    if not baseline and not scan and not febio:
        return

    st.markdown("### Simulation Summary")
    if febio:
        metrics = febio.get("simulation_metrics", {}) if isinstance(febio.get("simulation_metrics"), dict) else {}
        febio_metrics = [
            {"metric": key, "value": metrics.get(key, "NR")}
            for key in [
                "effective_stiffness",
                "peak_stress",
                "displacement_decay_length",
                "strain_heterogeneity",
                "target_mismatch_score",
            ]
        ]
        st.caption("FEBio")
        st.dataframe(febio_metrics, use_container_width=True, hide_index=True)
    if baseline:
        baseline_metrics = [
            {
                "metric": key,
                "value": baseline.get(key, "NR"),
            }
            for key in [
                "stiffness_mean",
                "stiffness_std",
                "anisotropy",
                "connectivity",
                "stress_propagation",
                "risk_index",
            ]
        ]
        st.dataframe(baseline_metrics, use_container_width=True, hide_index=True)
    if scan:
        ranking = scan.get("sensitivity_ranking", [])
        if ranking:
            st.caption("Sensitivity Ranking")
            st.dataframe(ranking, use_container_width=True, hide_index=True)


def latest_tool_result(tool_rows: List[Dict], tool_name: str) -> Dict:
    for row in reversed(tool_rows):
        if row.get("tool_name") != tool_name or row.get("status") != "ok":
            continue
        try:
            return json.loads(row.get("result", "{}"))
        except json.JSONDecodeError:
            return {}
    return {}


if __name__ == "__main__":
    main()
