from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Optional

from .artifacts import RunArtifacts, create_run_artifacts, utc_now_iso, write_json, write_stage_text
from .benchmarks import run_mechanics_benchmark_suite
from .calibration import calibrated_search_space_from_calibration_results, load_latest_calibration_results, predict_material_observables, run_calibration_pipeline
from .config import AppConfig
from .datasets import acquire_public_dataset, auto_register_loose_archives, list_public_dataset_specs, normalize_dataset_directory
from .febio import CandidateSimulationMappingOptions, candidate_requests_summary, compare_simulation_candidates_payloads, design_payload_to_simulation_requests
from .formulation import recommend_campaign_formulations, recommend_formulations_from_design_payload
from .model_client import build_model_client
from .tools import build_tools, sanitize_filename

DESIGN_STIFFNESS_RELATIVE_TOLERANCE = 0.20
DESIGN_ANISOTROPY_ABSOLUTE_TOLERANCE = 0.15
DESIGN_CONNECTIVITY_SHORTFALL_TOLERANCE = 0.05
DESIGN_STRESS_PROPAGATION_SHORTFALL_TOLERANCE = 0.25
DESIGN_RISK_INDEX_MAX = 1.0


@dataclass(frozen=True)
class ResearchRunResult:
    workflow: str
    report_path: Optional[Path]
    final_summary: str
    run_dir: Optional[Path] = None
    metadata_path: Optional[Path] = None
    planner_output: str = ""
    search_output: str = ""
    evidence_output: str = ""
    hypothesis_output: str = ""
    critic_output: str = ""
    mechanics_output: str = ""
    simulation_output: str = ""
    design_output: str = ""
    formulation_output: str = ""
    benchmark_output: str = ""
    dataset_output: str = ""
    calibration_output: str = ""


def load_weekly_template(project_dir: Path) -> str:
    template_path = project_dir / "templates" / "weekly_report_template.md"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return dedent(
        """
        # 本周研究周报
        ## 1. 本周研究问题
        ## 2. 检索策略
        ## 3. 文献快照比较表
        ## 4. 关键证据
        ## 5. 本地文献与历史笔记关联
        ## 6. 研究判断与模式总结
        ## 7. 证据空白与风险
        ## 8. 可验证假设
        ## 9. 下周行动清单
        ## 10. 参考文献
        """
    ).strip()


def build_single_agent_task_prompt(
    query: str, report_name: str, max_pubmed_results: int, template_text: str
) -> str:
    today = date.today().isoformat()
    return dedent(
        f"""
        Research topic:
        {query}

        Report date:
        {today}

        Work requirements:
        1. Use `search_pubmed` first and inspect up to {max_pubmed_results} papers.
        2. Use `search_crossref` to complement PubMed results, especially for DOI, journal metadata, and very recent records.
        3. Use `search_local_library` to check whether there are relevant notes, Markdown files, or PDFs in the local library folder.
        4. Produce a Markdown weekly report in Chinese. Keep paper titles, ECM material names, organoid names, and assay names in their original English when useful.
        5. Strictly separate:
           - Evidence from retrieved sources
           - Inference based on the evidence
           - Hypotheses and next-step experiments
        6. Include one comparison table with these columns:
           Study | ECM composition | organoid model | species | assay/readout | main finding | limitation
        7. If a field is unavailable from retrieved results, write `NR`.
        8. When citing studies, include PMID and DOI whenever available.
        9. Explicitly mention whether local files support, contradict, or extend the external literature.
        10. Follow this weekly report structure closely:

        {template_text}

        11. Save the full Markdown report by calling `save_report` with the filename `{report_name}`.
        12. In the final answer to the user, briefly state:
           - what you searched,
           - the main ECM-related patterns,
           - where the report was saved.
        """
    ).strip()


def build_team_task_prompt(query: str, report_name: str, max_pubmed_results: int, template_text: str) -> str:
    today = date.today().isoformat()
    return dedent(
        f"""
        Research topic:
        {query}

        Report date:
        {today}

        Team workflow:
        1. PlannerAgent decomposes the task and defines priorities.
        2. SearchAgent performs the evidence collection.
        3. EvidenceAgent converts the retrieved evidence into structured findings.
        4. HypothesisAgent proposes mechanisms, experiments, and decision points.
        5. CriticAgent checks unsupported claims, citation gaps, and evidence-vs-inference confusion.
        6. WriterAgent writes the final weekly report, saves it with `save_report`, then replies with `REPORT_COMPLETE`.

        Global requirements:
        1. SearchAgent should start with `search_pubmed` and inspect up to {max_pubmed_results} papers.
        2. SearchAgent may use `search_crossref` to complement DOI, journal metadata, and recent records.
        3. SearchAgent should use `search_local_library` once to check local notes, PDFs, or Markdown files.
        4. EvidenceAgent and CriticAgent must not invent papers, DOI, concentrations, stiffness values, or assay details not present in the retrieved material.
        5. The final report must be in Chinese, while keeping paper titles, ECM materials, organoid names, and assay/readout names in English when useful.
        6. The report must strictly separate:
           - Evidence from retrieved sources
           - Inference based on the evidence
           - Hypotheses and next-step experiments
        7. Include one comparison table with these columns:
           Study | ECM composition | organoid model | species | assay/readout | main finding | limitation
        8. If a field is unavailable from the retrieved results, write `NR`.
        9. Mention whether local files support, contradict, or extend the external literature.
        10. Follow this weekly report structure closely:

        {template_text}

        11. WriterAgent must save the report with filename `{report_name}`.
        12. WriterAgent must end the final reply with the exact token `REPORT_COMPLETE`.
        """
    ).strip()


def build_planner_stage_task(query: str, template_text: str) -> str:
    return dedent(
        f"""
        Research topic:
        {query}

        Your stage:
        - Decompose this research request before evidence collection starts.
        - Do not use tools.
        - Produce a compact plan that downstream agents can execute.

        Required output:
        ## Research Goal
        ## Subquestions
        ## Search Priorities
        ## Extraction Targets
        ## Report Constraints
        ## Stop Conditions

        Template reminder for the final report:
        {template_text}
        """
    ).strip()


def build_search_stage_task(query: str, max_pubmed_results: int) -> str:
    return dedent(
        f"""
        Research topic:
        {query}

        Your stage:
        - Collect the strongest literature evidence for this topic.
        - Use at most 4 tool calls total.
        - Start with `search_pubmed`.
        - Use `search_crossref` only if it adds useful DOI or recent metadata.
        - Use `search_local_library` exactly once.
        - After the local library check, stop searching and write your evidence ledger.

        Output format:
        ## Search Coverage
        ## Candidate Studies
        For each study include: title | PMID | DOI | journal/year | why it matters
        ## ECM Patterns
        ## Local Hits
        ## Evidence Gaps

        Rules:
        - Keep only the most relevant studies.
        - Write `NR` if a field is unavailable.
        - Do not write the final report.
        """
    ).strip()


def build_search_stage_task_with_plan(
    *,
    query: str,
    planner_output: str,
    max_pubmed_results: int,
) -> str:
    return dedent(
        f"""
        Research topic:
        {query}

        Planning output:
        {planner_output}

        Your stage:
        - Collect the strongest literature evidence for this topic.
        - Use at most 5 tool calls total.
        - Start with `search_pubmed`.
        - Use `search_crossref` only if it adds useful DOI or recent metadata.
        - Use `search_local_library` exactly once.
        - Prefer evidence that directly supports the planner's subquestions and extraction targets.
        - After the local library check, stop searching and write your evidence ledger.

        Output format:
        ## Search Coverage
        ## Candidate Studies
        For each study include: title | PMID | DOI | journal/year | why it matters
        ## ECM Patterns
        ## Local Hits
        ## Evidence Gaps

        Rules:
        - Keep only the most relevant studies.
        - Write `NR` if a field is unavailable.
        - Do not write the final report.
        - Inspect up to {max_pubmed_results} PubMed papers per query.
        """
    ).strip()


def build_evidence_stage_task(search_output: str, planner_output: str) -> str:
    return dedent(
        f"""
        You are given the literature search output below.
        Convert it into a structured synthesis for the writer.

        Planning output:
        {planner_output}

        Search output:
        {search_output}

        Produce:
        ## Evidence Table
        Include a Markdown table with columns:
        Study | ECM composition | organoid model | species | assay/readout | main finding | limitation

        ## Direct Evidence
        ## Inference
        ## Hypotheses
        ## Reference Ledger

        Rules:
        - Use only the evidence already present in the search output.
        - Do not use tools.
        - Do not expose internal reasoning.
        - Use `NR` for missing fields.
        """
    ).strip()


def build_hypothesis_stage_task(planner_output: str, evidence_output: str) -> str:
    return dedent(
        f"""
        Use the planner output and evidence synthesis below to propose mechanistic hypotheses and next-step experiments.

        Planning output:
        {planner_output}

        Evidence synthesis:
        {evidence_output}

        Produce:
        ## Supported Mechanistic Hypotheses
        ## Low-Confidence Hypotheses
        ## Next-Step Experiments
        ## Decision Points

        Rules:
        - Do not use tools.
        - Distinguish supported hypotheses from low-confidence speculation.
        - Link each proposed experiment to a concrete uncertainty in the evidence.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_critic_stage_task(
    planner_output: str,
    evidence_output: str,
    hypothesis_output: str,
) -> str:
    return dedent(
        f"""
        Review the full collaboration package below before final writing.

        Planning output:
        {planner_output}

        Evidence synthesis:
        {evidence_output}

        Hypothesis synthesis:
        {hypothesis_output}

        Output format:
        ## Review Verdict
        Write either `PASS_TO_WRITER` or `REVISION_NEEDED`
        ## Critical Issues
        ## Minor Fixes
        ## Citation Checks

        Rules:
        - Do not use tools.
        - Do not expose internal reasoning.
        - Focus on unsupported claims, citation gaps, missing `NR`, and evidence-vs-inference confusion.
        """
    ).strip()


def build_evidence_revision_task(
    *,
    planner_output: str,
    search_output: str,
    evidence_output: str,
    critic_output: str,
) -> str:
    return dedent(
        f"""
        Revise the evidence synthesis using the critic feedback below.

        Planning output:
        {planner_output}

        Search output:
        {search_output}

        Current evidence synthesis:
        {evidence_output}

        Critic feedback:
        {critic_output}

        Produce a corrected version with:
        ## Evidence Table
        ## Direct Evidence
        ## Inference
        ## Hypotheses
        ## Reference Ledger

        Rules:
        - Use only the evidence already present.
        - Fix only what the critic flagged.
        - Do not use tools.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_hypothesis_revision_task(
    *,
    planner_output: str,
    evidence_output: str,
    hypothesis_output: str,
    critic_output: str,
) -> str:
    return dedent(
        f"""
        Revise the hypothesis package using the critic feedback below.

        Planning output:
        {planner_output}

        Current evidence synthesis:
        {evidence_output}

        Current hypothesis synthesis:
        {hypothesis_output}

        Critic feedback:
        {critic_output}

        Produce a corrected version with:
        ## Supported Mechanistic Hypotheses
        ## Low-Confidence Hypotheses
        ## Next-Step Experiments
        ## Decision Points

        Rules:
        - Use only the evidence already present.
        - Fix only what the critic flagged.
        - Do not use tools.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_writer_stage_task(
    *,
    query: str,
    report_name: str,
    template_text: str,
    planner_output: str,
    search_output: str,
    evidence_output: str,
    hypothesis_output: str,
    critic_output: str,
) -> str:
    today = date.today().isoformat()
    return dedent(
        f"""
        Research topic:
        {query}

        Report date:
        {today}

        Planning output:
        {planner_output}

        Search output:
        {search_output}

        Evidence synthesis:
        {evidence_output}

        Hypothesis synthesis:
        {hypothesis_output}

        Critic review:
        {critic_output}

        Write the final Markdown weekly report in Chinese.
        Follow this structure closely:

        {template_text}

        Rules:
        - Use only the material above.
        - Apply critic fixes when valid.
        - Separate direct evidence, inference, and hypotheses.
        - Keep paper titles, ECM materials, organoid names, and assay names in English when useful.
        - Use `NR` for missing fields.
        - Save the report with `save_report` using filename `{report_name}`.
        - After the save succeeds, reply with:
          1. short search summary,
          2. main ECM-related patterns,
          3. saved path,
          4. the exact token `REPORT_COMPLETE`.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_single_agent_system_message() -> str:
    return dedent(
        """
        You are a personal research assistant focused on extracellular matrix and organoid synthesis.
        Your job is to help a biomedical researcher review literature, compare evidence, and generate cautious hypotheses.

        Rules:
        - Do not invent PMID, DOI, concentrations, stiffness values, passages, media compositions, or experimental conditions.
        - Use tools before making factual claims about papers.
        - Prefer PubMed plus Crossref together for literature grounding.
        - Clearly distinguish evidence from inference.
        - Prefer concise and structured Markdown with stable headings.
        - Default to Chinese unless the user explicitly requests another language.
        - When the evidence is thin, say so directly.
        - Before finishing the task, save the report with `save_report`.
        """
    ).strip()


def build_search_agent_system_message(max_pubmed_results: int) -> str:
    return dedent(
        f"""
        You are SearchAgent.
        Your only job is to collect literature evidence for the rest of the team.

        Rules:
        - Use tools. Do not rely on memory alone.
        - Start with `search_pubmed`, then use `search_crossref` only to supplement metadata or newer records.
        - Use `search_local_library` once to check local files.
        - Keep to the strongest and most relevant findings. You do not need to exhaust the literature.
        - Use at most 4 tool calls in this stage.
        - Prioritize intestinal organoids, synthetic ECM, PEG hydrogels, mechanical tuning, and translational relevance.
        - Return a concise evidence ledger with PMID/DOI/URL when available.
        - Clearly flag missing fields as `NR`.
        - Do not write the final report and do not call `save_report`.
        - Do not expose internal reasoning.
        - Inspect up to {max_pubmed_results} PubMed papers for each search.
        """
    ).strip()


def build_planner_agent_system_message() -> str:
    return dedent(
        """
        You are PlannerAgent.
        Your job is to decompose the user's research request into an executable multi-agent plan.

        Rules:
        - Do not use tools.
        - Define subquestions, extraction targets, and decision criteria.
        - Keep the plan concise but operational.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_evidence_agent_system_message() -> str:
    return dedent(
        """
        You are EvidenceAgent.
        Your job is to convert retrieved evidence into a structured synthesis for the writer.

        Rules:
        - Do not use tools.
        - Only use evidence already present in the conversation.
        - Separate:
          1. direct evidence,
          2. inference,
          3. hypotheses.
        - Produce a compact comparison table and note major ECM-related patterns.
        - Use `NR` for unavailable details.
        - Do not write the final report.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_hypothesis_agent_system_message() -> str:
    return dedent(
        """
        You are HypothesisAgent.
        Your job is to turn the structured evidence into mechanistic hypotheses and next-step experiments.

        Rules:
        - Do not use tools.
        - Clearly separate supported hypotheses from low-confidence speculation.
        - Link each proposed experiment to a concrete uncertainty in the evidence.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_mechanics_agent_system_message() -> str:
    return dedent(
        """
        You are MechanicsAgent.
        Your job is to fit simple constitutive ECM mechanics models to experimental data and interpret the fitted parameters.

        Rules:
        - Use tools.
        - Prefer deterministic model fitting over free-form speculation.
        - State when the selected model is likely too simple for the dataset.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_simulation_agent_system_message() -> str:
    return dedent(
        """
        You are SimulationAgent.
        Your job is to run deterministic bead-spring ECM fiber-network simulations and interpret the emergent network mechanics.

        Rules:
        - Use tools.
        - Prefer the provided default parameters unless evidence strongly suggests a change.
        - Keep the simulation assumptions explicit.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_critic_agent_system_message() -> str:
    return dedent(
        """
        You are CriticAgent.
        Your job is to review the evidence synthesis before the report is written.

        Rules:
        - Do not use tools.
        - Look for unsupported claims, missing PMID/DOI where available, evidence-vs-inference confusion, and missing `NR` placeholders.
        - Keep the review concise and actionable.
        - If the draft is usable, say `PASS_TO_WRITER` and list any minor fixes the writer should apply.
        - If the draft is not usable, say `REVISION_NEEDED` and explain the blockers.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_writer_agent_system_message(report_name: str, template_text: str) -> str:
    return dedent(
        f"""
        You are WriterAgent.
        Your job is to write the final weekly report and save it.

        Rules:
        - Do not use new facts beyond the conversation.
        - Write the final Markdown report in Chinese.
        - Follow this structure closely:

        {template_text}

        - Save the report by calling `save_report` with filename `{report_name}`.
        - After the save succeeds, reply with:
          1. a short search summary,
          2. the main ECM-related patterns,
          3. the saved path,
          4. the exact token `REPORT_COMPLETE`.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_mechanics_planner_stage_task(
    *,
    query: str,
    data_path: Optional[Path],
    experiment_type: str,
) -> str:
    return dedent(
        f"""
        Mechanics modeling objective:
        {query}

        Dataset path:
        {data_path or 'NR'}

        Requested experiment type:
        {experiment_type}

        Your stage:
        - Decompose the mechanics modeling task.
        - Identify what constitutive family is appropriate.
        - State the observables, key assumptions, and fit criteria.
        - Do not use tools.

        Required output:
        ## Modeling Goal
        ## Expected Experiment Type
        ## Required Columns
        ## Key Parameters To Estimate
        ## Assumptions
        ## Fit Acceptance Criteria
        """
    ).strip()


def build_mechanics_stage_task(
    *,
    query: str,
    planner_output: str,
    data_path: Path,
    experiment_type: str,
    time_column: str,
    stress_column: str,
    strain_column: str,
    applied_stress: Optional[float],
    applied_strain: Optional[float],
    delimiter: str,
) -> str:
    return dedent(
        f"""
        Mechanics modeling objective:
        {query}

        Planning output:
        {planner_output}

        Dataset path:
        {data_path}

        Your stage:
        - Fit the most appropriate simple mechanics model using `fit_mechanics_model`.
        - If the fit succeeds, optionally call `simulate_mechanics_model` once to generate a smooth response curve.
        - Summarize the fitted parameters, goodness-of-fit, assumptions, and what the model can or cannot support experimentally.

        Tool arguments to use:
        - experiment_type: {experiment_type}
        - time_column: {time_column}
        - stress_column: {stress_column}
        - strain_column: {strain_column}
        - delimiter: {delimiter}
        - applied_stress: {applied_stress if applied_stress is not None else 0.0}
        - applied_strain: {applied_strain if applied_strain is not None else 0.0}

        Output format:
        ## Dataset Summary
        ## Fitted Model
        ## Estimated Parameters
        ## Fit Quality
        ## Mechanical Interpretation
        ## Limits Of The Model

        Rules:
        - Use tools.
        - The backend may return different deterministic constitutive families depending on experiment type and fit quality.
        - Use the returned `selected_model` / `fit` fields instead of assuming a fixed model family.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_mechanics_critic_stage_task(
    *,
    planner_output: str,
    mechanics_output: str,
) -> str:
    return dedent(
        f"""
        Review the mechanics modeling package below.

        Planning output:
        {planner_output}

        Mechanics output:
        {mechanics_output}

        Output format:
        ## Review Verdict
        Write either `PASS_TO_WRITER` or `REVISION_NEEDED`
        ## Critical Issues
        ## Minor Fixes
        ## Identifiability And Units Check

        Rules:
        - Do not use tools.
        - Check parameter plausibility, unit consistency, identifiability, and model mismatch risk.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_mechanics_writer_stage_task(
    *,
    query: str,
    report_name: str,
    planner_output: str,
    mechanics_output: str,
    critic_output: str,
) -> str:
    return dedent(
        f"""
        Mechanics modeling objective:
        {query}

        Planning output:
        {planner_output}

        Mechanics analysis:
        {mechanics_output}

        Critic review:
        {critic_output}

        Write a concise Chinese Markdown mechanics report with these sections:
        ## 1. Modeling Goal
        ## 2. Dataset And Experiment Type
        ## 3. Constitutive Model
        ## 4. Estimated Parameters
        ## 5. Fit Quality
        ## 6. Mechanical Interpretation
        ## 7. Model Limitations
        ## 8. Experimental Implications

        Rules:
        - Use only the material above.
        - Save the report with `save_report` using filename `{report_name}`.
        - After the save succeeds, reply with:
          1. fitted model type,
          2. key parameters,
          3. saved path,
          4. the exact token `REPORT_COMPLETE`.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_simulation_stage_task(
    *,
    query: str,
    evidence_output: str,
    hypothesis_output: str,
    mechanics_output: str,
    simulation_defaults_json: str,
) -> str:
    return dedent(
        f"""
        Hybrid ECM engineering objective:
        {query}

        Evidence synthesis:
        {evidence_output}

        Hypothesis synthesis:
        {hypothesis_output}

        Mechanics analysis:
        {mechanics_output}

        Default simulation parameters:
        {simulation_defaults_json}

        Your stage:
        - Use `run_fiber_network_simulation` once.
        - Use `run_fiber_network_parameter_scan` once.
        - Start from the provided default simulation parameters.
        - Adjust them only if the evidence or mechanics analysis clearly supports doing so.
        - Summarize both the single baseline simulation and the sensitivity scan.
        - Use the provided `monte_carlo_runs` for the baseline simulation.
        - If `scan_monte_carlo_runs` is present in the defaults, use that lower budget for the sensitivity scan.

        Output format:
        ## Simulation Setup
        ## Baseline Simulation Result
        ## Sensitivity Scan
        ## Emergent Mechanical Interpretation
        ## Design Implications
        ## Model Caveats

        Rules:
        - Use tools.
        - Do not expose internal reasoning.
        - Do not write the final report.
        """
    ).strip()


def build_hybrid_critic_stage_task(
    *,
    planner_output: str,
    evidence_output: str,
    hypothesis_output: str,
    mechanics_output: str,
    simulation_output: str,
) -> str:
    return dedent(
        f"""
        Review the full hybrid ECM engineering package below.

        Planning output:
        {planner_output}

        Evidence synthesis:
        {evidence_output}

        Hypothesis synthesis:
        {hypothesis_output}

        Mechanics analysis:
        {mechanics_output}

        Simulation analysis:
        {simulation_output}

        Output format:
        ## Review Verdict
        Write either `PASS_TO_WRITER` or `REVISION_NEEDED`
        ## Critical Issues
        ## Minor Fixes
        ## Consistency Checks

        Rules:
        - Do not use tools.
        - Check whether literature suggestions, fitted parameters, and network simulation assumptions are mutually consistent.
        - Flag parameter mismatch, overclaiming, and unsupported structure decisions.
        - Do not expose internal reasoning.
        """
    ).strip()


def build_hybrid_writer_stage_task(
    *,
    query: str,
    report_name: str,
    planner_output: str,
    search_output: str,
    evidence_output: str,
    hypothesis_output: str,
    mechanics_output: str,
    simulation_output: str,
    critic_output: str,
) -> str:
    return dedent(
        f"""
        Hybrid ECM engineering objective:
        {query}

        Planning output:
        {planner_output}

        Search output:
        {search_output}

        Evidence synthesis:
        {evidence_output}

        Hypothesis synthesis:
        {hypothesis_output}

        Mechanics analysis:
        {mechanics_output}

        Simulation analysis:
        {simulation_output}

        Critic review:
        {critic_output}

        Write a Chinese Markdown engineering report with these sections:
        ## 1. Research And Engineering Goal
        ## 2. Literature Evidence
        ## 3. Suggested ECM Material Strategy
        ## 4. Mechanics Modeling Parameters
        ## 5. Fiber-Network Simulation Setup
        ## 6. Simulation Outputs
        ## 7. Design Decision
        ## 8. Model And Evidence Limitations
        ## 9. Next Experimental Step

        Rules:
        - Use only the material above.
        - Save the report with `save_report` using filename `{report_name}`.
        - After the save succeeds, reply with:
          1. literature conclusion,
          2. mechanics parameter conclusion,
          3. simulation conclusion,
          4. saved path,
          5. the exact token `REPORT_COMPLETE`.
        - Do not expose internal reasoning.
        """
    ).strip()


async def load_memory(config: AppConfig):
    from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

    memory = ListMemory(name="research_profile")
    for path in sorted(config.memory_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        await memory.add(
            MemoryContent(
                content=f"Source file: {path.name}\n\n{content}",
                mime_type=MemoryMimeType.MARKDOWN,
                metadata={"path": str(path)},
            )
        )
    return memory


def _tool_map(config: AppConfig) -> dict[str, Callable[..., object]]:
    tools = build_tools(config)
    return {tool.__name__: tool for tool in tools}


async def run_single_agent(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console

    model_client = build_model_client(config)
    memory = await load_memory(config)
    agent = AssistantAgent(
        name="ecm_organoid_assistant",
        model_client=model_client,
        tools=build_tools(config),
        memory=[memory],
        system_message=build_single_agent_system_message(),
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        model_client_stream=True,
    )

    try:
        _emit_progress(progress_callback, "single", "Single-agent workflow started.")
        prompt = build_single_agent_task_prompt(
            query,
            report_name,
            config.max_pubmed_results,
            load_weekly_template(config.project_dir),
        )
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "single_agent_prompt", prompt)
        await Console(agent.run_stream(task=prompt), output_stats=True)
        report_path = expected_report_path(config, report_name)
        summary = f"Single-agent workflow finished. Report path: {report_path}"
        metadata_path = None
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "final_summary", summary)
            metadata_path = _write_run_metadata(
                run_artifacts=run_artifacts,
                config=config,
                query=query,
                workflow="single",
                report_name=report_name,
                report_path=report_path,
                final_summary=summary,
            )
        _emit_progress(progress_callback, "done", summary)
        return ResearchRunResult(
            workflow="single",
            report_path=report_path,
            final_summary=summary,
            run_dir=run_artifacts.run_dir if run_artifacts else None,
            metadata_path=metadata_path,
        )
    finally:
        await model_client.close()
        await memory.close()


async def run_team_agent(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    from autogen_agentchat.agents import AssistantAgent

    model_client = build_model_client(config)
    memory = await load_memory(config)
    template_text = load_weekly_template(config.project_dir)
    tools = _tool_map(config)

    planner_agent = AssistantAgent(
        name="planner_agent",
        description="Plans the research task decomposition and collaboration strategy.",
        model_client=model_client,
        memory=[memory],
        system_message=build_planner_agent_system_message(),
    )
    search_agent = AssistantAgent(
        name="search_agent",
        description="Collects PubMed, Crossref, and local-library evidence.",
        model_client=model_client,
        tools=[
            tools["search_pubmed"],
            tools["search_crossref"],
            tools["search_local_library"],
        ],
        memory=[memory],
        system_message=build_search_agent_system_message(config.max_pubmed_results),
        reflect_on_tool_use=True,
        max_tool_iterations=4,
    )
    evidence_agent = AssistantAgent(
        name="evidence_agent",
        description="Converts raw search results into structured evidence, tables, and hypotheses.",
        model_client=model_client,
        memory=[memory],
        system_message=build_evidence_agent_system_message(),
    )
    hypothesis_agent = AssistantAgent(
        name="hypothesis_agent",
        description="Generates mechanistic hypotheses and next-step experiments from evidence.",
        model_client=model_client,
        memory=[memory],
        system_message=build_hypothesis_agent_system_message(),
    )
    critic_agent = AssistantAgent(
        name="critic_agent",
        description="Reviews the synthesis for unsupported claims and missing citations.",
        model_client=model_client,
        memory=[memory],
        system_message=build_critic_agent_system_message(),
    )
    writer_agent = AssistantAgent(
        name="writer_agent",
        description="Writes the final weekly report and saves it.",
        model_client=model_client,
        tools=[tools["save_report"]],
        memory=[memory],
        system_message=build_writer_agent_system_message(report_name, template_text),
        reflect_on_tool_use=True,
        max_tool_iterations=1,
    )

    try:
        _emit_progress(progress_callback, "planner", "[0/5] PlannerAgent decomposing task...")
        planner_result = await planner_agent.run(task=build_planner_stage_task(query, template_text))
        planner_output = _result_text(planner_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "planner_agent", planner_output)

        _emit_progress(progress_callback, "search", "[1/5] SearchAgent collecting evidence...")
        search_result = await search_agent.run(
            task=build_search_stage_task_with_plan(
                query=query,
                planner_output=planner_output,
                max_pubmed_results=config.max_pubmed_results,
            ),
        )
        search_output = _result_text(search_result)
        if _needs_search_fallback(search_output):
            _emit_progress(progress_callback, "search_fallback", "[1.5/5] SearchAgent output incomplete, applying deterministic evidence fallback...")
            search_output = await _build_fallback_search_ledger(
                config=config,
                query=query,
                max_pubmed_results=config.max_pubmed_results,
                tools=tools,
            )
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "search_agent", search_output)

        _emit_progress(progress_callback, "evidence", "[2/5] EvidenceAgent structuring evidence...")
        evidence_result = await evidence_agent.run(
            task=build_evidence_stage_task(search_output, planner_output)
        )
        evidence_output = _result_text(evidence_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "evidence_agent", evidence_output)

        _emit_progress(progress_callback, "hypothesis", "[3/5] HypothesisAgent generating hypotheses...")
        hypothesis_result = await hypothesis_agent.run(
            task=build_hypothesis_stage_task(planner_output, evidence_output)
        )
        hypothesis_output = _result_text(hypothesis_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "hypothesis_agent", hypothesis_output)

        _emit_progress(progress_callback, "critic", "[4/5] CriticAgent reviewing collaboration output...")
        critic_result = await critic_agent.run(
            task=build_critic_stage_task(planner_output, evidence_output, hypothesis_output)
        )
        critic_output = _result_text(critic_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "critic_agent", critic_output)

        if _critic_requests_revision(critic_output):
            _emit_progress(progress_callback, "revision", "[4.5/5] Applying critic-driven revisions...")
            evidence_result = await evidence_agent.run(
                task=build_evidence_revision_task(
                    planner_output=planner_output,
                    search_output=search_output,
                    evidence_output=evidence_output,
                    critic_output=critic_output,
                )
            )
            evidence_output = _result_text(evidence_result)
            if run_artifacts is not None:
                write_stage_text(run_artifacts.run_dir, "evidence_agent_revision", evidence_output)

            hypothesis_result = await hypothesis_agent.run(
                task=build_hypothesis_revision_task(
                    planner_output=planner_output,
                    evidence_output=evidence_output,
                    hypothesis_output=hypothesis_output,
                    critic_output=critic_output,
                )
            )
            hypothesis_output = _result_text(hypothesis_result)
            if run_artifacts is not None:
                write_stage_text(run_artifacts.run_dir, "hypothesis_agent_revision", hypothesis_output)

            critic_result = await critic_agent.run(
                task=build_critic_stage_task(planner_output, evidence_output, hypothesis_output)
            )
            critic_output = _result_text(critic_result)
            if run_artifacts is not None:
                write_stage_text(run_artifacts.run_dir, "critic_agent_revision", critic_output)

        _emit_progress(progress_callback, "writer", "[5/5] WriterAgent drafting and saving report...")
        writer_result = await writer_agent.run(
            task=build_writer_stage_task(
                query=query,
                report_name=report_name,
                template_text=template_text,
                planner_output=planner_output,
                search_output=search_output,
                evidence_output=evidence_output,
                hypothesis_output=hypothesis_output,
                critic_output=critic_output,
            )
        )
        final_summary = _result_text(writer_result)
        metadata_path = None
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
            metadata_path = _write_run_metadata(
                run_artifacts=run_artifacts,
                config=config,
                query=query,
                workflow="team",
                report_name=report_name,
                report_path=expected_report_path(config, report_name),
                final_summary=final_summary,
                stage_files={
                    "planner_agent": str(run_artifacts.run_dir / "planner_agent.md"),
                    "search_agent": str(run_artifacts.run_dir / "search_agent.md"),
                    "evidence_agent": str(run_artifacts.run_dir / "evidence_agent.md"),
                    "hypothesis_agent": str(run_artifacts.run_dir / "hypothesis_agent.md"),
                    "critic_agent": str(run_artifacts.run_dir / "critic_agent.md"),
                    "evidence_agent_revision": str(run_artifacts.run_dir / "evidence_agent_revision.md")
                    if (run_artifacts.run_dir / "evidence_agent_revision.md").exists()
                    else None,
                    "hypothesis_agent_revision": str(run_artifacts.run_dir / "hypothesis_agent_revision.md")
                    if (run_artifacts.run_dir / "hypothesis_agent_revision.md").exists()
                    else None,
                    "critic_agent_revision": str(run_artifacts.run_dir / "critic_agent_revision.md")
                    if (run_artifacts.run_dir / "critic_agent_revision.md").exists()
                    else None,
                    "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
                },
            )
        _emit_progress(progress_callback, "done", final_summary)
        return ResearchRunResult(
            workflow="team",
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            run_dir=run_artifacts.run_dir if run_artifacts else None,
            metadata_path=metadata_path,
            planner_output=planner_output,
            search_output=search_output,
            evidence_output=evidence_output,
            hypothesis_output=hypothesis_output,
            critic_output=critic_output,
        )
    finally:
        await model_client.close()
        await memory.close()


async def run_mechanics_agent(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    data_path: Path,
    experiment_type: str,
    time_column: str,
    stress_column: str,
    strain_column: str,
    applied_stress: Optional[float],
    applied_strain: Optional[float],
    delimiter: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    from autogen_agentchat.agents import AssistantAgent

    model_client = build_model_client(config)
    memory = await load_memory(config)
    tools = _tool_map(config)

    planner_agent = AssistantAgent(
        name="planner_agent",
        description="Plans the ECM mechanics modeling task.",
        model_client=model_client,
        memory=[memory],
        system_message=build_planner_agent_system_message(),
    )
    mechanics_agent = AssistantAgent(
        name="mechanics_agent",
        description="Fits deterministic constitutive models and interprets parameters.",
        model_client=model_client,
        tools=[tools["fit_mechanics_model"], tools["simulate_mechanics_model"]],
        memory=[memory],
        system_message=build_mechanics_agent_system_message(),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    critic_agent = AssistantAgent(
        name="critic_agent",
        description="Reviews model assumptions, fit quality, and identifiability.",
        model_client=model_client,
        memory=[memory],
        system_message=build_critic_agent_system_message(),
    )
    writer_agent = AssistantAgent(
        name="writer_agent",
        description="Writes the final mechanics report and saves it.",
        model_client=model_client,
        tools=[tools["save_report"]],
        memory=[memory],
        system_message=build_writer_agent_system_message(report_name, "## 1. Modeling Goal\n## 2. Dataset And Experiment Type\n## 3. Constitutive Model\n## 4. Estimated Parameters\n## 5. Fit Quality\n## 6. Mechanical Interpretation\n## 7. Model Limitations\n## 8. Experimental Implications"),
        reflect_on_tool_use=True,
        max_tool_iterations=1,
    )

    try:
        _emit_progress(progress_callback, "mechanics_planner", "[0/4] PlannerAgent preparing mechanics task...")
        planner_result = await planner_agent.run(
            task=build_mechanics_planner_stage_task(
                query=query,
                data_path=data_path,
                experiment_type=experiment_type,
            )
        )
        planner_output = _result_text(planner_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "mechanics_planner", planner_output)

        _emit_progress(progress_callback, "mechanics_fit", "[1/4] MechanicsAgent fitting constitutive model...")
        mechanics_result = await mechanics_agent.run(
            task=build_mechanics_stage_task(
                query=query,
                planner_output=planner_output,
                data_path=data_path,
                experiment_type=experiment_type,
                time_column=time_column,
                stress_column=stress_column,
                strain_column=strain_column,
                applied_stress=applied_stress,
                applied_strain=applied_strain,
                delimiter=delimiter,
            )
        )
        mechanics_output = _result_text(mechanics_result)
        if _needs_mechanics_fallback(mechanics_output):
            _emit_progress(progress_callback, "mechanics_fallback", "[1.5/4] MechanicsAgent output incomplete, applying deterministic mechanics fallback...")
            mechanics_output = _build_fallback_mechanics_ledger(
                tools=tools,
                data_path=data_path,
                experiment_type=experiment_type,
                time_column=time_column,
                stress_column=stress_column,
                strain_column=strain_column,
                applied_stress=applied_stress,
                applied_strain=applied_strain,
                delimiter=delimiter,
            )
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "mechanics_agent", mechanics_output)

        _emit_progress(progress_callback, "mechanics_critic", "[2/4] CriticAgent reviewing mechanics fit...")
        critic_result = await critic_agent.run(
            task=build_mechanics_critic_stage_task(
                planner_output=planner_output,
                mechanics_output=mechanics_output,
            )
        )
        critic_output = _result_text(critic_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "mechanics_critic", critic_output)

        _emit_progress(progress_callback, "mechanics_writer", "[3/4] WriterAgent drafting mechanics report...")
        writer_result = await writer_agent.run(
            task=build_mechanics_writer_stage_task(
                query=query,
                report_name=report_name,
                planner_output=planner_output,
                mechanics_output=mechanics_output,
                critic_output=critic_output,
            )
        )
        final_summary = _result_text(writer_result)
        metadata_path = None
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
            metadata_path = _write_run_metadata(
                run_artifacts=run_artifacts,
                config=config,
                query=query,
                workflow="mechanics",
                report_name=report_name,
                report_path=expected_report_path(config, report_name),
                final_summary=final_summary,
                stage_files={
                    "mechanics_planner": str(run_artifacts.run_dir / "mechanics_planner.md"),
                    "mechanics_agent": str(run_artifacts.run_dir / "mechanics_agent.md"),
                    "mechanics_critic": str(run_artifacts.run_dir / "mechanics_critic.md"),
                    "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
                },
            )
        _emit_progress(progress_callback, "done", final_summary)
        return ResearchRunResult(
            workflow="mechanics",
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            run_dir=run_artifacts.run_dir if run_artifacts else None,
            metadata_path=metadata_path,
            planner_output=planner_output,
            mechanics_output=mechanics_output,
            critic_output=critic_output,
        )
    finally:
        await model_client.close()
        await memory.close()


async def run_hybrid_agent(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    data_path: Path,
    experiment_type: str,
    time_column: str,
    stress_column: str,
    strain_column: str,
    applied_stress: Optional[float],
    applied_strain: Optional[float],
    delimiter: str,
    simulation_fiber_density: Optional[float],
    simulation_fiber_stiffness: Optional[float],
    simulation_bending_stiffness: Optional[float],
    simulation_crosslink_prob: Optional[float],
    simulation_domain_size: Optional[float],
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    from autogen_agentchat.agents import AssistantAgent

    model_client = build_model_client(config)
    memory = await load_memory(config)
    template_text = load_weekly_template(config.project_dir)
    tools = _tool_map(config)

    planner_agent = AssistantAgent(
        name="planner_agent",
        description="Plans the hybrid literature-mechanics-simulation task.",
        model_client=model_client,
        memory=[memory],
        system_message=build_planner_agent_system_message(),
    )
    search_agent = AssistantAgent(
        name="search_agent",
        description="Collects literature evidence for ECM materials and mechanics.",
        model_client=model_client,
        tools=[tools["search_pubmed"], tools["search_crossref"], tools["search_local_library"]],
        memory=[memory],
        system_message=build_search_agent_system_message(config.max_pubmed_results),
        reflect_on_tool_use=True,
        max_tool_iterations=4,
    )
    evidence_agent = AssistantAgent(
        name="evidence_agent",
        description="Structures literature evidence into design-relevant findings.",
        model_client=model_client,
        memory=[memory],
        system_message=build_evidence_agent_system_message(),
    )
    hypothesis_agent = AssistantAgent(
        name="hypothesis_agent",
        description="Suggests material and structural design directions from evidence.",
        model_client=model_client,
        memory=[memory],
        system_message=build_hypothesis_agent_system_message(),
    )
    mechanics_agent = AssistantAgent(
        name="mechanics_agent",
        description="Fits deterministic constitutive mechanics models from experimental data.",
        model_client=model_client,
        tools=[tools["fit_mechanics_model"], tools["simulate_mechanics_model"]],
        memory=[memory],
        system_message=build_mechanics_agent_system_message(),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    simulation_agent = AssistantAgent(
        name="simulation_agent",
        description="Runs 3D bead-spring ECM fiber-network simulations.",
        model_client=model_client,
        tools=[tools["run_fiber_network_simulation"], tools["run_fiber_network_parameter_scan"]],
        memory=[memory],
        system_message=build_simulation_agent_system_message(),
        reflect_on_tool_use=True,
        max_tool_iterations=2,
    )
    critic_agent = AssistantAgent(
        name="critic_agent",
        description="Checks cross-stage consistency across evidence, mechanics, and simulation.",
        model_client=model_client,
        memory=[memory],
        system_message=build_critic_agent_system_message(),
    )
    writer_agent = AssistantAgent(
        name="writer_agent",
        description="Writes the final hybrid engineering report.",
        model_client=model_client,
        tools=[tools["save_report"]],
        memory=[memory],
        system_message=build_writer_agent_system_message(report_name, template_text),
        reflect_on_tool_use=True,
        max_tool_iterations=1,
    )

    try:
        _emit_progress(progress_callback, "planner", "[0/7] PlannerAgent decomposing hybrid task...")
        planner_result = await planner_agent.run(task=build_planner_stage_task(query, template_text))
        planner_output = _result_text(planner_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "planner_agent", planner_output)

        _emit_progress(progress_callback, "search", "[1/7] SearchAgent collecting literature evidence...")
        search_result = await search_agent.run(
            task=build_search_stage_task_with_plan(
                query=query,
                planner_output=planner_output,
                max_pubmed_results=config.max_pubmed_results,
            )
        )
        search_output = _result_text(search_result)
        if _needs_search_fallback(search_output):
            _emit_progress(progress_callback, "search_fallback", "[1.5/7] Applying deterministic literature fallback...")
            search_output = await _build_fallback_search_ledger(
                config=config,
                query=query,
                max_pubmed_results=config.max_pubmed_results,
                tools=tools,
            )
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "search_agent", search_output)

        _emit_progress(progress_callback, "evidence", "[2/7] EvidenceAgent structuring literature evidence...")
        evidence_result = await evidence_agent.run(task=build_evidence_stage_task(search_output, planner_output))
        evidence_output = _result_text(evidence_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "evidence_agent", evidence_output)

        _emit_progress(progress_callback, "hypothesis", "[3/7] HypothesisAgent suggesting design directions...")
        hypothesis_result = await hypothesis_agent.run(
            task=build_hypothesis_stage_task(planner_output, evidence_output)
        )
        hypothesis_output = _result_text(hypothesis_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "hypothesis_agent", hypothesis_output)

        _emit_progress(progress_callback, "mechanics_fit", "[4/7] MechanicsAgent fitting constitutive model...")
        mechanics_result = await mechanics_agent.run(
            task=build_mechanics_stage_task(
                query=query,
                planner_output=planner_output,
                data_path=data_path,
                experiment_type=experiment_type,
                time_column=time_column,
                stress_column=stress_column,
                strain_column=strain_column,
                applied_stress=applied_stress,
                applied_strain=applied_strain,
                delimiter=delimiter,
            )
        )
        mechanics_output = _result_text(mechanics_result)
        if _needs_mechanics_fallback(mechanics_output):
            _emit_progress(progress_callback, "mechanics_fallback", "[4.5/7] Applying deterministic mechanics fallback...")
            mechanics_output = _build_fallback_mechanics_ledger(
                tools=tools,
                data_path=data_path,
                experiment_type=experiment_type,
                time_column=time_column,
                stress_column=stress_column,
                strain_column=strain_column,
                applied_stress=applied_stress,
                applied_strain=applied_strain,
                delimiter=delimiter,
            )
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "mechanics_agent", mechanics_output)

        simulation_defaults = _default_simulation_parameters(
            evidence_output=evidence_output,
            mechanics_output=mechanics_output,
            fiber_density=simulation_fiber_density,
            fiber_stiffness=simulation_fiber_stiffness,
            bending_stiffness=simulation_bending_stiffness,
            crosslink_prob=simulation_crosslink_prob,
            domain_size=simulation_domain_size,
        )

        _emit_progress(progress_callback, "simulation", "[5/7] SimulationAgent running fiber-network simulation...")
        simulation_result = await simulation_agent.run(
            task=build_simulation_stage_task(
                query=query,
                evidence_output=evidence_output,
                hypothesis_output=hypothesis_output,
                mechanics_output=mechanics_output,
                simulation_defaults_json=json.dumps(simulation_defaults, ensure_ascii=False, indent=2),
            )
        )
        simulation_output = _result_text(simulation_result)
        if _needs_simulation_fallback(simulation_output):
            _emit_progress(progress_callback, "simulation_fallback", "[5.5/7] Applying deterministic simulation fallback...")
            simulation_output = _build_fallback_simulation_ledger(tools=tools, parameters=simulation_defaults)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "simulation_agent", simulation_output)

        _emit_progress(progress_callback, "critic", "[6/7] CriticAgent reviewing cross-stage consistency...")
        critic_result = await critic_agent.run(
            task=build_hybrid_critic_stage_task(
                planner_output=planner_output,
                evidence_output=evidence_output,
                hypothesis_output=hypothesis_output,
                mechanics_output=mechanics_output,
                simulation_output=simulation_output,
            )
        )
        critic_output = _result_text(critic_result)
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "critic_agent", critic_output)

        _emit_progress(progress_callback, "writer", "[7/7] WriterAgent drafting hybrid engineering report...")
        writer_result = await writer_agent.run(
            task=build_hybrid_writer_stage_task(
                query=query,
                report_name=report_name,
                planner_output=planner_output,
                search_output=search_output,
                evidence_output=evidence_output,
                hypothesis_output=hypothesis_output,
                mechanics_output=mechanics_output,
                simulation_output=simulation_output,
                critic_output=critic_output,
            )
        )
        final_summary = _result_text(writer_result)
        metadata_path = None
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
            metadata_path = _write_run_metadata(
                run_artifacts=run_artifacts,
                config=config,
                query=query,
                workflow="hybrid",
                report_name=report_name,
                report_path=expected_report_path(config, report_name),
                final_summary=final_summary,
                stage_files={
                    "planner_agent": str(run_artifacts.run_dir / "planner_agent.md"),
                    "search_agent": str(run_artifacts.run_dir / "search_agent.md"),
                    "evidence_agent": str(run_artifacts.run_dir / "evidence_agent.md"),
                    "hypothesis_agent": str(run_artifacts.run_dir / "hypothesis_agent.md"),
                    "mechanics_agent": str(run_artifacts.run_dir / "mechanics_agent.md"),
                    "simulation_agent": str(run_artifacts.run_dir / "simulation_agent.md"),
                    "critic_agent": str(run_artifacts.run_dir / "critic_agent.md"),
                    "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
                },
            )
        _emit_progress(progress_callback, "done", final_summary)
        return ResearchRunResult(
            workflow="hybrid",
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            run_dir=run_artifacts.run_dir if run_artifacts else None,
            metadata_path=metadata_path,
            planner_output=planner_output,
            search_output=search_output,
            evidence_output=evidence_output,
            hypothesis_output=hypothesis_output,
            mechanics_output=mechanics_output,
            simulation_output=simulation_output,
            critic_output=critic_output,
        )
    finally:
        await model_client.close()
        await memory.close()


async def run_design_agent(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    target_stiffness: float,
    target_anisotropy: float,
    target_connectivity: float,
    target_stress_propagation: float,
    design_extra_targets: dict[str, Any] | None,
    constraint_max_anisotropy: Optional[float],
    constraint_min_connectivity: Optional[float],
    constraint_max_risk_index: Optional[float],
    constraint_min_stress_propagation: Optional[float],
    design_extra_constraints: dict[str, Any] | None,
    design_top_k: int,
    design_candidate_budget: int,
    design_monte_carlo_runs: int,
    design_run_simulation: bool = False,
    design_simulation_scenario: str = "bulk_mechanics",
    design_simulation_top_k: int = 0,
    cell_contractility: Optional[float] = None,
    organoid_radius: Optional[float] = None,
    matrix_youngs_modulus: Optional[float] = None,
    matrix_poisson_ratio: Optional[float] = None,
    simulation_request_json: str = "",
    condition_concentration_fraction: Optional[float] = None,
    condition_curing_seconds: Optional[float] = None,
    condition_overrides: dict[str, Any] | None = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    tools = _tool_map(config)

    targets = {
        "stiffness": float(target_stiffness),
        "anisotropy": float(target_anisotropy),
        "connectivity": float(target_connectivity),
        "stress_propagation": float(target_stress_propagation),
    }
    if design_extra_targets:
        targets.update({str(key): float(value) for key, value in design_extra_targets.items()})
    calibration_results = load_latest_calibration_results(config.project_dir)
    calibrated_search_space, calibration_context = _design_search_space_from_priors(
        calibration_results,
        inferred_family_hint=query,
        condition_concentration_fraction=condition_concentration_fraction,
        condition_curing_seconds=condition_curing_seconds,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffness,
        target_anisotropy=target_anisotropy,
        target_connectivity=target_connectivity,
        target_stress_propagation=target_stress_propagation,
    )
    constraints = _design_constraints_dict(
        constraint_max_anisotropy=constraint_max_anisotropy,
        constraint_min_connectivity=constraint_min_connectivity,
        constraint_max_risk_index=constraint_max_risk_index,
        constraint_min_stress_propagation=constraint_min_stress_propagation,
        extra_constraints=design_extra_constraints,
    )

    _emit_progress(progress_callback, "design_validation", "[0/4] Validating fiber-network physics core...")
    validation_text = tools["run_fiber_network_validation"]()
    validation_payload = _parse_json_object(validation_text)
    if not validation_payload:
        raise RuntimeError(f"Fiber-network validation failed before design search: {validation_text}")
    validation_output = _build_design_validation_ledger(validation_payload)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "design_validation", validation_output)

    _emit_progress(progress_callback, "design_search", "[1/4] Searching ECM design space against target mechanics...")
    design_text = tools["design_fiber_network_candidates"](
        float(target_stiffness),
        float(target_anisotropy),
        float(target_connectivity),
        float(target_stress_propagation),
        json.dumps(design_extra_targets or {}, ensure_ascii=False) if design_extra_targets else "",
        float(constraint_max_anisotropy or 0.0),
        float(constraint_min_connectivity or 0.0),
        float(constraint_max_risk_index or 0.0),
        float(constraint_min_stress_propagation or 0.0),
        json.dumps(design_extra_constraints or {}, ensure_ascii=False) if design_extra_constraints else "",
        int(design_top_k),
        int(design_candidate_budget),
        int(design_monte_carlo_runs),
        0.2,
        "x",
        0.15,
        1234,
        500,
        1e-5,
        8,
        json.dumps(calibrated_search_space, ensure_ascii=False) if calibrated_search_space else "",
    )
    design_payload = _parse_json_object(design_text)
    if not design_payload:
        raise RuntimeError(f"Fiber-network design search failed: {design_text}")
    formulation_recommendations = recommend_formulations_from_design_payload(design_payload, max_candidates=int(design_top_k))
    design_assessment = _assess_design_candidate(
        targets=targets,
        validation_payload=validation_payload,
        candidate=(design_payload.get("top_candidates", []) or [{}])[0] if isinstance(design_payload, dict) and design_payload.get("top_candidates") else {},
    )
    design_output = _build_design_stage_ledger(
        query=query,
        targets=targets,
        constraints=constraints,
        payload=design_payload,
    )
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "design_agent", design_output)

    formulation_output = _build_formulation_ledger(formulation_recommendations)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "formulation_mapping", formulation_output)

    sensitivity_output = "## Sensitivity Scan\n- No candidate designs were returned."
    scan_payload: dict[str, object] = {}
    top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
    if top_candidates:
        best_candidate = top_candidates[0]
        best_params = best_candidate.get("parameters", {})
        _emit_progress(progress_callback, "design_sensitivity", "[2/4] Running sensitivity scan for the top ECM candidate...")
        scan_text = tools["run_fiber_network_parameter_scan"](
            float(best_params.get("fiber_density", 0.35)),
            float(best_params.get("fiber_stiffness", 8.0)),
            float(best_params.get("bending_stiffness", 0.2)),
            float(best_params.get("crosslink_prob", 0.45)),
            float(best_params.get("domain_size", 1.0)),
            0.2,
            "x",
            0.15,
            1234,
            500,
            1e-5,
            max(3, min(int(design_monte_carlo_runs), 4)),
            8,
        )
        scan_payload = _parse_json_object(scan_text)
        sensitivity_output = _build_design_sensitivity_ledger(scan_payload)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "design_sensitivity", sensitivity_output)

    design_simulation_payload = _design_simulation_not_run_payload(
        enabled=design_run_simulation,
        scenario=design_simulation_scenario,
        reason=(
            config.febio_status_message
            if design_run_simulation and not config.febio_executable
            else "Design workflow simulation verification was not requested."
        ),
    )
    design_simulation_output = _build_design_simulation_ledger(design_simulation_payload)
    if design_run_simulation:
        _emit_progress(progress_callback, "design_febio", "[3/5] Running FEBio verification for top ECM candidates...")
        design_simulation_payload = _run_design_candidate_simulations(
            config=config,
            design_payload=design_payload,
            target_stiffness=target_stiffness,
            scenario=design_simulation_scenario,
            simulation_request_json=simulation_request_json,
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
            top_k=max(1, int(design_simulation_top_k or design_top_k)),
        )
        design_simulation_output = _build_design_simulation_ledger(design_simulation_payload)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "design_simulation", design_simulation_output)

    _emit_progress(progress_callback, "design_writer", "[4/5] Writing ECM design report...")
    report_content = _build_design_report_markdown(
        query=query,
        targets=targets,
        constraints=constraints,
        validation_payload=validation_payload,
        design_assessment=design_assessment,
        design_payload=design_payload,
        formulation_recommendations=formulation_recommendations,
        calibrated_search_space=calibrated_search_space,
        calibration_context=calibration_context,
        sensitivity_output=sensitivity_output,
        design_simulation=design_simulation_payload,
    )
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_design_final_summary(
        validation_payload=validation_payload,
        design_assessment=design_assessment,
        design_payload=design_payload,
        calibrated_search_space=calibrated_search_space,
        calibration_context=calibration_context,
        design_simulation=design_simulation_payload,
        saved_path=saved_path,
    )

    if run_artifacts is not None:
        design_summary_payload = _build_design_summary_payload(
            query=query,
            targets=targets,
            constraints=constraints,
            validation_payload=validation_payload,
            design_assessment=design_assessment,
            design_payload=design_payload,
            formulation_recommendations=formulation_recommendations,
            calibrated_search_space=calibrated_search_space,
            calibration_context=calibration_context,
            requested_condition_overrides=condition_overrides,
            scan_payload=scan_payload,
            sensitivity_output=sensitivity_output,
            design_simulation=design_simulation_payload,
            report_name=report_name,
            saved_report_path=saved_path,
        )
        write_json(run_artifacts.run_dir / "design_summary.json", design_summary_payload)

    metadata_path = None
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="design",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "design_validation": str(run_artifacts.run_dir / "design_validation.md"),
                "design_agent": str(run_artifacts.run_dir / "design_agent.md"),
                "design_sensitivity": str(run_artifacts.run_dir / "design_sensitivity.md"),
                "design_simulation": str(run_artifacts.run_dir / "design_simulation.md"),
                "formulation_mapping": str(run_artifacts.run_dir / "formulation_mapping.md"),
                "design_summary": str(run_artifacts.run_dir / "design_summary.json"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="design",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        design_output=design_output,
        simulation_output=design_simulation_output,
        formulation_output=formulation_output,
    )


async def run_design_campaign(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    target_stiffnesses: list[float],
    target_anisotropy: float,
    target_connectivity: float,
    target_stress_propagation: float,
    design_extra_targets: dict[str, Any] | None,
    constraint_max_anisotropy: Optional[float],
    constraint_min_connectivity: Optional[float],
    constraint_max_risk_index: Optional[float],
    constraint_min_stress_propagation: Optional[float],
    design_extra_constraints: dict[str, Any] | None,
    design_top_k: int,
    design_candidate_budget: int,
    design_monte_carlo_runs: int,
    condition_concentration_fraction: Optional[float] = None,
    condition_curing_seconds: Optional[float] = None,
    condition_overrides: dict[str, Any] | None = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    tools = _tool_map(config)
    constraints = _design_constraints_dict(
        constraint_max_anisotropy=constraint_max_anisotropy,
        constraint_min_connectivity=constraint_min_connectivity,
        constraint_max_risk_index=constraint_max_risk_index,
        constraint_min_stress_propagation=constraint_min_stress_propagation,
        extra_constraints=design_extra_constraints,
    )
    calibration_results = load_latest_calibration_results(config.project_dir)
    calibrated_search_space, calibration_context = _design_search_space_from_priors(
        calibration_results,
        inferred_family_hint=query,
        condition_concentration_fraction=condition_concentration_fraction,
        condition_curing_seconds=condition_curing_seconds,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffnesses[0] if target_stiffnesses else None,
        target_anisotropy=target_anisotropy,
        target_connectivity=target_connectivity,
        target_stress_propagation=target_stress_propagation,
    )

    _emit_progress(progress_callback, "campaign_validation", "[0/3] Validating fiber-network physics core...")
    validation_text = tools["run_fiber_network_validation"]()
    validation_payload = _parse_json_object(validation_text)
    if not validation_payload:
        raise RuntimeError(f"Fiber-network validation failed before campaign search: {validation_text}")
    validation_output = _build_design_validation_ledger(validation_payload)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "campaign_validation", validation_output)

    campaign_results: list[dict[str, object]] = []
    _emit_progress(progress_callback, "campaign_search", "[1/3] Running design campaign across target stiffness windows...")
    for index, target_stiffness in enumerate(target_stiffnesses, start=1):
        target_search_space, target_calibration_context = _design_search_space_from_priors(
            calibration_results,
            inferred_family_hint=query,
            condition_concentration_fraction=condition_concentration_fraction,
            condition_curing_seconds=condition_curing_seconds,
            condition_overrides=condition_overrides,
            target_stiffness=target_stiffness,
            target_anisotropy=target_anisotropy,
            target_connectivity=target_connectivity,
            target_stress_propagation=target_stress_propagation,
        )
        design_text = tools["design_fiber_network_candidates"](
            float(target_stiffness),
            float(target_anisotropy),
            float(target_connectivity),
            float(target_stress_propagation),
            json.dumps(design_extra_targets or {}, ensure_ascii=False) if design_extra_targets else "",
            float(constraint_max_anisotropy or 0.0),
            float(constraint_min_connectivity or 0.0),
            float(constraint_max_risk_index or 0.0),
            float(constraint_min_stress_propagation or 0.0),
            json.dumps(design_extra_constraints or {}, ensure_ascii=False) if design_extra_constraints else "",
            int(design_top_k),
            int(design_candidate_budget),
            int(design_monte_carlo_runs),
            0.2,
            "x",
            0.15,
            1234 + 1000 * index,
            500,
            1e-5,
            8,
            json.dumps(target_search_space, ensure_ascii=False) if target_search_space else "",
        )
        design_payload = _parse_json_object(design_text)
        if not design_payload:
            raise RuntimeError(f"Design campaign target {target_stiffness} failed: {design_text}")
        top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
        best_candidate = top_candidates[0] if top_candidates else {}
        campaign_results.append(
            {
                "target_stiffness": float(target_stiffness),
                "targets": {
                "stiffness": float(target_stiffness),
                "anisotropy": float(target_anisotropy),
                "connectivity": float(target_connectivity),
                "stress_propagation": float(target_stress_propagation),
                **{str(key): float(value) for key, value in (design_extra_targets or {}).items()},
            },
                "constraints": constraints,
                "design_payload": design_payload,
                "best_candidate": best_candidate,
                "design_assessment": _assess_design_candidate(
                    targets={
                        "stiffness": float(target_stiffness),
                        "anisotropy": float(target_anisotropy),
                        "connectivity": float(target_connectivity),
                        "stress_propagation": float(target_stress_propagation),
                    },
                    validation_payload=validation_payload,
                    candidate=best_candidate if isinstance(best_candidate, dict) else {},
                ),
                "calibrated_search_space": target_search_space,
                "calibration_context": target_calibration_context,
            }
        )
    campaign_formulations = recommend_campaign_formulations(campaign_results)
    campaign_calibration_context = _summarize_campaign_calibration_contexts(campaign_results)
    campaign_assessment = _assess_design_campaign(
        validation_payload=validation_payload,
        campaign_results=campaign_results,
    )

    campaign_output = _build_design_campaign_ledger(query=query, campaign_results=campaign_results, constraints=constraints)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "campaign_agent", campaign_output)
    formulation_output = _build_campaign_formulation_ledger(campaign_formulations)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "formulation_mapping", formulation_output)

    _emit_progress(progress_callback, "campaign_writer", "[2/3] Writing design campaign report...")
    report_content = _build_design_campaign_report_markdown(
        query=query,
        validation_payload=validation_payload,
        campaign_assessment=campaign_assessment,
        constraints=constraints,
        campaign_results=campaign_results,
        formulation_recommendations=campaign_formulations,
        calibrated_search_space=calibrated_search_space,
        calibration_context=campaign_calibration_context,
    )
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_design_campaign_final_summary(
        validation_payload=validation_payload,
        campaign_assessment=campaign_assessment,
        campaign_results=campaign_results,
        calibrated_search_space=calibrated_search_space,
        calibration_context=campaign_calibration_context,
        saved_path=saved_path,
    )

    metadata_path = None
    if run_artifacts is not None:
        write_json(
            run_artifacts.run_dir / "campaign_summary.json",
            _build_design_campaign_summary_payload(
                query=query,
                constraints=constraints,
                validation_payload=validation_payload,
                campaign_assessment=campaign_assessment,
                campaign_results=campaign_results,
                formulation_recommendations=campaign_formulations,
                calibrated_search_space=calibrated_search_space,
                calibration_context=campaign_calibration_context,
                requested_condition_overrides=condition_overrides,
                report_name=report_name,
                saved_report_path=saved_path,
            ),
        )
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="design_campaign",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "campaign_validation": str(run_artifacts.run_dir / "campaign_validation.md"),
                "campaign_agent": str(run_artifacts.run_dir / "campaign_agent.md"),
                "formulation_mapping": str(run_artifacts.run_dir / "formulation_mapping.md"),
                "campaign_summary": str(run_artifacts.run_dir / "campaign_summary.json"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="design_campaign",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        design_output=campaign_output,
        formulation_output=formulation_output,
    )


def _build_simulation_workflow_ledger(execution_payload: dict[str, object]) -> str:
    result_payload = execution_payload.get("simulation_result", {}) if isinstance(execution_payload.get("simulation_result"), dict) else {}
    metrics_payload = execution_payload.get("simulation_metrics", {}) if isinstance(execution_payload.get("simulation_metrics"), dict) else {}
    warnings = result_payload.get("warnings", []) if isinstance(result_payload.get("warnings"), list) else []
    lines = [
        "## FEBio Simulation Workflow",
        f"- scenario: {execution_payload.get('request', {}).get('scenario', 'NR') if isinstance(execution_payload.get('request'), dict) else 'NR'}",
        f"- status: {execution_payload.get('status', 'NR')}",
        f"- effective_stiffness: {metrics_payload.get('effective_stiffness', 'NR')}",
        f"- peak_stress: {metrics_payload.get('peak_stress', 'NR')}",
        f"- displacement_decay_length: {metrics_payload.get('displacement_decay_length', 'NR')}",
        f"- strain_heterogeneity: {metrics_payload.get('strain_heterogeneity', 'NR')}",
        f"- target_mismatch_score: {metrics_payload.get('target_mismatch_score', 'NR')}",
        f"- feasibility_flags: {json.dumps(metrics_payload.get('feasibility_flags', {}), ensure_ascii=False)}",
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines)


def _build_simulation_report_markdown(
    *,
    query: str,
    request_payload: dict[str, object],
    execution_payload: dict[str, object],
) -> str:
    result_payload = execution_payload.get("simulation_result", {}) if isinstance(execution_payload.get("simulation_result"), dict) else {}
    metrics_payload = execution_payload.get("simulation_metrics", {}) if isinstance(execution_payload.get("simulation_metrics"), dict) else {}
    recommendation_lines = [
        "- Use bulk effective stiffness and peak stress to judge basic material feasibility.",
        "- Use stress propagation distance and strain heterogeneity to judge whether local cell/organoid perturbations remain localized or spread excessively.",
    ]
    if metrics_payload.get("candidate_suitability_score_components"):
        recommendation_lines.append(
            f"- Candidate suitability score: {metrics_payload.get('candidate_suitability_score_components', {}).get('suitability_score', 'NR')}"
        )
    uncertainty_lines = [
        "- This simulation is template-driven and only supports a constrained phase-1 geometry/material space.",
        f"- FEBio warnings: {json.dumps(result_payload.get('warnings', []), ensure_ascii=False)}",
        f"- Error message: {result_payload.get('error_message', 'NR')}",
    ]
    return "\n".join(
        [
            "# FEBio Simulation Report",
            "",
            "## 1. Goal",
            f"- Query: {query}",
            f"- Request: {json.dumps(request_payload, ensure_ascii=False)}",
            "",
            "## 2. Execution Status",
            f"- status: {execution_payload.get('status', 'NR')}",
            f"- runner: {json.dumps(execution_payload.get('runner', {}), ensure_ascii=False)}",
            f"- warnings: {json.dumps(result_payload.get('warnings', []), ensure_ascii=False)}",
            f"- error_message: {result_payload.get('error_message', 'NR')}",
            "",
            "## 3. Simulation Evidence",
            f"- extracted_fields: {json.dumps(result_payload.get('extracted_fields', {}), ensure_ascii=False)}",
            "",
            "## 4. Key Mechanical Metrics",
            f"- effective_stiffness: {metrics_payload.get('effective_stiffness', 'NR')}",
            f"- peak_stress: {metrics_payload.get('peak_stress', 'NR')}",
            f"- peak_matrix_stress: {metrics_payload.get('peak_matrix_stress', 'NR')}",
            f"- displacement_decay_length: {metrics_payload.get('displacement_decay_length', 'NR')}",
            f"- stress_propagation_distance: {metrics_payload.get('stress_propagation_distance', 'NR')}",
            f"- strain_heterogeneity: {metrics_payload.get('strain_heterogeneity', 'NR')}",
            f"- target_mismatch_score: {metrics_payload.get('target_mismatch_score', 'NR')}",
            f"- displacement_field_summary: {json.dumps(metrics_payload.get('displacement_field_summary', {}), ensure_ascii=False)}",
            f"- matrix_stress_distribution_summary: {json.dumps(metrics_payload.get('matrix_stress_distribution_summary', {}), ensure_ascii=False)}",
            f"- interface_deformation: {metrics_payload.get('interface_deformation', 'NR')}",
            f"- compression_tension_region_summary: {json.dumps(metrics_payload.get('compression_tension_region_summary', {}), ensure_ascii=False)}",
            f"- candidate_suitability_score_components: {json.dumps(metrics_payload.get('candidate_suitability_score_components', {}), ensure_ascii=False)}",
            f"- feasibility_flags: {json.dumps(metrics_payload.get('feasibility_flags', {}), ensure_ascii=False)}",
            "",
            "## 5. Recommendation Rationale",
            *recommendation_lines,
            "",
            "## 6. Uncertainty and Failure Modes",
            *uncertainty_lines,
            "",
            "## 7. Artifact Paths",
            f"- simulation_dir: {execution_payload.get('simulation_dir', 'NR')}",
            f"- final_summary_path: {execution_payload.get('final_summary_path', 'NR')}",
        ]
    )


def _build_simulation_final_summary(execution_payload: dict[str, object], saved_path: str) -> str:
    metrics_payload = execution_payload.get("simulation_metrics", {}) if isinstance(execution_payload.get("simulation_metrics"), dict) else {}
    return (
        f"Simulation workflow finished. status={execution_payload.get('status', 'NR')}, "
        f"scenario={execution_payload.get('request', {}).get('scenario', 'NR') if isinstance(execution_payload.get('request'), dict) else 'NR'}, "
        f"effective_stiffness={metrics_payload.get('effective_stiffness', 'NR')}, "
        f"peak_stress={metrics_payload.get('peak_stress', 'NR')}. Report saved to {saved_path}."
    )


async def run_simulation_workflow(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    simulation_scenario: str,
    simulation_request_json: str,
    target_stiffness: Optional[float],
    cell_contractility: Optional[float],
    organoid_radius: Optional[float],
    matrix_youngs_modulus: Optional[float],
    matrix_poisson_ratio: Optional[float],
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    tools = _tool_map(config)
    _emit_progress(progress_callback, "simulation_request", "[0/3] Building constrained FEBio simulation request...")
    request_text = tools["build_febio_simulation_request"](
        simulation_scenario,
        simulation_request_json,
        float(target_stiffness or 0.0),
        float(cell_contractility or 0.0),
        float(organoid_radius or 0.0),
        float(matrix_youngs_modulus or 0.0),
        float(matrix_poisson_ratio if matrix_poisson_ratio is not None else 0.3),
    )
    request_payload = _parse_json_object(request_text)
    if not request_payload:
        raise RuntimeError(f"Failed to build FEBio simulation request: {request_text}")

    _emit_progress(progress_callback, "simulation_runner", "[1/3] Running FEBio simulation...")
    run_text = tools["run_febio_simulation"](json.dumps(request_payload, ensure_ascii=False), "")
    execution_payload = _parse_json_object(run_text)
    if not execution_payload:
        raise RuntimeError(f"FEBio simulation workflow failed: {run_text}")

    simulation_output = _build_simulation_workflow_ledger(execution_payload)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "simulation_agent", simulation_output)

    _emit_progress(progress_callback, "simulation_writer", "[2/3] Writing FEBio simulation report...")
    report_content = _build_simulation_report_markdown(
        query=query,
        request_payload=request_payload,
        execution_payload=execution_payload,
    )
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_simulation_final_summary(execution_payload, saved_path)

    metadata_path = None
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="simulation",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "simulation_agent": str(run_artifacts.run_dir / "simulation_agent.md"),
                "simulation_dir": str(run_artifacts.run_dir / "simulation"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="simulation",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        simulation_output=simulation_output,
    )


async def run_benchmark_workflow(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    _emit_progress(progress_callback, "benchmark_solver", "[0/3] Running solver benchmark suite...")
    payload = run_mechanics_benchmark_suite(project_dir=config.project_dir)

    solver_output = _build_solver_benchmark_ledger(payload["solver_benchmark"])
    load_output = _build_load_ladder_benchmark_ledger(payload["load_ladder_benchmark"])
    scaling_output = _build_scaling_benchmark_ledger(payload["scaling_benchmark"])
    design_output = _build_inverse_design_benchmark_ledger(payload["inverse_design_benchmark"])
    property_design_output = _build_property_target_design_benchmark_ledger(payload["property_target_design_benchmark"])
    repeatability_output = _build_repeatability_benchmark_ledger(payload["repeatability_benchmark"])
    identifiability_output = _build_identifiability_benchmark_ledger(payload["identifiability_proxy_benchmark"])
    fit_output = _build_fit_benchmark_ledger(payload["mechanics_fit_benchmark"])
    simulation_smoke_output = _build_simulation_smoke_benchmark_ledger(payload["simulation_smoke_benchmark"])
    calibration_design_output = _build_calibration_design_benchmark_ledger(payload["calibration_design_benchmark"])
    benchmark_output = _build_benchmark_summary_ledger(query=query, payload=payload)

    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "benchmark_solver", solver_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_load_ladder", load_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_scaling", scaling_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_design", design_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_property_design", property_design_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_repeatability", repeatability_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_identifiability", identifiability_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_fit", fit_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_simulation_smoke", simulation_smoke_output)
        write_stage_text(run_artifacts.run_dir, "benchmark_calibration_design", calibration_design_output)
        write_json(run_artifacts.run_dir / "benchmark_summary.json", payload)

    tools = _tool_map(config)
    _emit_progress(progress_callback, "benchmark_writer", "[1/3] Writing benchmark report...")
    report_content = _build_benchmark_report_markdown(
        query=query,
        payload=payload,
        solver_output=solver_output,
        load_output=load_output,
        scaling_output=scaling_output,
        design_output=design_output,
        property_design_output=property_design_output,
        repeatability_output=repeatability_output,
        identifiability_output=identifiability_output,
        fit_output=fit_output,
        simulation_smoke_output=simulation_smoke_output,
        calibration_design_output=calibration_design_output,
    )
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_benchmark_final_summary(payload=payload, saved_path=saved_path)

    metadata_path = None
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="benchmark",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "benchmark_solver": str(run_artifacts.run_dir / "benchmark_solver.md"),
                "benchmark_load_ladder": str(run_artifacts.run_dir / "benchmark_load_ladder.md"),
                "benchmark_scaling": str(run_artifacts.run_dir / "benchmark_scaling.md"),
                "benchmark_design": str(run_artifacts.run_dir / "benchmark_design.md"),
                "benchmark_property_design": str(run_artifacts.run_dir / "benchmark_property_design.md"),
                "benchmark_repeatability": str(run_artifacts.run_dir / "benchmark_repeatability.md"),
                "benchmark_identifiability": str(run_artifacts.run_dir / "benchmark_identifiability.md"),
                "benchmark_fit": str(run_artifacts.run_dir / "benchmark_fit.md"),
                "benchmark_simulation_smoke": str(run_artifacts.run_dir / "benchmark_simulation_smoke.md"),
                "benchmark_calibration_design": str(run_artifacts.run_dir / "benchmark_calibration_design.md"),
                "benchmark_summary": str(run_artifacts.run_dir / "benchmark_summary.json"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="benchmark",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        benchmark_output="\n\n".join([benchmark_output, simulation_smoke_output]),
        mechanics_output=fit_output,
        simulation_output="\n\n".join([solver_output, load_output, scaling_output, simulation_smoke_output]),
        design_output="\n\n".join([design_output, property_design_output, repeatability_output, identifiability_output, calibration_design_output]),
    )


async def run_dataset_workflow(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    _emit_progress(progress_callback, "dataset_search", "[0/3] Listing curated public mechanics datasets...")
    listing = list_public_dataset_specs(query=query)
    listing_output = _build_dataset_listing_ledger(query=query, payload=listing)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "dataset_search", listing_output)

    loose_payloads = auto_register_loose_archives(config.project_dir)
    loose_output = _build_loose_dataset_ledger(loose_payloads)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "dataset_register", loose_output)

    datasets = listing.get("datasets", [])
    if not datasets:
        final_summary = "Dataset workflow finished. No curated public datasets matched the query."
        if run_artifacts is not None:
            write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
            metadata_path = _write_run_metadata(
                run_artifacts=run_artifacts,
                config=config,
                query=query,
                workflow="datasets",
                report_name=report_name,
                report_path=None,
                final_summary=final_summary,
                stage_files={"dataset_search": str(run_artifacts.run_dir / "dataset_search.md"), "final_summary": str(run_artifacts.run_dir / "final_summary.md")},
            )
        else:
            metadata_path = None
        return ResearchRunResult(
            workflow="datasets",
            report_path=None,
            final_summary=final_summary,
            run_dir=run_artifacts.run_dir if run_artifacts else None,
            metadata_path=metadata_path,
            dataset_output="\n\n".join([listing_output, loose_output]),
        )

    acquired_payloads = []
    _emit_progress(progress_callback, "dataset_download", "[1/3] Downloading matching public datasets...")
    for dataset in datasets:
        payload = acquire_public_dataset(project_dir=config.project_dir, dataset_id=str(dataset["dataset_id"]))
        acquired_payloads.append(payload)

    acquisition_output = _build_dataset_acquisition_ledger(acquired_payloads)
    normalized_payloads = []
    for payload in acquired_payloads + loose_payloads:
        if payload.get("status") in {"manual_local_ingested", "manual_ingested", "downloaded"}:
            normalized_payloads.append(normalize_dataset_directory(project_dir=config.project_dir, dataset_id=str(payload["dataset_id"])))
    normalization_output = _build_dataset_normalization_ledger(normalized_payloads)
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "dataset_download", acquisition_output)
        write_stage_text(run_artifacts.run_dir, "dataset_normalize", normalization_output)
        write_json(run_artifacts.run_dir / "dataset_manifest_snapshot.json", {"datasets": acquired_payloads, "query": query})

    tools = _tool_map(config)
    _emit_progress(progress_callback, "dataset_writer", "[2/3] Writing dataset acquisition report...")
    report_content = _build_dataset_report_markdown(
        query=query,
        listing=listing,
        acquired_payloads=acquired_payloads,
        loose_payloads=loose_payloads,
        normalized_payloads=normalized_payloads,
    )
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_dataset_final_summary(acquired_payloads=acquired_payloads, saved_path=saved_path)

    metadata_path = None
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="datasets",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "dataset_search": str(run_artifacts.run_dir / "dataset_search.md"),
                "dataset_register": str(run_artifacts.run_dir / "dataset_register.md"),
                "dataset_download": str(run_artifacts.run_dir / "dataset_download.md"),
                "dataset_normalize": str(run_artifacts.run_dir / "dataset_normalize.md"),
                "dataset_manifest_snapshot": str(run_artifacts.run_dir / "dataset_manifest_snapshot.json"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="datasets",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        dataset_output="\n\n".join([listing_output, loose_output, acquisition_output, normalization_output]),
    )


async def run_calibration_workflow(
    *,
    config: AppConfig,
    query: str,
    report_name: str,
    dataset_id: str,
    calibration_max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    run_artifacts: Optional[RunArtifacts] = None,
) -> ResearchRunResult:
    _emit_progress(progress_callback, "calibration_extract", "[0/3] Extracting calibration targets from normalized experimental data...")
    payload = run_calibration_pipeline(
        project_dir=config.project_dir,
        dataset_id=dataset_id,
        max_samples=calibration_max_samples,
    )

    targets_output = _build_calibration_targets_ledger(payload["calibration_targets"])
    results_output = _build_calibration_results_ledger(payload["calibration_results"])
    impact_output = _build_calibration_impact_ledger(payload.get("calibration_impact_assessment", {}))
    summary_output = _build_calibration_summary_ledger(query=query, payload=payload)

    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "calibration_targets", targets_output)
        write_stage_text(run_artifacts.run_dir, "calibration_results", results_output)
        write_stage_text(run_artifacts.run_dir, "calibration_impact", impact_output)
        write_json(run_artifacts.run_dir / "calibration_targets.json", {"calibration_targets": payload["calibration_targets"]})
        write_json(run_artifacts.run_dir / "calibration_results.json", payload["calibration_results"])
        write_json(run_artifacts.run_dir / "calibration_impact.json", payload.get("calibration_impact_assessment", {}))

    tools = _tool_map(config)
    _emit_progress(progress_callback, "calibration_writer", "[1/3] Writing calibration report...")
    report_content = _build_calibration_report_markdown(query=query, dataset_id=dataset_id, payload=payload)
    saved_path = tools["save_report"](report_name, report_content)
    final_summary = _build_calibration_final_summary(payload=payload, saved_path=saved_path)

    metadata_path = None
    if run_artifacts is not None:
        write_stage_text(run_artifacts.run_dir, "final_summary", final_summary)
        metadata_path = _write_run_metadata(
            run_artifacts=run_artifacts,
            config=config,
            query=query,
            workflow="calibration",
            report_name=report_name,
            report_path=expected_report_path(config, report_name),
            final_summary=final_summary,
            stage_files={
                "calibration_targets": str(run_artifacts.run_dir / "calibration_targets.md"),
                "calibration_results": str(run_artifacts.run_dir / "calibration_results.md"),
                "calibration_impact": str(run_artifacts.run_dir / "calibration_impact.md"),
                "calibration_targets_json": str(run_artifacts.run_dir / "calibration_targets.json"),
                "calibration_results_json": str(run_artifacts.run_dir / "calibration_results.json"),
                "calibration_impact_json": str(run_artifacts.run_dir / "calibration_impact.json"),
                "final_summary": str(run_artifacts.run_dir / "final_summary.md"),
            },
        )

    _emit_progress(progress_callback, "done", final_summary)
    return ResearchRunResult(
        workflow="calibration",
        report_path=expected_report_path(config, report_name),
        final_summary=final_summary,
        run_dir=run_artifacts.run_dir if run_artifacts else None,
        metadata_path=metadata_path,
        calibration_output="\n\n".join([targets_output, results_output, impact_output, summary_output]),
    )


async def run_research_agent(
    *,
    project_dir: Path,
    query: str,
    report_name: str,
    workflow: str = "team",
    model: Optional[str] = None,
    library_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    max_pubmed_results: Optional[int] = None,
    data_path: Optional[Path] = None,
    experiment_type: str = "auto",
    time_column: str = "time",
    stress_column: str = "stress",
    strain_column: str = "strain",
    applied_stress: Optional[float] = None,
    applied_strain: Optional[float] = None,
    delimiter: str = ",",
    simulation_fiber_density: Optional[float] = None,
    simulation_fiber_stiffness: Optional[float] = None,
    simulation_bending_stiffness: Optional[float] = None,
    simulation_crosslink_prob: Optional[float] = None,
    simulation_domain_size: Optional[float] = None,
    target_stiffness: Optional[float] = None,
    simulation_scenario: str = "bulk_mechanics",
    simulation_request_json: str = "",
    cell_contractility: Optional[float] = None,
    organoid_radius: Optional[float] = None,
    matrix_youngs_modulus: Optional[float] = None,
    matrix_poisson_ratio: Optional[float] = None,
    target_anisotropy: float = 0.1,
    target_connectivity: float = 0.95,
    target_stress_propagation: float = 0.5,
    design_extra_targets_json: str = "",
    constraint_max_anisotropy: Optional[float] = None,
    constraint_min_connectivity: Optional[float] = None,
    constraint_max_risk_index: Optional[float] = None,
    constraint_min_stress_propagation: Optional[float] = None,
    design_extra_constraints_json: str = "",
    design_top_k: int = 3,
    design_candidate_budget: int = 12,
    design_monte_carlo_runs: int = 4,
    design_run_simulation: bool = False,
    design_simulation_scenario: str = "bulk_mechanics",
    design_simulation_top_k: int = 2,
    condition_concentration_fraction: Optional[float] = None,
    condition_curing_seconds: Optional[float] = None,
    condition_overrides_json: str = "",
    campaign_target_stiffnesses: str = "",
    dataset_id: Optional[str] = None,
    calibration_max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> ResearchRunResult:
    config = AppConfig.from_project_dir(project_dir).with_overrides(
        library_dir=library_dir,
        report_dir=report_dir,
        model=model,
        max_pubmed_results=max_pubmed_results,
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.library_dir.mkdir(parents=True, exist_ok=True)
    config.memory_dir.mkdir(parents=True, exist_ok=True)
    config.template_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    condition_overrides = _parse_optional_json_object(
        condition_overrides_json,
        flag_name="--condition-overrides-json",
    )
    design_extra_targets = _parse_optional_json_object(
        design_extra_targets_json,
        flag_name="--design-extra-targets-json",
    )
    design_extra_constraints = _parse_optional_json_object(
        design_extra_constraints_json,
        flag_name="--design-extra-constraints-json",
    )
    run_artifacts = create_run_artifacts(
        runs_dir=config.runs_dir,
        query=query,
        workflow=workflow.strip().lower(),
    )
    config = config.with_overrides(active_run_dir=run_artifacts.run_dir)

    workflow = workflow.strip().lower()
    if workflow == "single":
        return await run_single_agent(
            config=config,
            query=query,
            report_name=report_name,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "team":
        return await run_team_agent(
            config=config,
            query=query,
            report_name=report_name,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "mechanics":
        if data_path is None:
            raise RuntimeError("`mechanics` workflow requires --data-path.")
        return await run_mechanics_agent(
            config=config,
            query=query,
            report_name=report_name,
            data_path=data_path.resolve(),
            experiment_type=experiment_type,
            time_column=time_column,
            stress_column=stress_column,
            strain_column=strain_column,
            applied_stress=applied_stress,
            applied_strain=applied_strain,
            delimiter=delimiter,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "datasets":
        return await run_dataset_workflow(
            config=config,
            query=query,
            report_name=report_name,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "calibration":
        if not dataset_id:
            raise RuntimeError("`calibration` workflow requires --dataset-id.")
        return await run_calibration_workflow(
            config=config,
            query=query,
            report_name=report_name,
            dataset_id=dataset_id,
            calibration_max_samples=calibration_max_samples,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "benchmark":
        return await run_benchmark_workflow(
            config=config,
            query=query,
            report_name=report_name,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "simulation":
        return await run_simulation_workflow(
            config=config,
            query=query,
            report_name=report_name,
            simulation_scenario=simulation_scenario,
            simulation_request_json=simulation_request_json,
            target_stiffness=target_stiffness,
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "design":
        if target_stiffness is None:
            raise RuntimeError("`design` workflow requires --target-stiffness.")
        return await run_design_agent(
            config=config,
            query=query,
            report_name=report_name,
            target_stiffness=target_stiffness,
            target_anisotropy=target_anisotropy,
            target_connectivity=target_connectivity,
            target_stress_propagation=target_stress_propagation,
            design_extra_targets=design_extra_targets,
            constraint_max_anisotropy=constraint_max_anisotropy,
            constraint_min_connectivity=constraint_min_connectivity,
            constraint_max_risk_index=constraint_max_risk_index,
            constraint_min_stress_propagation=constraint_min_stress_propagation,
            design_extra_constraints=design_extra_constraints,
            design_top_k=design_top_k,
            design_candidate_budget=design_candidate_budget,
            design_monte_carlo_runs=design_monte_carlo_runs,
            design_run_simulation=design_run_simulation,
            design_simulation_scenario=design_simulation_scenario,
            design_simulation_top_k=design_simulation_top_k,
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
            simulation_request_json=simulation_request_json,
            condition_concentration_fraction=condition_concentration_fraction,
            condition_curing_seconds=condition_curing_seconds,
            condition_overrides=condition_overrides,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "design_campaign":
        target_stiffnesses = _parse_campaign_target_stiffnesses(campaign_target_stiffnesses)
        if not target_stiffnesses:
            raise RuntimeError("`design_campaign` workflow requires --campaign-target-stiffnesses.")
        return await run_design_campaign(
            config=config,
            query=query,
            report_name=report_name,
            target_stiffnesses=target_stiffnesses,
            target_anisotropy=target_anisotropy,
            target_connectivity=target_connectivity,
            target_stress_propagation=target_stress_propagation,
            design_extra_targets=design_extra_targets,
            constraint_max_anisotropy=constraint_max_anisotropy,
            constraint_min_connectivity=constraint_min_connectivity,
            constraint_max_risk_index=constraint_max_risk_index,
            constraint_min_stress_propagation=constraint_min_stress_propagation,
            design_extra_constraints=design_extra_constraints,
            design_top_k=design_top_k,
            design_candidate_budget=design_candidate_budget,
            design_monte_carlo_runs=design_monte_carlo_runs,
            condition_concentration_fraction=condition_concentration_fraction,
            condition_curing_seconds=condition_curing_seconds,
            condition_overrides=condition_overrides,
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    if workflow == "hybrid":
        if data_path is None:
            raise RuntimeError("`hybrid` workflow requires --data-path.")
        return await run_hybrid_agent(
            config=config,
            query=query,
            report_name=report_name,
            data_path=data_path.resolve(),
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
            progress_callback=progress_callback,
            run_artifacts=run_artifacts,
        )
    raise RuntimeError(f"Unsupported workflow: {workflow}")


def _result_text(task_result) -> str:
    texts: list[str] = []
    for message in task_result.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return texts[-1] if texts else ""


def expected_report_path(config: AppConfig, report_name: str) -> Path:
    return config.report_dir / sanitize_filename(report_name)


def run_research_agent_sync(
    *,
    project_dir: Path,
    query: str,
    report_name: str,
    workflow: str = "team",
    model: Optional[str] = None,
    library_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    max_pubmed_results: Optional[int] = None,
    data_path: Optional[Path] = None,
    experiment_type: str = "auto",
    time_column: str = "time",
    stress_column: str = "stress",
    strain_column: str = "strain",
    applied_stress: Optional[float] = None,
    applied_strain: Optional[float] = None,
    delimiter: str = ",",
    simulation_fiber_density: Optional[float] = None,
    simulation_fiber_stiffness: Optional[float] = None,
    simulation_bending_stiffness: Optional[float] = None,
    simulation_crosslink_prob: Optional[float] = None,
    simulation_domain_size: Optional[float] = None,
    target_stiffness: Optional[float] = None,
    simulation_scenario: str = "bulk_mechanics",
    simulation_request_json: str = "",
    cell_contractility: Optional[float] = None,
    organoid_radius: Optional[float] = None,
    matrix_youngs_modulus: Optional[float] = None,
    matrix_poisson_ratio: Optional[float] = None,
    target_anisotropy: float = 0.1,
    target_connectivity: float = 0.95,
    target_stress_propagation: float = 0.5,
    design_extra_targets_json: str = "",
    constraint_max_anisotropy: Optional[float] = None,
    constraint_min_connectivity: Optional[float] = None,
    constraint_max_risk_index: Optional[float] = None,
    constraint_min_stress_propagation: Optional[float] = None,
    design_extra_constraints_json: str = "",
    design_top_k: int = 3,
    design_candidate_budget: int = 12,
    design_monte_carlo_runs: int = 4,
    design_run_simulation: bool = False,
    design_simulation_scenario: str = "bulk_mechanics",
    design_simulation_top_k: int = 2,
    condition_concentration_fraction: Optional[float] = None,
    condition_curing_seconds: Optional[float] = None,
    condition_overrides_json: str = "",
    campaign_target_stiffnesses: str = "",
    dataset_id: Optional[str] = None,
    calibration_max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> ResearchRunResult:
    import asyncio

    return asyncio.run(
        run_research_agent(
            project_dir=project_dir,
            query=query,
            report_name=report_name,
            workflow=workflow,
            model=model,
            library_dir=library_dir,
            report_dir=report_dir,
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
            progress_callback=progress_callback,
        )
    )


def _emit_progress(
    progress_callback: Optional[Callable[[str, str], None]],
    stage: str,
    message: str,
) -> None:
    print(message, flush=True)
    if progress_callback is not None:
        progress_callback(stage, message)


def _write_run_metadata(
    *,
    run_artifacts: RunArtifacts,
    config: AppConfig,
    query: str,
    workflow: str,
    report_name: str,
    report_path: Optional[Path],
    final_summary: str,
    stage_files: Optional[dict[str, Optional[str]]] = None,
) -> Path:
    metadata = {
        "run_id": run_artifacts.run_id,
        "created_at": utc_now_iso(),
        "workflow": workflow,
        "query": query,
        "report_name": sanitize_filename(report_name),
        "report_path": str(report_path) if report_path else None,
        "model_provider": config.model_provider,
        "model": config.model,
        "library_dir": str(config.library_dir),
        "report_dir": str(config.report_dir),
        "cache_dir": str(config.cache_dir),
        "tool_log_path": str(run_artifacts.tool_log_path),
        "stage_files": stage_files or {"final_summary": str(run_artifacts.run_dir / "final_summary.md")},
        "final_summary": final_summary,
    }
    return write_json(run_artifacts.metadata_path, metadata)


def _critic_requests_revision(critic_output: str) -> bool:
    return "REVISION_NEEDED" in critic_output.upper()


def _needs_search_fallback(search_output: str) -> bool:
    stripped = search_output.strip()
    if not stripped:
        return True
    if "<｜DSML｜function_calls>" in stripped:
        return True
    return "## Search Coverage" not in stripped or "## Candidate Studies" not in stripped


def _needs_mechanics_fallback(mechanics_output: str) -> bool:
    stripped = mechanics_output.strip()
    if not stripped:
        return True
    if "<｜DSML｜function_calls>" in stripped:
        return True
    return "## Dataset Summary" not in stripped or "## Fitted Model" not in stripped


def _needs_simulation_fallback(simulation_output: str) -> bool:
    stripped = simulation_output.strip()
    if not stripped:
        return True
    if "<｜DSML｜function_calls>" in stripped:
        return True
    return "## Simulation Setup" not in stripped or "## Simulation Result" not in stripped


async def _build_fallback_search_ledger(
    *,
    config: AppConfig,
    query: str,
    max_pubmed_results: int,
    tools: dict[str, Callable[..., object]],
) -> str:
    pubmed_text = await tools["search_pubmed"](query, max(3, min(max_pubmed_results, 6)))
    crossref_text = await tools["search_crossref"](query, 3)
    library_text = tools["search_local_library"](query, 5)

    pubmed_records = _parse_json_list(pubmed_text)
    crossref_records = _parse_json_list(crossref_text)
    library_payload = _parse_json_object(library_text)
    candidate_lines: list[str] = []
    seen_keys: set[str] = set()

    for record in pubmed_records[:6]:
        pmid = _safe_str(record.get("pmid"))
        doi = _safe_str(record.get("doi"))
        key = pmid or doi or _safe_str(record.get("title"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidate_lines.append(
            f"- **{_safe_str(record.get('title')) or 'NR'}** | PMID: {pmid or 'NR'} | DOI: {doi or 'NR'} | "
            f"{_safe_str(record.get('journal')) or 'NR'}, {_safe_str(record.get('pubdate')) or 'NR'} | "
            f"Why it matters: {_short_why(record.get('abstract_excerpt'))}"
        )

    for record in crossref_records[:3]:
        title = _safe_str(record.get("title"))
        doi = _safe_str(record.get("doi"))
        key = doi or title
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        candidate_lines.append(
            f"- **{title or 'NR'}** | PMID: NR | DOI: {doi or 'NR'} | "
            f"{_safe_str(record.get('journal')) or 'NR'}, {_safe_str(record.get('published')) or 'NR'} | "
            f"Why it matters: {_short_why(record.get('abstract_excerpt'))}"
        )

    local_hits = library_payload.get("matches", []) if isinstance(library_payload, dict) else []
    local_lines = []
    for hit in local_hits[:5]:
        local_lines.append(
            f"- {hit.get('file_name', 'NR')} | score={hit.get('score', 'NR')} | "
            f"title={hit.get('title_guess', 'NR')}"
        )
    if not local_lines:
        local_lines = ["- No relevant local files found."]

    pattern_lines = _keyword_pattern_lines(pubmed_records)
    if not pattern_lines:
        pattern_lines = [
            "- Synthetic ECM papers were found, but recurring material/mechanics terms were sparse in the extracted abstracts."
        ]

    evidence_gaps = [
        "- Quantitative parameters such as stiffness, ligand density, and long-term passage stability are often underreported in abstracts.",
        "- Cross-study comparability remains limited when materials and assay readouts differ.",
    ]

    return "\n".join(
        [
            "## Search Coverage",
            f"- PubMed query: `{query}`",
            f"- Crossref query: `{query}`",
            f"- Local library query: `{query}`",
            "",
            "## Candidate Studies",
            *(candidate_lines or ["- No high-confidence candidate studies were extracted."]),
            "",
            "## ECM Patterns",
            *pattern_lines,
            "",
            "## Local Hits",
            *local_lines,
            "",
            "## Evidence Gaps",
            *evidence_gaps,
        ]
    )


def _parse_json_list(text: str) -> list[dict]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _parse_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _short_why(value: object) -> str:
    text = _safe_str(value)
    if not text:
        return "Relevant to synthetic ECM and intestinal organoid culture."
    sentence = text.split(".")[0].strip()
    return sentence[:220] + ("..." if len(sentence) > 220 else "")


def _keyword_pattern_lines(pubmed_records: list[dict]) -> list[str]:
    joined = " ".join(_safe_str(record.get("abstract_excerpt")).lower() for record in pubmed_records)
    patterns = []
    for keyword in ("peg", "alginate", "laminin", "stiffness", "yap", "mmp"):
        if keyword in joined:
            patterns.append(f"- Recurrent keyword in abstracts: `{keyword}`")
    return patterns


def _build_fallback_mechanics_ledger(
    *,
    tools: dict[str, Callable[..., object]],
    data_path: Path,
    experiment_type: str,
    time_column: str,
    stress_column: str,
    strain_column: str,
    applied_stress: Optional[float],
    applied_strain: Optional[float],
    delimiter: str,
) -> str:
    fit_text = tools["fit_mechanics_model"](
        str(data_path),
        experiment_type,
        time_column,
        stress_column,
        strain_column,
        applied_stress or 0.0,
        applied_strain or 0.0,
        delimiter,
    )
    payload = _parse_json_object(fit_text)
    fit = payload.get("fit", {}) if isinstance(payload, dict) else {}
    fit_type = _safe_str(payload.get("experiment_type")) or experiment_type
    selected_model = _safe_str(payload.get("selected_model")) or _safe_str(payload.get("model_type")) or fit_type
    params_lines = []
    quality_lines = []
    interpretation_lines = []
    limits = []
    scalar_fit_keys = [
        key
        for key, value in fit.items()
        if isinstance(value, (int, float)) and key not in {"mse", "r2", "n_points"}
    ]
    for key in scalar_fit_keys:
        params_lines.append(f"- {key}: {fit.get(key, 'NR')}")

    quality_lines.append(f"- Selected model: {selected_model}")
    quality_lines.append(f"- MSE: {fit.get('mse', 'NR')}")
    if "r2" in fit:
        quality_lines.append(f"- R²: {fit.get('r2', 'NR')}")
    if isinstance(payload.get("identifiability"), dict):
        quality_lines.append(f"- Identifiability risk: {payload['identifiability'].get('risk', 'NR')}")
    if isinstance(payload.get("candidate_models"), list):
        quality_lines.append(f"- Candidate models evaluated: {len(payload['candidate_models'])}")

    interpretation_lookup = {
        "linear_elastic": "- A linear elastic fit treats the ECM as time-independent near the measured strain window.",
        "power_law_elastic": "- A power-law elastic fit captures nonlinear strain-dependent stiffening or softening in the stress-strain data.",
        "kelvin_voigt_creep": "- Kelvin-Voigt captures saturating delayed strain under step stress.",
        "standard_linear_solid_creep": "- Standard Linear Solid creep adds both instantaneous and long-time elastic structure to delayed creep.",
        "burgers_creep": "- Burgers creep captures both retardation and long-time viscous flow components.",
        "maxwell_relaxation": "- Maxwell relaxation captures stress decay under step strain but not an equilibrium modulus plateau.",
        "standard_linear_solid_relaxation": "- Standard Linear Solid relaxation captures both instantaneous and equilibrium modulus behavior.",
        "generalized_maxwell_frequency_sweep": "- The frequency-sweep fit captures storage/loss modulus balance around a dominant relaxation timescale.",
        "cyclic_loop_metrics": "- The cyclic analysis summarizes hysteresis, loss factor, and residual strain rather than fitting a full constitutive law.",
    }
    limit_lookup = {
        "linear_elastic": "- This model cannot represent time-dependent creep or relaxation.",
        "power_law_elastic": "- This model is still quasi-static and does not represent explicit viscoelastic memory.",
        "kelvin_voigt_creep": "- Kelvin-Voigt cannot represent an instantaneous elastic jump.",
        "standard_linear_solid_creep": "- Standard Linear Solid is still a single-timescale viscoelastic approximation.",
        "burgers_creep": "- Burgers remains a low-order lumped model and may miss multi-timescale or poroelastic behavior.",
        "maxwell_relaxation": "- Maxwell cannot represent a non-zero long-time equilibrium modulus.",
        "standard_linear_solid_relaxation": "- Standard Linear Solid remains a single-branch approximation.",
        "generalized_maxwell_frequency_sweep": "- The current frequency-sweep backend is a single-mode generalized Maxwell approximation.",
        "cyclic_loop_metrics": "- The cyclic backend currently reports descriptive loop metrics instead of a fatigue damage law.",
    }
    interpretation_lines.append(interpretation_lookup.get(selected_model, "- Mechanical interpretation unavailable."))
    limits.append(limit_lookup.get(selected_model, "- The deterministic fitter returned a model family without a custom limit description."))

    return "\n".join(
        [
            "## Dataset Summary",
            f"- Data path: `{data_path}`",
            f"- Experiment type: `{fit_type}`",
            f"- Available columns: {payload.get('available_columns', 'NR')}",
            f"- Sample count: {payload.get('sample_count', 'NR')}",
            "",
            "## Fitted Model",
            f"- Backend deterministic fit selected: `{selected_model}`",
            "",
            "## Estimated Parameters",
            *(params_lines or ["- NR"]),
            "",
            "## Fit Quality",
            *(quality_lines or ["- NR"]),
            "",
            "## Mechanical Interpretation",
            *(interpretation_lines or ["- NR"]),
            "",
            "## Limits Of The Model",
            *(limits or ["- NR"]),
        ]
    )


def _default_simulation_parameters(
    *,
    evidence_output: str,
    mechanics_output: str,
    fiber_density: Optional[float],
    fiber_stiffness: Optional[float],
    bending_stiffness: Optional[float],
    crosslink_prob: Optional[float],
    domain_size: Optional[float],
) -> dict[str, float | int | str]:
    mechanics_lines = mechanics_output.lower()
    evidence_lines = evidence_output.lower()

    inferred_density = fiber_density if fiber_density is not None else (0.42 if "dense" in evidence_lines else 0.32)
    inferred_stiffness = fiber_stiffness if fiber_stiffness is not None else _extract_first_float(mechanics_output, fallback=10.0)
    inferred_bending = (
        bending_stiffness
        if bending_stiffness is not None
        else max(0.05 * inferred_stiffness, 0.1)
    )
    inferred_crosslink = crosslink_prob if crosslink_prob is not None else (0.5 if "laminin" in evidence_lines else 0.38)
    inferred_domain = domain_size if domain_size is not None else 1.0

    return {
        "fiber_density": float(min(max(inferred_density, 0.05), 1.0)),
        "fiber_stiffness": float(max(inferred_stiffness, 1e-3)),
        "bending_stiffness": float(max(inferred_bending, 0.0)),
        "crosslink_prob": float(min(max(inferred_crosslink, 0.01), 1.0)),
        "domain_size": float(max(inferred_domain, 0.1)),
        "total_force": 0.2,
        "axis": "x",
        "boundary_fraction": 0.15,
        "seed": 42,
        "max_iterations": 500,
        "tolerance": 1e-5,
        "monte_carlo_runs": 10,
        "scan_monte_carlo_runs": 4,
        "target_nodes": 8,
    }


def _build_fallback_simulation_ledger(
    *,
    tools: dict[str, Callable[..., object]],
    parameters: dict[str, float | int | str],
) -> str:
    baseline_text = tools["run_fiber_network_simulation"](
        float(parameters["fiber_density"]),
        float(parameters["fiber_stiffness"]),
        float(parameters["bending_stiffness"]),
        float(parameters["crosslink_prob"]),
        float(parameters["domain_size"]),
        float(parameters["total_force"]),
        str(parameters["axis"]),
        float(parameters["boundary_fraction"]),
        int(parameters["seed"]),
        int(parameters["max_iterations"]),
        float(parameters["tolerance"]),
        int(parameters["monte_carlo_runs"]),
        int(parameters["target_nodes"]),
    )
    baseline_payload = _parse_json_object(baseline_text)
    scan_text = tools["run_fiber_network_parameter_scan"](
        float(parameters["fiber_density"]),
        float(parameters["fiber_stiffness"]),
        float(parameters["bending_stiffness"]),
        float(parameters["crosslink_prob"]),
        float(parameters["domain_size"]),
        float(parameters["total_force"]),
        str(parameters["axis"]),
        float(parameters["boundary_fraction"]),
        int(parameters["seed"]),
        int(parameters["max_iterations"]),
        float(parameters["tolerance"]),
        int(parameters.get("scan_monte_carlo_runs", parameters["monte_carlo_runs"])),
        int(parameters["target_nodes"]),
    )
    scan_payload = _parse_json_object(scan_text)
    ranking = scan_payload.get("sensitivity_ranking", []) if isinstance(scan_payload, dict) else []
    ranking_lines = []
    for item in ranking[:4]:
        ranking_lines.append(
            f"- {item.get('parameter', 'NR')}: normalized stiffness span = {item.get('normalized_stiffness_span', 'NR')}"
        )
    if not ranking_lines:
        ranking_lines = ["- Sensitivity ranking unavailable."]

    return "\n".join(
        [
            "## Simulation Setup",
            f"- Parameters: {json.dumps(parameters, ensure_ascii=False)}",
            "",
            "## Baseline Simulation Result",
            f"- stiffness: {baseline_payload.get('stiffness', 'NR')}",
            f"- anisotropy: {baseline_payload.get('anisotropy', 'NR')}",
            f"- connectivity: {baseline_payload.get('connectivity', 'NR')}",
            f"- mean_displacement: {baseline_payload.get('mean_displacement', 'NR')}",
            "",
            "## Sensitivity Scan",
            *ranking_lines,
            "",
            "## Emergent Mechanical Interpretation",
            "- The bead-spring network translates local fiber and crosslink assumptions into emergent bulk stiffness and anisotropy.",
            "",
            "## Design Implications",
            "- Use stiffness and connectivity as screening metrics for candidate ECM architectures before wet-lab fabrication.",
            "",
            "## Model Caveats",
            "- This is a coarse 3D random fiber-network model, not a material-specific finite element calibration.",
        ]
    )


def _build_design_validation_ledger(validation_payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "## Physics Validation",
            f"- solver_converged: {validation_payload.get('solver_converged', 'NR')}",
            f"- monotonicity_valid: {validation_payload.get('monotonicity_valid', 'NR')}",
            f"- nonlinearity_valid: {validation_payload.get('nonlinearity_valid', 'NR')}",
            f"- physics_valid: {validation_payload.get('physics_valid', 'NR')}",
        ]
    )


def _build_design_stage_ledger(
    *,
    query: str,
    targets: dict[str, float],
    constraints: dict[str, float],
    payload: dict[str, object],
) -> str:
    top_candidates = payload.get("top_candidates", []) if isinstance(payload, dict) else []
    candidate_lines = []
    for candidate in top_candidates[:5]:
        params = candidate.get("parameters", {})
        features = candidate.get("features", {})
        violations = candidate.get("constraint_violations", {})
        candidate_lines.extend(
            [
                f"### Rank {candidate.get('rank', 'NR')}",
                f"- score: {candidate.get('score', 'NR')}",
                f"- feasible: {candidate.get('feasible', 'NR')}",
                f"- params: {json.dumps(params, ensure_ascii=False)}",
                f"- stiffness_mean: {features.get('stiffness_mean', 'NR')}",
                f"- anisotropy: {features.get('anisotropy', 'NR')}",
                f"- connectivity: {features.get('connectivity', 'NR')}",
                f"- stress_propagation: {features.get('stress_propagation', 'NR')}",
                f"- risk_index: {features.get('risk_index', 'NR')}",
                f"- constraint_violations: {json.dumps(violations, ensure_ascii=False)}",
            ]
        )
    if not candidate_lines:
        candidate_lines = ["- No candidate designs were returned."]

    return "\n".join(
        [
            "## Design Objective",
            f"- Query: {query}",
            f"- Targets: {json.dumps(targets, ensure_ascii=False)}",
            f"- Constraints: {json.dumps(constraints, ensure_ascii=False)}",
            "",
            "## Search Summary",
            f"- Candidate budget: {payload.get('candidate_budget', 'NR')}",
            f"- Evaluated candidates: {payload.get('evaluated_candidate_count', 'NR')}",
            f"- Feasible candidates: {payload.get('feasible_candidate_count', 'NR')}",
            f"- Top-k returned: {payload.get('top_k', 'NR')}",
            "",
            "## Top Candidates",
            *candidate_lines,
        ]
    )


def _build_design_sensitivity_ledger(scan_payload: dict[str, object]) -> str:
    ranking = scan_payload.get("sensitivity_ranking", []) if isinstance(scan_payload, dict) else []
    ranking_lines = []
    for row in ranking[:5]:
        ranking_lines.append(
            f"- {row.get('parameter', 'NR')}: normalized stiffness span = {row.get('normalized_stiffness_span', 'NR')}"
        )
    if not ranking_lines:
        ranking_lines = ["- Sensitivity ranking unavailable."]
    return "\n".join(["## Sensitivity Scan", *ranking_lines])


def _build_formulation_ledger(recommendations: list[dict[str, object]]) -> str:
    if not recommendations:
        return "## Formulation Mapping\n- No formulation recommendations available."
    lines = ["## Formulation Mapping"]
    for item in recommendations:
        recipe = item.get("primary_recipe", {})
        lines.extend(
            [
                f"### Candidate Rank {item.get('candidate_rank', 'NR')}",
                f"- material_family: {item.get('material_family', 'NR')}",
                f"- template_name: {item.get('template_name', 'NR')}",
                f"- crosslinking_strategy: {item.get('crosslinking_strategy', 'NR')}",
                f"- mapping_confidence: {item.get('mapping_confidence', 'NR')}",
                f"- primary_recipe: {json.dumps(recipe, ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def _build_campaign_formulation_ledger(recommendations: list[dict[str, object]]) -> str:
    if not recommendations:
        return "## Campaign Formulation Mapping\n- No formulation recommendations available."
    lines = ["## Campaign Formulation Mapping"]
    for item in recommendations:
        recipe = item.get("primary_recipe", {})
        lines.extend(
            [
                f"### Target stiffness {item.get('target_stiffness', 'NR')}",
                f"- material_family: {item.get('material_family', 'NR')}",
                f"- template_name: {item.get('template_name', 'NR')}",
                f"- feasible: {item.get('feasible', 'NR')}",
                f"- primary_recipe: {json.dumps(recipe, ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _assess_design_candidate(
    *,
    targets: dict[str, float],
    validation_payload: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    has_candidate = bool(candidate)
    feasible = bool(candidate.get("feasible", False)) if has_candidate else False
    physics_valid = bool(validation_payload.get("physics_valid", False))
    features = candidate.get("features", {}) if has_candidate and isinstance(candidate.get("features"), dict) else {}

    target_stiffness = _float_or_none(targets.get("stiffness"))
    stiffness_mean = _float_or_none(features.get("stiffness_mean"))
    target_anisotropy = _float_or_none(targets.get("anisotropy"))
    anisotropy = _float_or_none(features.get("anisotropy"))
    target_connectivity = _float_or_none(targets.get("connectivity"))
    connectivity = _float_or_none(features.get("connectivity"))
    target_stress_propagation = _float_or_none(targets.get("stress_propagation"))
    stress_propagation = _float_or_none(features.get("stress_propagation"))
    risk_index = _float_or_none(features.get("risk_index"))

    stiffness_rel_error = (
        abs(stiffness_mean - target_stiffness) / max(abs(target_stiffness), 1e-9)
        if stiffness_mean is not None and target_stiffness not in (None, 0.0)
        else None
    )
    anisotropy_abs_error = (
        abs(anisotropy - target_anisotropy)
        if anisotropy is not None and target_anisotropy is not None
        else None
    )
    connectivity_shortfall = (
        max(0.0, target_connectivity - connectivity)
        if connectivity is not None and target_connectivity is not None
        else None
    )
    stress_propagation_shortfall = (
        max(0.0, target_stress_propagation - stress_propagation)
        if stress_propagation is not None and target_stress_propagation is not None
        else None
    )

    failed_checks: list[str] = []
    notes: list[str] = []

    if not physics_valid:
        failed_checks.append("physics_validation")
        notes.append("Physics validation gate did not pass.")
    if not has_candidate:
        failed_checks.append("no_candidate")
        notes.append("No candidate design was returned.")
    if has_candidate and not feasible:
        failed_checks.append("constraint_feasibility")
        notes.append("Top candidate violates one or more hard constraints.")
    if stiffness_rel_error is not None and stiffness_rel_error > DESIGN_STIFFNESS_RELATIVE_TOLERANCE:
        failed_checks.append("stiffness_accuracy")
        notes.append(
            f"Stiffness relative error {stiffness_rel_error:.3f} exceeds {DESIGN_STIFFNESS_RELATIVE_TOLERANCE:.3f}."
        )
    if anisotropy_abs_error is not None and anisotropy_abs_error > DESIGN_ANISOTROPY_ABSOLUTE_TOLERANCE:
        failed_checks.append("anisotropy_accuracy")
        notes.append(
            f"Anisotropy absolute error {anisotropy_abs_error:.3f} exceeds {DESIGN_ANISOTROPY_ABSOLUTE_TOLERANCE:.3f}."
        )
    if connectivity_shortfall is not None and connectivity_shortfall > DESIGN_CONNECTIVITY_SHORTFALL_TOLERANCE:
        failed_checks.append("connectivity_shortfall")
        notes.append(
            f"Connectivity shortfall {connectivity_shortfall:.3f} exceeds {DESIGN_CONNECTIVITY_SHORTFALL_TOLERANCE:.3f}."
        )
    if (
        stress_propagation_shortfall is not None
        and stress_propagation_shortfall > DESIGN_STRESS_PROPAGATION_SHORTFALL_TOLERANCE
    ):
        failed_checks.append("stress_propagation_shortfall")
        notes.append(
            "Stress propagation shortfall "
            f"{stress_propagation_shortfall:.3f} exceeds {DESIGN_STRESS_PROPAGATION_SHORTFALL_TOLERANCE:.3f}."
        )
    if risk_index is not None and risk_index > DESIGN_RISK_INDEX_MAX:
        failed_checks.append("risk_index")
        notes.append(f"Risk index {risk_index:.3f} exceeds {DESIGN_RISK_INDEX_MAX:.3f}.")

    recommended_for_screening = not failed_checks
    if recommended_for_screening:
        status = "ready_for_screening"
    elif physics_valid and has_candidate and feasible:
        status = "caution"
    else:
        status = "not_ready"

    return {
        "status": status,
        "recommended_for_screening": recommended_for_screening,
        "has_candidate": has_candidate,
        "physics_valid": physics_valid,
        "feasible": feasible,
        "failed_checks": failed_checks,
        "notes": notes,
        "thresholds": {
            "stiffness_rel_error_max": DESIGN_STIFFNESS_RELATIVE_TOLERANCE,
            "anisotropy_abs_error_max": DESIGN_ANISOTROPY_ABSOLUTE_TOLERANCE,
            "connectivity_shortfall_max": DESIGN_CONNECTIVITY_SHORTFALL_TOLERANCE,
            "stress_propagation_shortfall_max": DESIGN_STRESS_PROPAGATION_SHORTFALL_TOLERANCE,
            "risk_index_max": DESIGN_RISK_INDEX_MAX,
        },
        "metrics": {
            "stiffness_mean": stiffness_mean,
            "target_stiffness": target_stiffness,
            "stiffness_rel_error": stiffness_rel_error,
            "anisotropy": anisotropy,
            "target_anisotropy": target_anisotropy,
            "anisotropy_abs_error": anisotropy_abs_error,
            "connectivity": connectivity,
            "target_connectivity": target_connectivity,
            "connectivity_shortfall": connectivity_shortfall,
            "stress_propagation": stress_propagation,
            "target_stress_propagation": target_stress_propagation,
            "stress_propagation_shortfall": stress_propagation_shortfall,
            "risk_index": risk_index,
        },
    }


def _design_simulation_not_run_payload(*, enabled: bool, scenario: str, reason: str) -> dict[str, object]:
    return {
        "enabled": enabled,
        "scenario": scenario,
        "status": "not_run" if not enabled else "unavailable",
        "reason": reason,
        "candidate_simulations": [],
        "comparison": {},
    }


def _candidate_simulation_request(
    *,
    candidate: dict[str, object],
    target_stiffness: float,
    scenario: str,
    simulation_request_json: str,
    cell_contractility: Optional[float],
    organoid_radius: Optional[float],
    matrix_youngs_modulus: Optional[float],
    matrix_poisson_ratio: Optional[float],
) -> dict[str, object]:
    return design_payload_to_simulation_requests(
        {"top_candidates": [candidate]},
        top_k=1,
        options=CandidateSimulationMappingOptions(
            scenario=scenario,
            target_stiffness=target_stiffness,
            simulation_request_overrides=_parse_json_object(simulation_request_json) if simulation_request_json.strip() else {},
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
        ),
    )[0]


def _run_design_candidate_simulations(
    *,
    config: AppConfig,
    design_payload: dict[str, object],
    target_stiffness: float,
    scenario: str,
    simulation_request_json: str,
    cell_contractility: Optional[float],
    organoid_radius: Optional[float],
    matrix_youngs_modulus: Optional[float],
    matrix_poisson_ratio: Optional[float],
    top_k: int,
) -> dict[str, object]:
    tools = _tool_map(config)
    top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
    if not top_candidates:
        return {
            "enabled": True,
            "scenario": scenario,
            "status": "not_run",
            "reason": "No top candidates were available for FEBio verification.",
            "candidate_simulations": [],
            "mapped_requests": [],
            "comparison": {},
        }
    mapped_requests = design_payload_to_simulation_requests(
        design_payload,
        top_k=max(1, int(top_k)),
        options=CandidateSimulationMappingOptions(
            scenario=scenario,
            target_stiffness=target_stiffness,
            simulation_request_overrides=_parse_json_object(simulation_request_json) if simulation_request_json.strip() else {},
            cell_contractility=cell_contractility,
            organoid_radius=organoid_radius,
            matrix_youngs_modulus=matrix_youngs_modulus,
            matrix_poisson_ratio=matrix_poisson_ratio,
        ),
    )
    candidate_payloads: list[dict[str, object]] = []
    for candidate, request in zip(top_candidates[: max(1, int(top_k))], mapped_requests):
        candidate_id = f"candidate_rank_{int(candidate.get('rank', 0)):02d}"
        run_text = tools["run_febio_simulation"](request.to_json(), candidate_id)
        run_payload = _parse_json_object(run_text)
        if not run_payload:
            run_payload = {
                "status": "failed",
                "request": request.to_dict(),
                "final_summary": run_text,
                "simulation_metrics": {},
                "simulation_result": {"warnings": [run_text]},
            }
        candidate_payloads.append(
            {
                "candidate_id": candidate_id,
                "candidate_rank": candidate.get("rank", "NR"),
                "design_candidate": candidate,
                **run_payload,
            }
        )
    comparison = compare_simulation_candidates_payloads(candidate_payloads)
    available = any(row.get("status") == "succeeded" for row in candidate_payloads)
    return {
        "enabled": True,
        "scenario": scenario,
        "status": "completed" if available else "unavailable",
        "reason": config.febio_status_message if not available and not config.febio_executable else "",
        "candidate_simulations": candidate_payloads,
        "mapped_requests": candidate_requests_summary(mapped_requests),
        "comparison": comparison,
    }


def _build_design_simulation_ledger(payload: dict[str, object]) -> str:
    lines = [
        "## FEBio Simulation Verification",
        f"- enabled: {payload.get('enabled', False)}",
        f"- scenario: {payload.get('scenario', 'NR')}",
        f"- status: {payload.get('status', 'NR')}",
    ]
    if payload.get("reason"):
        lines.append(f"- note: {payload.get('reason')}")
    mapped_requests = payload.get("mapped_requests", []) if isinstance(payload.get("mapped_requests"), list) else []
    if mapped_requests:
        lines.append(f"- mapped_request_count: {len(mapped_requests)}")
    ranking = payload.get("comparison", {}).get("ranking", []) if isinstance(payload.get("comparison"), dict) else []
    if ranking:
        lines.extend(
            [
                "",
                "| Rank | candidate_id | comparison_score | feasible | target_mismatch_score | peak_stress | stress_propagation_distance | suitability_score |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in ranking[:5]:
            lines.append(
                "| {rank} | {candidate_id} | {comparison_score:.4f} | {feasible} | {target_mismatch_score} | {peak_stress} | {stress_propagation_distance} | {candidate_suitability_score} |".format(
                    rank=int(row.get("rank", 0)),
                    candidate_id=row.get("candidate_id", "NR"),
                    comparison_score=float(row.get("comparison_score", 0.0)),
                    feasible="yes" if row.get("feasible") else "no",
                    target_mismatch_score=row.get("target_mismatch_score", "NR"),
                    peak_stress=row.get("peak_stress", "NR"),
                    stress_propagation_distance=row.get("stress_propagation_distance", "NR"),
                    candidate_suitability_score=row.get("candidate_suitability_score", "NR"),
                )
            )
    candidate_payloads = payload.get("candidate_simulations", [])
    if candidate_payloads:
        lines.append("")
        for item in candidate_payloads[:5]:
            metrics = item.get("simulation_metrics", {}) if isinstance(item.get("simulation_metrics"), dict) else {}
            lines.extend(
                [
                    f"### {item.get('candidate_id', 'NR')}",
                    f"- status: {item.get('status', 'NR')}",
                    f"- effective_stiffness: {metrics.get('effective_stiffness', 'NR')}",
                    f"- peak_stress: {metrics.get('peak_stress', 'NR')}",
                    f"- displacement_decay_length: {metrics.get('displacement_decay_length', 'NR')}",
                    f"- stress_propagation_distance: {metrics.get('stress_propagation_distance', 'NR')}",
                    f"- strain_heterogeneity: {metrics.get('strain_heterogeneity', 'NR')}",
                    f"- interface_deformation: {metrics.get('interface_deformation', 'NR')}",
                    f"- candidate_suitability_score: {metrics.get('candidate_suitability_score_components', {}).get('suitability_score', 'NR') if isinstance(metrics.get('candidate_suitability_score_components'), dict) else 'NR'}",
                    f"- target_mismatch_score: {metrics.get('target_mismatch_score', 'NR')}",
                ]
            )
    elif not payload.get("reason"):
        lines.append("- No FEBio verification payloads were generated.")
    return "\n".join(lines)


def _assess_design_campaign(
    *,
    validation_payload: dict[str, object],
    campaign_results: list[dict[str, object]],
) -> dict[str, object]:
    target_assessments = [
        row.get("design_assessment", {})
        for row in campaign_results
        if isinstance(row.get("design_assessment"), dict)
    ]
    ready_count = sum(1 for assessment in target_assessments if assessment.get("recommended_for_screening"))
    caution_count = sum(1 for assessment in target_assessments if assessment.get("status") == "caution")
    not_ready_count = sum(1 for assessment in target_assessments if assessment.get("status") == "not_ready")
    physics_valid = bool(validation_payload.get("physics_valid", False))

    if physics_valid and target_assessments and ready_count == len(target_assessments):
        status = "ready_for_screening"
    elif physics_valid and ready_count > 0:
        status = "partial"
    elif physics_valid and caution_count > 0:
        status = "caution"
    else:
        status = "not_ready"

    return {
        "status": status,
        "physics_valid": physics_valid,
        "target_count": len(campaign_results),
        "ready_target_count": ready_count,
        "caution_target_count": caution_count,
        "not_ready_target_count": not_ready_count,
        "recommended_for_screening": physics_valid and bool(target_assessments) and ready_count == len(target_assessments),
    }


def _build_design_summary_payload(
    *,
    query: str,
    targets: dict[str, float],
    constraints: dict[str, float],
    validation_payload: dict[str, object],
    design_assessment: dict[str, object],
    design_payload: dict[str, object],
    formulation_recommendations: list[dict[str, object]],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
    requested_condition_overrides: dict[str, object] | None,
    scan_payload: dict[str, object],
    sensitivity_output: str,
    design_simulation: dict[str, object] | None,
    report_name: str,
    saved_report_path: str,
) -> dict[str, object]:
    top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
    best_candidate = top_candidates[0] if top_candidates else {}
    best_params = best_candidate.get("parameters", {}) if isinstance(best_candidate, dict) else {}
    return {
        "workflow": "design",
        "query": query,
        "targets": targets,
        "constraints": constraints,
        "calibrated_search_space": calibrated_search_space,
        "calibration_context": calibration_context,
        "requested_condition_overrides": requested_condition_overrides or {},
        "validation_payload": validation_payload,
        "design_assessment": design_assessment,
        "design_payload": design_payload,
        "formulation_recommendations": formulation_recommendations,
        "sensitivity_payload": scan_payload,
        "sensitivity_summary": sensitivity_output,
        "design_simulation": design_simulation or {},
        "top_candidates": top_candidates,
        "best_candidate": best_candidate,
        "predicted_material_observables": predict_material_observables(best_params) if best_params else {},
        "report_name": report_name,
        "report_path": saved_report_path,
    }


def _build_design_report_markdown(
    *,
    query: str,
    targets: dict[str, float],
    constraints: dict[str, float],
    validation_payload: dict[str, object],
    design_assessment: dict[str, object],
    design_payload: dict[str, object],
    formulation_recommendations: list[dict[str, object]],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
    sensitivity_output: str,
    design_simulation: dict[str, object] | None,
) -> str:
    top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
    table_lines = [
        "| Rank | feasible | fiber_density | fiber_stiffness | bending_stiffness | crosslink_prob | domain_size | stiffness_mean | anisotropy | connectivity | stress_propagation | risk_index | score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in top_candidates[:5]:
        params = candidate.get("parameters", {})
        features = candidate.get("features", {})
        table_lines.append(
            "| {rank} | {feasible} | {fiber_density:.3f} | {fiber_stiffness:.3f} | {bending_stiffness:.3f} | {crosslink_prob:.3f} | {domain_size:.3f} | {stiffness_mean:.3f} | {anisotropy:.3f} | {connectivity:.3f} | {stress_propagation:.3f} | {risk_index:.3f} | {score:.3f} |".format(
                rank=int(candidate.get("rank", 0)),
                feasible="yes" if candidate.get("feasible") else "no",
                fiber_density=float(params.get("fiber_density", 0.0)),
                fiber_stiffness=float(params.get("fiber_stiffness", 0.0)),
                bending_stiffness=float(params.get("bending_stiffness", 0.0)),
                crosslink_prob=float(params.get("crosslink_prob", 0.0)),
                domain_size=float(params.get("domain_size", 0.0)),
                stiffness_mean=float(features.get("stiffness_mean", 0.0)),
                anisotropy=float(features.get("anisotropy", 0.0)),
                connectivity=float(features.get("connectivity", 0.0)),
                stress_propagation=float(features.get("stress_propagation", 0.0)),
                risk_index=float(features.get("risk_index", 0.0)),
                score=float(candidate.get("score", 0.0)),
            )
        )
    if len(table_lines) == 2:
        table_lines.append("| NR | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |")
    formulation_lines = []
    for item in formulation_recommendations[:3]:
        recipe = item.get("primary_recipe", {})
        formulation_lines.extend(
            [
                f"### Rank {item.get('candidate_rank', 'NR')} | {item.get('material_family', 'NR')}",
                f"- template: {item.get('template_name', 'NR')}",
                f"- crosslinking: {item.get('crosslinking_strategy', 'NR')}",
                f"- suggested recipe: {json.dumps(recipe, ensure_ascii=False)}",
                *[f"- rationale: {text}" for text in item.get("rationale", [])[:2]],
            ]
        )
    if not formulation_lines:
        formulation_lines = ["- No formulation translations available."]

    best_candidate = top_candidates[0] if top_candidates else {}
    best_features = best_candidate.get("features", {}) if isinstance(best_candidate, dict) else {}
    proxy_lines = []
    for key in (
        "mesh_size_proxy",
        "permeability_proxy",
        "compressibility_proxy",
        "swelling_ratio_proxy",
        "loss_tangent_proxy",
        "poroelastic_time_constant_proxy",
        "strain_stiffening_proxy",
    ):
        if key in best_features:
            proxy_lines.append(f"- {key}: {best_features.get(key, 'NR')}")
    if not proxy_lines:
        proxy_lines = ["- No material-property proxy outputs available."]

    simulation_lines = _build_design_simulation_ledger(design_simulation or {})
    comparison = (design_simulation or {}).get("comparison", {}) if isinstance(design_simulation, dict) else {}
    best_simulated = comparison.get("best_candidate", {}) if isinstance(comparison, dict) else {}
    recommendation_lines = [
        "- Use the top-ranked candidate from the mechanics-informed search as the primary fabrication target when FEBio verification is unavailable.",
        "- Use the next two candidates as contingency designs and validate them experimentally.",
    ]
    if best_simulated:
        recommendation_lines = [
            f"- FEBio-backed best candidate: {best_simulated.get('candidate_id', 'NR')} with comparison_score={best_simulated.get('comparison_score', 'NR')}.",
            "- Treat the FEBio ranking as an additional screening layer on top of the fiber-network search, not as a replacement for wet-lab validation.",
        ]
    simulation_metrics = best_simulated.get("simulation_metrics", {}) if isinstance(best_simulated, dict) and isinstance(best_simulated.get("simulation_metrics"), dict) else {}
    key_mechanical_metrics_lines = [
        f"- target_stiffness: {targets.get('stiffness', 'NR')}",
        f"- design_best_stiffness_mean: {best_features.get('stiffness_mean', 'NR') if best_features else 'NR'}",
        f"- design_best_anisotropy: {best_features.get('anisotropy', 'NR') if best_features else 'NR'}",
        f"- design_best_connectivity: {best_features.get('connectivity', 'NR') if best_features else 'NR'}",
        f"- design_best_stress_propagation: {best_features.get('stress_propagation', 'NR') if best_features else 'NR'}",
        f"- febio_effective_stiffness: {simulation_metrics.get('effective_stiffness', 'NR')}",
        f"- febio_peak_matrix_stress: {simulation_metrics.get('peak_matrix_stress', 'NR')}",
        f"- febio_stress_propagation_distance: {simulation_metrics.get('stress_propagation_distance', 'NR')}",
        f"- febio_strain_heterogeneity: {simulation_metrics.get('strain_heterogeneity', 'NR')}",
    ]
    uncertainty_lines = [
        "- fiber-network ranking and FEBio verification are two different models; disagreement between them should be treated as a review signal, not automatically as a failure.",
        "- The FEBio module remains template-driven and only covers fixed bulk, single-cell, and spheroid scenarios.",
        "- Candidate-to-simulation mapping is deterministic and explainable, but still heuristic when translating abstract design parameters into geometric FE parameters.",
    ]
    if not design_simulation or (isinstance(design_simulation, dict) and design_simulation.get("status") in {"not_run", "unavailable"}):
        uncertainty_lines.append("- FEBio was unavailable or not requested, so the recommendation is mechanics-informed but not FEBio-verified.")
    if design_simulation and isinstance(design_simulation, dict):
        for item in design_simulation.get("candidate_simulations", [])[:3]:
            warnings = item.get("simulation_result", {}).get("warnings", []) if isinstance(item.get("simulation_result"), dict) else []
            if warnings:
                uncertainty_lines.append(f"- {item.get('candidate_id', 'candidate')}: warnings={json.dumps(warnings, ensure_ascii=False)}")

    return "\n".join(
        [
            "# ECM Mechanics Design Report",
            "",
            "## 1. Design Goal",
            f"- Query: {query}",
            f"- Target mechanics: {json.dumps(targets, ensure_ascii=False)}",
            f"- Hard constraints: {json.dumps(constraints, ensure_ascii=False)}",
            f"- Calibrated search space: {json.dumps(calibrated_search_space or {}, ensure_ascii=False)}",
            f"- Calibration context: {json.dumps(calibration_context or {}, ensure_ascii=False)}",
            "",
            "## 2. Physics Validation Gate",
            f"- solver_converged: {validation_payload.get('solver_converged', 'NR')}",
            f"- monotonicity_valid: {validation_payload.get('monotonicity_valid', 'NR')}",
            f"- nonlinearity_valid: {validation_payload.get('nonlinearity_valid', 'NR')}",
            f"- physics_valid: {validation_payload.get('physics_valid', 'NR')}",
            "",
            "## 2.1 Screening Readiness",
            f"- status: {design_assessment.get('status', 'NR')}",
            f"- recommended_for_screening: {design_assessment.get('recommended_for_screening', 'NR')}",
            f"- failed_checks: {json.dumps(design_assessment.get('failed_checks', []), ensure_ascii=False)}",
            *[f"- note: {note}" for note in design_assessment.get("notes", [])],
            "",
            "## 3. Top ECM Candidates",
            *table_lines,
            "",
            "## 4. Sensitivity Of The Best Candidate",
            sensitivity_output,
            "",
            "## 5. Simulation Evidence",
            simulation_lines,
            "",
            "## 6. Key Mechanical Metrics",
            *key_mechanical_metrics_lines,
            "",
            "## 7. Recommendation Rationale",
            *recommendation_lines,
            "- These candidates are ranked by distance to target mechanics plus robustness penalties, and infeasible candidates are pushed behind feasible ones.",
            "- Treat FEBio evidence as an explicit second gate over the mechanics-informed design shortlist.",
            "",
            "## 8. Uncertainty and Failure Modes",
            *uncertainty_lines,
            "",
            "## 9. Material Property Proxies",
            *proxy_lines,
            "",
            "## 10. Formulation Translation",
            *formulation_lines,
        ]
    )


def _build_design_final_summary(
    *,
    validation_payload: dict[str, object],
    design_assessment: dict[str, object],
    design_payload: dict[str, object],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
    design_simulation: dict[str, object] | None,
    saved_path: str,
) -> str:
    top_candidates = design_payload.get("top_candidates", []) if isinstance(design_payload, dict) else []
    if top_candidates:
        best = top_candidates[0]
        params = best.get("parameters", {})
        features = best.get("features", {})
        best_summary = (
            f"Top candidate rank={best.get('rank', 'NR')}, score={best.get('score', 'NR')}, "
            f"feasible={best.get('feasible', 'NR')}, "
            f"stiffness_mean={features.get('stiffness_mean', 'NR')}, "
            f"fiber_density={params.get('fiber_density', 'NR')}, crosslink_prob={params.get('crosslink_prob', 'NR')}."
        )
    else:
        best_summary = "No candidate designs were returned."
    simulation_status = (design_simulation or {}).get("status", "not_run") if isinstance(design_simulation, dict) else "not_run"
    simulation_best = (
        (design_simulation or {}).get("comparison", {}).get("best_candidate", {})
        if isinstance(design_simulation, dict)
        else {}
    )
    simulation_summary = (
        f" FEBio_verification={simulation_status}, best_simulated_candidate={simulation_best.get('candidate_id', 'NR')}."
        if simulation_status != "not_run"
        else " FEBio_verification=not_run."
    )
    return (
        f"Design workflow finished. physics_valid={validation_payload.get('physics_valid', 'NR')}. "
        f"screening_status={design_assessment.get('status', 'NR')}, "
        f"recommended_for_screening={design_assessment.get('recommended_for_screening', 'NR')}. "
        f"{best_summary} calibrated_search_space={'yes' if calibrated_search_space else 'no'}, "
        f"calibration_context={json.dumps(calibration_context or {}, ensure_ascii=False)}."
        f"{simulation_summary} Report saved to {saved_path}."
    )


def _build_design_campaign_ledger(
    *,
    query: str,
    campaign_results: list[dict[str, object]],
    constraints: dict[str, float],
) -> str:
    lines = [
        "## Design Campaign Objective",
        f"- Query: {query}",
        f"- Constraints: {json.dumps(constraints, ensure_ascii=False)}",
        "",
        "## Campaign Targets",
    ]
    for row in campaign_results:
        best = row.get("best_candidate", {})
        features = best.get("features", {}) if isinstance(best, dict) else {}
        params = best.get("parameters", {}) if isinstance(best, dict) else {}
        lines.extend(
            [
                f"### Target stiffness {row.get('target_stiffness', 'NR')}",
                f"- best_score: {best.get('score', 'NR') if isinstance(best, dict) else 'NR'}",
                f"- feasible: {best.get('feasible', 'NR') if isinstance(best, dict) else 'NR'}",
                f"- calibration_context: {json.dumps(row.get('calibration_context', {}), ensure_ascii=False)}",
                f"- stiffness_mean: {features.get('stiffness_mean', 'NR')}",
                f"- anisotropy: {features.get('anisotropy', 'NR')}",
                f"- connectivity: {features.get('connectivity', 'NR')}",
                f"- risk_index: {features.get('risk_index', 'NR')}",
                f"- params: {json.dumps(params, ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def _build_design_campaign_summary_payload(
    *,
    query: str,
    constraints: dict[str, float],
    validation_payload: dict[str, object],
    campaign_assessment: dict[str, object],
    campaign_results: list[dict[str, object]],
    formulation_recommendations: list[dict[str, object]],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
    requested_condition_overrides: dict[str, object] | None,
    report_name: str,
    saved_report_path: str,
) -> dict[str, object]:
    enriched_results = []
    for row in campaign_results:
        best = row.get("best_candidate", {}) if isinstance(row.get("best_candidate"), dict) else {}
        params = best.get("parameters", {}) if isinstance(best.get("parameters"), dict) else {}
        enriched_results.append(
            {
                **row,
                "predicted_material_observables": predict_material_observables(params) if params else {},
            }
        )
    return {
        "workflow": "design_campaign",
        "query": query,
        "constraints": constraints,
        "calibrated_search_space": calibrated_search_space,
        "calibration_context": calibration_context,
        "requested_condition_overrides": requested_condition_overrides or {},
        "validation_payload": validation_payload,
        "campaign_assessment": campaign_assessment,
        "campaign_results": enriched_results,
        "formulation_recommendations": formulation_recommendations,
        "report_name": report_name,
        "report_path": saved_report_path,
    }


def _build_design_campaign_report_markdown(
    *,
    query: str,
    validation_payload: dict[str, object],
    campaign_assessment: dict[str, object],
    constraints: dict[str, float],
    campaign_results: list[dict[str, object]],
    formulation_recommendations: list[dict[str, object]],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
) -> str:
    table_lines = [
        "| target_stiffness | feasible | recommended | top_score | stiffness_mean | anisotropy | connectivity | stress_propagation | risk_index | fiber_density | fiber_stiffness | crosslink_prob |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in campaign_results:
        best = row.get("best_candidate", {})
        features = best.get("features", {}) if isinstance(best, dict) else {}
        params = best.get("parameters", {}) if isinstance(best, dict) else {}
        assessment = row.get("design_assessment", {}) if isinstance(row.get("design_assessment"), dict) else {}
        table_lines.append(
            "| {target_stiffness:.3f} | {feasible} | {recommended} | {score:.3f} | {stiffness_mean:.3f} | {anisotropy:.3f} | {connectivity:.3f} | {stress_propagation:.3f} | {risk_index:.3f} | {fiber_density:.3f} | {fiber_stiffness:.3f} | {crosslink_prob:.3f} |".format(
                target_stiffness=float(row.get("target_stiffness", 0.0)),
                feasible="yes" if isinstance(best, dict) and best.get("feasible") else "no",
                recommended="yes" if assessment.get("recommended_for_screening") else "no",
                score=float(best.get("score", 0.0)) if isinstance(best, dict) else 0.0,
                stiffness_mean=float(features.get("stiffness_mean", 0.0)),
                anisotropy=float(features.get("anisotropy", 0.0)),
                connectivity=float(features.get("connectivity", 0.0)),
                stress_propagation=float(features.get("stress_propagation", 0.0)),
                risk_index=float(features.get("risk_index", 0.0)),
                fiber_density=float(params.get("fiber_density", 0.0)),
                fiber_stiffness=float(params.get("fiber_stiffness", 0.0)),
                crosslink_prob=float(params.get("crosslink_prob", 0.0)),
            )
        )
    formulation_lines = []
    for item in formulation_recommendations:
        recipe = item.get("primary_recipe", {})
        formulation_lines.extend(
            [
                f"### Target stiffness {item.get('target_stiffness', 'NR')} | {item.get('material_family', 'NR')}",
                f"- template: {item.get('template_name', 'NR')}",
                f"- crosslinking: {item.get('crosslinking_strategy', 'NR')}",
                f"- suggested recipe: {json.dumps(recipe, ensure_ascii=False)}",
            ]
        )
    if not formulation_lines:
        formulation_lines = ["- No formulation translations available."]

    return "\n".join(
        [
            "# ECM Design Campaign Report",
            "",
            "## 1. Campaign Goal",
            f"- Query: {query}",
            f"- Hard constraints: {json.dumps(constraints, ensure_ascii=False)}",
            f"- Calibrated search space: {json.dumps(calibrated_search_space or {}, ensure_ascii=False)}",
            f"- Calibration context: {json.dumps(calibration_context or {}, ensure_ascii=False)}",
            "",
            "## 2. Physics Validation Gate",
            f"- solver_converged: {validation_payload.get('solver_converged', 'NR')}",
            f"- monotonicity_valid: {validation_payload.get('monotonicity_valid', 'NR')}",
            f"- nonlinearity_valid: {validation_payload.get('nonlinearity_valid', 'NR')}",
            f"- physics_valid: {validation_payload.get('physics_valid', 'NR')}",
            "",
            "## 2.1 Campaign Screening Readiness",
            f"- status: {campaign_assessment.get('status', 'NR')}",
            f"- recommended_for_screening: {campaign_assessment.get('recommended_for_screening', 'NR')}",
            f"- ready_target_count: {campaign_assessment.get('ready_target_count', 'NR')}",
            f"- caution_target_count: {campaign_assessment.get('caution_target_count', 'NR')}",
            f"- not_ready_target_count: {campaign_assessment.get('not_ready_target_count', 'NR')}",
            "",
            "## 3. Best Candidate Per Target Window",
            *table_lines,
            "",
            "## 3.1 Per-Target Calibration Context",
            *[
                f"- target {row.get('target_stiffness', 'NR')}: {json.dumps(row.get('calibration_context', {}), ensure_ascii=False)}"
                for row in campaign_results
            ],
            "",
            "## 4. Formulation Translation",
            *formulation_lines,
            "",
            "## 5. Interpretation",
            "- This campaign compares the best feasible candidate across multiple target stiffness windows under a shared constraint set.",
            "- Use it to decide whether a single ECM design family can span your desired mechanics regime or if separate formulations are needed.",
        ]
    )


def _build_design_campaign_final_summary(
    *,
    validation_payload: dict[str, object],
    campaign_assessment: dict[str, object],
    campaign_results: list[dict[str, object]],
    calibrated_search_space: dict[str, object] | None,
    calibration_context: dict[str, object] | None,
    saved_path: str,
) -> str:
    feasible_count = sum(
        1
        for row in campaign_results
        if isinstance(row.get("best_candidate"), dict) and row["best_candidate"].get("feasible")
    )
    return (
        f"Design campaign finished. physics_valid={validation_payload.get('physics_valid', 'NR')}. "
        f"Targets evaluated={len(campaign_results)}, feasible best-candidates={feasible_count}, "
        f"screening_status={campaign_assessment.get('status', 'NR')}, "
        f"recommended_targets={campaign_assessment.get('ready_target_count', 'NR')}/{campaign_assessment.get('target_count', 'NR')}, "
        f"calibrated_search_space={'yes' if calibrated_search_space else 'no'}, calibration_context={json.dumps(calibration_context or {}, ensure_ascii=False)}. "
        f"Report saved to {saved_path}."
    )


def _summarize_campaign_calibration_contexts(campaign_results: list[dict[str, object]]) -> dict[str, object]:
    contexts = [
        row.get("calibration_context", {})
        for row in campaign_results
        if isinstance(row.get("calibration_context"), dict) and row.get("calibration_context")
    ]
    if not contexts:
        return {}

    grouped_contexts: dict[str, list[dict[str, object]]] = {}
    for context in contexts:
        key = json.dumps(
            {
                "prior_level": context.get("prior_level"),
                "material_family": context.get("material_family"),
                "concentration_fraction": context.get("concentration_fraction"),
                "curing_seconds": context.get("curing_seconds"),
                "interpolation": context.get("interpolation"),
                "selection_mode": context.get("selection_reason", {}).get("mode")
                if isinstance(context.get("selection_reason"), dict)
                else None,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        grouped_contexts.setdefault(key, []).append(context)

    unique_keys = list(grouped_contexts.values())
    if len(unique_keys) == 1:
        representative = dict(unique_keys[0][0])
        if len(unique_keys[0]) > 1:
            target_values = [
                float(item["target_stiffness_mean"])
                for item in unique_keys[0]
                if item.get("target_stiffness_mean") is not None
            ]
            representative["context_count"] = len(unique_keys[0])
            if target_values:
                representative["target_stiffness_window"] = {
                    "min": min(target_values),
                    "max": max(target_values),
                }
        return representative

    return {
        "prior_level": "mixed",
        "context_count": len(unique_keys),
        "contexts": [group[0] for group in unique_keys],
    }


def _design_constraints_dict(
    *,
    constraint_max_anisotropy: Optional[float],
    constraint_min_connectivity: Optional[float],
    constraint_max_risk_index: Optional[float],
    constraint_min_stress_propagation: Optional[float],
    extra_constraints: dict[str, Any] | None = None,
) -> dict[str, float]:
    constraints = {
        key: value
        for key, value in {
            "max_anisotropy": float(constraint_max_anisotropy) if constraint_max_anisotropy is not None else None,
            "min_connectivity": float(constraint_min_connectivity) if constraint_min_connectivity is not None else None,
            "max_risk_index": float(constraint_max_risk_index) if constraint_max_risk_index is not None else None,
            "min_stress_propagation": float(constraint_min_stress_propagation) if constraint_min_stress_propagation is not None else None,
        }.items()
        if value is not None
    }
    if extra_constraints:
        constraints.update({str(key): float(value) for key, value in extra_constraints.items()})
    return constraints


def _design_search_space_from_priors(
    calibration_results: dict[str, object],
    *,
    inferred_family_hint: str,
    condition_concentration_fraction: float | None = None,
    condition_curing_seconds: float | None = None,
    condition_overrides: dict[str, Any] | None = None,
    target_stiffness: float | None = None,
    target_anisotropy: float | None = None,
    target_connectivity: float | None = None,
    target_stress_propagation: float | None = None,
) -> tuple[dict[str, dict[str, float]] | None, dict[str, object] | None]:
    hint = inferred_family_hint.lower()
    target_family = "GelMA" if "gelma" in hint else ("PEGDA" if "pegda" in hint else None)
    family_priors = calibration_results.get("family_priors", []) if isinstance(calibration_results, dict) else []
    if target_family is None and family_priors:
        target_family = str(family_priors[0].get("material_family", ""))
    if not target_family:
        return None, None

    inferred_concentration, inferred_curing = _infer_condition_hints_from_query(inferred_family_hint)
    concentration_fraction = (
        float(condition_concentration_fraction)
        if condition_concentration_fraction is not None
        else inferred_concentration
    )
    curing_seconds = (
        float(condition_curing_seconds)
        if condition_curing_seconds is not None
        else inferred_curing
    )
    return calibrated_search_space_from_calibration_results(
        calibration_results,
        material_family=target_family,
        concentration_fraction=concentration_fraction,
        curing_seconds=curing_seconds,
        condition_overrides=condition_overrides,
        target_stiffness=target_stiffness,
        target_anisotropy=target_anisotropy,
        target_connectivity=target_connectivity,
        target_stress_propagation=target_stress_propagation,
    )


def _parse_optional_json_object(raw_value: str, *, flag_name: str) -> dict[str, Any]:
    raw_value = raw_value.strip()
    if not raw_value:
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {flag_name} payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{flag_name} must decode to a JSON object.")
    return payload


def _infer_condition_hints_from_query(query: str) -> tuple[float | None, float | None]:
    import re

    concentration_fraction = None
    curing_seconds = None

    concentration_match = re.search(r"(\d+(?:\.\d+)?)\s*%", query)
    if concentration_match:
        numeric = float(concentration_match.group(1))
        concentration_fraction = numeric / 100.0 if numeric > 1.0 else numeric
    else:
        ratio_match = re.search(r"\b0\.\d{1,3}\b", query)
        if ratio_match:
            concentration_fraction = float(ratio_match.group(0))

    curing_match = re.search(r"(\d+(?:\.\d+)?)\s*s\b", query, re.IGNORECASE)
    if curing_match:
        curing_seconds = float(curing_match.group(1))

    return concentration_fraction, curing_seconds


def _parse_campaign_target_stiffnesses(raw_value: str) -> list[float]:
    if not raw_value.strip():
        return []
    values = []
    for piece in raw_value.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def _build_solver_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Solver Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('name', 'NR')}",
                f"- convergence_rate: {row.get('convergence_rate', 'NR')}",
                f"- max_residual: {row.get('max_residual', 'NR')}",
                f"- mean_residual: {row.get('mean_residual', 'NR')}",
                f"- mean_iterations: {row.get('mean_iterations', 'NR')}",
                f"- stiffness_mean: {row.get('stiffness_mean', 'NR')}",
                f"- stiffness_std: {row.get('stiffness_std', 'NR')}",
                f"- pass: {row.get('pass', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- case_count: {summary.get('case_count', 'NR')}",
            f"- pass_rate: {summary.get('pass_rate', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_load_ladder_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Load Ladder Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### total_force={row.get('total_force', 'NR')}",
                f"- convergence_rate: {row.get('convergence_rate', 'NR')}",
                f"- max_residual: {row.get('max_residual', 'NR')}",
                f"- mean_displacement: {row.get('mean_displacement', 'NR')}",
                f"- stiffness_mean: {row.get('stiffness_mean', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- force_count: {summary.get('force_count', 'NR')}",
            f"- displacement_monotonic: {summary.get('displacement_monotonic', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_scaling_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Scaling Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### target_nodes={row.get('target_nodes', 'NR')}",
                f"- convergence_rate: {row.get('convergence_rate', 'NR')}",
                f"- max_residual: {row.get('max_residual', 'NR')}",
                f"- mean_iterations: {row.get('mean_iterations', 'NR')}",
                f"- stiffness_mean: {row.get('stiffness_mean', 'NR')}",
                f"- pass: {row.get('pass', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- node_count_levels: {summary.get('node_count_levels', 'NR')}",
            f"- pass_count: {summary.get('pass_count', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_inverse_design_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Inverse Design Benchmark"]
    for row in rows:
        target = row.get("target", {})
        top = row.get("top_candidate", {})
        features = top.get("features", {}) if isinstance(top, dict) else {}
        lines.extend(
            [
                f"### target_stiffness={target.get('stiffness', 'NR')}",
                f"- abs_error: {row.get('abs_error', 'NR')}",
                f"- rel_error: {row.get('rel_error', 'NR')}",
                f"- feasible: {row.get('feasible', 'NR')}",
                f"- top_score: {top.get('score', 'NR') if isinstance(top, dict) else 'NR'}",
                f"- top_stiffness_mean: {features.get('stiffness_mean', 'NR')}",
                f"- pass: {row.get('pass', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- target_count: {summary.get('target_count', 'NR')}",
            f"- feasible_count: {summary.get('feasible_count', 'NR')}",
            f"- mean_abs_error: {summary.get('mean_abs_error', 'NR')}",
            f"- max_abs_error: {summary.get('max_abs_error', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_property_target_design_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Property-Target Design Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('name', 'NR')}",
                f"- targets: {json.dumps(row.get('targets', {}), ensure_ascii=False)}",
                f"- constraints: {json.dumps(row.get('constraints', {}), ensure_ascii=False)}",
                f"- mean_property_error: {row.get('mean_property_error', 'NR')}",
                f"- stiffness_rel_error: {row.get('stiffness_rel_error', 'NR')}",
                f"- property_errors: {json.dumps(row.get('property_errors', {}), ensure_ascii=False)}",
                f"- pass: {row.get('pass', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- case_count: {summary.get('case_count', 'NR')}",
            f"- pass_count: {summary.get('pass_count', 'NR')}",
            f"- mean_property_error: {summary.get('mean_property_error', 'NR')}",
            f"- max_property_error: {summary.get('max_property_error', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_repeatability_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Inverse Design Repeatability Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### seed={row.get('seed', 'NR')}",
                f"- feasible: {row.get('feasible', 'NR')}",
                f"- score: {row.get('score', 'NR')}",
                f"- stiffness_mean: {row.get('stiffness_mean', 'NR')}",
                f"- fiber_density: {row.get('fiber_density', 'NR')}",
                f"- fiber_stiffness: {row.get('fiber_stiffness', 'NR')}",
                f"- crosslink_prob: {row.get('crosslink_prob', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- repeat_count: {summary.get('repeat_count', 'NR')}",
            f"- feasible_count: {summary.get('feasible_count', 'NR')}",
            f"- stiffness_std: {summary.get('stiffness_std', 'NR')}",
            f"- score_std: {summary.get('score_std', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_identifiability_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Identifiability Proxy Benchmark"]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### equivalent_candidate_{idx}",
                f"- score: {row.get('score', 'NR')}",
                f"- feasible: {row.get('feasible', 'NR')}",
                f"- parameters: {json.dumps(row.get('parameters', {}), ensure_ascii=False)}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    recommendations = summary.get("recommended_measurements", []) if isinstance(summary, dict) else []
    lines.extend(
        [
            "",
            "## Summary",
            f"- equivalent_candidate_count: {summary.get('equivalent_candidate_count', 'NR')}",
            f"- parameter_spread: {json.dumps(summary.get('parameter_spread', {}), ensure_ascii=False)}",
            f"- max_parameter_spread: {summary.get('max_parameter_spread', 'NR')}",
            f"- dominant_degenerate_parameters: {json.dumps(summary.get('dominant_degenerate_parameters', []), ensure_ascii=False)}",
            f"- observable_spread: {json.dumps(summary.get('observable_spread', {}), ensure_ascii=False)}",
            f"- identifiability_risk: {summary.get('identifiability_risk', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    if recommendations:
        lines.extend(["", "## Recommended Additional Measurements"])
        for item in recommendations:
            lines.extend(
                [
                    f"### {item.get('measurement', 'NR')}",
                    f"- observable: {item.get('observable', 'NR')}",
                    f"- spread: {item.get('spread', 'NR')}",
                    f"- experiment: {item.get('experiment', 'NR')}",
                    f"- why: {item.get('why', 'NR')}",
                ]
            )
    return "\n".join(lines)


def _build_fit_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Mechanics Fit Benchmark"]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('name', 'NR')}",
                f"- true_parameters: {json.dumps(row.get('true_parameters', {}), ensure_ascii=False)}",
                f"- fitted_parameters: {json.dumps(row.get('fitted_parameters', {}), ensure_ascii=False)}",
                f"- relative_errors: {json.dumps(row.get('relative_errors', {}), ensure_ascii=False)}",
                f"- pass: {row.get('pass', 'NR')}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- case_count: {summary.get('case_count', 'NR')}",
            f"- mean_relative_error: {summary.get('mean_relative_error', 'NR')}",
            f"- max_relative_error: {summary.get('max_relative_error', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_simulation_smoke_benchmark_ledger(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return "\n".join(
        [
            "## FEBio Simulation Smoke Benchmark",
            f"- available: {summary.get('available', 'NR')}",
            f"- status: {summary.get('status', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
            f"- solver_converged: {summary.get('solver_converged', 'NR')}",
            f"- effective_stiffness: {summary.get('effective_stiffness', 'NR')}",
            f"- target_mismatch_score: {summary.get('target_mismatch_score', 'NR')}",
            f"- peak_stress: {summary.get('peak_stress', 'NR')}",
            f"- reason: {summary.get('reason', 'NR')}",
        ]
    )


def _build_calibration_design_benchmark_ledger(payload: dict[str, object]) -> str:
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    lines = ["## Calibration-Aware Design Benchmark"]
    for row in rows[:20]:
        calibration_context = row.get("calibration_context", {})
        selection_reason = calibration_context.get("selection_reason", {}) if isinstance(calibration_context, dict) else {}
        lines.extend(
            [
                f"### {row.get('sample_key', 'NR')}",
                f"- material_family: {row.get('material_family', 'NR')}",
                f"- target_stiffness: {row.get('target_stiffness', 'NR')}",
                f"- default_abs_error: {row.get('default_abs_error', 'NR')}",
                f"- calibrated_abs_error: {row.get('calibrated_abs_error', 'NR')}",
                f"- default_combined_error: {row.get('default_combined_error', 'NR')}",
                f"- calibrated_combined_error: {row.get('calibrated_combined_error', 'NR')}",
                f"- improved: {row.get('improved', 'NR')}",
                f"- calibration_context: {json.dumps(calibration_context, ensure_ascii=False)}",
                f"- selection_reason: {json.dumps(selection_reason, ensure_ascii=False)}",
                f"- calibrated_search_space: {json.dumps(row.get('calibrated_search_space', {}), ensure_ascii=False)}",
            ]
        )
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- available: {summary.get('available', 'NR')}",
            f"- dataset_id: {summary.get('dataset_id', 'NR')}",
            f"- cached_from_run: {summary.get('cached_from_run', 'NR')}",
            f"- target_count: {summary.get('target_count', 'NR')}",
            f"- default_mean_abs_error: {summary.get('default_mean_abs_error', 'NR')}",
            f"- calibrated_mean_abs_error: {summary.get('calibrated_mean_abs_error', 'NR')}",
            f"- mean_abs_error_improvement: {summary.get('mean_abs_error_improvement', 'NR')}",
            f"- default_mean_combined_error: {summary.get('default_mean_combined_error', 'NR')}",
            f"- calibrated_mean_combined_error: {summary.get('calibrated_mean_combined_error', 'NR')}",
            f"- mean_combined_error_improvement: {summary.get('mean_combined_error_improvement', 'NR')}",
            f"- improved_count: {summary.get('improved_count', 'NR')}",
            f"- evaluation_mode: {summary.get('evaluation_mode', 'NR')}",
            f"- routing_modes: {json.dumps(summary.get('routing_modes', []), ensure_ascii=False)}",
            f"- routing_mode_counts: {json.dumps(summary.get('routing_mode_counts', {}), ensure_ascii=False)}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_benchmark_summary_ledger(*, query: str, payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return "\n".join(
        [
            "## Mechanics Benchmark Summary",
            f"- Query: {query}",
            f"- solver_pass_rate: {summary.get('solver_pass_rate', 'NR')}",
            f"- load_ladder_monotonic: {summary.get('load_ladder_monotonic', 'NR')}",
            f"- scaling_pass_count: {summary.get('scaling_pass_count', 'NR')}",
            f"- inverse_design_mean_abs_error: {summary.get('inverse_design_mean_abs_error', 'NR')}",
            f"- property_target_mean_error: {summary.get('property_target_mean_error', 'NR')}",
            f"- repeatability_stiffness_std: {summary.get('repeatability_stiffness_std', 'NR')}",
            f"- identifiability_risk: {summary.get('identifiability_risk', 'NR')}",
            f"- fit_mean_relative_error: {summary.get('fit_mean_relative_error', 'NR')}",
            f"- simulation_smoke_available: {summary.get('simulation_smoke_available', 'NR')}",
            f"- simulation_smoke_status: {summary.get('simulation_smoke_status', 'NR')}",
            f"- simulation_smoke_pass: {summary.get('simulation_smoke_pass', 'NR')}",
            f"- calibration_benchmark_available: {summary.get('calibration_benchmark_available', 'NR')}",
            f"- calibration_design_improvement: {summary.get('calibration_design_improvement', 'NR')}",
            f"- calibration_cached_from_run: {summary.get('calibration_cached_from_run', 'NR')}",
            f"- calibration_routing_modes: {json.dumps(summary.get('calibration_routing_modes', []), ensure_ascii=False)}",
            f"- calibration_routing_mode_counts: {json.dumps(summary.get('calibration_routing_mode_counts', {}), ensure_ascii=False)}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )


def _build_benchmark_report_markdown(
    *,
    query: str,
    payload: dict[str, object],
    solver_output: str,
    load_output: str,
    scaling_output: str,
    design_output: str,
    property_design_output: str,
    repeatability_output: str,
    identifiability_output: str,
    fit_output: str,
    simulation_smoke_output: str,
    calibration_design_output: str,
) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return "\n".join(
        [
            "# ECM Mechanics Benchmark Report",
            "",
            "## 1. Benchmark Goal",
            f"- Query: {query}",
            "- This report evaluates solver convergence, inverse-design recovery, and synthetic mechanics fitting.",
            "",
            "## 2. Overall Summary",
            f"- solver_pass_rate: {summary.get('solver_pass_rate', 'NR')}",
            f"- load_ladder_monotonic: {summary.get('load_ladder_monotonic', 'NR')}",
            f"- scaling_pass_count: {summary.get('scaling_pass_count', 'NR')}",
            f"- inverse_design_mean_abs_error: {summary.get('inverse_design_mean_abs_error', 'NR')}",
            f"- property_target_mean_error: {summary.get('property_target_mean_error', 'NR')}",
            f"- repeatability_stiffness_std: {summary.get('repeatability_stiffness_std', 'NR')}",
            f"- identifiability_risk: {summary.get('identifiability_risk', 'NR')}",
            f"- fit_mean_relative_error: {summary.get('fit_mean_relative_error', 'NR')}",
            f"- simulation_smoke_available: {summary.get('simulation_smoke_available', 'NR')}",
            f"- simulation_smoke_status: {summary.get('simulation_smoke_status', 'NR')}",
            f"- simulation_smoke_pass: {summary.get('simulation_smoke_pass', 'NR')}",
            f"- simulation_smoke_effective_stiffness: {summary.get('simulation_smoke_effective_stiffness', 'NR')}",
            f"- simulation_smoke_target_mismatch_score: {summary.get('simulation_smoke_target_mismatch_score', 'NR')}",
            f"- calibration_benchmark_available: {summary.get('calibration_benchmark_available', 'NR')}",
            f"- calibration_design_improvement: {summary.get('calibration_design_improvement', 'NR')}",
            f"- calibration_cached_from_run: {summary.get('calibration_cached_from_run', 'NR')}",
            f"- calibration_routing_modes: {json.dumps(summary.get('calibration_routing_modes', []), ensure_ascii=False)}",
            f"- calibration_routing_mode_counts: {json.dumps(summary.get('calibration_routing_mode_counts', {}), ensure_ascii=False)}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
            "",
            "## 3. Solver Benchmark",
            solver_output,
            "",
            "## 4. Load Ladder Benchmark",
            load_output,
            "",
            "## 5. Scaling Benchmark",
            scaling_output,
            "",
            "## 6. Inverse Design Benchmark",
            design_output,
            "",
            "## 7. Property-Target Design Benchmark",
            property_design_output,
            "",
            "## 8. Inverse Design Repeatability",
            repeatability_output,
            "",
            "## 9. Identifiability Proxy",
            identifiability_output,
            "",
            "## 10. Mechanics Fit Benchmark",
            fit_output,
            "",
            "## 11. FEBio Simulation Smoke Benchmark",
            simulation_smoke_output,
            "",
            "## 12. Calibration-Aware Design Benchmark",
            calibration_design_output,
            "",
            "## 13. Interpretation",
            "- Use solver residuals and convergence rate to judge numerical stability.",
            "- Use load-ladder and scaling results to judge whether the solver remains trustworthy as load and network size increase.",
            "- Use inverse-design recovery error to judge whether target stiffness windows are realistically reachable.",
            "- Use property-target design results to judge whether the backend can match material-property proxies, not only bulk stiffness.",
            "- Use repeatability and identifiability proxy outputs to judge whether the inverse problem is stable or degenerate.",
            "- Use synthetic fit recovery to judge whether low-dimensional constitutive fitting is numerically reliable.",
            "- Use the FEBio smoke result to verify that the template-driven FEBio integration layer still runs end-to-end in the current environment.",
            "- Use calibration-aware design benchmarking to judge whether experimental priors actually improve mechanics matching.",
        ]
    )


def _build_benchmark_final_summary(*, payload: dict[str, object], saved_path: str) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return (
        f"Benchmark workflow finished. overall_pass={summary.get('overall_pass', 'NR')}, "
        f"solver_pass_rate={summary.get('solver_pass_rate', 'NR')}, "
        f"load_ladder_monotonic={summary.get('load_ladder_monotonic', 'NR')}, "
        f"scaling_pass_count={summary.get('scaling_pass_count', 'NR')}, "
        f"inverse_design_mean_abs_error={summary.get('inverse_design_mean_abs_error', 'NR')}, "
        f"property_target_mean_error={summary.get('property_target_mean_error', 'NR')}, "
        f"repeatability_stiffness_std={summary.get('repeatability_stiffness_std', 'NR')}, "
        f"identifiability_risk={summary.get('identifiability_risk', 'NR')}, "
        f"fit_mean_relative_error={summary.get('fit_mean_relative_error', 'NR')}, "
        f"simulation_smoke_status={summary.get('simulation_smoke_status', 'NR')}. "
        f"calibration_design_improvement={summary.get('calibration_design_improvement', 'NR')}. "
        f"Report saved to {saved_path}."
    )


def _build_dataset_listing_ledger(*, query: str, payload: dict[str, object]) -> str:
    datasets = payload.get("datasets", []) if isinstance(payload, dict) else []
    lines = [
        "## Curated Public Dataset Search",
        f"- Query: {query}",
        f"- Matches: {payload.get('count', 0) if isinstance(payload, dict) else 0}",
    ]
    for dataset in datasets:
        lines.extend(
            [
                f"### {dataset.get('dataset_id', 'NR')}",
                f"- title: {dataset.get('title', 'NR')}",
                f"- source: {dataset.get('source', 'NR')}",
                f"- landing_page: {dataset.get('landing_page', 'NR')}",
                f"- tags: {json.dumps(dataset.get('tags', []), ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def _build_dataset_acquisition_ledger(acquired_payloads: list[dict[str, object]]) -> str:
    if not acquired_payloads:
        return "## Dataset Acquisition\n- No datasets were downloaded."
    lines = ["## Dataset Acquisition"]
    for payload in acquired_payloads:
        lines.extend(
            [
                f"### {payload.get('dataset_id', 'NR')}",
                f"- title: {payload.get('title', 'NR')}",
                f"- source: {payload.get('source', 'NR')}",
                f"- status: {payload.get('status', 'NR')}",
                f"- archive_path: {payload.get('archive_path', 'NR')}",
                f"- extracted_dir: {payload.get('extracted_dir', 'NR')}",
                f"- file_count: {payload.get('file_count', 'NR')}",
            ]
        )
        if payload.get("manual_download_instructions"):
            lines.append(f"- manual_download_instructions: {payload.get('manual_download_instructions')}")
    return "\n".join(lines)


def _build_loose_dataset_ledger(loose_payloads: list[dict[str, object]]) -> str:
    if not loose_payloads:
        return "## Local Dataset Registration\n- No loose manually downloaded archives were found."
    lines = ["## Local Dataset Registration"]
    for payload in loose_payloads:
        lines.extend(
            [
                f"### {payload.get('dataset_id', 'NR')}",
                f"- title: {payload.get('title', 'NR')}",
                f"- archive_path: {payload.get('archive_path', 'NR')}",
                f"- extracted_dir: {payload.get('extracted_dir', 'NR')}",
                f"- file_count: {payload.get('file_count', 'NR')}",
            ]
        )
    return "\n".join(lines)


def _build_dataset_normalization_ledger(normalized_payloads: list[dict[str, object]]) -> str:
    if not normalized_payloads:
        return "## Dataset Normalization\n- No datasets were normalized."
    lines = ["## Dataset Normalization"]
    for payload in normalized_payloads:
        lines.extend(
            [
                f"### {payload.get('dataset_id', 'NR')}",
                f"- normalized_dir: {payload.get('normalized_dir', 'NR')}",
                f"- file_count: {payload.get('file_count', 'NR')}",
            ]
        )
        for row in payload.get("files", [])[:6]:
            lines.append(
                "- {file_name} | material={material_guess} | measurement={measurement_guess} | condition={condition}".format(
                    file_name=row.get("file_name", "NR"),
                    material_guess=row.get("material_guess", "NR"),
                    measurement_guess=row.get("measurement_guess", "NR"),
                    condition=json.dumps(row.get("condition_guess", {}), ensure_ascii=False),
                )
            )
    return "\n".join(lines)


def _build_dataset_report_markdown(
    *,
    query: str,
    listing: dict[str, object],
    acquired_payloads: list[dict[str, object]],
    loose_payloads: list[dict[str, object]],
    normalized_payloads: list[dict[str, object]],
) -> str:
    table_lines = [
        "| dataset_id | source | file_count | extracted_dir | landing_page |",
        "| --- | --- | --- | --- | --- |",
    ]
    for payload in acquired_payloads:
        table_lines.append(
            "| {dataset_id} | {source} | {file_count} | {extracted_dir} | {landing_page} |".format(
                dataset_id=payload.get("dataset_id", "NR"),
                source=payload.get("source", "NR"),
                file_count=payload.get("file_count", "NR"),
                extracted_dir=payload.get("extracted_dir", "NR"),
                landing_page=payload.get("landing_page", "NR"),
            )
        )
    for payload in loose_payloads:
        table_lines.append(
            "| {dataset_id} | {source} | {file_count} | {extracted_dir} | {landing_page} |".format(
                dataset_id=payload.get("dataset_id", "NR"),
                source=payload.get("source", "NR"),
                file_count=payload.get("file_count", "NR"),
                extracted_dir=payload.get("extracted_dir", "NR"),
                landing_page=payload.get("landing_page", "NR"),
            )
        )
    if len(table_lines) == 2:
        table_lines.append("| NR | NR | NR | NR | NR |")
    normalized_lines = []
    for payload in normalized_payloads:
        normalized_lines.append(f"### {payload.get('dataset_id', 'NR')}")
        for row in payload.get("files", [])[:6]:
            normalized_lines.append(
                "- {file_name} | material={material_guess} | measurement={measurement_guess} | condition={condition}".format(
                    file_name=row.get("file_name", "NR"),
                    material_guess=row.get("material_guess", "NR"),
                    measurement_guess=row.get("measurement_guess", "NR"),
                    condition=json.dumps(row.get("condition_guess", {}), ensure_ascii=False),
                )
            )
    if not normalized_lines:
        normalized_lines = ["- No normalized dataset summaries available."]
    return "\n".join(
        [
            "# Public ECM Mechanics Dataset Report",
            "",
            "## 1. Query",
            f"- {query}",
            "",
            "## 2. Curated Dataset Matches",
            f"- Count: {listing.get('count', 0) if isinstance(listing, dict) else 0}",
            "",
            "## 3. Downloaded Datasets",
            *table_lines,
            "",
            "## 4. Normalized File Overview",
            *normalized_lines,
            "",
            "## 5. Next Use",
            "- Use these datasets as priors, calibration templates, and benchmark inputs before integrating your own experimental measurements.",
            "- Standardize raw columns into a shared schema before feeding them into mechanics calibration workflows.",
        ]
    )


def _build_dataset_final_summary(*, acquired_payloads: list[dict[str, object]], saved_path: str) -> str:
    return (
        f"Dataset workflow finished. downloaded={len(acquired_payloads)} curated public dataset(s). "
        f"Report saved to {saved_path}."
    )


def _build_calibration_targets_ledger(calibration_targets: list[dict[str, object]]) -> str:
    if not calibration_targets:
        return "## Calibration Targets\n- No calibration targets extracted."
    lines = ["## Calibration Targets"]
    for target in calibration_targets[:20]:
        lines.extend(
            [
                f"### {target.get('sample_key', 'NR')}",
                f"- material_family: {target.get('material_family', 'NR')}",
                f"- target_stiffness: {target.get('target_stiffness', 'NR')}",
                f"- target_metric: {target.get('target_stiffness_metric', 'NR')}",
                f"- concentration_fraction: {target.get('concentration_fraction', 'NR')}",
                f"- curing_seconds: {target.get('curing_seconds', 'NR')}",
            ]
        )
    return "\n".join(lines)


def _build_calibration_results_ledger(calibration_results: dict[str, object]) -> str:
    rows = calibration_results.get("cases", []) if isinstance(calibration_results, dict) else []
    lines = ["## Calibration Results"]
    for row in rows[:20]:
        best = row.get("best_candidate", {})
        params = best.get("parameters", {}) if isinstance(best, dict) else {}
        lines.extend(
            [
                f"### {row.get('sample_key', 'NR')}",
                f"- material_family: {row.get('material_family', 'NR')}",
                f"- target_stiffness: {row.get('target_stiffness', 'NR')}",
                f"- abs_error: {row.get('abs_error', 'NR')}",
                f"- rel_error: {row.get('rel_error', 'NR')}",
                f"- auxiliary_errors: {json.dumps(row.get('auxiliary_errors', {}), ensure_ascii=False)}",
                f"- calibrated_params: {json.dumps(params, ensure_ascii=False)}",
            ]
        )
    lines.extend(["", "## Family Priors"])
    for prior in calibration_results.get("family_priors", []) if isinstance(calibration_results, dict) else []:
        lines.extend(
            [
                f"### {prior.get('material_family', 'NR')}",
                f"- sample_count: {prior.get('sample_count', 'NR')}",
                f"- mean_abs_error: {prior.get('mean_abs_error', 'NR')}",
                f"- parameter_priors: {json.dumps(prior.get('parameter_priors', {}), ensure_ascii=False)}",
                f"- auxiliary_error_summary: {json.dumps(prior.get('auxiliary_error_summary', {}), ensure_ascii=False)}",
            ]
        )
    condition_priors = calibration_results.get("condition_priors", []) if isinstance(calibration_results, dict) else []
    if condition_priors:
        lines.extend(["", "## Condition Priors"])
        for prior in condition_priors:
            lines.extend(
                [
                    f"### {prior.get('material_family', 'NR')} | concentration={prior.get('concentration_fraction', 'NR')} | curing={prior.get('curing_seconds', 'NR')}",
                    f"- sample_count: {prior.get('sample_count', 'NR')}",
                    f"- mean_abs_error: {prior.get('mean_abs_error', 'NR')}",
                    f"- parameter_priors: {json.dumps(prior.get('parameter_priors', {}), ensure_ascii=False)}",
                    f"- auxiliary_error_summary: {json.dumps(prior.get('auxiliary_error_summary', {}), ensure_ascii=False)}",
                ]
            )
    return "\n".join(lines)


def _build_calibration_impact_ledger(impact_payload: dict[str, object]) -> str:
    rows = impact_payload.get("cases", []) if isinstance(impact_payload, dict) else []
    lines = ["## Calibration Impact Assessment"]
    for row in rows[:20]:
        baseline = row.get("baseline", {}) if isinstance(row, dict) else {}
        calibrated = row.get("calibrated", {}) if isinstance(row, dict) else {}
        lines.extend(
            [
                f"### {row.get('sample_key', 'NR')}",
                f"- material_family: {row.get('material_family', 'NR')}",
                f"- baseline_abs_error: {baseline.get('abs_error', 'NR')}",
                f"- calibrated_abs_error: {calibrated.get('abs_error', 'NR')}",
                f"- baseline_total_error: {baseline.get('total_error', 'NR')}",
                f"- calibrated_total_error: {calibrated.get('total_error', 'NR')}",
                f"- improved_abs_error: {row.get('improved_abs_error', 'NR')}",
                f"- improved_total_error: {row.get('improved_total_error', 'NR')}",
                f"- prior_source: {row.get('prior_source', 'NR')}",
                f"- calibrated_search_space: {json.dumps(row.get('calibrated_search_space', {}), ensure_ascii=False)}",
            ]
        )
    summary = impact_payload.get("summary", {}) if isinstance(impact_payload, dict) else {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- available: {summary.get('available', 'NR')}",
            f"- eligible_case_count: {summary.get('eligible_case_count', 'NR')}",
            f"- mean_abs_error_baseline: {summary.get('mean_abs_error_baseline', 'NR')}",
            f"- mean_abs_error_calibrated: {summary.get('mean_abs_error_calibrated', 'NR')}",
            f"- mean_abs_error_delta: {summary.get('mean_abs_error_delta', 'NR')}",
            f"- mean_total_error_baseline: {summary.get('mean_total_error_baseline', 'NR')}",
            f"- mean_total_error_calibrated: {summary.get('mean_total_error_calibrated', 'NR')}",
            f"- mean_total_error_delta: {summary.get('mean_total_error_delta', 'NR')}",
            f"- improved_abs_case_count: {summary.get('improved_abs_case_count', 'NR')}",
            f"- improved_total_case_count: {summary.get('improved_total_case_count', 'NR')}",
            f"- evaluation_mode: {summary.get('evaluation_mode', 'NR')}",
            f"- overall_pass: {summary.get('overall_pass', 'NR')}",
        ]
    )
    return "\n".join(lines)


def _build_calibration_summary_ledger(*, query: str, payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return "\n".join(
        [
            "## Calibration Summary",
            f"- Query: {query}",
            f"- dataset_id: {payload.get('dataset_id', 'NR') if isinstance(payload, dict) else 'NR'}",
            f"- measurement_count: {summary.get('measurement_count', 'NR')}",
            f"- target_count: {summary.get('target_count', 'NR')}",
            f"- family_count: {summary.get('family_count', 'NR')}",
            f"- condition_prior_count: {len(payload.get('calibration_results', {}).get('condition_priors', [])) if isinstance(payload, dict) else 'NR'}",
            f"- metrics_covered: {json.dumps(summary.get('metrics_covered', []), ensure_ascii=False)}",
            f"- mean_abs_error: {summary.get('mean_abs_error', 'NR')}",
            f"- impact_available: {summary.get('impact_available', 'NR')}",
            f"- impact_mean_abs_error_delta: {summary.get('impact_mean_abs_error_delta', 'NR')}",
            f"- impact_mean_total_error_delta: {summary.get('impact_mean_total_error_delta', 'NR')}",
        ]
    )


def _build_calibration_report_markdown(
    *,
    query: str,
    dataset_id: str,
    payload: dict[str, object],
) -> str:
    targets = payload.get("calibration_targets", []) if isinstance(payload, dict) else []
    results = payload.get("calibration_results", {}).get("cases", []) if isinstance(payload, dict) else []
    impact = payload.get("calibration_impact_assessment", {}) if isinstance(payload, dict) else {}
    impact_summary = impact.get("summary", {}) if isinstance(impact, dict) else {}
    table_lines = [
        "| sample_key | material_family | target_stiffness | abs_error | rel_error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results[:20]:
        table_lines.append(
            "| {sample_key} | {material_family} | {target_stiffness:.3f} | {abs_error:.3f} | {rel_error:.3f} |".format(
                sample_key=row.get("sample_key", "NR"),
                material_family=row.get("material_family", "NR"),
                target_stiffness=float(row.get("target_stiffness", 0.0)),
                abs_error=float(row.get("abs_error", 0.0)),
                rel_error=float(row.get("rel_error", 0.0)),
            )
        )
    if len(table_lines) == 2:
        table_lines.append("| NR | NR | NR | NR | NR |")
    return "\n".join(
        [
            "# ECM Calibration Report",
            "",
            "## 1. Calibration Goal",
            f"- Query: {query}",
            f"- Dataset: {dataset_id}",
            "",
            "## 2. Extracted Calibration Targets",
            f"- Count: {len(targets)}",
            f"- Metrics covered: {json.dumps(payload.get('summary', {}).get('metrics_covered', []), ensure_ascii=False)}",
            "",
            "## 3. Calibration Results",
            *table_lines,
            "",
            "## 4. Family Priors",
            json.dumps(payload.get("calibration_results", {}).get("family_priors", []), ensure_ascii=False, indent=2),
            "",
            "## 5. Condition Priors",
            json.dumps(payload.get("calibration_results", {}).get("condition_priors", []), ensure_ascii=False, indent=2),
            "",
            "## 6. Calibration-to-Design Impact",
            f"- available: {impact_summary.get('available', 'NR')}",
            f"- eligible_case_count: {impact_summary.get('eligible_case_count', 'NR')}",
            f"- mean_abs_error_baseline: {impact_summary.get('mean_abs_error_baseline', 'NR')}",
            f"- mean_abs_error_calibrated: {impact_summary.get('mean_abs_error_calibrated', 'NR')}",
            f"- mean_abs_error_delta: {impact_summary.get('mean_abs_error_delta', 'NR')}",
            f"- mean_total_error_baseline: {impact_summary.get('mean_total_error_baseline', 'NR')}",
            f"- mean_total_error_calibrated: {impact_summary.get('mean_total_error_calibrated', 'NR')}",
            f"- mean_total_error_delta: {impact_summary.get('mean_total_error_delta', 'NR')}",
            f"- improved_total_case_count: {impact_summary.get('improved_total_case_count', 'NR')}",
            f"- evaluation_mode: {impact_summary.get('evaluation_mode', 'NR')}",
            "",
            "## 7. Next Use",
            "- Use these family-level priors to initialize design search and these condition priors to specialize around concentration / curing regimes.",
            "- Refine with your own laboratory measurements once repeated rheology / compression data are available.",
        ]
    )


def _build_calibration_final_summary(*, payload: dict[str, object], saved_path: str) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return (
        f"Calibration workflow finished. dataset_id={payload.get('dataset_id', 'NR') if isinstance(payload, dict) else 'NR'}, "
        f"target_count={summary.get('target_count', 'NR')}, "
        f"family_count={summary.get('family_count', 'NR')}, "
        f"mean_abs_error={summary.get('mean_abs_error', 'NR')}. "
        f"impact_mean_total_error_delta={summary.get('impact_mean_total_error_delta', 'NR')}. "
        f"Report saved to {saved_path}."
    )


def _extract_first_float(text: str, *, fallback: float) -> float:
    import re

    match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else fallback
