from __future__ import annotations

import html
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from .artifacts import append_jsonl, utc_now_iso
from .config import AppConfig
from .fiber_network import (
    design_ecm_candidates as design_fiber_network_candidates_backend,
    run_parameter_scan as run_fiber_network_parameter_scan_backend,
    run_simulation as run_fiber_network_simulation_backend,
    run_validation as run_fiber_network_validation_backend,
    simulate_ecm as simulate_ecm_backend,
)
from .mechanics import fit_mechanics_dataset, simulate_mechanics_curve

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf"}
MAX_TEXT_CHARS = 30000
MAX_PDF_PAGES = 10
FIBER_NETWORK_CACHE_VERSION = "2026-03-27-search-refine-v1"
SECTION_STOP_PATTERNS = [
    re.compile(r"^(keywords?|index terms?)\b", re.IGNORECASE),
    re.compile(r"^(introduction|background|methods?|materials? and methods?)\b", re.IGNORECASE),
    re.compile(r"^\d+(\.\d+)*\s+(introduction|background|methods?|results?|discussion)\b", re.IGNORECASE),
]


def normalize_query_terms(query: str) -> List[str]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", query.lower())
    if terms:
        return terms
    fallback = query.strip().lower()
    return [fallback] if fallback else []


def sanitize_filename(filename: str) -> str:
    safe_name = re.sub(r"[^\w.-]+", "_", filename, flags=re.UNICODE).strip("._")
    if not safe_name:
        safe_name = "research_report"
    if not safe_name.endswith(".md"):
        safe_name = safe_name + ".md"
    return safe_name


def make_snippet(text: str, terms: Sequence[str], *, window: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    for term in terms:
        index = compact.lower().find(term.lower())
        if index >= 0:
            start = max(0, index - window // 2)
            end = min(len(compact), index + window // 2)
            snippet = compact[start:end]
            return snippet if start == 0 else "..." + snippet
    return compact[:window] + ("..." if len(compact) > window else "")


def _meaningful_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def guess_title_from_text(text: str, fallback_name: str) -> str:
    for line in _meaningful_lines(text)[:12]:
        compact = re.sub(r"\s+", " ", line).strip(" .")
        if not compact:
            continue
        lowered = compact.lower()
        if lowered in {"abstract", "summary", "introduction"}:
            continue
        if len(compact) < 15 or len(compact) > 220:
            continue
        if compact.count(" ") < 2:
            continue
        return compact
    return fallback_name.replace("_", " ").replace("-", " ")


def extract_abstract_excerpt(text: str, *, max_chars: int = 1600) -> str:
    lines = _meaningful_lines(text)
    if not lines:
        return ""

    for idx, line in enumerate(lines):
        if re.fullmatch(r"abstract", line, flags=re.IGNORECASE) or re.match(
            r"^abstract[:\s-]", line, flags=re.IGNORECASE
        ):
            parts: List[str] = []
            remainder = re.sub(r"^abstract[:\s-]*", "", line, flags=re.IGNORECASE).strip()
            if remainder:
                parts.append(remainder)
            for next_line in lines[idx + 1 :]:
                if any(pattern.match(next_line) for pattern in SECTION_STOP_PATTERNS):
                    break
                parts.append(next_line)
                if len(" ".join(parts)) >= max_chars:
                    break
            compact = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if compact:
                return compact[:max_chars]

    start = 1 if len(lines) > 1 else 0
    fallback = re.sub(r"\s+", " ", " ".join(lines[start:])).strip()
    return fallback[:max_chars]


def extract_text_from_file(path: Path) -> Tuple[str, Optional[str]]:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_CHARS], None
        except OSError as exc:
            return "", f"Failed to read {path.name}: {exc}"

    if suffix == ".pdf":
        if PdfReader is None:
            return "", "Skipping PDF files because pypdf is not installed."
        try:
            reader = PdfReader(str(path))
            chunks: List[str] = []
            for page in reader.pages[:MAX_PDF_PAGES]:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks)[:MAX_TEXT_CHARS], None
        except Exception as exc:  # pragma: no cover
            return "", f"Failed to extract PDF text from {path.name}: {exc}"

    return "", f"Unsupported file type for {path.name}."


def summarize_document(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text, note = extract_text_from_file(path)
    if not text:
        return None, note
    title_guess = guess_title_from_text(text, path.stem)
    abstract_excerpt = extract_abstract_excerpt(text)
    return (
        {
            "path": str(path),
            "file_name": path.name,
            "document_type": path.suffix.lower().lstrip("."),
            "text": text,
            "title_guess": title_guess,
            "abstract_excerpt": abstract_excerpt,
        },
        note,
    )


def search_library(query: str, library_dir: Path, *, max_results: int = 5) -> Dict[str, object]:
    terms = normalize_query_terms(query)
    payload: Dict[str, object] = {"library_dir": str(library_dir), "matches": [], "notes": []}
    if not library_dir.exists():
        payload["notes"] = [f"Library directory does not exist: {library_dir}"]
        return payload
    if not terms:
        payload["notes"] = ["Search query is empty."]
        return payload

    matches: List[Dict[str, object]] = []
    notes: List[str] = []

    for path in sorted(library_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        summary, note = summarize_document(path)
        if note:
            notes.append(note)
        if summary is None:
            continue

        text = str(summary["text"])
        title_guess = str(summary["title_guess"])
        abstract_excerpt = str(summary["abstract_excerpt"])
        lowered = text.lower()
        title_lower = title_guess.lower()
        abstract_lower = abstract_excerpt.lower()
        score = (
            sum(lowered.count(term) for term in terms)
            + 3 * sum(title_lower.count(term) for term in terms)
            + 2 * sum(abstract_lower.count(term) for term in terms)
        )
        if score <= 0:
            continue

        matches.append(
            {
                "path": str(summary["path"]),
                "file_name": str(summary["file_name"]),
                "document_type": str(summary["document_type"]),
                "score": score,
                "title_guess": title_guess,
                "abstract_excerpt": abstract_excerpt[:500],
                "snippet": make_snippet(text, terms),
            }
        )

    matches.sort(key=lambda item: int(item["score"]), reverse=True)
    payload["matches"] = matches[:max_results]
    payload["notes"] = notes
    return payload


def save_markdown_report(report_dir: Path, filename: str, content: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / sanitize_filename(filename)
    path.write_text(content, encoding="utf-8")
    return path


def _ncbi_common_params(config: AppConfig) -> Dict[str, str]:
    params = {"tool": "autogen-ecm-organoid-assistant"}
    if config.ncbi_email:
        params["email"] = config.ncbi_email
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return params


def _extract_doi(article_ids: Sequence[Dict[str, object]]) -> str:
    for article_id in article_ids:
        if article_id.get("idtype") == "doi":
            return str(article_id.get("value", ""))
    return ""


def _parse_abstracts(xml_text: str) -> Dict[str, str]:
    root = ET.fromstring(xml_text)
    abstracts: Dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext("./MedlineCitation/PMID")
        if not pmid:
            continue

        parts: List[str] = []
        for abstract_node in article.findall(".//Abstract/AbstractText"):
            text = "".join(abstract_node.itertext()).strip()
            if not text:
                continue
            label = abstract_node.attrib.get("Label")
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        abstracts[pmid] = "\n".join(parts)
    return abstracts


def _format_date_parts(parts: Sequence[object]) -> str:
    filtered = [str(part) for part in parts if part is not None]
    return "-".join(filtered[:3])


def _crossref_published_date(item: Dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued", "created"):
        date_parts = item.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return _format_date_parts(date_parts[0])
    return ""


def _clean_markup(text: str) -> str:
    compact = html.unescape(text or "")
    compact = re.sub(r"<[^>]+>", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _join_field(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value or "").strip()


def normalize_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    title = _join_field(item.get("title", []))
    journal = _join_field(item.get("container-title", []))
    authors = []
    for author in item.get("author", []):
        given = str(author.get("given", "")).strip()
        family = str(author.get("family", "")).strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)

    doi = str(item.get("DOI", "")).strip()
    abstract_excerpt = _clean_markup(str(item.get("abstract", "")))[:1500]
    url = str(item.get("URL", "")).strip() or (f"https://doi.org/{doi}" if doi else "")
    return {
        "title": title,
        "journal": journal,
        "published": _crossref_published_date(item),
        "type": str(item.get("type", "")).strip(),
        "doi": doi,
        "url": url,
        "authors": authors[:6],
        "abstract_excerpt": abstract_excerpt,
    }


def _cache_path(config: AppConfig, namespace: str, payload: Dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return config.cache_dir / namespace / f"{digest}.json"


def _load_cached_text(config: AppConfig, namespace: str, payload: Dict[str, Any]) -> Optional[str]:
    if config.cache_ttl_hours <= 0:
        return None
    path = _cache_path(config, namespace, payload)
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > config.cache_ttl_hours * 3600:
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = cached.get("value")
    return value if isinstance(value, str) else None


def _save_cached_text(config: AppConfig, namespace: str, payload: Dict[str, Any], value: str) -> None:
    path = _cache_path(config, namespace, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cached_at": utc_now_iso(),
                "namespace": namespace,
                "payload": payload,
                "value": value,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _log_tool_call(
    config: AppConfig,
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    result: str,
    cached: Optional[bool] = None,
    status: str = "ok",
) -> None:
    if config.active_run_dir is None:
        return
    append_jsonl(
        config.active_run_dir / "tool_calls.jsonl",
        {
            "timestamp": utc_now_iso(),
            "tool_name": tool_name,
            "arguments": arguments,
            "cached": cached,
            "status": status,
            "result": result,
        },
    )


def build_tools(config: AppConfig):
    async def search_pubmed(query: str, max_results: int = 5) -> str:
        """Search PubMed and return JSON records with PMID, journal metadata, DOI, and abstract excerpt."""
        import httpx

        max_results = max(1, min(max_results, 20))
        cache_payload = {"query": query, "max_results": max_results}
        cached_text = _load_cached_text(config, "pubmed", cache_payload)
        if cached_text is not None:
            _log_tool_call(
                config,
                tool_name="search_pubmed",
                arguments=cache_payload,
                result=cached_text,
                cached=True,
            )
            return cached_text
        common_params = _ncbi_common_params(config)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                search_response = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={
                        "db": "pubmed",
                        "term": query,
                        "retmax": max_results,
                        "retmode": "json",
                        "sort": "relevance",
                        **common_params,
                    },
                )
                search_response.raise_for_status()
                ids = search_response.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    result = "No PubMed results found."
                    _save_cached_text(config, "pubmed", cache_payload, result)
                    _log_tool_call(
                        config,
                        tool_name="search_pubmed",
                        arguments=cache_payload,
                        result=result,
                        cached=False,
                    )
                    return result

                summary_response = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "json",
                        **common_params,
                    },
                )
                summary_response.raise_for_status()
                summary_payload = summary_response.json().get("result", {})

                fetch_response = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        **common_params,
                    },
                )
                fetch_response.raise_for_status()
                abstracts = _parse_abstracts(fetch_response.text)
        except httpx.HTTPError as exc:
            result = f"PubMed request failed: {exc}"
            _log_tool_call(
                config,
                tool_name="search_pubmed",
                arguments=cache_payload,
                result=result,
                cached=False,
                status="error",
            )
            return result
        except ET.ParseError as exc:
            result = f"PubMed abstract parsing failed: {exc}"
            _log_tool_call(
                config,
                tool_name="search_pubmed",
                arguments=cache_payload,
                result=result,
                cached=False,
                status="error",
            )
            return result

        records: List[Dict[str, object]] = []
        for pmid in ids:
            article = summary_payload.get(pmid, {})
            abstract = abstracts.get(pmid, "")
            authors = [item.get("name", "") for item in article.get("authors", []) if item.get("name")]
            records.append(
                {
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "journal": article.get("fulljournalname", article.get("source", "")),
                    "pubdate": article.get("pubdate", ""),
                    "doi": _extract_doi(article.get("articleids", [])),
                    "authors": authors[:6],
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "abstract_excerpt": abstract[:1500],
                }
            )
        result = json.dumps(records, ensure_ascii=False, indent=2)
        _save_cached_text(config, "pubmed", cache_payload, result)
        _log_tool_call(
            config,
            tool_name="search_pubmed",
            arguments=cache_payload,
            result=result,
            cached=False,
        )
        return result

    async def search_crossref(query: str, max_results: int = 5) -> str:
        """Search Crossref for complementary metadata such as DOI, title, journal, and abstract when available."""
        import httpx

        max_results = max(1, min(max_results, 20))
        cache_payload = {"query": query, "max_results": max_results}
        cached_text = _load_cached_text(config, "crossref", cache_payload)
        if cached_text is not None:
            _log_tool_call(
                config,
                tool_name="search_crossref",
                arguments=cache_payload,
                result=cached_text,
                cached=True,
            )
            return cached_text
        user_agent = "autogen-ecm-organoid-assistant/0.2"
        if config.crossref_mailto:
            user_agent = f"{user_agent} (mailto:{config.crossref_mailto})"

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": user_agent},
            ) as client:
                response = await client.get(
                    "https://api.crossref.org/works",
                    params={
                        key: value
                        for key, value in {
                            "query": query,
                            "rows": max_results,
                            "mailto": config.crossref_mailto,
                        }.items()
                        if value not in (None, "")
                    },
                )
                response.raise_for_status()
                items = response.json().get("message", {}).get("items", [])
        except httpx.HTTPError as exc:
            result = f"Crossref request failed: {exc}"
            _log_tool_call(
                config,
                tool_name="search_crossref",
                arguments=cache_payload,
                result=result,
                cached=False,
                status="error",
            )
            return result

        if not items:
            result = "No Crossref results found."
            _save_cached_text(config, "crossref", cache_payload, result)
            _log_tool_call(
                config,
                tool_name="search_crossref",
                arguments=cache_payload,
                result=result,
                cached=False,
            )
            return result

        records = [normalize_crossref_item(item) for item in items]
        result = json.dumps(records, ensure_ascii=False, indent=2)
        _save_cached_text(config, "crossref", cache_payload, result)
        _log_tool_call(
            config,
            tool_name="search_crossref",
            arguments=cache_payload,
            result=result,
            cached=False,
        )
        return result

    def search_local_library(query: str, max_results: int = 5) -> str:
        """Search local Markdown, text, and PDF files and return ranked matches with title and abstract-like excerpts."""
        payload = search_library(query, config.library_dir, max_results=max_results)
        result = json.dumps(payload, ensure_ascii=False, indent=2)
        _log_tool_call(
            config,
            tool_name="search_local_library",
            arguments={"query": query, "max_results": max_results},
            result=result,
            cached=False,
        )
        return result

    def save_report(filename: str, content: str) -> str:
        """Save a Markdown report into the reports directory and return the saved path."""
        path = save_markdown_report(config.report_dir, filename, content)
        result = str(path)
        _log_tool_call(
            config,
            tool_name="save_report",
            arguments={"filename": filename},
            result=result,
            cached=False,
        )
        return result

    def fit_mechanics_model(
        data_path: str,
        experiment_type: str = "auto",
        time_column: str = "time",
        stress_column: str = "stress",
        strain_column: str = "strain",
        applied_stress: float = 0.0,
        applied_strain: float = 0.0,
        delimiter: str = ",",
    ) -> str:
        """Fit a simple ECM mechanics model from a CSV/TSV dataset of stress/strain/time measurements."""
        arguments = {
            "data_path": data_path,
            "experiment_type": experiment_type,
            "time_column": time_column,
            "stress_column": stress_column,
            "strain_column": strain_column,
            "applied_stress": applied_stress,
            "applied_strain": applied_strain,
            "delimiter": delimiter,
        }
        try:
            result_payload = fit_mechanics_dataset(
                data_path,
                experiment_type=experiment_type,
                delimiter=delimiter,
                time_column=time_column,
                stress_column=stress_column,
                strain_column=strain_column,
                applied_stress=applied_stress if applied_stress > 0 else None,
                applied_strain=applied_strain if applied_strain > 0 else None,
            )
            result = json.dumps(result_payload, ensure_ascii=False, indent=2)
            _log_tool_call(
                config,
                tool_name="fit_mechanics_model",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Mechanics fit failed: {exc}"
            _log_tool_call(
                config,
                tool_name="fit_mechanics_model",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    def simulate_mechanics_model(
        model_type: str,
        parameters_json: str,
        x_values_json: str,
        applied_stress: float = 0.0,
        applied_strain: float = 0.0,
    ) -> str:
        """Simulate a mechanics response curve from fitted parameters."""
        arguments = {
            "model_type": model_type,
            "applied_stress": applied_stress,
            "applied_strain": applied_strain,
        }
        try:
            parameters = json.loads(parameters_json)
            x_values = json.loads(x_values_json)
            payload = simulate_mechanics_curve(
                model_type=model_type,
                x_values=x_values,
                parameters=parameters,
                applied_stress=applied_stress if applied_stress > 0 else None,
                applied_strain=applied_strain if applied_strain > 0 else None,
            )
            result = json.dumps(payload, ensure_ascii=False, indent=2)
            _log_tool_call(
                config,
                tool_name="simulate_mechanics_model",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Mechanics simulation failed: {exc}"
            _log_tool_call(
                config,
                tool_name="simulate_mechanics_model",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    def run_fiber_network_simulation(
        fiber_density: float,
        fiber_stiffness: float,
        bending_stiffness: float,
        crosslink_prob: float,
        domain_size: float,
        total_force: float = 1.0,
        axis: str = "x",
        boundary_fraction: float = 0.15,
        seed: int = 0,
        max_iterations: int = 600,
        tolerance: float = 1e-6,
        monte_carlo_runs: int = 10,
        target_nodes: int = 8,
    ) -> str:
        """Run a 3D bead-spring ECM fiber-network mechanics simulation."""
        arguments = {
            "cache_version": FIBER_NETWORK_CACHE_VERSION,
            "fiber_density": fiber_density,
            "fiber_stiffness": fiber_stiffness,
            "bending_stiffness": bending_stiffness,
            "crosslink_prob": crosslink_prob,
            "domain_size": domain_size,
            "total_force": total_force,
            "axis": axis,
            "boundary_fraction": boundary_fraction,
            "seed": seed,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "monte_carlo_runs": monte_carlo_runs,
            "target_nodes": target_nodes,
        }
        cached_text = _load_cached_text(config, "fiber_network_simulation", arguments)
        if cached_text is not None:
            _log_tool_call(
                config,
                tool_name="run_fiber_network_simulation",
                arguments=arguments,
                result=cached_text,
                cached=True,
            )
            return cached_text
        try:
            payload = simulate_ecm_backend(
                {
                    "fiber_density": fiber_density,
                    "fiber_stiffness": fiber_stiffness,
                    "bending_stiffness": bending_stiffness,
                    "crosslink_prob": crosslink_prob,
                    "domain_size": domain_size,
                    "total_force": total_force,
                    "axis": axis,
                    "boundary_fraction": boundary_fraction,
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "tolerance": tolerance,
                    "monte_carlo_runs": monte_carlo_runs,
                    "target_nodes": target_nodes,
                }
            )
            result = json.dumps(payload, ensure_ascii=False, indent=2)
            _save_cached_text(config, "fiber_network_simulation", arguments, result)
            _log_tool_call(
                config,
                tool_name="run_fiber_network_simulation",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Fiber network simulation failed: {exc}"
            _log_tool_call(
                config,
                tool_name="run_fiber_network_simulation",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    def run_fiber_network_parameter_scan(
        fiber_density: float,
        fiber_stiffness: float,
        bending_stiffness: float,
        crosslink_prob: float,
        domain_size: float,
        total_force: float = 1.0,
        axis: str = "x",
        boundary_fraction: float = 0.15,
        seed: int = 0,
        max_iterations: int = 300,
        tolerance: float = 1e-6,
        monte_carlo_runs: int = 10,
        target_nodes: int = 8,
    ) -> str:
        """Run one-factor-at-a-time sensitivity scans around a baseline fiber-network design."""
        arguments = {
            "cache_version": FIBER_NETWORK_CACHE_VERSION,
            "fiber_density": fiber_density,
            "fiber_stiffness": fiber_stiffness,
            "bending_stiffness": bending_stiffness,
            "crosslink_prob": crosslink_prob,
            "domain_size": domain_size,
            "total_force": total_force,
            "axis": axis,
            "boundary_fraction": boundary_fraction,
            "seed": seed,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "monte_carlo_runs": monte_carlo_runs,
            "target_nodes": target_nodes,
        }
        cached_text = _load_cached_text(config, "fiber_network_scan", arguments)
        if cached_text is not None:
            _log_tool_call(
                config,
                tool_name="run_fiber_network_parameter_scan",
                arguments=arguments,
                result=cached_text,
                cached=True,
            )
            return cached_text
        try:
            payload = run_fiber_network_parameter_scan_backend(
                fiber_density=fiber_density,
                fiber_stiffness=fiber_stiffness,
                bending_stiffness=bending_stiffness,
                crosslink_prob=crosslink_prob,
                domain_size=domain_size,
                total_force=total_force,
                axis=axis,
                boundary_fraction=boundary_fraction,
                seed=seed,
                max_iterations=max_iterations,
                tolerance=tolerance,
                monte_carlo_runs=monte_carlo_runs,
                target_nodes=target_nodes,
            )
            result = json.dumps(payload, ensure_ascii=False, indent=2)
            _save_cached_text(config, "fiber_network_scan", arguments, result)
            _log_tool_call(
                config,
                tool_name="run_fiber_network_parameter_scan",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Fiber network parameter scan failed: {exc}"
            _log_tool_call(
                config,
                tool_name="run_fiber_network_parameter_scan",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    def design_fiber_network_candidates(
        target_stiffness: float,
        target_anisotropy: float = 0.1,
        target_connectivity: float = 0.95,
        target_stress_propagation: float = 0.5,
        extra_targets_json: str = "",
        constraint_max_anisotropy: float = 0.0,
        constraint_min_connectivity: float = 0.0,
        constraint_max_risk_index: float = 0.0,
        constraint_min_stress_propagation: float = 0.0,
        extra_constraints_json: str = "",
        top_k: int = 3,
        candidate_budget: int = 12,
        monte_carlo_runs: int = 4,
        total_force: float = 0.2,
        axis: str = "x",
        boundary_fraction: float = 0.15,
        seed: int = 1234,
        max_iterations: int = 500,
        tolerance: float = 1e-5,
        target_nodes: int = 8,
        search_space_json: str = "",
    ) -> str:
        """Search ECM parameter space for top-k candidate designs that match target mechanics."""

        arguments = {
            "cache_version": FIBER_NETWORK_CACHE_VERSION,
            "target_stiffness": target_stiffness,
            "target_anisotropy": target_anisotropy,
            "target_connectivity": target_connectivity,
            "target_stress_propagation": target_stress_propagation,
            "extra_targets_json": extra_targets_json,
            "constraint_max_anisotropy": constraint_max_anisotropy,
            "constraint_min_connectivity": constraint_min_connectivity,
            "constraint_max_risk_index": constraint_max_risk_index,
            "constraint_min_stress_propagation": constraint_min_stress_propagation,
            "extra_constraints_json": extra_constraints_json,
            "top_k": top_k,
            "candidate_budget": candidate_budget,
            "monte_carlo_runs": monte_carlo_runs,
            "total_force": total_force,
            "axis": axis,
            "boundary_fraction": boundary_fraction,
            "seed": seed,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "target_nodes": target_nodes,
            "search_space_json": search_space_json,
        }
        cached_text = _load_cached_text(config, "fiber_network_design", arguments)
        if cached_text is not None:
            _log_tool_call(
                config,
                tool_name="design_fiber_network_candidates",
                arguments=arguments,
                result=cached_text,
                cached=True,
            )
            return cached_text

        try:
            search_space = json.loads(search_space_json) if search_space_json.strip() else None
            extra_targets = json.loads(extra_targets_json) if extra_targets_json.strip() else {}
            extra_constraints = json.loads(extra_constraints_json) if extra_constraints_json.strip() else {}
            constraints = {
                key: value
                for key, value in {
                    "max_anisotropy": constraint_max_anisotropy if constraint_max_anisotropy > 0 else None,
                    "min_connectivity": constraint_min_connectivity if constraint_min_connectivity > 0 else None,
                    "max_risk_index": constraint_max_risk_index if constraint_max_risk_index > 0 else None,
                    "min_stress_propagation": constraint_min_stress_propagation if constraint_min_stress_propagation > 0 else None,
                }.items()
                if value is not None
            }
            if isinstance(extra_constraints, dict):
                constraints.update({str(key): float(value) for key, value in extra_constraints.items()})
            payload = design_fiber_network_candidates_backend(
                {
                    "stiffness": target_stiffness,
                    "anisotropy": target_anisotropy,
                    "connectivity": target_connectivity,
                    "stress_propagation": target_stress_propagation,
                    **({str(key): float(value) for key, value in extra_targets.items()} if isinstance(extra_targets, dict) else {}),
                },
                search_space=search_space,
                constraints=constraints,
                top_k=top_k,
                candidate_budget=candidate_budget,
                monte_carlo_runs=monte_carlo_runs,
                total_force=total_force,
                axis=axis,
                boundary_fraction=boundary_fraction,
                seed=seed,
                max_iterations=max_iterations,
                tolerance=tolerance,
                target_nodes=target_nodes,
            )
            result = json.dumps(payload, ensure_ascii=False, indent=2)
            _save_cached_text(config, "fiber_network_design", arguments, result)
            _log_tool_call(
                config,
                tool_name="design_fiber_network_candidates",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Fiber network design search failed: {exc}"
            _log_tool_call(
                config,
                tool_name="design_fiber_network_candidates",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    def run_fiber_network_validation() -> str:
        """Run the built-in physics and solver validation suite for the ECM fiber-network model."""
        arguments: dict[str, object] = {}
        try:
            payload = run_fiber_network_validation_backend()
            result = json.dumps(payload, ensure_ascii=False, indent=2)
            _log_tool_call(
                config,
                tool_name="run_fiber_network_validation",
                arguments=arguments,
                result=result,
                cached=False,
            )
            return result
        except Exception as exc:
            result = f"Fiber network validation failed: {exc}"
            _log_tool_call(
                config,
                tool_name="run_fiber_network_validation",
                arguments=arguments,
                result=result,
                cached=False,
                status="error",
            )
            return result

    return [
        search_pubmed,
        search_crossref,
        search_local_library,
        save_report,
        fit_mechanics_model,
        simulate_mechanics_model,
        run_fiber_network_simulation,
        run_fiber_network_parameter_scan,
        design_fiber_network_candidates,
        run_fiber_network_validation,
    ]
