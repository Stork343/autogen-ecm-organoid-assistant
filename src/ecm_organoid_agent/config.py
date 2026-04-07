from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


def _load_dotenv(project_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(project_dir / ".env", override=False)


@dataclass(frozen=True)
class AppConfig:
    project_dir: Path
    memory_dir: Path
    library_dir: Path
    report_dir: Path
    runs_dir: Path
    cache_dir: Path
    template_dir: Path
    model_provider: str
    model: str
    model_api_key: Optional[str]
    model_base_url: Optional[str]
    active_run_dir: Optional[Path]
    cache_ttl_hours: int
    max_pubmed_results: int
    ncbi_email: Optional[str]
    ncbi_api_key: Optional[str]
    crossref_mailto: Optional[str]
    frontend_require_login: bool
    frontend_username: Optional[str]
    frontend_password: Optional[str]
    frontend_password_sha256: Optional[str]
    frontend_public_host: str
    frontend_public_port: int

    @classmethod
    def from_project_dir(cls, project_dir: Path) -> "AppConfig":
        project_dir = project_dir.resolve()
        _load_dotenv(project_dir)
        model_provider = os.getenv(
            "MODEL_PROVIDER",
            "deepseek" if os.getenv("DEEPSEEK_API_KEY") else "openai",
        ).strip().lower()
        default_model = "deepseek-chat" if model_provider == "deepseek" else "gpt-4.1-mini"
        model = (
            os.getenv("MODEL_NAME")
            or os.getenv("DEEPSEEK_MODEL" if model_provider == "deepseek" else "OPENAI_MODEL")
            or default_model
        )
        model_api_key = (
            os.getenv("MODEL_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY" if model_provider == "deepseek" else "OPENAI_API_KEY")
        )
        model_base_url = (
            os.getenv("MODEL_BASE_URL")
            or (
                os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                if model_provider == "deepseek"
                else os.getenv("OPENAI_BASE_URL")
            )
        )
        return cls(
            project_dir=project_dir,
            memory_dir=project_dir / "memory",
            library_dir=project_dir / "library",
            report_dir=project_dir / "reports",
            runs_dir=project_dir / "runs",
            cache_dir=project_dir / ".cache",
            template_dir=project_dir / "templates",
            model_provider=model_provider,
            model=model,
            model_api_key=model_api_key,
            model_base_url=model_base_url,
            active_run_dir=None,
            cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "168")),
            max_pubmed_results=int(os.getenv("PUBMED_MAX_RESULTS", "8")),
            ncbi_email=os.getenv("NCBI_EMAIL"),
            ncbi_api_key=os.getenv("NCBI_API_KEY"),
            crossref_mailto=os.getenv("CROSSREF_MAILTO"),
            frontend_require_login=os.getenv("FRONTEND_REQUIRE_LOGIN", "false").strip().lower() in {"1", "true", "yes", "on"},
            frontend_username=os.getenv("FRONTEND_USERNAME"),
            frontend_password=os.getenv("FRONTEND_PASSWORD"),
            frontend_password_sha256=os.getenv("FRONTEND_PASSWORD_SHA256"),
            frontend_public_host=os.getenv("FRONTEND_PUBLIC_HOST", "0.0.0.0").strip(),
            frontend_public_port=int(os.getenv("FRONTEND_PUBLIC_PORT", "8525")),
        )

    def with_overrides(
        self,
        *,
        library_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None,
        model: Optional[str] = None,
        max_pubmed_results: Optional[int] = None,
        active_run_dir: Optional[Path] = None,
    ) -> "AppConfig":
        return replace(
            self,
            library_dir=library_dir.resolve() if library_dir else self.library_dir,
            report_dir=report_dir.resolve() if report_dir else self.report_dir,
            model=model or self.model,
            active_run_dir=active_run_dir.resolve() if active_run_dir else self.active_run_dir,
            max_pubmed_results=(
                max_pubmed_results if max_pubmed_results is not None else self.max_pubmed_results
            ),
        )
