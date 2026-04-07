from __future__ import annotations

from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import AppConfig


def _deepseek_model_info(model_name: str) -> dict[str, object]:
    model_lower = model_name.lower()
    is_reasoner = "reasoner" in model_lower or model_lower.startswith("deepseek-r1")
    return {
        "vision": False,
        "function_calling": not is_reasoner,
        "json_output": False,
        "structured_output": False,
        "family": ModelFamily.R1 if is_reasoner else ModelFamily.UNKNOWN,
        "multiple_system_messages": True,
    }


def build_model_client(config: AppConfig) -> OpenAIChatCompletionClient:
    provider = config.model_provider.lower()
    common_kwargs = {
        "model": config.model,
        "parallel_tool_calls": False,
    }

    if provider == "openai":
        if config.model_api_key:
            common_kwargs["api_key"] = config.model_api_key
        if config.model_base_url:
            common_kwargs["base_url"] = config.model_base_url
        return OpenAIChatCompletionClient(**common_kwargs)

    if provider == "deepseek":
        if not config.model_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing. Set it in .env or the shell environment.")
        return OpenAIChatCompletionClient(
            **common_kwargs,
            api_key=config.model_api_key,
            base_url=config.model_base_url or "https://api.deepseek.com/v1",
            model_info=_deepseek_model_info(config.model),
        )

    raise RuntimeError(f"Unsupported MODEL_PROVIDER: {config.model_provider}")
