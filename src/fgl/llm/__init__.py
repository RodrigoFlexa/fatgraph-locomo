"""LLM access: one `complete()` interface, Azure or offline fake, always cached."""

from fgl.llm.client import (
    FakeLLM,
    LLMUnhealthy,
    LLMClient,
    LLMError,
    Usage,
    build_llm,
    default_fake_responder,
    parse_json_loose,
)
from fgl.llm.prompts import (
    SYSTEM_ANSWERER,
    SYSTEM_EXTRACTOR,
    SYSTEM_JUDGE,
    PromptLibrary,
)


def __getattr__(name: str):
    """`AzureLLM` needs the openai SDK, so import it only when asked for."""
    if name == "AzureLLM":
        from fgl.llm.azure import AzureLLM

        return AzureLLM
    if name == "is_reasoning_deployment":
        from fgl.llm.azure import is_reasoning_deployment

        return is_reasoning_deployment
    raise AttributeError(name)

__all__ = [
    "AzureLLM", "FakeLLM", "LLMClient", "LLMError", "Usage", "build_llm",
    "default_fake_responder", "parse_json_loose", "PromptLibrary",
    "LLMUnhealthy", "is_reasoning_deployment",
    "SYSTEM_ANSWERER", "SYSTEM_EXTRACTOR", "SYSTEM_JUDGE",
]
