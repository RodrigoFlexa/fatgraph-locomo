"""LLM access: one `complete()` interface, Azure or offline fake, always cached."""

from fgl.llm.client import (
    AzureLLM,
    FakeLLM,
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

__all__ = [
    "AzureLLM", "FakeLLM", "LLMClient", "LLMError", "Usage", "build_llm",
    "default_fake_responder", "parse_json_loose", "PromptLibrary",
    "SYSTEM_ANSWERER", "SYSTEM_EXTRACTOR", "SYSTEM_JUDGE",
]
