"""Thin LLM interface (DRAFT, not frozen; freeze with docs/preregistration.md).

The graph never talks to a provider SDK. Real clients (Bedrock/Anthropic) will implement
`LLMClient` behind the disk cache and spend limits; tests use `ScriptedLLM`. This keeps the
cache key, cost accounting and mocking independent of LangGraph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class LLMReply:
    """Either final `text` (expected to be the JSON decision) or `tool_calls`."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, messages: Sequence[dict[str, Any]], *, tools_allowed: bool) -> LLMReply: ...


@dataclass
class ScriptedLLM:
    """Test double: returns pre-written replies in order and records what it was shown."""

    replies: list[LLMReply]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages: Sequence[dict[str, Any]], *, tools_allowed: bool) -> LLMReply:
        self.calls.append({"n_messages": len(messages), "tools_allowed": tools_allowed})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)
