"""Core conversation types.

Mirrors the shapes in `runtime/src/` and `api/src/types.rs` from claw-code.
Names are kept identical to the Rust originals wherever Python allows it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class RuntimeError(Exception):  # noqa: A001 - deliberately mirrors runtime::RuntimeError
    """Turn-level failure. Shadows the builtin inside this package on purpose."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class ContentBlock:
    """One block inside an assistant message: `Text` or `ToolUse`."""

    kind: str  # "text" | "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text_block(cls, text: str) -> "ContentBlock":
        return cls(kind="text", text=text)

    @classmethod
    def tool_use(cls, id: str, name: str, input: dict[str, Any]) -> "ContentBlock":
        return cls(kind="tool_use", id=id, name=name, input=input)


@dataclass
class ConversationMessage:
    """A single message in the session history."""

    role: str  # "user" | "assistant" | "tool"
    blocks: list[ContentBlock] = field(default_factory=list)
    tool_use_id: str = ""
    tool_name: str = ""
    is_error: bool = False

    @classmethod
    def user_text(cls, text: str) -> "ConversationMessage":
        return cls(role="user", blocks=[ContentBlock.text_block(text)])

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        tool_name: str,
        output: str,
        is_error: bool,
    ) -> "ConversationMessage":
        return cls(
            role="tool",
            blocks=[ContentBlock.text_block(output)],
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            is_error=is_error,
        )

    def text(self) -> str:
        return "".join(b.text for b in self.blocks if b.kind == "text")

    def tool_uses(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.kind == "tool_use"]

    def to_wire(self) -> dict[str, Any]:
        """Render into the Ollama /api/chat message format."""
        if self.role == "tool":
            prefix = "ERROR: " if self.is_error else ""
            return {
                "role": "tool",
                "content": f"{prefix}{self.text()}",
                "tool_name": self.tool_name,
            }

        wire: dict[str, Any] = {"role": self.role, "content": self.text()}
        calls = self.tool_uses()
        if calls:
            wire["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.input}} for c in calls
            ]
        return wire


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class UsageTracker:
    def __init__(self) -> None:
        self._cumulative = Usage()

    def record(self, usage: Usage) -> None:
        self._cumulative = self._cumulative + usage

    def cumulative_usage(self) -> Usage:
        return self._cumulative


@dataclass
class CompactionRecord:
    """Set on the session once history has been rewritten."""

    summary: str
    dropped_messages: int
    before_tokens: int
    after_tokens: int


@dataclass
class Session:
    """Ordered message history plus the compaction marker."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[ConversationMessage] = field(default_factory=list)
    compaction: Optional[CompactionRecord] = None
    # Set by replay_session: the system prompt the history was produced under.
    # A resumed session must not silently get a different one.
    system_prompt: Optional[str] = None

    def push_user_text(self, text: str) -> None:
        if not text.strip():
            raise RuntimeError("cannot push an empty user message")
        self.messages.append(ConversationMessage.user_text(text))

    def push_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)

    def fork_session(self, branch_name: Optional[str] = None) -> "Session":
        return Session(
            session_id=branch_name or uuid.uuid4().hex[:12],
            messages=list(self.messages),
            compaction=self.compaction,
            system_prompt=self.system_prompt,
        )


@dataclass
class ApiRequest:
    system_prompt: str
    messages: list[ConversationMessage]


@dataclass
class TurnSummary:
    assistant_messages: list[ConversationMessage] = field(default_factory=list)
    tool_results: list[ConversationMessage] = field(default_factory=list)
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    auto_compaction: Optional[CompactionRecord] = None
