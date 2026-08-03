"""Lifecycle hooks.

Mirrors `runtime/src/hooks.rs`. A PreToolUse hook is the strongest actor in the
pipeline: it can rewrite the tool input, inject a permission override, or
short-circuit the call entirely before the policy is ever consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class HookEvent(Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"

    def as_str(self) -> str:
        return self.value


@dataclass
class HookResult:
    decision: str = "continue"  # continue | deny | cancel | fail
    _updated_input: Optional[dict[str, Any]] = None
    _permission_override: Optional[str] = None
    _permission_reason: Optional[str] = None
    _messages: list[str] = field(default_factory=list)

    # Accessors named after the Rust methods the runtime calls.
    def updated_input(self) -> Optional[dict[str, Any]]:
        return self._updated_input

    def permission_override(self) -> Optional[str]:
        return self._permission_override

    def permission_reason(self) -> Optional[str]:
        return self._permission_reason

    def messages(self) -> list[str]:
        return self._messages

    def is_denied(self) -> bool:
        return self.decision == "deny"

    def is_cancelled(self) -> bool:
        return self.decision == "cancel"

    def is_failed(self) -> bool:
        return self.decision == "fail"

    # Constructors for hook authors.
    @classmethod
    def proceed(cls, message: Optional[str] = None) -> "HookResult":
        return cls(_messages=[message] if message else [])

    @classmethod
    def rewrite(cls, updated_input: dict[str, Any], message: Optional[str] = None) -> "HookResult":
        return cls(_updated_input=updated_input, _messages=[message] if message else [])

    @classmethod
    def deny(cls, reason: str) -> "HookResult":
        return cls(decision="deny", _messages=[reason])

    @classmethod
    def cancel(cls, reason: str) -> "HookResult":
        return cls(decision="cancel", _messages=[reason])

    @classmethod
    def override(cls, permission_override: str, reason: str) -> "HookResult":
        return cls(
            _permission_override=permission_override,
            _permission_reason=reason,
            _messages=[reason],
        )


HookFn = Callable[[dict[str, Any]], HookResult]


class HookRegistry:
    """Hooks run in registration order; the first short-circuit wins."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookFn]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, hook: HookFn) -> None:
        self._hooks[event].append(hook)

    def run(self, event: HookEvent, payload: dict[str, Any]) -> HookResult:
        folded = HookResult()
        for hook in self._hooks[event]:
            result = hook(payload)
            folded._messages.extend(result.messages())

            if result.updated_input() is not None:
                folded._updated_input = result.updated_input()
                payload = {**payload, "input": result.updated_input()}
            if result.permission_override() is not None:
                folded._permission_override = result.permission_override()
                folded._permission_reason = result.permission_reason()
            if result.decision != "continue":
                folded.decision = result.decision
                return folded
        return folded


def format_hook_message(result: HookResult, fallback: str) -> str:
    messages = [m for m in result.messages() if m]
    if not messages:
        return fallback
    return f"{fallback}: " + "; ".join(messages)


def merge_hook_feedback(messages: list[str], output: str, is_error: bool) -> str:
    """Splice hook commentary into the tool output the model will see."""
    notes = [m for m in messages if m]
    if not notes:
        return output
    label = "hook error" if is_error else "hook note"
    joined = "\n".join(f"[{label}] {note}" for note in notes)
    return f"{joined}\n{output}" if output else joined
