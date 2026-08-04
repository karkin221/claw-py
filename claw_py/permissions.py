"""Permission policy.

Mirrors `runtime/src/permissions.rs`. The policy is consulted only if the
PreToolUse hook did not already short-circuit; a hook-supplied
`permission_override` inside the `PermissionContext` outranks the mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

# Fallback classification, used when no risk_lookup is supplied.
WRITE_TOOLS = {"write_file", "edit_file", "bash", "todo_write"}
ESCALATED_TOOLS = {"bash"}

RISK_READ = "read"
RISK_WRITE = "write"
RISK_ESCALATE = "escalate"


class PermissionMode(Enum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"
    PROMPT = "prompt"
    ALLOW = "allow"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "PermissionMode":
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"unknown permission mode `{value}`")


@dataclass
class PermissionContext:
    """Carries a hook's override into the policy decision."""

    permission_override: Optional[str] = None  # "allow" | "deny" | None
    permission_reason: Optional[str] = None


# Which layer actually decided. Without this, every denial looks identical in
# the trace and you cannot tell a hook veto from a policy refusal.
SOURCE_HOOK = "hook"                # PreToolUse short-circuited
SOURCE_HOOK_OVERRIDE = "hook_override"  # hook supplied permission_override
SOURCE_ALLOWLIST = "allowlist"      # tool not in allowed_tools (subagents)
SOURCE_MODE = "mode"                # the permission mode alone decided
SOURCE_WORKSPACE = "workspace"      # write outside the workspace root
SOURCE_PROMPTER = "prompter"        # a human answered
SOURCE_NO_PROMPTER = "no_prompter"  # escalation needed, nobody to ask


@dataclass
class PermissionOutcome:
    allowed: bool
    reason: str = ""
    source: str = SOURCE_MODE

    @classmethod
    def Allow(cls, source: str = SOURCE_MODE) -> "PermissionOutcome":  # noqa: N802
        return cls(allowed=True, source=source)

    @classmethod
    def Deny(cls, reason: str, source: str = SOURCE_MODE) -> "PermissionOutcome":  # noqa: N802
        return cls(allowed=False, reason=reason, source=source)


class PermissionPrompter(Protocol):
    """Interactive escalation path. Absent in one-shot mode."""

    def confirm(self, tool_name: str, effective_input: dict[str, Any]) -> bool: ...


class ConsolePrompter:
    def confirm(self, tool_name: str, effective_input: dict[str, Any]) -> bool:
        preview = str(effective_input)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"\n  permission required: {tool_name} {preview}")
        answer = input("  allow? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


class PermissionPolicy:
    def __init__(
        self,
        mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
        workspace_root: Optional[Path] = None,
        allowed_tools: Optional[set[str]] = None,
        risk_lookup: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.mode = mode
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.allowed_tools = allowed_tools
        # Where a tool's risk class comes from. Without this the policy falls
        # back to hardcoded name sets, which cannot classify bridged MCP tools.
        self.risk_lookup = risk_lookup

    def risk_for(self, tool_name: str) -> str:
        if self.risk_lookup is not None:
            return self.risk_lookup(tool_name)
        if tool_name in ESCALATED_TOOLS:
            return RISK_ESCALATE
        if tool_name in WRITE_TOOLS:
            return RISK_WRITE
        return RISK_READ

    def authorize_with_context(
        self,
        tool_name: str,
        effective_input: dict[str, Any],
        permission_context: PermissionContext,
        prompter: Optional[PermissionPrompter] = None,
    ) -> PermissionOutcome:
        # 1. A hook override outranks everything below.
        override = permission_context.permission_override
        if override == "deny":
            return PermissionOutcome.Deny(
                permission_context.permission_reason or "denied by hook override",
                SOURCE_HOOK_OVERRIDE,
            )
        if override == "allow":
            return PermissionOutcome.Allow(SOURCE_HOOK_OVERRIDE)

        # 2. An explicit allowlist, if configured.
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return PermissionOutcome.Deny(
                f"`{tool_name}` is not in the allowed tool list", SOURCE_ALLOWLIST
            )

        # 3. Mode.
        if self.mode in (PermissionMode.ALLOW, PermissionMode.DANGER_FULL_ACCESS):
            return PermissionOutcome.Allow(SOURCE_MODE)

        risk = self.risk_for(tool_name)

        if self.mode is PermissionMode.READ_ONLY:
            if risk != RISK_READ:
                return PermissionOutcome.Deny(
                    f"`{tool_name}` mutates state and the session is read-only",
                    SOURCE_MODE,
                )
            return PermissionOutcome.Allow(SOURCE_MODE)

        if self.mode is PermissionMode.PROMPT:
            return self._escalate(tool_name, effective_input, prompter)

        # workspace-write
        if risk == RISK_ESCALATE:
            return self._escalate(tool_name, effective_input, prompter)
        if risk == RISK_WRITE:
            target = effective_input.get("path", "")
            # A write tool with no path argument cannot be workspace-checked.
            if not target:
                return PermissionOutcome.Allow(SOURCE_MODE)
            if not self._inside_workspace(target):
                return PermissionOutcome.Deny(
                    f"`{target}` is outside the workspace root {self.workspace_root}",
                    SOURCE_WORKSPACE,
                )
        return PermissionOutcome.Allow(SOURCE_MODE)

    def _escalate(
        self,
        tool_name: str,
        effective_input: dict[str, Any],
        prompter: Optional[PermissionPrompter],
    ) -> PermissionOutcome:
        # No prompter means non-interactive: deny rather than silently allow.
        if prompter is None:
            return PermissionOutcome.Deny(
                f"`{tool_name}` needs approval but no prompter is attached",
                SOURCE_NO_PROMPTER,
            )
        if prompter.confirm(tool_name, effective_input):
            return PermissionOutcome.Allow(SOURCE_PROMPTER)
        return PermissionOutcome.Deny(f"user declined `{tool_name}`", SOURCE_PROMPTER)

    def _inside_workspace(self, target: str) -> bool:
        if not target:
            return False
        try:
            resolved = Path(target).expanduser().resolve()
        except OSError:
            return False
        return self.workspace_root == resolved or self.workspace_root in resolved.parents
