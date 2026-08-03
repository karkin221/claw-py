"""Subagents.

Mirrors `execute_agent` / `build_agent_system_prompt` / `allowed_tools_for_subagent`
in `tools/src/lib.rs`.

A subagent is not a special mechanism. It is an ordinary tool whose handler
builds a nested `ConversationRuntime` with:

  * its own system prompt, scoped to one job
  * a restricted `allowed_tools` set enforced by the permission policy
  * a permission mode that can only ever be *narrower* than the parent's
  * `prompter=None`, so it can never escalate to the human
  * its own fresh `Session`, so its intermediate steps never enter parent context

That last point is the reason subagents exist at all. An `explore` agent may
burn fifteen iterations grepping a codebase; the parent sees one paragraph of
findings, not fifteen tool results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .api import ApiClient
from .hooks import HookRegistry
from .permissions import PermissionMode, PermissionPolicy
from .telemetry import SessionTracer
from .tools import ToolError, ToolExecutor, ToolSpec, default_tool_executor
from .types import Session

# Narrower modes sort first. A subagent takes min(parent, spec).
MODE_RANK = {
    PermissionMode.READ_ONLY: 0,
    PermissionMode.PROMPT: 1,
    PermissionMode.WORKSPACE_WRITE: 2,
    PermissionMode.ALLOW: 3,
    PermissionMode.DANGER_FULL_ACCESS: 4,
}

READ_TOOLS = {"read_file", "glob_search", "grep_search"}


@dataclass(frozen=True)
class AgentSpec:
    """One subagent type: its prompt, its tools, its ceiling."""

    subagent_type: str
    purpose: str
    instructions: str
    allowed_tools: frozenset[str]
    mode: PermissionMode
    max_iterations: int = 8


AGENT_SPECS: dict[str, AgentSpec] = {
    "explore": AgentSpec(
        subagent_type="explore",
        purpose="Search a codebase and report what is there.",
        instructions=(
            "Locate the relevant files and report concretely: paths, symbol names, "
            "and how the pieces connect. Do not modify anything. Do not speculate "
            "about code you have not read."
        ),
        allowed_tools=frozenset(READ_TOOLS),
        mode=PermissionMode.READ_ONLY,
        max_iterations=10,
    ),
    "plan": AgentSpec(
        subagent_type="plan",
        purpose="Turn a goal into an ordered, concrete plan.",
        instructions=(
            "Read enough to ground the plan in the actual code. Produce numbered "
            "steps, each naming the file it touches. Flag anything you could not "
            "verify. Do not implement anything."
        ),
        allowed_tools=frozenset(READ_TOOLS | {"todo_write"}),
        mode=PermissionMode.READ_ONLY,
        max_iterations=8,
    ),
    "verification": AgentSpec(
        subagent_type="verification",
        purpose="Check whether a claim about the code actually holds.",
        instructions=(
            "Verify by reading files and running read-only commands such as tests "
            "or linters. Report pass or fail with the evidence that decided it. "
            "Never edit code to make a check pass."
        ),
        allowed_tools=frozenset(READ_TOOLS | {"bash"}),
        mode=PermissionMode.WORKSPACE_WRITE,
        max_iterations=10,
    ),
    "general-purpose": AgentSpec(
        subagent_type="general-purpose",
        purpose="Carry out a self-contained task end to end.",
        instructions=(
            "Complete the task, then report what you did and what remains. Prefer "
            "reading before writing. Stop as soon as the task is done."
        ),
        allowed_tools=frozenset(
            READ_TOOLS | {"write_file", "edit_file", "bash", "todo_write"}
        ),
        mode=PermissionMode.WORKSPACE_WRITE,
        max_iterations=12,
    ),
}

DEFAULT_SUBAGENT_TYPE = "general-purpose"


def normalize_subagent_type(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_SUBAGENT_TYPE
    candidate = value.strip().lower().replace("_", "-").replace(" ", "-")
    if candidate in AGENT_SPECS:
        return candidate
    for alias, target in (
        ("explorer", "explore"), ("search", "explore"), ("research", "explore"),
        ("planner", "plan"), ("planning", "plan"),
        ("verify", "verification"), ("verifier", "verification"), ("test", "verification"),
        ("general", "general-purpose"), ("generalpurpose", "general-purpose"),
    ):
        if candidate == alias:
            return target
    raise ToolError(
        f"unknown subagent_type `{value}`. Valid types: {', '.join(sorted(AGENT_SPECS))}"
    )


def allowed_tools_for_subagent(subagent_type: str) -> set[str]:
    return set(AGENT_SPECS[subagent_type].allowed_tools)


def build_agent_system_prompt(subagent_type: str, workspace_root: Path, tool_names: list[str]) -> str:
    spec = AGENT_SPECS[subagent_type]
    return (
        f"You are a `{spec.subagent_type}` subagent.\n\n"
        f"{spec.purpose}\n\n"
        f"{spec.instructions}\n\n"
        "You are running inside a parent agent's task. You cannot ask the user "
        "anything — there is nobody to ask. Work with what the tools give you.\n\n"
        "Your final message is the only thing the parent will see, so make it "
        "self-contained: state findings and conclusions, not a narration of your "
        "process. When you are done, answer in plain text and stop calling tools.\n\n"
        f"Tools available to you: {', '.join(tool_names)}\n"
        f"Working directory: {workspace_root}"
    )


def narrower_mode(parent: PermissionMode, spec: PermissionMode) -> PermissionMode:
    """A subagent never gets more authority than its parent."""
    return parent if MODE_RANK[parent] <= MODE_RANK[spec] else spec


@dataclass
class AgentConfig:
    """Everything a nested runtime needs, captured once at wiring time."""

    client_factory: Callable[[str], ApiClient]
    workspace_root: Path
    hook_registry: HookRegistry
    session_tracer: SessionTracer
    parent_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
    subagent_model: str = ""
    max_depth: int = 2
    extra_tools: list[ToolSpec] = field(default_factory=list)


AGENT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "A short label for this delegation, 3-6 words.",
        },
        "prompt": {
            "type": "string",
            "description": "The full self-contained task. The subagent sees nothing else.",
        },
        "subagent_type": {
            "type": "string",
            "enum": sorted(AGENT_SPECS),
            "description": "Which subagent to spawn.",
        },
    },
    "required": ["description", "prompt", "subagent_type"],
}

AGENT_TOOL_DESCRIPTION = (
    "Delegate a self-contained task to a subagent with its own context window. "
    "Use this when a task needs many exploratory steps whose intermediate output "
    "you do not need to see. The subagent cannot ask questions, so the `prompt` "
    "must contain everything it needs. You receive only its final report."
)


def build_tool_executor(config: AgentConfig, depth: int) -> ToolExecutor:
    """Tools for a runtime at `depth`. The agent tool is omitted at the ceiling."""
    executor = default_tool_executor()
    for spec in config.extra_tools:
        executor.register(spec)
    if depth < config.max_depth:
        executor.register(make_agent_tool(config, depth + 1))
    return executor


def make_agent_tool(config: AgentConfig, depth: int) -> ToolSpec:
    """The `agent` tool. `depth` is the depth of the runtime it will spawn."""
    return ToolSpec(
        name="agent",
        description=AGENT_TOOL_DESCRIPTION,
        input_schema=AGENT_INPUT_SCHEMA,
        handler=lambda input: execute_agent(input, config, depth),
    )


def execute_agent(input: dict[str, Any], config: AgentConfig, depth: int) -> str:
    """Run one nested turn and return only the subagent's final message."""
    from .conversation import ConversationRuntime  # deferred: avoids a cycle

    prompt = str(input.get("prompt", "")).strip()
    if not prompt:
        raise ToolError("`prompt` is required and must describe the whole task")

    description = str(input.get("description", "")).strip() or "delegated task"
    subagent_type = normalize_subagent_type(input.get("subagent_type"))
    spec = AGENT_SPECS[subagent_type]

    if depth > config.max_depth:
        raise ToolError(
            f"subagent depth limit reached ({config.max_depth}); do this task yourself"
        )

    agent_id = f"{subagent_type}-{uuid.uuid4().hex[:6]}"
    allowed_tools = allowed_tools_for_subagent(subagent_type)

    tool_executor = build_tool_executor(config, depth)
    if depth < config.max_depth:
        allowed_tools = allowed_tools | {"agent"}
    # Only offer the model tools it is actually permitted to call.
    offered = allowed_tools & set(tool_executor.names())

    api_client = config.client_factory(config.subagent_model)
    api_client.tool_specs = tool_executor.wire_specs(offered)

    session = Session(session_id=agent_id)
    tracer = config.session_tracer.child(agent_id)
    tracer.emit(
        "subagent_started",
        depth=depth,
        subagent_type=subagent_type,
        description=description,
        allowed_tools=sorted(offered),
    )

    runtime = ConversationRuntime(
        api_client=api_client,
        tool_executor=tool_executor,
        permission_policy=PermissionPolicy(
            mode=narrower_mode(config.parent_mode, spec.mode),
            workspace_root=config.workspace_root,
            allowed_tools=allowed_tools,
        ),
        system_prompt=build_agent_system_prompt(
            subagent_type, config.workspace_root, sorted(offered)
        ),
        session=session,
        hook_registry=config.hook_registry,  # subagents inherit every gate
        session_tracer=tracer,
        max_iterations=spec.max_iterations,
    )

    try:
        summary = runtime.run_turn(prompt, prompter=None)  # nobody to ask
    except Exception as error:  # noqa: BLE001 - surfaced to the parent as a tool error
        tracer.emit("subagent_failed", depth=depth, error=str(error))
        raise ToolError(f"subagent `{agent_id}` failed: {error}") from error

    report = summary.assistant_messages[-1].text().strip() if summary.assistant_messages else ""
    tracer.emit(
        "subagent_finished",
        depth=depth,
        iterations=summary.iterations,
        tool_results=len(summary.tool_results),
        report_chars=len(report),
    )

    if not report:
        return (
            f"[{agent_id}] finished after {summary.iterations} iteration(s) "
            "but produced no final report."
        )
    return (
        f"[{agent_id}] {description}\n"
        f"({summary.iterations} iteration(s), {len(summary.tool_results)} tool call(s))\n\n"
        f"{report}"
    )
