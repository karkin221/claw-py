"""CLI: REPL and one-shot prompt.

Mirrors `rusty-claude-cli`. Also registers a couple of demonstration hooks so
the PreToolUse rewrite / override / deny paths are visible in a real run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .agents import AGENT_SPECS, AgentConfig, build_tool_executor
from .api import DEFAULT_BASE_URL, DEFAULT_MODEL, ApiClient
from .compact import CompactionConfig
from .conversation import ConversationRuntime
from .hooks import HookEvent, HookRegistry, HookResult
from .permissions import ConsolePrompter, PermissionMode, PermissionPolicy
from .prompt import build_system_prompt
from .telemetry import SessionTracer
from .tools import default_tool_executor
from .types import RuntimeError, Session

DESTRUCTIVE_FRAGMENTS = ("rm -rf /", "mkfs", ":(){", "dd if=", "shutdown", "> /dev/sd")


# ---------------------------------------------------------------------------
# demonstration hooks
# ---------------------------------------------------------------------------


def block_destructive_bash(payload: dict[str, Any]) -> HookResult:
    """PreToolUse: veto a command the permission mode might otherwise allow."""
    if payload["tool_name"] != "bash":
        return HookResult.proceed()
    command = str(payload["input"].get("command", ""))
    for fragment in DESTRUCTIVE_FRAGMENTS:
        if fragment in command:
            return HookResult.deny(f"command contains blocked fragment `{fragment}`")
    return HookResult.proceed()


def autoscope_reads(payload: dict[str, Any]) -> HookResult:
    """PreToolUse: rewrite input, then hand the policy a pre-approval."""
    if payload["tool_name"] not in {"glob_search", "grep_search"}:
        return HookResult.proceed()
    input = payload["input"]
    if input.get("path"):
        return HookResult.proceed()
    return HookResult.rewrite(
        {**input, "path": "."}, message="defaulted search path to cwd"
    )


def flag_empty_results(payload: dict[str, Any]) -> HookResult:
    """PostToolUse: annotate an unhelpful-but-successful result."""
    if payload["output"].strip() in {"no matches", "(empty)", "(no output)"}:
        return HookResult.proceed("tool succeeded but returned nothing usable")
    return HookResult.proceed()


def default_hook_registry() -> HookRegistry:
    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, block_destructive_bash)
    registry.register(HookEvent.PRE_TOOL_USE, autoscope_reads)
    registry.register(HookEvent.POST_TOOL_USE, flag_empty_results)
    return registry


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def build_runtime(args: argparse.Namespace) -> ConversationRuntime:
    cwd = Path(args.cwd).expanduser().resolve()
    permission_mode = PermissionMode.from_str(args.permission_mode)
    session = Session()
    session_tracer = SessionTracer(
        session.session_id,
        path=Path(args.trace) if args.trace else None,
        echo=args.verbose,
    )
    hook_registry = default_hook_registry()

    def announce_route(role: str, model: str, attempt: int) -> None:
        if attempt > 0:
            print(f"  route `{role}` fell back to {model}", file=sys.stderr)

    # One factory, so parent and subagents are built the same way.
    if args.router == "litellm":
        from .routing import routed_client_factory

        client_factory = routed_client_factory(args.model, on_route=announce_route)
    else:
        def client_factory(model: str) -> ApiClient:
            return ApiClient(model=model or args.model, base_url=args.base_url)

    if args.no_subagents:
        tool_executor = default_tool_executor()
    else:
        agent_config = AgentConfig(
            client_factory=client_factory,
            workspace_root=cwd,
            hook_registry=hook_registry,
            session_tracer=session_tracer,
            parent_mode=permission_mode,
            subagent_model=args.subagent_model or args.model,
            max_depth=args.max_agent_depth,
        )
        tool_executor = build_tool_executor(agent_config, depth=0)

    system_prompt, memory_files = build_system_prompt(cwd, tool_executor.names())

    api_client = client_factory(args.model)
    api_client.tool_specs = tool_executor.wire_specs()

    permission_policy = PermissionPolicy(mode=permission_mode, workspace_root=cwd)

    if memory_files:
        loaded = ", ".join(str(m.path.name) for m in memory_files)
        print(f"  loaded project memory: {loaded}", file=sys.stderr)

    return ConversationRuntime(
        api_client=api_client,
        tool_executor=tool_executor,
        permission_policy=permission_policy,
        system_prompt=system_prompt,
        session=session,
        hook_registry=hook_registry,
        session_tracer=session_tracer,
        max_iterations=args.max_iterations,
        compaction_config=CompactionConfig(threshold_tokens=args.compact_threshold),
        on_text=lambda chunk: print(chunk, end="", flush=True),
    )


def render_summary(runtime: ConversationRuntime, summary: Any) -> None:
    usage = summary.usage
    line = (
        f"  [{summary.iterations} iteration(s), "
        f"{len(summary.tool_results)} tool call(s), "
        f"{usage.input_tokens} in / {usage.output_tokens} out, "
        f"~{runtime.estimated_tokens()} tok in session]"
    )
    if summary.auto_compaction is not None:
        line += f" compacted {summary.auto_compaction.dropped_messages} msg(s)"
    print(f"\n{line}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claw-py", description="minimal agent harness")
    parser.add_argument("prompt", nargs="*", help="one-shot prompt; omit for a REPL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--permission-mode",
        default="workspace-write",
        choices=[mode.value for mode in PermissionMode],
    )
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--compact-threshold", type=int, default=3000)
    parser.add_argument("--trace", default=None, help="write JSONL trace to this path")
    parser.add_argument(
        "--router",
        default="ollama",
        choices=["ollama", "litellm"],
        help="ollama: stdlib client. litellm: route across providers by role name.",
    )
    parser.add_argument(
        "--subagent-model",
        default=None,
        help="model (or litellm role) for subagents; defaults to --model",
    )
    parser.add_argument(
        "--max-agent-depth",
        type=int,
        default=2,
        help="how deep subagents may nest; 0 disables the agent tool",
    )
    parser.add_argument(
        "--no-subagents", action="store_true", help="do not register the agent tool"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="echo trace events")
    args = parser.parse_args(argv)

    runtime = build_runtime(args)
    prompter = ConsolePrompter()

    if args.prompt:
        try:
            summary = runtime.run_turn(" ".join(args.prompt), prompter)
        except RuntimeError as error:
            print(f"\nerror: {error.message}", file=sys.stderr)
            return 1
        render_summary(runtime, summary)
        return 0

    banner = f"claw-py · {args.model} · {args.permission_mode}"
    if not args.no_subagents and args.max_agent_depth > 0:
        banner += f" · subagents: {', '.join(sorted(AGENT_SPECS))}"
    print(f"{banner} · ctrl-d to exit")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        try:
            summary = runtime.run_turn(user_input, prompter)
        except RuntimeError as error:
            print(f"\nerror: {error.message}", file=sys.stderr)
            continue
        render_summary(runtime, summary)


if __name__ == "__main__":
    raise SystemExit(main())
