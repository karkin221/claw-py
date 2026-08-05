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
from .api import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_REQUEST_TIMEOUT,
    ApiClient,
)
from .compact import CompactionConfig
from .conversation import ConversationRuntime
from .mcp import McpError, McpServerManager, load_mcp_config
from .persistence import (
    SessionEnvironment,
    describe_environment_drift,
    format_session_list,
    list_sessions,
    replay_environment,
    replay_session,
)
from .hooks import HookEvent, HookRegistry, HookResult
from .permissions import ConsolePrompter, PermissionMode, PermissionPolicy
from .prompt import build_system_prompt
from .rag import RagClient, RagConfig, build_rag_tools, retrieve_for_prompt
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


def make_retrieval_hook(client: RagClient, k: int):
    """Always-on retrieval. Runs once per turn, before the prompt is pushed."""

    def hook(payload: dict[str, Any]) -> HookResult:
        context = retrieve_for_prompt(client, payload["user_input"], k)
        if not context:
            return HookResult.proceed()
        return HookResult.with_context(context, "retrieved corpus context")

    return hook


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
    recorded_environment = None
    if args.resume:
        if not args.trace:
            raise SystemExit("--resume needs --trace pointing at the trace file")
        wanted = None if args.resume == "last" else args.resume
        session = replay_session(Path(args.trace), wanted)
        recorded_environment = replay_environment(Path(args.trace), session.session_id)
        print(
            f"  resumed session {session.session_id} "
            f"({len(session.messages)} message(s))",
            file=sys.stderr,
        )
    else:
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
            return ApiClient(
                model=model or args.model,
                base_url=args.base_url,
                request_timeout=args.request_timeout,
                num_ctx=args.num_ctx,
            )

    # MCP servers start before tool assembly so their tools join the registry
    # like any other, and inherit the same hook and permission pipeline.
    # Retrieval tools register like any other, so they inherit the same gate.
    rag_specs: list = []
    rag_client = None
    if args.rag_url:
        rag_client = RagClient(RagConfig(base_url=args.rag_url, k=args.rag_k))
        reachable, reason = rag_client.check()
        if reachable:
            rag_specs = build_rag_tools(rag_client)
            print(
                f"  rag: connected to {rag_client.config.base_url}", file=sys.stderr
            )
            if args.rag_auto:
                hook_registry.register(
                    HookEvent.USER_PROMPT_SUBMIT,
                    make_retrieval_hook(rag_client, args.rag_k),
                )
                print("  rag: always-on retrieval enabled", file=sys.stderr)
        else:
            print(f"  rag: unavailable — {reason}", file=sys.stderr)
            print("  rag: continuing without retrieval", file=sys.stderr)
            rag_client = None

    mcp_specs: list = []
    mcp_manager = None
    if args.mcp_config:
        try:
            configs = load_mcp_config(Path(args.mcp_config).expanduser())
            mcp_manager = McpServerManager(configs)
            mcp_specs = mcp_manager.start_all()
            if mcp_specs:
                print(
                    f"  mcp: {len(mcp_specs)} tool(s) from "
                    f"{', '.join(sorted(mcp_manager.clients))}",
                    file=sys.stderr,
                )
            for name, reason in mcp_manager.failures.items():
                print(f"  mcp: server `{name}` unavailable ({reason})", file=sys.stderr)
        except (McpError, OSError, ValueError) as error:
            print(f"  mcp: {error}", file=sys.stderr)

    extra_tools = rag_specs + mcp_specs

    if args.no_subagents:
        tool_executor = default_tool_executor()
        for spec in extra_tools:
            tool_executor.register(spec)
    else:
        agent_config = AgentConfig(
            client_factory=client_factory,
            workspace_root=cwd,
            hook_registry=hook_registry,
            session_tracer=session_tracer,
            parent_mode=permission_mode,
            subagent_model=args.subagent_model or args.model,
            max_depth=args.max_agent_depth,
            extra_tools=extra_tools,  # subagents see these too
        )
        tool_executor = build_tool_executor(agent_config, depth=0)

    system_prompt, memory_files = build_system_prompt(cwd, tool_executor.names())

    # A resumed session must run under the prompt that produced its history,
    # or the model silently gets different instructions than it had before.
    if recorded_environment is not None:
        current_environment = SessionEnvironment(
            system_prompt=system_prompt,
            tool_names=tool_executor.names(),
            permission_mode=permission_mode.as_str(),
            workspace_root=str(cwd),
            model=args.model,
        )
        drift = describe_environment_drift(recorded_environment, current_environment)
        for note in drift:
            print(f"  drift: {note}", file=sys.stderr)
        if recorded_environment.system_prompt and not args.rebuild_prompt:
            if system_prompt != recorded_environment.system_prompt:
                print(
                    "  using the recorded system prompt "
                    "(pass --rebuild-prompt to use the current one)",
                    file=sys.stderr,
                )
            system_prompt = recorded_environment.system_prompt
    elif args.resume:
        print(
            "  note: this trace predates session_started; "
            "the system prompt was rebuilt from the current directory",
            file=sys.stderr,
        )

    api_client = client_factory(args.model)
    api_client.tool_specs = tool_executor.wire_specs()

    permission_policy = PermissionPolicy(
        mode=permission_mode,
        workspace_root=cwd,
        # Classify by declared risk, not by name. Bridged MCP tools default to
        # `escalate`, so an unfamiliar remote tool is never auto-allowed.
        risk_lookup=tool_executor.risk_for,
    )

    if memory_files:
        loaded = ", ".join(str(m.path.name) for m in memory_files)
        print(f"  loaded project memory: {loaded}", file=sys.stderr)

    runtime = ConversationRuntime(
        api_client=api_client,
        tool_executor=tool_executor,
        permission_policy=permission_policy,
        system_prompt=system_prompt,
        session=session,
        hook_registry=hook_registry,
        session_tracer=session_tracer,
        max_iterations=args.max_iterations,
        compaction_config=CompactionConfig(threshold_tokens=args.compact_threshold),
        parallel_tools=args.parallel_tools,
        on_text=lambda chunk: print(chunk, end="", flush=True),
    )
    runtime.mcp_manager = mcp_manager  # so main() can shut it down
    return runtime


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


def report_error(error: RuntimeError) -> None:
    print(f"\nerror: {error.message}", file=sys.stderr)
    partial = getattr(error, "partial_text", "")
    if partial:
        print(
            f"\n[the model had produced {len(partial)} characters before this; "
            "shown above and recorded in the trace]",
            file=sys.stderr,
        )


def shutdown(runtime: ConversationRuntime) -> None:
    manager = getattr(runtime, "mcp_manager", None)
    if manager is not None:
        manager.stop_all()
    runtime.session_tracer.close()


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
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="seconds to wait on one provider request (default %(default)s)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="model context window (default %(default)s). Ollama defaults to "
        "4096 and truncates silently past it, so this is set explicitly.",
    )
    parser.add_argument(
        "--compact-threshold",
        type=int,
        default=None,
        help="compact above this many tokens (default: half of --num-ctx)",
    )
    parser.add_argument("--trace", default=None, help="write JSONL trace to this path")
    parser.add_argument(
        "--mcp-config", default=None, help="path to a .mcp.json server config"
    )
    parser.add_argument(
        "--rag-url",
        nargs="?",
        const="http://127.0.0.1:8000",
        default=None,
        metavar="URL",
        help="register rag_search/rag_doc against a retrieval service",
    )
    parser.add_argument(
        "--rag-auto",
        action="store_true",
        help="retrieve before every turn instead of waiting for the model to ask",
    )
    parser.add_argument(
        "--rag-k", type=int, default=5, help="passages per retrieval (default 5)"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        metavar="SESSION_ID",
        help="resume a session by replaying --trace; omit the id for the latest",
    )
    parser.add_argument(
        "--list-sessions", action="store_true", help="list resumable sessions in --trace"
    )
    parser.add_argument(
        "--rebuild-prompt",
        action="store_true",
        help="on resume, use the current system prompt instead of the recorded one",
    )
    parser.add_argument(
        "--parallel-tools",
        type=int,
        default=1,
        help="max concurrent read-only tool calls; 1 disables parallelism",
    )
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
    if args.compact_threshold is None:
        # Coupled deliberately: a threshold larger than the window means the
        # server truncates before compaction ever fires, and a tiny threshold
        # against a large window compacts for no reason.
        args.compact_threshold = max(2000, args.num_ctx // 2)

    if args.list_sessions:
        if not args.trace:
            raise SystemExit("--list-sessions needs --trace pointing at the trace file")
        print(format_session_list(list_sessions(Path(args.trace))))
        return 0

    runtime = build_runtime(args)
    prompter = ConsolePrompter()

    if args.prompt:
        try:
            summary = runtime.run_turn(" ".join(args.prompt), prompter)
        except RuntimeError as error:
            report_error(error)
            return 1
        finally:
            shutdown(runtime)
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
            shutdown(runtime)
            return 0
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            shutdown(runtime)
            return 0
        try:
            summary = runtime.run_turn(user_input, prompter)
        except RuntimeError as error:
            report_error(error)
            continue
        render_summary(runtime, summary)


if __name__ == "__main__":
    raise SystemExit(main())
