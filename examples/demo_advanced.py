"""MCP tools, parallel dispatch, and resume — all offline, no model needed.

  1. MCP        a real stdio server subprocess; its tools join the same
                registry and hit the same permission gate as built-ins
  2. PARALLEL   four read-only tools, sequential vs concurrent, timed
  3. RESUME     kill the session, rebuild it by replaying the trace JSONL
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claw_py.api import ApiClient  # noqa: E402
from claw_py.conversation import ConversationRuntime  # noqa: E402
from claw_py.mcp import McpServerConfig, McpServerManager  # noqa: E402
from claw_py.permissions import ConsolePrompter, PermissionMode, PermissionPolicy  # noqa: E402
from claw_py.persistence import (  # noqa: E402
    describe_environment_drift,
    list_sessions,
    replay_environment,
    replay_session,
)
from claw_py.persistence import SessionEnvironment  # noqa: E402
from claw_py.telemetry import SessionTracer  # noqa: E402
from claw_py.tools import RISK_READ, ToolSpec, default_tool_executor  # noqa: E402
from claw_py.types import ApiRequest, Session  # noqa: E402

SERVER = Path(__file__).with_name("mcp_echo_server.py")


class ScriptedApiClient(ApiClient):
    def __init__(self, script) -> None:
        super().__init__(model="scripted", tool_specs=[])
        self.script = script
        self.turn = 0

    def stream(self, request: ApiRequest):
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {"type": "text_delta", "text": f"step {self.turn}"}
        for index, (name, input) in enumerate(calls, 1):
            yield {"type": "tool_use", "id": f"c{self.turn}_{index}", "name": name, "input": input}
        yield {"type": "usage", "input_tokens": 10, "output_tokens": 4}
        yield {"type": "message_stop"}

    def complete(self, system_prompt, user_text):
        return "scripted summary"


class AutoApprove(ConsolePrompter):
    """Stands in for a human at the terminal."""

    def confirm(self, tool_name, effective_input):
        return True


def slow_read(delay: float = 0.4) -> ToolSpec:
    """A read tool with latency, so parallelism is measurable."""

    def handler(input):
        time.sleep(delay)
        return f"read {input.get('path', '?')}"

    return ToolSpec(
        name="slow_read",
        description="A slow read.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=handler,
        risk=RISK_READ,
    )


def section(title: str) -> None:
    print(f"\n{'─' * 62}\n{title}\n{'─' * 62}")


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="claw-py-final-"))
    trace_path = workspace / "trace.jsonl"

    # ------------------------------------------------------------------
    section("1. MCP — a real subprocess, bridged into the same registry")

    config = McpServerConfig(name="echo", command=sys.executable, args=[str(SERVER)])
    with McpServerManager([config]) as manager:
        client = manager.clients["echo"]
        from claw_py.mcp import bridge_mcp_tool

        bridged = [bridge_mcp_tool(client, t) for t in client.list_tools()]
        print(f"  server     : {client.server_info.get('name')} "
              f"v{client.server_info.get('version')}")
        print(f"  bridged    : {', '.join(s.name for s in bridged)}")

        tool_executor = default_tool_executor()
        for spec in bridged:
            tool_executor.register(spec)

        session = Session()
        tracer = SessionTracer(session.session_id, path=trace_path)
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([
                [("mcp__echo__word_count", {"text": "the quick brown fox jumps"})],
                [("mcp__echo__always_fails", {})],
                [],
            ]),
            tool_executor=tool_executor,
            permission_policy=PermissionPolicy(
                mode=PermissionMode.WORKSPACE_WRITE,
                workspace_root=workspace,
                risk_lookup=tool_executor.risk_for,
            ),
            system_prompt="(scripted)",
            session=session,
            session_tracer=tracer,
        )

        print(f"  risk class : {tool_executor.risk_for('mcp__echo__word_count')} "
              "(remote tools are never auto-allowed)")

        # No prompter: the escalation is refused because nobody is watching.
        denied = runtime.run_turn("count some words", prompter=None)
        print(f"  no prompter: {denied.tool_results[0].text()[:70]}")

        runtime.api_client.turn = 0  # replay the script for the approved run
        approved = runtime.run_turn("count some words", prompter=AutoApprove())
        print("  with prompter:")
        for result in approved.tool_results:
            flag = "ERR" if result.is_error else "ok "
            print(f"  {flag} {result.tool_name:<24} {result.text()[:40]}")

    # ------------------------------------------------------------------
    section("2. PARALLEL — same four reads, sequential vs concurrent")

    four_reads = [
        [("slow_read", {"path": f"file{i}.txt"}) for i in range(1, 5)],
        [],
    ]

    for parallel_tools in (1, 4):
        executor = default_tool_executor()
        executor.register(slow_read())
        session = Session()
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient(four_reads),
            tool_executor=executor,
            permission_policy=PermissionPolicy(workspace_root=workspace),
            system_prompt="(scripted)",
            session=session,
            session_tracer=SessionTracer(session.session_id),
            parallel_tools=parallel_tools,
        )
        started = time.perf_counter()
        summary = runtime.run_turn("read all four")
        elapsed = time.perf_counter() - started
        label = "sequential" if parallel_tools == 1 else f"parallel ({parallel_tools})"
        order = " → ".join(r.text().split()[-1] for r in summary.tool_results)
        print(f"  {label:<15} {elapsed:5.2f}s   results in order: {order}")

    print("\n  Results keep request order regardless. Only idempotent reads")
    print("  batch; a write never overtakes a read requested before it.")

    # ------------------------------------------------------------------
    section("3. RESUME — rebuild the session by replaying the trace")

    sessions = list_sessions(trace_path)
    print(f"  sessions in {trace_path.name}:")
    for info in sessions:
        print(f"    {info.session_id}  turns={info.turns}  msgs={info.messages}"
              f"  first={info.first_prompt[:32]!r}")

    original_id = sessions[0].session_id
    resumed = replay_session(trace_path, original_id)
    recorded = replay_environment(trace_path, original_id)
    drifted = SessionEnvironment(
        system_prompt="(scripted) plus a CLAUDE.md the user added later",
        tool_names=recorded.tool_names,
        permission_mode=recorded.permission_mode,
        workspace_root=recorded.workspace_root,
        model=recorded.model,
    )
    print(f"\n  replayed   : {resumed.session_id} → {len(resumed.messages)} messages")
    print(f"  prompt     : restored {len(resumed.system_prompt or '')} chars from the trace")
    for note in describe_environment_drift(recorded, drifted):
        print(f"  drift      : {note}")
    print("  roles      :", " ".join(m.role[:4] for m in resumed.messages))
    print(f"  last text  : {resumed.messages[-1].text()[:52]!r}")

    # The resumed session is a normal Session: keep talking to it.
    executor = default_tool_executor()
    continued = ConversationRuntime(
        api_client=ScriptedApiClient([[]]),
        tool_executor=executor,
        permission_policy=PermissionPolicy(workspace_root=workspace),
        system_prompt="(scripted)",
        session=resumed,
        session_tracer=SessionTracer(resumed.session_id),
    )
    summary = continued.run_turn("and now continue")
    print(f"  continued  : {len(continued.session.messages)} messages after one more turn")


if __name__ == "__main__":
    main()
