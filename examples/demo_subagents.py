"""Subagent delegation with a scripted provider — no model, no network.

Shows the four properties that make subagents worth having:

  1. CONTEXT ISOLATION  the explore agent burns 3 iterations; the parent's
                        history gains one paragraph, not three tool results
  2. TOOL RESTRICTION   the explore agent is denied write_file even though the
                        parent is running in workspace-write
  3. HOOK INHERITANCE   the parent's PreToolUse hooks still gate the subagent
  4. DEPTH LIMITING     at max_depth the agent tool is not registered at all
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claw_py.agents import AgentConfig, build_tool_executor  # noqa: E402
from claw_py.api import ApiClient  # noqa: E402
from claw_py.cli import default_hook_registry  # noqa: E402
from claw_py.conversation import ConversationRuntime  # noqa: E402
from claw_py.permissions import PermissionMode, PermissionPolicy  # noqa: E402
from claw_py.telemetry import SessionTracer  # noqa: E402
from claw_py.types import ApiRequest, Session  # noqa: E402

# Scripts keyed by which runtime is asking: the parent, or a subagent type.
PARENT_SCRIPT = [
    [("agent", {
        "description": "map the package",
        "subagent_type": "explore",
        "prompt": "List the python modules here and say what each one is for.",
    })],
    [("agent", {
        "description": "try to write from a read-only agent",
        "subagent_type": "explore",
        "prompt": "Write a file called notes.md.",
    })],
    [],
]

EXPLORE_SCRIPTS = [
    # first delegation: three iterations of real work, one paragraph out
    [
        [("glob_search", {"pattern": "*.py"})],
        [("read_file", {"path": "PLACEHOLDER/alpha.py"})],
        [("grep_search", {"pattern": "def "})],
        [],
    ],
    # second delegation: tries to write, gets refused by the allowlist
    [
        [("write_file", {"path": "PLACEHOLDER/notes.md", "content": "hi"})],
        [],
    ],
]


class ScriptedApiClient(ApiClient):
    def __init__(self, script, workspace, label) -> None:
        super().__init__(model=f"scripted-{label}", tool_specs=[])
        self.script = script
        self.workspace = workspace
        self.label = label
        self.turn = 0

    def stream(self, request: ApiRequest):
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {
            "type": "text_delta",
            "text": (
                f"{self.label} finished: 3 modules, 7 functions."
                if not calls
                else f"{self.label} step {self.turn}"
            ),
        }
        for index, (name, input) in enumerate(calls, 1):
            resolved = {
                key: (str(value).replace("PLACEHOLDER", str(self.workspace)))
                for key, value in input.items()
            }
            if name == "agent":
                resolved = input
            yield {
                "type": "tool_use",
                "id": f"{self.label}_{self.turn}_{index}",
                "name": name,
                "input": resolved,
            }
        yield {"type": "usage", "input_tokens": 50, "output_tokens": 20}
        yield {"type": "message_stop"}

    def complete(self, system_prompt: str, user_text: str) -> str:
        return "(scripted summary)"


class CapturingTracer(SessionTracer):
    """Records events so the demo can show what happened inside a subagent."""

    def __init__(self, session_id: str, sink: list | None = None) -> None:
        super().__init__(session_id)
        self.sink = sink if sink is not None else []

    def child(self, session_id: str) -> "CapturingTracer":
        return CapturingTracer(session_id, self.sink)

    def emit(self, kind: str, **fields) -> None:
        self.sink.append({"session_id": self.session_id, "kind": kind, **fields})


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="claw-py-agents-"))
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (workspace / name).write_text("def hello():\n    return 1\n")

    delegations = {"n": 0}

    def client_factory(model: str) -> ApiClient:
        script = EXPLORE_SCRIPTS[min(delegations["n"], len(EXPLORE_SCRIPTS) - 1)]
        delegations["n"] += 1
        return ScriptedApiClient(script, workspace, f"explore#{delegations['n']}")

    session = Session()
    tracer = CapturingTracer(session.session_id)
    hook_registry = default_hook_registry()

    config = AgentConfig(
        client_factory=client_factory,
        workspace_root=workspace,
        hook_registry=hook_registry,
        session_tracer=tracer,
        parent_mode=PermissionMode.WORKSPACE_WRITE,
        max_depth=1,
    )
    tool_executor = build_tool_executor(config, depth=0)

    runtime = ConversationRuntime(
        api_client=ScriptedApiClient(PARENT_SCRIPT, workspace, "parent"),
        tool_executor=tool_executor,
        permission_policy=PermissionPolicy(
            mode=PermissionMode.WORKSPACE_WRITE, workspace_root=workspace
        ),
        system_prompt="(scripted parent)",
        session=session,
        hook_registry=hook_registry,
        session_tracer=tracer,
    )

    summary = runtime.run_turn("Understand this package, then try to document it.")

    print("PARENT")
    print(f"  iterations       : {summary.iterations}")
    print(f"  tool results     : {len(summary.tool_results)}")
    print(f"  session messages : {len(runtime.session.messages)}")
    print(f"  parent tools     : {', '.join(tool_executor.names())}\n")

    for result in summary.tool_results:
        flag = "ERR " if result.is_error else "ok  "
        body = result.text().replace("\n", " ⏎ ")
        print(f"{flag}{result.tool_name:<7} {body[:150]}")

    print("\nINSIDE THE SUBAGENTS (from the trace)")
    for event in tracer.sink:
        if event["kind"] == "subagent_started":
            print(
                f"  {event['session_id']}  spawned at depth {event['depth']}"
                f"  tools={','.join(event['allowed_tools'])}"
            )
        elif event["kind"] == "tool_finished" and event["session_id"] != session.session_id:
            flag = "DENIED " if event["is_error"] else "ok     "
            print(f"  {event['session_id']}  {flag}{event['tool_name']}")

    print("\nDEPTH LIMIT")
    nested = build_tool_executor(config, depth=1)
    print(f"  tools at depth 0 : {'agent' in tool_executor.names()}")
    print(f"  tools at depth 1 : {'agent' in nested.names()}   (max_depth=1)")


if __name__ == "__main__":
    main()
