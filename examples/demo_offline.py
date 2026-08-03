"""Run the loop with a scripted provider — no Ollama, no model, no network.

Exercises each branch of the tool pipeline in one turn:
  1. read_file          -> allowed, succeeds
  2. grep_search        -> PreToolUse hook rewrites the input
  3. bash rm -rf /      -> PreToolUse hook denies before the policy runs
  4. write_file to /etc -> policy denies (outside workspace root)
  5. read_file missing  -> tool raises, returned as an is_error result
  6. (no tool calls)    -> loop breaks, turn summary returned
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claw_py.api import ApiClient  # noqa: E402
from claw_py.cli import default_hook_registry  # noqa: E402
from claw_py.conversation import ConversationRuntime  # noqa: E402
from claw_py.permissions import PermissionMode, PermissionPolicy  # noqa: E402
from claw_py.telemetry import SessionTracer  # noqa: E402
from claw_py.tools import default_tool_executor  # noqa: E402
from claw_py.types import ApiRequest, Session  # noqa: E402

SCRIPT = [
    [("read_file", {"path": "PLACEHOLDER/notes.txt"})],
    [("grep_search", {"pattern": "beta"})],
    [("bash", {"command": "rm -rf / --no-preserve-root"})],
    [("write_file", {"path": "/etc/claw-should-not-write", "content": "nope"})],
    [("read_file", {"path": "PLACEHOLDER/does-not-exist.txt"})],
    [],
]


class ScriptedApiClient(ApiClient):
    """Replays a fixed sequence of tool calls in place of a real model."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(model="scripted", tool_specs=[])
        self.workspace = workspace
        self.turn = 0

    def stream(self, request: ApiRequest):
        calls = SCRIPT[min(self.turn, len(SCRIPT) - 1)]
        self.turn += 1

        if not calls:
            yield {"type": "text_delta", "text": "Done. Five tool calls attempted."}
        else:
            yield {"type": "text_delta", "text": f"Step {self.turn}: calling tools."}

        for index, (name, input) in enumerate(calls, 1):
            resolved = {
                key: str(value).replace("PLACEHOLDER", str(self.workspace))
                for key, value in input.items()
            }
            yield {
                "type": "tool_use",
                "id": f"call_{self.turn}_{index}",
                "name": name,
                "input": resolved,
            }

        yield {"type": "usage", "input_tokens": 120, "output_tokens": 40}
        yield {"type": "message_stop"}

    def complete(self, system_prompt: str, user_text: str) -> str:
        return "(scripted summary)"


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="claw-py-"))
    (workspace / "notes.txt").write_text("alpha\nbeta\ngamma\n")

    session = Session()
    runtime = ConversationRuntime(
        api_client=ScriptedApiClient(workspace),
        tool_executor=default_tool_executor(),
        permission_policy=PermissionPolicy(
            mode=PermissionMode.WORKSPACE_WRITE, workspace_root=workspace
        ),
        system_prompt="(scripted)",
        session=session,
        hook_registry=default_hook_registry(),
        session_tracer=SessionTracer(session.session_id, echo=False),
        max_iterations=10,
    )

    summary = runtime.run_turn("Inspect the workspace.", prompter=None)

    print(f"iterations       : {summary.iterations}")
    print(f"assistant msgs   : {len(summary.assistant_messages)}")
    print(f"tool results     : {len(summary.tool_results)}")
    print(f"usage            : {summary.usage}")
    print(f"session messages : {len(runtime.session.messages)}")
    print(f"estimated tokens : {runtime.estimated_tokens()}\n")

    for result in summary.tool_results:
        flag = "ERR " if result.is_error else "ok  "
        body = result.text().replace("\n", " ⏎ ")
        print(f"{flag}{result.tool_name:<12} {body[:110]}")


if __name__ == "__main__":
    main()
