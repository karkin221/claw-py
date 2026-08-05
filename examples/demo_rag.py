"""The three retrieval seams, measured. No model, no external service.

Spins up `fake_rag_server.py` and runs the same question through:

  1. MODEL-DECIDED   rag_search as a tool; the model chooses when
  2. ALWAYS-ON       UserPromptSubmit hook; retrieved before the model is called
  3. DELEGATED       a research subagent; passages stay in its context

The last section is the point: the same retrieval costs very different amounts
depending on where it enters, because Stage 2 re-sends the whole history on
every iteration.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_rag_server import serve  # noqa: E402

from claw_py.agents import AgentConfig, build_tool_executor  # noqa: E402
from claw_py.api import ApiClient  # noqa: E402
from claw_py.compact import estimate_session_tokens  # noqa: E402
from claw_py.conversation import ConversationRuntime  # noqa: E402
from claw_py.hooks import HookEvent, HookRegistry, HookResult  # noqa: E402
from claw_py.permissions import PermissionPolicy  # noqa: E402
from claw_py.rag import RagClient, RagConfig, build_rag_tools, retrieve_for_prompt  # noqa: E402
from claw_py.telemetry import SessionTracer  # noqa: E402
from claw_py.tools import default_tool_executor  # noqa: E402
from claw_py.types import ApiRequest, Session  # noqa: E402

PORT = 8741
QUESTION = "what do the authors say about cartel discipline"


class ScriptedApiClient(ApiClient):
    def __init__(self, script, label="model") -> None:
        super().__init__(model=f"scripted-{label}", tool_specs=[])
        self.script = script
        self.turn = 0

    def stream(self, request: ApiRequest):
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {"type": "text_delta", "text": "The cartel's discipline has broken down."}
        for index, (name, input) in enumerate(calls, 1):
            yield {"type": "tool_use", "id": f"c{self.turn}_{index}", "name": name, "input": input}
        yield {"type": "usage", "input_tokens": 1, "output_tokens": 1}
        yield {"type": "message_stop"}

    def complete(self, system_prompt, user_text):
        return "summary"


def section(title: str) -> None:
    print(f"\n{'─' * 64}\n{title}\n{'─' * 64}")


def main() -> None:
    server = serve(PORT)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    client = RagClient(RagConfig(base_url=f"http://127.0.0.1:{PORT}", k=3))
    workspace = Path(tempfile.mkdtemp(prefix="claw-py-rag-"))
    rag_specs = build_rag_tools(client)
    results = {}

    # ------------------------------------------------------------------
    section("1. MODEL-DECIDED — rag_search as a tool")

    executor = default_tool_executor()
    for spec in rag_specs:
        executor.register(spec)
    session = Session()
    runtime = ConversationRuntime(
        api_client=ScriptedApiClient([[("rag_search", {"query": "cartel discipline"})], []]),
        tool_executor=executor,
        permission_policy=PermissionPolicy(
            workspace_root=workspace, risk_lookup=executor.risk_for
        ),
        system_prompt="(scripted)",
        session=session,
        session_tracer=SessionTracer(session.session_id),
    )
    summary = runtime.run_turn(QUESTION)
    results["model-decided"] = (summary.iterations, estimate_session_tokens(session))
    print(f"  iterations : {summary.iterations}  (one spent deciding to search)")
    print(f"  retrieved  : {len(summary.tool_results)} tool result in parent history")
    print(f"  risk class : {executor.risk_for('rag_search')}  → batches under --parallel-tools")

    # ------------------------------------------------------------------
    section("2. ALWAYS-ON — UserPromptSubmit hook")

    registry = HookRegistry()

    def retrieve(payload):
        context = retrieve_for_prompt(client, payload["user_input"], 3)
        return HookResult.with_context(context) if context else HookResult.proceed()

    registry.register(HookEvent.USER_PROMPT_SUBMIT, retrieve)

    session = Session()
    runtime = ConversationRuntime(
        api_client=ScriptedApiClient([[]]),
        tool_executor=default_tool_executor(),
        permission_policy=PermissionPolicy(workspace_root=workspace),
        system_prompt="(scripted)",
        session=session,
        hook_registry=registry,
        session_tracer=SessionTracer(session.session_id),
    )
    summary = runtime.run_turn(QUESTION)
    results["always-on"] = (summary.iterations, estimate_session_tokens(session))
    first = session.messages[0].text()
    print(f"  iterations : {summary.iterations}  (no round trip spent asking)")
    print(f"  prompt     : {len(QUESTION)} chars typed → {len(first)} chars sent")
    print(f"  fenced     : {'<retrieved_context' in first}")

    # ------------------------------------------------------------------
    section("3. DELEGATED — research subagent")

    sink: list = []

    class Collecting(SessionTracer):
        def __init__(self, session_id, s=sink):
            super().__init__(session_id)
            self.s = s

        def child(self, session_id):
            return Collecting(session_id, self.s)

        def emit(self, kind, **fields):
            self.s.append({"kind": kind, **fields})

    session = Session()
    tracer = Collecting(session.session_id)
    config = AgentConfig(
        client_factory=lambda model: ScriptedApiClient([
            [("rag_search", {"query": "cartel discipline"})],
            [("rag_doc", {"slug": "a-new-oil-era"})],
            [],
        ], "sub"),
        workspace_root=workspace,
        hook_registry=HookRegistry(),
        session_tracer=tracer,
        max_depth=1,
        extra_tools=rag_specs,
    )
    executor = build_tool_executor(config, depth=0)
    runtime = ConversationRuntime(
        api_client=ScriptedApiClient([
            [("agent", {"description": "corpus lookup", "prompt": QUESTION,
                        "subagent_type": "research"})],
            [],
        ]),
        tool_executor=executor,
        permission_policy=PermissionPolicy(workspace_root=workspace),
        system_prompt="(scripted)",
        session=session,
        session_tracer=tracer,
    )
    summary = runtime.run_turn(QUESTION)
    results["delegated"] = (summary.iterations, estimate_session_tokens(session))
    finished = [e for e in sink if e["kind"] == "subagent_finished"][0]
    started = [e for e in sink if e["kind"] == "subagent_started"][0]
    print(f"  subagent   : {finished['iterations']} iterations, "
          f"{finished['tool_results']} retrievals, tools={','.join(started['allowed_tools'])}")
    print(f"  parent sees: {len(summary.tool_results)} result, "
          f"{finished['report_chars']} chars")

    # ------------------------------------------------------------------
    section("WHAT EACH COSTS THE PARENT")

    print(f"  {'seam':<16}{'iterations':>12}{'session tokens':>16}")
    print("  " + "-" * 44)
    for name, (iterations, tokens) in results.items():
        print(f"  {name:<16}{iterations:>12}{tokens:>16}")
    print()
    print("  Stage 2 re-sends the whole history every iteration, so passages")
    print("  living in parent context are paid for on every later request.")
    print("  A research subagent burns them in a context it then discards.")

    server.shutdown()


if __name__ == "__main__":
    main()
