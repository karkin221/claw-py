"""Tests for MCP bridging, trace-replay persistence, and parallel dispatch.

The MCP tests spawn `examples/mcp_echo_server.py` as a real subprocess, so the
JSON-RPC transport is exercised rather than mocked. Still no network.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from claw_py.conversation import ConversationRuntime
from claw_py.mcp import (
    McpError,
    McpServerConfig,
    McpServerManager,
    bridge_mcp_tool,
    flatten_content,
    load_mcp_config,
    normalize_schema,
)
from claw_py.hooks import HookEvent, HookRegistry, HookResult
from claw_py.permissions import PermissionMode, PermissionPolicy
from claw_py.persistence import (
    deserialize_message,
    list_sessions,
    replay_session,
    serialize_message,
)
from claw_py.telemetry import SessionTracer
from claw_py.tools import (
    RISK_ESCALATE,
    RISK_READ,
    RISK_WRITE,
    ToolError,
    ToolSpec,
    default_tool_executor,
)
from claw_py.types import ApiRequest, ContentBlock, ConversationMessage, RuntimeError, Session

SERVER = Path(__file__).resolve().parent.parent / "examples" / "mcp_echo_server.py"


class ScriptedApiClient:
    """Minimal stand-in; only `stream` and `complete` are used by the runtime."""

    def __init__(self, script) -> None:
        self.script = script
        self.turn = 0
        self.model = "scripted"
        self.tool_specs: list = []

    def stream(self, request: ApiRequest):
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {"type": "text_delta", "text": f"step {self.turn}"}
        for index, (name, input) in enumerate(calls, 1):
            yield {"type": "tool_use", "id": f"c{self.turn}_{index}", "name": name, "input": input}
        yield {"type": "usage", "input_tokens": 3, "output_tokens": 1}
        yield {"type": "message_stop"}

    def complete(self, system_prompt, user_text):
        return "scripted summary"


class AlwaysApprove:
    def confirm(self, tool_name, effective_input):
        return True


def make_runtime(script, workspace, executor=None, **kw):
    executor = executor or default_tool_executor()
    session = kw.pop("session", None) or Session()
    return ConversationRuntime(
        api_client=ScriptedApiClient(script),
        tool_executor=executor,
        permission_policy=kw.pop("policy", None)
        or PermissionPolicy(workspace_root=workspace, risk_lookup=executor.risk_for),
        system_prompt="(test)",
        session=session,
        session_tracer=kw.pop("tracer", None) or SessionTracer(session.session_id),
        **kw,
    )


# ---------------------------------------------------------------------------
# risk classification
# ---------------------------------------------------------------------------


class RiskTests(unittest.TestCase):
    def test_builtins_are_classified(self) -> None:
        executor = default_tool_executor()
        self.assertEqual(executor.risk_for("read_file"), RISK_READ)
        self.assertEqual(executor.risk_for("write_file"), RISK_WRITE)
        self.assertEqual(executor.risk_for("bash"), RISK_ESCALATE)

    def test_unknown_tools_are_treated_as_dangerous(self) -> None:
        self.assertEqual(default_tool_executor().risk_for("mystery"), RISK_ESCALATE)

    def test_policy_uses_risk_lookup_over_name_sets(self) -> None:
        policy = PermissionPolicy(
            mode=PermissionMode.READ_ONLY,
            risk_lookup=lambda name: RISK_WRITE if name == "harmless_sounding" else RISK_READ,
        )
        from claw_py.permissions import PermissionContext

        outcome = policy.authorize_with_context(
            "harmless_sounding", {}, PermissionContext()
        )
        self.assertFalse(outcome.allowed)

    def test_policy_falls_back_to_name_sets_without_lookup(self) -> None:
        from claw_py.permissions import PermissionContext

        policy = PermissionPolicy(mode=PermissionMode.READ_ONLY)
        self.assertFalse(
            policy.authorize_with_context("write_file", {}, PermissionContext()).allowed
        )
        self.assertTrue(
            policy.authorize_with_context("read_file", {}, PermissionContext()).allowed
        )


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


class McpTransportTests(unittest.TestCase):
    """Runs against a real server subprocess."""

    def setUp(self) -> None:
        self.config = McpServerConfig(name="echo", command=sys.executable, args=[str(SERVER)])
        self.manager = McpServerManager([self.config])
        self.specs = self.manager.start_all()
        self.addCleanup(self.manager.stop_all)

    def test_handshake_reports_server_info(self) -> None:
        client = self.manager.clients["echo"]
        self.assertEqual(client.server_info.get("name"), "echo-server")

    def test_tools_are_listed_and_namespaced(self) -> None:
        names = sorted(spec.name for spec in self.specs)
        self.assertEqual(
            names,
            ["mcp__echo__always_fails", "mcp__echo__reverse", "mcp__echo__word_count"],
        )

    def test_bridged_tool_executes(self) -> None:
        executor = default_tool_executor()
        for spec in self.specs:
            executor.register(spec)
        self.assertEqual(
            executor.execute("mcp__echo__word_count", {"text": "a b c"}), "3 words"
        )
        self.assertEqual(
            executor.execute("mcp__echo__reverse", {"text": "abc"}), "cba"
        )

    def test_server_reported_error_becomes_tool_error(self) -> None:
        executor = default_tool_executor()
        for spec in self.specs:
            executor.register(spec)
        with self.assertRaises(ToolError):
            executor.execute("mcp__echo__always_fails", {})

    def test_bridged_tools_default_to_escalate(self) -> None:
        for spec in self.specs:
            self.assertEqual(spec.risk, RISK_ESCALATE)

    def test_concurrent_calls_do_not_interleave_on_the_wire(self) -> None:
        """The client lock must keep request/response pairs matched."""
        client = self.manager.clients["echo"]
        results: dict[int, str] = {}

        def call(index: int) -> None:
            results[index] = client.call_tool("reverse", {"text": str(index) * 4})

        threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 8)
        for index, value in results.items():
            self.assertEqual(value, str(index) * 4)

    def test_unknown_method_raises_mcp_error(self) -> None:
        with self.assertRaises(McpError):
            self.manager.clients["echo"].request("nonexistent/method", {})


class McpPipelineTests(unittest.TestCase):
    """A bridged tool must pass through the same gate as a built-in."""

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-mcp-"))
        self.manager = McpServerManager(
            [McpServerConfig(name="echo", command=sys.executable, args=[str(SERVER)])]
        )
        specs = self.manager.start_all()
        self.addCleanup(self.manager.stop_all)
        self.executor = default_tool_executor()
        for spec in specs:
            self.executor.register(spec)

    def test_mcp_tool_is_denied_without_a_prompter(self) -> None:
        runtime = make_runtime(
            [[("mcp__echo__word_count", {"text": "a b"})], []],
            self.workspace,
            executor=self.executor,
        )
        summary = runtime.run_turn("count", prompter=None)
        self.assertTrue(summary.tool_results[0].is_error)
        self.assertIn("needs approval", summary.tool_results[0].text())

    def test_mcp_tool_runs_once_approved(self) -> None:
        runtime = make_runtime(
            [[("mcp__echo__word_count", {"text": "a b"})], []],
            self.workspace,
            executor=self.executor,
        )
        summary = runtime.run_turn("count", prompter=AlwaysApprove())
        self.assertFalse(summary.tool_results[0].is_error)
        self.assertIn("2 words", summary.tool_results[0].text())

    def test_hooks_gate_mcp_tools_too(self) -> None:
        from claw_py.hooks import HookEvent, HookRegistry, HookResult

        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, lambda p: HookResult.deny("no remote tools"))
        runtime = make_runtime(
            [[("mcp__echo__word_count", {"text": "a b"})], []],
            self.workspace,
            executor=self.executor,
            hook_registry=registry,
        )
        summary = runtime.run_turn("count", prompter=AlwaysApprove())
        self.assertTrue(summary.tool_results[0].is_error)
        self.assertIn("no remote tools", summary.tool_results[0].text())


class McpHelperTests(unittest.TestCase):
    def test_flatten_text_blocks(self) -> None:
        self.assertEqual(
            flatten_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
            "a\nb",
        )

    def test_flatten_handles_resources_and_images(self) -> None:
        self.assertIn("resource", flatten_content([{"type": "resource", "resource": {"uri": "u"}}]))
        self.assertIn("image", flatten_content([{"type": "image", "mimeType": "image/png"}]))

    def test_normalize_schema_fills_defaults(self) -> None:
        self.assertEqual(normalize_schema({}), {"type": "object", "properties": {}})

    def test_config_file_is_parsed(self) -> None:
        path = Path(tempfile.mkdtemp()) / ".mcp.json"
        path.write_text('{"mcpServers": {"a": {"command": "echo", "args": ["hi"]}}}')
        configs = load_mcp_config(path)
        self.assertEqual(configs[0].name, "a")
        self.assertEqual(configs[0].args, ["hi"])

    def test_config_without_command_is_rejected(self) -> None:
        path = Path(tempfile.mkdtemp()) / ".mcp.json"
        path.write_text('{"mcpServers": {"a": {}}}')
        with self.assertRaises(McpError):
            load_mcp_config(path)

    def test_broken_server_degrades_instead_of_crashing(self) -> None:
        manager = McpServerManager(
            [McpServerConfig(name="nope", command="/nonexistent/binary")]
        )
        specs = manager.start_all()
        self.assertEqual(specs, [])
        self.assertIn("nope", manager.failures)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class SerializationTests(unittest.TestCase):
    def test_message_round_trips(self) -> None:
        original = ConversationMessage(
            role="assistant",
            blocks=[
                ContentBlock.text_block("hello"),
                ContentBlock.tool_use("id1", "read_file", {"path": "a.py"}),
            ],
        )
        restored = deserialize_message(serialize_message(original))
        self.assertEqual(restored.text(), "hello")
        self.assertEqual(restored.tool_uses()[0].input, {"path": "a.py"})

    def test_tool_result_round_trips(self) -> None:
        original = ConversationMessage.tool_result("id1", "bash", "boom", True)
        restored = deserialize_message(serialize_message(original))
        self.assertTrue(restored.is_error)
        self.assertEqual(restored.tool_name, "bash")


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-replay-"))
        (self.workspace / "a.txt").write_text("alpha\n")
        self.trace = self.workspace / "trace.jsonl"

    def _run(self, script, **kw):
        session = kw.pop("session", None) or Session()
        tracer = SessionTracer(session.session_id, path=self.trace)
        runtime = make_runtime(
            script, self.workspace, session=session, tracer=tracer, **kw
        )
        summary = runtime.run_turn("do the thing")
        tracer.close()
        return runtime, summary

    def test_replay_reproduces_the_live_session(self) -> None:
        runtime, _ = self._run([
            [("read_file", {"path": str(self.workspace / "a.txt")})],
            [],
        ])
        replayed = replay_session(self.trace, runtime.session.session_id)
        self.assertEqual(len(replayed.messages), len(runtime.session.messages))
        self.assertEqual(
            [m.role for m in replayed.messages],
            [m.role for m in runtime.session.messages],
        )
        self.assertEqual(
            [m.text() for m in replayed.messages],
            [m.text() for m in runtime.session.messages],
        )

    def test_replay_preserves_tool_error_flags(self) -> None:
        runtime, _ = self._run([
            [("read_file", {"path": str(self.workspace / "missing.txt")})],
            [],
        ])
        replayed = replay_session(self.trace, runtime.session.session_id)
        errors = [m for m in replayed.messages if m.is_error]
        self.assertEqual(len(errors), 1)

    def test_replay_reapplies_compaction_as_an_operation(self) -> None:
        from claw_py.compact import CompactionConfig

        session = Session()
        tracer = SessionTracer(session.session_id, path=self.trace)
        runtime = make_runtime(
            [[("read_file", {"path": str(self.workspace / "a.txt")})]] * 3 + [[]],
            self.workspace,
            session=session,
            tracer=tracer,
            compaction_config=CompactionConfig(threshold_tokens=1, keep_last=2),
        )
        runtime.run_turn("x" * 200)
        tracer.close()

        self.assertIsNotNone(runtime.session.compaction)
        replayed = replay_session(self.trace, session.session_id)
        self.assertEqual(len(replayed.messages), len(runtime.session.messages))
        self.assertIsNotNone(replayed.compaction)

    def test_resumed_session_continues_normally(self) -> None:
        runtime, _ = self._run([
            [("read_file", {"path": str(self.workspace / "a.txt")})],
            [],
        ])
        replayed = replay_session(self.trace, runtime.session.session_id)
        before = len(replayed.messages)
        continued = make_runtime([[]], self.workspace, session=replayed)
        continued.run_turn("carry on")
        self.assertGreater(len(continued.session.messages), before)

    def test_truncated_trailing_line_still_replays(self) -> None:
        runtime, _ = self._run([[], []])
        with self.trace.open("a") as handle:
            handle.write('{"ts": 1, "kind": "message_app')  # crash mid-write
        replayed = replay_session(self.trace, runtime.session.session_id)
        self.assertEqual(len(replayed.messages), len(runtime.session.messages))

    def test_list_sessions_summarises(self) -> None:
        runtime, _ = self._run([[], []])
        infos = list_sessions(self.trace)
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].session_id, runtime.session.session_id)
        self.assertEqual(infos[0].turns, 1)
        self.assertIn("do the thing", infos[0].first_prompt)

    def test_subagent_sessions_are_excluded_by_default(self) -> None:
        from claw_py.agents import AgentConfig, build_tool_executor
        from claw_py.hooks import HookRegistry

        session = Session()
        tracer = SessionTracer(session.session_id, path=self.trace)
        config = AgentConfig(
            client_factory=lambda model: ScriptedApiClient([[]]),
            workspace_root=self.workspace,
            hook_registry=HookRegistry(),
            session_tracer=tracer,
            max_depth=1,
        )
        executor = build_tool_executor(config, depth=0)
        runtime = make_runtime(
            [[("agent", {"description": "d", "prompt": "p", "subagent_type": "explore"})], []],
            self.workspace,
            executor=executor,
            session=session,
            tracer=tracer,
        )
        runtime.run_turn("delegate")
        tracer.close()

        top_level = list_sessions(self.trace)
        everything = list_sessions(self.trace, include_subagents=True)
        self.assertEqual(len(top_level), 1)
        self.assertGreater(len(everything), 1)
        child = [i for i in everything if i.is_subagent][0]
        self.assertEqual(child.subagent_of, session.session_id)

    def test_missing_session_is_reported(self) -> None:
        self._run([[], []])
        with self.assertRaises(RuntimeError):
            replay_session(self.trace, "nonexistent")

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(RuntimeError):
            replay_session(self.workspace / "nope.jsonl")


# ---------------------------------------------------------------------------
# parallel dispatch
# ---------------------------------------------------------------------------


def slow_tool(name: str, risk: str, delay: float, log: list) -> ToolSpec:
    def handler(input):
        log.append(("start", name))
        time.sleep(delay)
        log.append(("end", name))
        return f"{name}:{input.get('path', '')}"

    return ToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=handler,
        risk=risk,
    )


class ParallelDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-par-"))
        self.log: list = []

    def _executor(self, delay=0.15):
        executor = default_tool_executor()
        executor.register(slow_tool("slow_read", RISK_READ, delay, self.log))
        executor.register(slow_tool("slow_write", RISK_WRITE, delay, self.log))
        return executor

    def test_reads_run_concurrently(self) -> None:
        script = [[("slow_read", {"path": f"{i}"}) for i in range(4)], []]
        runtime = make_runtime(
            script, self.workspace, executor=self._executor(), parallel_tools=4
        )
        started = time.perf_counter()
        runtime.run_turn("go")
        self.assertLess(time.perf_counter() - started, 0.45)  # not 4 x 0.15

    def test_sequential_by_default(self) -> None:
        script = [[("slow_read", {"path": f"{i}"}) for i in range(3)], []]
        runtime = make_runtime(script, self.workspace, executor=self._executor())
        started = time.perf_counter()
        runtime.run_turn("go")
        self.assertGreater(time.perf_counter() - started, 0.4)  # 3 x 0.15

    def test_results_keep_request_order(self) -> None:
        script = [[("slow_read", {"path": str(i)}) for i in range(4)], []]
        runtime = make_runtime(
            script, self.workspace, executor=self._executor(0.05), parallel_tools=4
        )
        summary = runtime.run_turn("go")
        self.assertEqual(
            [r.text() for r in summary.tool_results],
            [f"slow_read:{i}" for i in range(4)],
        )

    def test_writes_never_overlap(self) -> None:
        script = [
            [("slow_write", {"path": str(self.workspace / str(i))}) for i in range(3)],
            [],
        ]
        runtime = make_runtime(
            script, self.workspace, executor=self._executor(0.05), parallel_tools=4
        )
        runtime.run_turn("go")
        # start/end must strictly alternate if nothing ran concurrently
        self.assertEqual([entry[0] for entry in self.log], ["start", "end"] * 3)

    def test_a_write_does_not_overtake_an_earlier_read(self) -> None:
        script = [
            [
                ("slow_read", {"path": "r1"}),
                ("slow_write", {"path": str(self.workspace / "w1")}),
                ("slow_read", {"path": "r2"}),
            ],
            [],
        ]
        runtime = make_runtime(
            script, self.workspace, executor=self._executor(0.05), parallel_tools=4
        )
        runtime.run_turn("go")
        order = [name for phase, name in self.log if phase == "start"]
        self.assertEqual(order, ["slow_read", "slow_write", "slow_read"])

    def test_denied_calls_still_land_in_order(self) -> None:
        script = [
            [
                ("slow_read", {"path": "ok"}),
                ("write_file", {"path": "/etc/nope", "content": "x"}),
                ("slow_read", {"path": "ok2"}),
            ],
            [],
        ]
        runtime = make_runtime(
            script, self.workspace, executor=self._executor(0.05), parallel_tools=4
        )
        summary = runtime.run_turn("go")
        self.assertEqual(
            [r.is_error for r in summary.tool_results], [False, True, False]
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# regressions found by reading a real trace
# ---------------------------------------------------------------------------


class TokenEstimateTests(unittest.TestCase):
    """estimate_session_tokens counted text() only, missing tool_use inputs.

    A file-writing session was undercounted ~6x, so auto-compaction never fired
    on exactly the workload that grows history fastest.
    """

    def _session_with_a_written_file(self, content_chars: int = 1659) -> Session:
        session = Session()
        session.push_user_text("write me a game")
        session.push_message(
            ConversationMessage(
                role="assistant",
                blocks=[
                    ContentBlock.tool_use(
                        "call_1", "write_file", {"path": "g.py", "content": "x" * content_chars}
                    )
                ],
            )
        )
        session.push_message(
            ConversationMessage.tool_result("call_1", "write_file", "wrote it", False)
        )
        return session

    def test_tool_use_arguments_are_counted(self) -> None:
        from claw_py.compact import estimate_session_tokens

        session = self._session_with_a_written_file()
        # text() sees ~35 chars; the wire payload is over 1600.
        self.assertGreater(estimate_session_tokens(session), 400)

    def test_estimate_tracks_the_actual_wire_payload(self) -> None:
        import json as _json

        from claw_py.compact import CHARS_PER_TOKEN, estimate_session_tokens

        session = self._session_with_a_written_file()
        wire = sum(
            len(_json.dumps(m.to_wire(), ensure_ascii=False)) for m in session.messages
        )
        self.assertEqual(estimate_session_tokens(session), wire // CHARS_PER_TOKEN)

    def test_bigger_file_means_bigger_estimate(self) -> None:
        from claw_py.compact import estimate_session_tokens

        small = estimate_session_tokens(self._session_with_a_written_file(100))
        large = estimate_session_tokens(self._session_with_a_written_file(8000))
        self.assertGreater(large - small, 1500)

    def test_compaction_now_fires_on_a_file_writing_session(self) -> None:
        from claw_py.compact import CompactionConfig, should_compact

        session = self._session_with_a_written_file(8000)
        session.push_user_text("now run it")
        session.push_message(ConversationMessage(role="assistant", blocks=[]))
        session.push_user_text("and again")
        session.push_message(ConversationMessage(role="assistant", blocks=[]))
        config = CompactionConfig(threshold_tokens=1000, keep_last=4)
        self.assertTrue(should_compact(session, config))

        # The same session measured the old way stays under the threshold,
        # which is exactly why this never fired before.
        old_estimate = sum(len(m.text()) for m in session.messages) // 4
        self.assertLess(old_estimate, 1000)


class CallIdTests(unittest.TestCase):
    """Tool-use ids restarted at call_1 every stream, so they collided."""

    class _Chunks:
        def __init__(self, calls) -> None:
            self.calls = calls

        def __iter__(self):
            import json as _json

            for name, args in self.calls:
                yield _json.dumps(
                    {"message": {"tool_calls": [{"function": {"name": name, "arguments": args}}]}}
                ).encode()
            yield b'{"done": true, "prompt_eval_count": 1, "eval_count": 1}'

    def test_ids_are_unique_across_streams(self) -> None:
        from claw_py.api import ApiClient

        client = ApiClient()
        first = [e["id"] for e in client._decode(self._Chunks([("read_file", {})])) if e["type"] == "tool_use"]
        second = [e["id"] for e in client._decode(self._Chunks([("bash", {})])) if e["type"] == "tool_use"]
        self.assertEqual(first, ["call_1"])
        self.assertEqual(second, ["call_2"])  # not call_1 again

    def test_ids_are_unique_within_a_stream(self) -> None:
        from claw_py.api import ApiClient

        client = ApiClient()
        ids = [
            e["id"]
            for e in client._decode(self._Chunks([("read_file", {}), ("bash", {})]))
            if e["type"] == "tool_use"
        ]
        self.assertEqual(len(set(ids)), 2)

    def test_trace_events_carry_the_tool_use_id(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="claw-py-ids-"))
        (workspace / "a.txt").write_text("x\n")
        sink: list = []

        class Collecting(SessionTracer):
            def emit(self, kind, **fields):
                sink.append({"kind": kind, **fields})

        session = Session()
        runtime = make_runtime(
            [[("read_file", {"path": str(workspace / "a.txt")})], []],
            workspace,
            session=session,
            tracer=Collecting(session.session_id),
        )
        runtime.run_turn("read it")
        started = [e for e in sink if e["kind"] == "tool_started"][0]
        finished = [e for e in sink if e["kind"] == "tool_finished"][0]
        self.assertTrue(started["tool_use_id"])
        self.assertEqual(started["tool_use_id"], finished["tool_use_id"])


class SessionEnvironmentTests(unittest.TestCase):
    """The system prompt and tool list are sent every request but live outside
    session.messages, so a trace without them cannot reproduce its own history.
    """

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-env-"))
        self.trace = self.workspace / "trace.jsonl"

    def _run(self, system_prompt="(original prompt)", executor=None, **kw):
        from claw_py.permissions import PermissionPolicy

        executor = executor or default_tool_executor()
        session = kw.pop("session", None) or Session()
        tracer = SessionTracer(session.session_id, path=self.trace)
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([[]]),
            tool_executor=executor,
            permission_policy=PermissionPolicy(workspace_root=self.workspace),
            system_prompt=system_prompt,
            session=session,
            session_tracer=tracer,
            **kw,
        )
        runtime.run_turn("do the thing")
        tracer.close()
        return runtime

    def test_session_started_records_the_environment(self) -> None:
        from claw_py.persistence import replay_environment

        runtime = self._run()
        env = replay_environment(self.trace, runtime.session.session_id)
        self.assertIsNotNone(env)
        self.assertEqual(env.system_prompt, "(original prompt)")
        self.assertIn("read_file", env.tool_names)
        self.assertEqual(env.permission_mode, "workspace-write")

    def test_replay_restores_the_system_prompt(self) -> None:
        from claw_py.persistence import replay_session

        runtime = self._run(system_prompt="you are a very specific agent")
        resumed = replay_session(self.trace, runtime.session.session_id)
        self.assertEqual(resumed.system_prompt, "you are a very specific agent")

    def test_drift_is_reported_when_the_prompt_changes(self) -> None:
        from claw_py.persistence import (
            SessionEnvironment,
            describe_environment_drift,
            replay_environment,
        )

        runtime = self._run(system_prompt="original")
        recorded = replay_environment(self.trace, runtime.session.session_id)
        current = SessionEnvironment(
            system_prompt="original plus a CLAUDE.md section",
            tool_names=recorded.tool_names,
            permission_mode=recorded.permission_mode,
            workspace_root=recorded.workspace_root,
            model=recorded.model,
        )
        drift = describe_environment_drift(recorded, current)
        self.assertEqual(len(drift), 1)
        self.assertIn("system prompt changed", drift[0])

    def test_drift_reports_added_and_removed_tools(self) -> None:
        from claw_py.persistence import SessionEnvironment, describe_environment_drift

        recorded = SessionEnvironment(system_prompt="p", tool_names=["read_file", "bash"])
        current = SessionEnvironment(system_prompt="p", tool_names=["read_file", "agent"])
        drift = describe_environment_drift(recorded, current)
        self.assertIn("added agent", drift[0])
        self.assertIn("removed bash", drift[0])

    def test_identical_environments_report_no_drift(self) -> None:
        from claw_py.persistence import (
            SessionEnvironment,
            describe_environment_drift,
            replay_environment,
        )

        runtime = self._run()
        recorded = replay_environment(self.trace, runtime.session.session_id)
        same = SessionEnvironment(
            system_prompt=recorded.system_prompt,
            tool_names=list(recorded.tool_names),
            permission_mode=recorded.permission_mode,
            workspace_root=recorded.workspace_root,
            model=recorded.model,
        )
        self.assertEqual(describe_environment_drift(recorded, same), [])

    def test_older_traces_without_the_event_return_none(self) -> None:
        from claw_py.persistence import replay_environment, replay_session

        legacy = self.workspace / "legacy.jsonl"
        legacy.write_text(
            '{"ts": 1, "session_id": "old1", "kind": "turn_started", "chars": 3}\n'
            '{"ts": 2, "session_id": "old1", "kind": "message_appended",'
            ' "message": {"role": "user", "blocks": [{"kind": "text", "text": "hi"}]}}\n'
        )
        self.assertIsNone(replay_environment(legacy, "old1"))
        session = replay_session(legacy, "old1")  # still resumable
        self.assertIsNone(session.system_prompt)
        self.assertEqual(len(session.messages), 1)

    def test_subagents_record_their_own_environment(self) -> None:
        from claw_py.agents import AgentConfig, build_tool_executor
        from claw_py.hooks import HookRegistry
        from claw_py.permissions import PermissionPolicy
        from claw_py.persistence import list_sessions, replay_environment

        session = Session()
        tracer = SessionTracer(session.session_id, path=self.trace)
        config = AgentConfig(
            client_factory=lambda model: ScriptedApiClient([[]]),
            workspace_root=self.workspace,
            hook_registry=HookRegistry(),
            session_tracer=tracer,
            max_depth=1,
        )
        executor = build_tool_executor(config, depth=0)
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([
                [("agent", {"description": "d", "prompt": "p", "subagent_type": "explore"})],
                [],
            ]),
            tool_executor=executor,
            permission_policy=PermissionPolicy(workspace_root=self.workspace),
            system_prompt="(parent)",
            session=session,
            session_tracer=tracer,
        )
        runtime.run_turn("delegate")
        tracer.close()

        child = [i for i in list_sessions(self.trace, include_subagents=True) if i.is_subagent][0]
        child_env = replay_environment(self.trace, child.session_id)
        self.assertIn("explore", child_env.system_prompt)
        self.assertEqual(child_env.permission_mode, "read-only")
        self.assertNotIn("write_file", child_env.tool_names)

    def test_forked_session_carries_the_prompt(self) -> None:
        session = Session(system_prompt="carried")
        self.assertEqual(session.fork_session().system_prompt, "carried")


class PermissionDecisionTests(unittest.TestCase):
    """A denial used to be visible only as a missing tool_started, with no way
    to tell a hook veto from a policy refusal from a declined prompt.
    """

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-decision-"))
        (self.workspace / "a.txt").write_text("x\n")
        self.sink: list = []
        outer = self

        class Collecting(SessionTracer):
            def emit(self, kind, **fields):
                outer.sink.append({"kind": kind, **fields})

        self.tracer_cls = Collecting

    def _run(self, calls, hooks=None, prompter=None, mode=None):
        from claw_py.permissions import PermissionMode, PermissionPolicy

        executor = default_tool_executor()
        session = Session()
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([calls, []]),
            tool_executor=executor,
            permission_policy=PermissionPolicy(
                mode=mode or PermissionMode.WORKSPACE_WRITE,
                workspace_root=self.workspace,
                risk_lookup=executor.risk_for,
            ),
            system_prompt="(test)",
            session=session,
            hook_registry=hooks or HookRegistry(),
            session_tracer=self.tracer_cls(session.session_id),
        )
        runtime.run_turn("go", prompter=prompter)
        return [e for e in self.sink if e["kind"] == "permission_decision"]

    def test_allowed_call_records_a_decision(self) -> None:
        decisions = self._run([("read_file", {"path": str(self.workspace / "a.txt")})])
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0]["allowed"])
        self.assertEqual(decisions[0]["source"], "mode")
        self.assertEqual(decisions[0]["risk"], RISK_READ)

    def test_hook_denial_is_attributed_to_the_hook(self) -> None:
        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, lambda p: HookResult.deny("nope"))
        decisions = self._run(
            [("read_file", {"path": str(self.workspace / "a.txt")})], hooks=registry
        )
        self.assertFalse(decisions[0]["allowed"])
        self.assertEqual(decisions[0]["source"], "hook")

    def test_hook_override_is_distinguishable_from_the_mode(self) -> None:
        registry = HookRegistry()
        registry.register(
            HookEvent.PRE_TOOL_USE, lambda p: HookResult.override("allow", "trusted")
        )
        decisions = self._run([("todo_write", {"todos": ["a"]})], hooks=registry)
        self.assertTrue(decisions[0]["allowed"])
        self.assertEqual(decisions[0]["source"], "hook_override")

    def test_workspace_denial_is_attributed_to_the_workspace(self) -> None:
        decisions = self._run([("write_file", {"path": "/etc/nope", "content": "x"})])
        self.assertEqual(decisions[0]["source"], "workspace")

    def test_missing_prompter_is_distinguishable_from_a_declined_prompt(self) -> None:
        without = self._run([("bash", {"command": "echo hi"})], prompter=None)
        self.assertEqual(without[0]["source"], "no_prompter")

        self.sink.clear()

        class Decline:
            def confirm(self, tool_name, effective_input):
                return False

        declined = self._run([("bash", {"command": "echo hi"})], prompter=Decline())
        self.assertEqual(declined[0]["source"], "prompter")
        self.assertFalse(declined[0]["allowed"])

    def test_approved_prompt_is_recorded(self) -> None:
        class Approve:
            def confirm(self, tool_name, effective_input):
                return True

        decisions = self._run([("bash", {"command": "echo hi"})], prompter=Approve())
        self.assertTrue(decisions[0]["allowed"])
        self.assertEqual(decisions[0]["source"], "prompter")

    def test_input_rewrites_are_flagged(self) -> None:
        registry = HookRegistry()
        target = str(self.workspace / "a.txt")
        registry.register(
            HookEvent.PRE_TOOL_USE, lambda p: HookResult.rewrite({"path": target})
        )
        decisions = self._run([("read_file", {"path": "wrong"})], hooks=registry)
        self.assertTrue(decisions[0]["input_rewritten"])

    def test_decision_carries_the_tool_use_id(self) -> None:
        decisions = self._run([("read_file", {"path": str(self.workspace / "a.txt")})])
        started = [e for e in self.sink if e["kind"] == "tool_started"]
        self.assertEqual(decisions[0]["tool_use_id"], started[0]["tool_use_id"])

    def test_denied_calls_emit_a_decision_but_no_tool_started(self) -> None:
        decisions = self._run([("write_file", {"path": "/etc/nope", "content": "x"})])
        self.assertEqual(len(decisions), 1)
        self.assertEqual([e for e in self.sink if e["kind"] == "tool_started"], [])

    def test_post_hook_rejection_is_distinct_from_tool_failure(self) -> None:
        registry = HookRegistry()
        registry.register(
            HookEvent.POST_TOOL_USE, lambda p: HookResult.deny("failed validation")
        )
        self._run([("read_file", {"path": str(self.workspace / "a.txt")})], hooks=registry)
        rejected = [e for e in self.sink if e["kind"] == "post_tool_use_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["decision"], "deny")
        # the tool itself succeeded; only the post-hook failed it
        finished = [e for e in self.sink if e["kind"] == "tool_finished"][0]
        self.assertTrue(finished["is_error"])

    def test_tool_failure_emits_no_rejection_event(self) -> None:
        self._run([("read_file", {"path": str(self.workspace / "missing.txt")})])
        self.assertEqual([e for e in self.sink if e["kind"] == "post_tool_use_rejected"], [])

    def test_session_listing_counts_denials(self) -> None:
        from claw_py.permissions import PermissionPolicy
        from claw_py.persistence import list_sessions

        trace = self.workspace / "trace.jsonl"
        executor = default_tool_executor()
        session = Session()
        tracer = SessionTracer(session.session_id, path=trace)
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([
                [
                    ("read_file", {"path": str(self.workspace / "a.txt")}),
                    ("write_file", {"path": "/etc/nope", "content": "x"}),
                ],
                [],
            ]),
            tool_executor=executor,
            permission_policy=PermissionPolicy(
                workspace_root=self.workspace, risk_lookup=executor.risk_for
            ),
            system_prompt="(test)",
            session=session,
            session_tracer=tracer,
        )
        runtime.run_turn("go")
        tracer.close()

        info = list_sessions(trace)[0]
        self.assertEqual(info.tool_calls, 2)
        self.assertEqual(info.denials, 1)


class ProviderFailureTests(unittest.TestCase):
    """A read-phase timeout raises bare TimeoutError, not URLError. It used to
    escape every handler, crash the CLI with a traceback, discard whatever the
    model had already streamed, and never emit turn_failed.
    """

    @staticmethod
    def _slow_server(port: int, words: int = 4):
        import json as _json
        import threading as _threading
        import time as _time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Slow(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for index in range(words):
                    chunk = _json.dumps({"message": {"content": f"word{index} "}})
                    self.wfile.write((chunk + "\n").encode())
                    self.wfile.flush()
                _time.sleep(3)  # stall mid-generation; client times out well before

        # Threading + daemon threads, or a stalled handler blocks shutdown()
        # and each test pays the full stall.
        server = ThreadingHTTPServer(("127.0.0.1", port), Slow)
        server.daemon_threads = True
        _threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_read_timeout_becomes_a_clean_runtime_error(self) -> None:
        from claw_py.api import ApiClient, build_assistant_message
        from claw_py.types import ApiRequest, ConversationMessage
        from claw_py.types import RuntimeError as ClawError

        server = self._slow_server(8771)
        try:
            client = ApiClient(model="stub", base_url="http://127.0.0.1:8771", request_timeout=1)
            request = ApiRequest("sys", [ConversationMessage.user_text("hi")])
            with self.assertRaises(ClawError) as ctx:
                build_assistant_message(client.stream(request))
            self.assertIn("stopped sending", ctx.exception.message)
            self.assertIn("--request-timeout", ctx.exception.message)
        finally:
            server.shutdown()
            server.server_close()

    def test_streamed_text_survives_the_timeout(self) -> None:
        from claw_py.api import ApiClient, build_assistant_message
        from claw_py.types import ApiRequest, ConversationMessage
        from claw_py.types import RuntimeError as ClawError

        server = self._slow_server(8772, words=6)
        try:
            client = ApiClient(model="stub", base_url="http://127.0.0.1:8772", request_timeout=1)
            request = ApiRequest("sys", [ConversationMessage.user_text("hi")])
            with self.assertRaises(ClawError) as ctx:
                build_assistant_message(client.stream(request))
            self.assertIn("word0", ctx.exception.partial_text)
            self.assertIn("word5", ctx.exception.partial_text)
        finally:
            server.shutdown()
            server.server_close()

    def test_turn_failed_is_recorded_with_the_partial_output(self) -> None:
        from claw_py.api import ApiClient
        from claw_py.permissions import PermissionPolicy
        from claw_py.types import RuntimeError as ClawError

        sink: list = []

        class Collecting(SessionTracer):
            def emit(self, kind, **fields):
                sink.append({"kind": kind, **fields})

        server = self._slow_server(8773)
        try:
            workspace = Path(tempfile.mkdtemp(prefix="claw-py-timeout-"))
            session = Session()
            runtime = ConversationRuntime(
                api_client=ApiClient(
                    model="stub", base_url="http://127.0.0.1:8773", request_timeout=1
                ),
                tool_executor=default_tool_executor(),
                permission_policy=PermissionPolicy(workspace_root=workspace),
                system_prompt="(test)",
                session=session,
                session_tracer=Collecting(session.session_id),
            )
            with self.assertRaises(ClawError):
                runtime.run_turn("generate something long")
        finally:
            server.shutdown()
            server.server_close()

        failed = [e for e in sink if e["kind"] == "turn_failed"]
        self.assertEqual(len(failed), 1)
        self.assertGreater(failed[0]["partial_chars"], 0)
        self.assertIn("word0", failed[0]["partial_text"])

    def test_connection_refused_still_names_the_service(self) -> None:
        from claw_py.api import ApiClient
        from claw_py.types import ApiRequest, ConversationMessage
        from claw_py.types import RuntimeError as ClawError

        client = ApiClient(model="stub", base_url="http://127.0.0.1:9", request_timeout=1)
        request = ApiRequest("sys", [ConversationMessage.user_text("hi")])
        with self.assertRaises(ClawError) as ctx:
            list(client.stream(request))
        self.assertIn("ollama serve", ctx.exception.message)

    def test_timeout_is_configurable(self) -> None:
        from claw_py.api import DEFAULT_REQUEST_TIMEOUT, ApiClient

        self.assertEqual(ApiClient().request_timeout, DEFAULT_REQUEST_TIMEOUT)
        self.assertEqual(ApiClient(request_timeout=42).request_timeout, 42)
