"""Tests for the turn loop and the tool-gating pipeline.

Stdlib unittest, no network, no model. Run with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from claw_py.api import ApiClient, build_assistant_message, strip_reasoning
from claw_py.compact import (
    CompactionConfig,
    compact_session,
    estimate_session_tokens,
    should_compact,
)
from claw_py.conversation import ConversationRuntime
from claw_py.hooks import HookEvent, HookRegistry, HookResult, merge_hook_feedback
from claw_py.permissions import (
    PermissionContext,
    PermissionMode,
    PermissionPolicy,
)
from claw_py.telemetry import SessionTracer
from claw_py.tools import default_tool_executor
from claw_py.types import (
    ApiRequest,
    ContentBlock,
    ConversationMessage,
    RuntimeError,
    Session,
)


class StubApiClient(ApiClient):
    """Replays a scripted list of turns. Each turn is a list of tool calls."""

    def __init__(self, script: list[list[tuple[str, dict]]]) -> None:
        super().__init__(model="stub", tool_specs=[])
        self.script = script
        self.turn = 0
        self.seen_requests: list[ApiRequest] = []

    def stream(self, request: ApiRequest):
        self.seen_requests.append(request)
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {"type": "text_delta", "text": f"step {self.turn}"}
        for index, (name, input) in enumerate(calls, 1):
            yield {
                "type": "tool_use",
                "id": f"call_{self.turn}_{index}",
                "name": name,
                "input": input,
            }
        yield {"type": "usage", "input_tokens": 10, "output_tokens": 5}
        yield {"type": "message_stop"}

    def complete(self, system_prompt: str, user_text: str) -> str:
        return "stub summary"


def make_runtime(script, workspace, hook_registry=None, mode=PermissionMode.WORKSPACE_WRITE, **kw):
    session = Session()
    return ConversationRuntime(
        api_client=StubApiClient(script),
        tool_executor=default_tool_executor(),
        permission_policy=PermissionPolicy(mode=mode, workspace_root=workspace),
        system_prompt="(test)",
        session=session,
        hook_registry=hook_registry or HookRegistry(),
        session_tracer=SessionTracer(session.session_id),
        **kw,
    )


class TurnLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-test-"))
        (self.workspace / "notes.txt").write_text("alpha\nbeta\n")

    def test_loop_breaks_when_no_tool_calls(self) -> None:
        runtime = make_runtime([[]], self.workspace)
        summary = runtime.run_turn("hello")
        self.assertEqual(summary.iterations, 1)
        self.assertEqual(summary.tool_results, [])

    def test_loop_continues_while_tools_pending(self) -> None:
        script = [
            [("read_file", {"path": str(self.workspace / "notes.txt")})],
            [("read_file", {"path": str(self.workspace / "notes.txt")})],
            [],
        ]
        summary = make_runtime(script, self.workspace).run_turn("read it twice")
        self.assertEqual(summary.iterations, 3)
        self.assertEqual(len(summary.tool_results), 2)
        self.assertFalse(any(r.is_error for r in summary.tool_results))

    def test_max_iterations_trips(self) -> None:
        script = [[("read_file", {"path": str(self.workspace / "notes.txt")})]]
        runtime = make_runtime(script, self.workspace, max_iterations=3)
        with self.assertRaises(RuntimeError) as ctx:
            runtime.run_turn("loop forever")
        self.assertIn("maximum number of iterations", ctx.exception.message)

    def test_usage_accumulates_across_iterations(self) -> None:
        script = [[("read_file", {"path": str(self.workspace / "notes.txt")})], []]
        summary = make_runtime(script, self.workspace).run_turn("go")
        self.assertEqual(summary.usage.input_tokens, 20)
        self.assertEqual(summary.usage.output_tokens, 10)

    def test_full_history_resent_each_iteration(self) -> None:
        script = [[("read_file", {"path": str(self.workspace / "notes.txt")})], []]
        runtime = make_runtime(script, self.workspace)
        runtime.run_turn("go")
        sent = runtime.api_client.seen_requests
        self.assertEqual(len(sent), 2)
        self.assertGreater(len(sent[1].messages), len(sent[0].messages))


class ToolFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-test-"))

    def test_tool_error_becomes_is_error_result_not_exception(self) -> None:
        script = [[("read_file", {"path": str(self.workspace / "missing.txt")})], []]
        summary = make_runtime(script, self.workspace).run_turn("read missing")
        self.assertEqual(len(summary.tool_results), 1)
        self.assertTrue(summary.tool_results[0].is_error)
        self.assertIn("no such file", summary.tool_results[0].text())

    def test_unknown_tool_is_reported_to_the_model(self) -> None:
        script = [[("nonexistent_tool", {})], []]
        summary = make_runtime(script, self.workspace).run_turn("call junk")
        self.assertTrue(summary.tool_results[0].is_error)
        self.assertIn("unknown tool", summary.tool_results[0].text())


class PermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-test-"))
        self.policy = PermissionPolicy(
            mode=PermissionMode.WORKSPACE_WRITE, workspace_root=self.workspace
        )

    def test_read_only_blocks_writes(self) -> None:
        policy = PermissionPolicy(mode=PermissionMode.READ_ONLY, workspace_root=self.workspace)
        outcome = policy.authorize_with_context(
            "write_file", {"path": str(self.workspace / "x")}, PermissionContext()
        )
        self.assertFalse(outcome.allowed)

    def test_read_only_allows_reads(self) -> None:
        policy = PermissionPolicy(mode=PermissionMode.READ_ONLY, workspace_root=self.workspace)
        self.assertTrue(
            policy.authorize_with_context("read_file", {"path": "x"}, PermissionContext()).allowed
        )

    def test_workspace_write_blocks_paths_outside_root(self) -> None:
        outcome = self.policy.authorize_with_context(
            "write_file", {"path": "/etc/passwd"}, PermissionContext()
        )
        self.assertFalse(outcome.allowed)
        self.assertIn("outside the workspace root", outcome.reason)

    def test_workspace_write_allows_paths_inside_root(self) -> None:
        outcome = self.policy.authorize_with_context(
            "write_file", {"path": str(self.workspace / "ok.txt")}, PermissionContext()
        )
        self.assertTrue(outcome.allowed)

    def test_escalation_denied_without_prompter(self) -> None:
        outcome = self.policy.authorize_with_context(
            "bash", {"command": "ls"}, PermissionContext(), prompter=None
        )
        self.assertFalse(outcome.allowed)
        self.assertIn("no prompter", outcome.reason)

    def test_hook_override_beats_the_mode(self) -> None:
        policy = PermissionPolicy(mode=PermissionMode.READ_ONLY, workspace_root=self.workspace)
        outcome = policy.authorize_with_context(
            "write_file",
            {"path": "/etc/passwd"},
            PermissionContext(permission_override="allow", permission_reason="hook says yes"),
        )
        self.assertTrue(outcome.allowed)

    def test_danger_full_access_allows_everything(self) -> None:
        policy = PermissionPolicy(mode=PermissionMode.DANGER_FULL_ACCESS)
        self.assertTrue(
            policy.authorize_with_context("bash", {"command": "ls"}, PermissionContext()).allowed
        )


class HookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-test-"))
        (self.workspace / "notes.txt").write_text("alpha\n")

    def test_pre_hook_rewrites_input(self) -> None:
        registry = HookRegistry()
        target = self.workspace / "notes.txt"
        registry.register(
            HookEvent.PRE_TOOL_USE,
            lambda payload: HookResult.rewrite({"path": str(target)}, "redirected"),
        )
        script = [[("read_file", {"path": "/nowhere"})], []]
        summary = make_runtime(script, self.workspace, registry).run_turn("read")
        result = summary.tool_results[0]
        self.assertFalse(result.is_error)
        self.assertIn("alpha", result.text())
        self.assertIn("redirected", result.text())

    def test_pre_hook_denies_before_policy_runs(self) -> None:
        registry = HookRegistry()
        registry.register(
            HookEvent.PRE_TOOL_USE, lambda payload: HookResult.deny("blocked by policy hook")
        )
        script = [[("read_file", {"path": str(self.workspace / "notes.txt")})], []]
        summary = make_runtime(
            script, self.workspace, registry, mode=PermissionMode.ALLOW
        ).run_turn("read")
        self.assertTrue(summary.tool_results[0].is_error)
        self.assertIn("blocked by policy hook", summary.tool_results[0].text())

    def test_post_hook_denial_flips_a_successful_tool_to_error(self) -> None:
        registry = HookRegistry()
        registry.register(
            HookEvent.POST_TOOL_USE, lambda payload: HookResult.deny("output failed validation")
        )
        script = [[("read_file", {"path": str(self.workspace / "notes.txt")})], []]
        summary = make_runtime(script, self.workspace, registry).run_turn("read")
        self.assertTrue(summary.tool_results[0].is_error)

    def test_failure_hook_fires_only_on_error_path(self) -> None:
        fired: list[str] = []
        registry = HookRegistry()
        registry.register(
            HookEvent.POST_TOOL_USE, lambda p: (fired.append("success"), HookResult.proceed())[1]
        )
        registry.register(
            HookEvent.POST_TOOL_USE_FAILURE,
            lambda p: (fired.append("failure"), HookResult.proceed())[1],
        )
        script = [[("read_file", {"path": str(self.workspace / "missing.txt")})], []]
        make_runtime(script, self.workspace, registry).run_turn("read")
        self.assertEqual(fired, ["failure"])

    def test_merge_hook_feedback_prefixes_notes(self) -> None:
        merged = merge_hook_feedback(["note one"], "tool output", False)
        self.assertIn("[hook note] note one", merged)
        self.assertIn("tool output", merged)


class CompactionTests(unittest.TestCase):
    def _big_session(self, pairs: int = 14) -> Session:
        session = Session()
        for index in range(pairs):
            session.push_user_text("x" * 900 + f" msg{index}")
            session.push_message(ConversationMessage(role="assistant", blocks=[]))
        return session

    def test_threshold_gate(self) -> None:
        small = Session()
        small.push_user_text("hi")
        self.assertFalse(should_compact(small, CompactionConfig()))
        self.assertTrue(should_compact(self._big_session(), CompactionConfig()))

    def test_compaction_shrinks_history_and_records_it(self) -> None:
        session = self._big_session()
        before = estimate_session_tokens(session)
        result = compact_session(
            session, CompactionConfig(keep_last=4), lambda sp, ut: "a summary"
        )
        self.assertLess(estimate_session_tokens(session), before)
        self.assertEqual(len(session.messages), 5)
        self.assertEqual(session.messages[0].role, "user")
        self.assertIsNotNone(result.record)
        self.assertEqual(session.compaction, result.record)

    def test_health_probe_rejects_corrupted_history(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="claw-py-test-"))
        runtime = make_runtime([[]], workspace)
        runtime.session = self._big_session()
        runtime.run_session_health_probe()
        runtime.session.messages[0].role = "bogus"
        with self.assertRaises(ValueError):
            runtime.run_session_health_probe()


class MessageTests(unittest.TestCase):
    def test_tool_result_wire_format_marks_errors(self) -> None:
        message = ConversationMessage.tool_result("id1", "bash", "boom", True)
        self.assertEqual(message.to_wire()["role"], "tool")
        self.assertTrue(message.to_wire()["content"].startswith("ERROR: "))

    def test_assistant_wire_format_carries_tool_calls(self) -> None:
        message = ConversationMessage(
            role="assistant",
            blocks=[
                ContentBlock.text_block("thinking"),
                ContentBlock.tool_use("id1", "read_file", {"path": "a"}),
            ],
        )
        wire = message.to_wire()
        self.assertEqual(wire["tool_calls"][0]["function"]["name"], "read_file")

    def test_build_assistant_message_folds_events(self) -> None:
        events = iter(
            [
                {"type": "text_delta", "text": "hel"},
                {"type": "text_delta", "text": "lo"},
                {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
                {"type": "usage", "input_tokens": 7, "output_tokens": 3},
                {"type": "message_stop"},
            ]
        )
        message, usage = build_assistant_message(events)
        self.assertEqual(message.text(), "hello")
        self.assertEqual(len(message.tool_uses()), 1)
        self.assertEqual(usage.input_tokens, 7)

    def test_reasoning_blocks_are_stripped(self) -> None:
        self.assertEqual(strip_reasoning("<think>hmm</think>answer"), "answer")

    def test_empty_user_message_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            Session().push_user_text("   ")


if __name__ == "__main__":
    unittest.main()
