"""Tests for subagents and provider routing.

Stdlib unittest, no network, no model. Run with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from claw_py.agents import (
    AGENT_SPECS,
    AgentConfig,
    allowed_tools_for_subagent,
    build_agent_system_prompt,
    build_tool_executor,
    execute_agent,
    narrower_mode,
    normalize_subagent_type,
)
from claw_py.api import ApiClient, build_assistant_message
from claw_py.conversation import ConversationRuntime
from claw_py.hooks import HookEvent, HookRegistry, HookResult
from claw_py.permissions import PermissionMode, PermissionPolicy
from claw_py.routing import _parse_arguments
from claw_py.telemetry import SessionTracer
from claw_py.tools import ToolError
from claw_py.types import ApiRequest, Session


class ScriptedApiClient(ApiClient):
    def __init__(self, script, label="stub") -> None:
        super().__init__(model=label, tool_specs=[])
        self.script = script
        self.turn = 0

    def stream(self, request: ApiRequest):
        calls = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        yield {"type": "text_delta", "text": f"report from step {self.turn}"}
        for index, (name, input) in enumerate(calls, 1):
            yield {
                "type": "tool_use",
                "id": f"c{self.turn}_{index}",
                "name": name,
                "input": input,
            }
        yield {"type": "usage", "input_tokens": 5, "output_tokens": 2}
        yield {"type": "message_stop"}

    def complete(self, system_prompt: str, user_text: str) -> str:
        return "summary"


class CollectingTracer(SessionTracer):
    def __init__(self, session_id, sink=None) -> None:
        super().__init__(session_id)
        self.sink = sink if sink is not None else []

    def child(self, session_id):
        return CollectingTracer(session_id, self.sink)

    def emit(self, kind, **fields):
        self.sink.append({"session_id": self.session_id, "kind": kind, **fields})


class AgentTypeTests(unittest.TestCase):
    def test_normalize_accepts_canonical_names(self) -> None:
        for name in AGENT_SPECS:
            self.assertEqual(normalize_subagent_type(name), name)

    def test_normalize_accepts_aliases_and_casing(self) -> None:
        self.assertEqual(normalize_subagent_type("Explorer"), "explore")
        self.assertEqual(normalize_subagent_type("general_purpose"), "general-purpose")
        self.assertEqual(normalize_subagent_type("VERIFY"), "verification")

    def test_normalize_defaults_when_missing(self) -> None:
        self.assertEqual(normalize_subagent_type(None), "general-purpose")

    def test_normalize_rejects_unknown(self) -> None:
        with self.assertRaises(ToolError):
            normalize_subagent_type("wizard")

    def test_read_only_agents_cannot_write(self) -> None:
        for name in ("explore", "plan"):
            tools = allowed_tools_for_subagent(name)
            self.assertNotIn("write_file", tools)
            self.assertNotIn("edit_file", tools)
            self.assertNotIn("bash", tools)

    def test_system_prompt_states_no_human_is_available(self) -> None:
        prompt = build_agent_system_prompt("explore", Path("/tmp"), ["read_file"])
        self.assertIn("cannot ask the user", prompt)
        self.assertIn("explore", prompt)


class ModeNarrowingTests(unittest.TestCase):
    def test_subagent_never_widens_parent_authority(self) -> None:
        self.assertEqual(
            narrower_mode(PermissionMode.READ_ONLY, PermissionMode.WORKSPACE_WRITE),
            PermissionMode.READ_ONLY,
        )

    def test_spec_can_narrow_a_permissive_parent(self) -> None:
        self.assertEqual(
            narrower_mode(PermissionMode.DANGER_FULL_ACCESS, PermissionMode.READ_ONLY),
            PermissionMode.READ_ONLY,
        )


class SubagentExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="claw-py-agent-"))
        (self.workspace / "alpha.py").write_text("def hello():\n    return 1\n")
        self.sink: list = []
        self.tracer = CollectingTracer("parent", self.sink)

    def make_config(self, subagent_script, max_depth=2, parent_mode=PermissionMode.WORKSPACE_WRITE, hook_registry=None):
        return AgentConfig(
            client_factory=lambda model: ScriptedApiClient(subagent_script, "sub"),
            workspace_root=self.workspace,
            hook_registry=hook_registry or HookRegistry(),
            session_tracer=self.tracer,
            parent_mode=parent_mode,
            max_depth=max_depth,
        )

    def test_returns_only_the_final_report(self) -> None:
        config = self.make_config([
            [("read_file", {"path": str(self.workspace / "alpha.py")})],
            [],
        ])
        output = execute_agent(
            {"description": "look", "prompt": "read alpha", "subagent_type": "explore"},
            config,
            depth=1,
        )
        self.assertIn("report from step 2", output)
        self.assertNotIn("def hello", output)  # intermediate tool output stayed inside

    def test_restricted_tool_is_denied_inside_the_subagent(self) -> None:
        config = self.make_config([
            [("write_file", {"path": str(self.workspace / "x.md"), "content": "hi"})],
            [],
        ])
        execute_agent(
            {"description": "write", "prompt": "write a file", "subagent_type": "explore"},
            config,
            depth=1,
        )
        denials = [
            e for e in self.sink
            if e["kind"] == "tool_finished" and e.get("is_error")
        ]
        self.assertEqual(len(denials), 1)
        self.assertEqual(denials[0]["tool_name"], "write_file")
        self.assertFalse((self.workspace / "x.md").exists())

    def test_restricted_tools_are_not_even_offered(self) -> None:
        config = self.make_config([[]])
        execute_agent(
            {"description": "look", "prompt": "explore", "subagent_type": "explore"},
            config,
            depth=1,
        )
        started = [e for e in self.sink if e["kind"] == "subagent_started"][0]
        self.assertNotIn("write_file", started["allowed_tools"])
        self.assertIn("read_file", started["allowed_tools"])

    def test_parent_hooks_still_gate_the_subagent(self) -> None:
        registry = HookRegistry()
        registry.register(
            HookEvent.PRE_TOOL_USE, lambda payload: HookResult.deny("gated by parent hook")
        )
        config = self.make_config(
            [[("read_file", {"path": str(self.workspace / "alpha.py")})], []],
            hook_registry=registry,
        )
        execute_agent(
            {"description": "look", "prompt": "read", "subagent_type": "explore"},
            config,
            depth=1,
        )
        denials = [e for e in self.sink if e["kind"] == "tool_finished" and e.get("is_error")]
        self.assertEqual(len(denials), 1)

    def test_depth_limit_refuses_to_spawn(self) -> None:
        config = self.make_config([[]], max_depth=1)
        with self.assertRaises(ToolError) as ctx:
            execute_agent(
                {"description": "x", "prompt": "y", "subagent_type": "explore"},
                config,
                depth=2,
            )
        self.assertIn("depth limit", str(ctx.exception))

    def test_agent_tool_absent_at_the_ceiling(self) -> None:
        config = self.make_config([[]], max_depth=1)
        self.assertIn("agent", build_tool_executor(config, depth=0).names())
        self.assertNotIn("agent", build_tool_executor(config, depth=1).names())

    def test_max_depth_zero_disables_delegation(self) -> None:
        config = self.make_config([[]], max_depth=0)
        self.assertNotIn("agent", build_tool_executor(config, depth=0).names())

    def test_missing_prompt_is_rejected(self) -> None:
        config = self.make_config([[]])
        with self.assertRaises(ToolError):
            execute_agent({"description": "x", "subagent_type": "explore"}, config, depth=1)

    def test_subagent_failure_surfaces_as_tool_error(self) -> None:
        # A subagent that never stops calling tools trips its own iteration cap.
        config = self.make_config([[("read_file", {"path": str(self.workspace / "alpha.py")})]])
        with self.assertRaises(ToolError) as ctx:
            execute_agent(
                {"description": "spin", "prompt": "loop", "subagent_type": "explore"},
                config,
                depth=1,
            )
        self.assertIn("failed", str(ctx.exception))


class ParentIntegrationTests(unittest.TestCase):
    def test_parent_context_stays_small_across_delegation(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="claw-py-agent-"))
        (workspace / "alpha.py").write_text("x = 1\n")
        sink: list = []
        tracer = CollectingTracer("parent", sink)

        subagent_script = [
            [("read_file", {"path": str(workspace / "alpha.py")})],
            [("glob_search", {"pattern": "*.py", "path": str(workspace)})],
            [("grep_search", {"pattern": "x", "path": str(workspace)})],
            [],
        ]
        config = AgentConfig(
            client_factory=lambda model: ScriptedApiClient(subagent_script, "sub"),
            workspace_root=workspace,
            hook_registry=HookRegistry(),
            session_tracer=tracer,
            max_depth=1,
        )
        tool_executor = build_tool_executor(config, depth=0)

        session = Session(session_id="parent")
        runtime = ConversationRuntime(
            api_client=ScriptedApiClient([
                [("agent", {
                    "description": "map",
                    "prompt": "map the package",
                    "subagent_type": "explore",
                })],
                [],
            ], "parent"),
            tool_executor=tool_executor,
            permission_policy=PermissionPolicy(workspace_root=workspace),
            system_prompt="(test)",
            session=session,
            session_tracer=tracer,
        )
        summary = runtime.run_turn("understand this")

        # Subagent ran 4 iterations and 3 tools; parent sees 1 tool result.
        self.assertEqual(summary.iterations, 2)
        self.assertEqual(len(summary.tool_results), 1)
        finished = [e for e in sink if e["kind"] == "subagent_finished"][0]
        self.assertEqual(finished["iterations"], 4)
        self.assertEqual(finished["tool_results"], 3)


class RoutingTests(unittest.TestCase):
    @staticmethod
    def _client():
        """Build without __init__ so these run whether or not litellm is present."""
        from claw_py.routing import RoutedApiClient

        return RoutedApiClient.__new__(RoutedApiClient)

    @staticmethod
    def _frag(content=None, calls=None, usage=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=calls))],
            usage=usage,
        )

    @staticmethod
    def _call(index, id=None, name=None, arguments=None):
        return SimpleNamespace(
            index=index,
            id=id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    def test_parses_streamed_argument_fragments(self) -> None:
        self.assertEqual(_parse_arguments('{"path": "a.py"}'), {"path": "a.py"})

    def test_survives_malformed_arguments(self) -> None:
        self.assertEqual(_parse_arguments("{not json"), {"_raw": "{not json"})

    def test_empty_arguments_become_empty_dict(self) -> None:
        self.assertEqual(_parse_arguments("   "), {})

    def test_decode_reassembles_split_tool_arguments(self) -> None:
        stream = [
            self._frag(content="checking"),
            self._frag(calls=[self._call(0, id="call_a", name="read_file", arguments='{"pa')]),
            self._frag(calls=[self._call(0, arguments='th": "alpha.py"}')]),
            self._frag(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4)),
        ]
        events = list(self._client()._decode(stream))
        tool = [e for e in events if e["type"] == "tool_use"][0]
        self.assertEqual(tool["name"], "read_file")
        self.assertEqual(tool["input"], {"path": "alpha.py"})

    def test_decode_keeps_parallel_calls_separate(self) -> None:
        stream = [
            self._frag(calls=[self._call(0, id="a", name="read_file", arguments='{"path":"x"}')]),
            self._frag(calls=[self._call(1, id="b", name="grep_search", arguments='{"pattern":"y"}')]),
        ]
        tools = [e for e in self._client()._decode(stream) if e["type"] == "tool_use"]
        self.assertEqual([t["name"] for t in tools], ["read_file", "grep_search"])
        self.assertEqual([t["id"] for t in tools], ["a", "b"])

    def test_decode_emits_usage_and_stop(self) -> None:
        stream = [
            self._frag(content="hi", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        ]
        events = list(self._client()._decode(stream))
        self.assertEqual(events[-1]["type"], "message_stop")
        usage = [e for e in events if e["type"] == "usage"][0]
        self.assertEqual(usage["input_tokens"], 7)

    def test_decoded_events_fold_like_the_ollama_client(self) -> None:
        """The whole point of the router: identical event contract."""
        stream = [
            self._frag(content="one "),
            self._frag(content="two"),
            self._frag(calls=[self._call(0, id="a", name="bash", arguments='{"command":"ls"}')]),
            self._frag(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ]
        message, usage = build_assistant_message(iter(self._client()._decode(stream)))
        self.assertEqual(message.text(), "one two")
        self.assertEqual(len(message.tool_uses()), 1)
        self.assertEqual(usage.output_tokens, 2)

    def test_missing_litellm_gives_an_actionable_error(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            from claw_py.routing import RoutedApiClient
            from claw_py.types import RuntimeError as ClawRuntimeError

            with self.assertRaises(ClawRuntimeError) as ctx:
                RoutedApiClient()
            self.assertIn("pip install litellm", str(ctx.exception))
        else:
            self.skipTest("litellm is installed")

    def test_unknown_route_is_rejected(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            self.skipTest("litellm not installed")
        from claw_py.routing import RoutedApiClient
        from claw_py.types import RuntimeError as ClawRuntimeError

        with self.assertRaises(ClawRuntimeError):
            RoutedApiClient(role="nonexistent-role")


if __name__ == "__main__":
    unittest.main()
