"""The agentic loop.

Mirrors `runtime/src/conversation.rs`. This module is the whole architecture;
everything else is a collaborator it calls into. Read `run_turn` top to bottom
and you have the design.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .api import ApiClient, build_assistant_message
from .compact import CompactionConfig, compact_session, should_compact, estimate_session_tokens
from .hooks import (
    HookEvent,
    HookRegistry,
    HookResult,
    format_hook_message,
    merge_hook_feedback,
)
from .permissions import (
    PermissionContext,
    PermissionOutcome,
    PermissionPolicy,
    PermissionPrompter,
)
from .telemetry import SessionTracer
from .tools import ToolError, ToolExecutor
from .types import (
    ApiRequest,
    CompactionRecord,
    ConversationMessage,
    RuntimeError,
    Session,
    TurnSummary,
    UsageTracker,
)


class ConversationRuntime:
    def __init__(
        self,
        api_client: ApiClient,
        tool_executor: ToolExecutor,
        permission_policy: PermissionPolicy,
        system_prompt: str,
        session: Optional[Session] = None,
        hook_registry: Optional[HookRegistry] = None,
        session_tracer: Optional[SessionTracer] = None,
        max_iterations: int = 12,
        compaction_config: Optional[CompactionConfig] = None,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.api_client = api_client
        self.tool_executor = tool_executor
        self.permission_policy = permission_policy
        self.system_prompt = system_prompt
        self.session = session or Session()
        self.hook_registry = hook_registry or HookRegistry()
        self.session_tracer = session_tracer or SessionTracer(self.session.session_id)
        self.max_iterations = max_iterations
        self.compaction_config = compaction_config or CompactionConfig()
        self.usage_tracker = UsageTracker()
        self.on_text = on_text

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def run_turn(
        self,
        user_input: str,
        prompter: Optional[PermissionPrompter] = None,
    ) -> TurnSummary:
        # Session-health canary: if history was rewritten by a previous
        # compaction, verify the session still round-trips before building on it.
        if self.session.compaction is not None:
            try:
                self.run_session_health_probe()
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(
                    f"Session health probe failed after compaction: {error}. "
                    "The session may be in an inconsistent state. "
                    "Consider starting a fresh session."
                ) from error

        self.record_turn_started(user_input)
        self.session.push_user_text(user_input)

        assistant_messages: list[ConversationMessage] = []
        tool_results: list[ConversationMessage] = []
        iterations = 0
        auto_compaction: Optional[CompactionRecord] = None

        while True:
            iterations += 1
            if iterations > self.max_iterations:
                error = RuntimeError(
                    "conversation loop exceeded the maximum number of iterations"
                )
                self.record_turn_failed(iterations, error)
                raise error

            request = ApiRequest(
                system_prompt=self.system_prompt,
                messages=list(self.session.messages),
            )
            try:
                events = self.api_client.stream(request)
                assistant_message, usage = build_assistant_message(events, self.on_text)
            except RuntimeError as error:
                self.record_turn_failed(iterations, error)
                raise

            if usage is not None:
                self.usage_tracker.record(usage)

            pending_tool_uses = [
                (block.id, block.name, block.input)
                for block in assistant_message.tool_uses()
            ]
            self.record_assistant_iteration(
                iterations, assistant_message, len(pending_tool_uses)
            )

            self.session.push_message(assistant_message)
            assistant_messages.append(assistant_message)

            # Runs before the next API call, including on the terminal
            # (no-tool) iteration, to prevent unbounded session growth.
            compaction = self.maybe_auto_compact()
            if compaction is not None:
                auto_compaction = compaction

            if not pending_tool_uses:
                break

            for tool_use_id, tool_name, input in pending_tool_uses:
                result_message = self._run_tool_call(
                    iterations, tool_use_id, tool_name, input, prompter
                )
                self.session.push_message(result_message)
                self.record_tool_finished(iterations, result_message)
                tool_results.append(result_message)

        summary = TurnSummary(
            assistant_messages=assistant_messages,
            tool_results=tool_results,
            iterations=iterations,
            usage=self.usage_tracker.cumulative_usage(),
            auto_compaction=auto_compaction,
        )
        self.record_turn_completed(summary)
        return summary

    # ------------------------------------------------------------------
    # tool call pipeline: hook -> permission -> execute -> hook
    # ------------------------------------------------------------------

    def _run_tool_call(
        self,
        iterations: int,
        tool_use_id: str,
        tool_name: str,
        input: dict[str, Any],
        prompter: Optional[PermissionPrompter],
    ) -> ConversationMessage:
        pre_hook_result = self.run_pre_tool_use_hook(tool_name, input)

        # A hook may rewrite the arguments. The model is never told.
        effective_input = pre_hook_result.updated_input()
        if effective_input is None:
            effective_input = dict(input)

        permission_context = PermissionContext(
            permission_override=pre_hook_result.permission_override(),
            permission_reason=pre_hook_result.permission_reason(),
        )

        if pre_hook_result.is_cancelled():
            permission_outcome = PermissionOutcome.Deny(
                format_hook_message(
                    pre_hook_result, f"PreToolUse hook cancelled tool `{tool_name}`"
                )
            )
        elif pre_hook_result.is_failed():
            permission_outcome = PermissionOutcome.Deny(
                format_hook_message(
                    pre_hook_result, f"PreToolUse hook failed for tool `{tool_name}`"
                )
            )
        elif pre_hook_result.is_denied():
            permission_outcome = PermissionOutcome.Deny(
                format_hook_message(
                    pre_hook_result, f"PreToolUse hook denied tool `{tool_name}`"
                )
            )
        else:
            permission_outcome = self.permission_policy.authorize_with_context(
                tool_name, effective_input, permission_context, prompter
            )

        if not permission_outcome.allowed:
            return ConversationMessage.tool_result(
                tool_use_id,
                tool_name,
                merge_hook_feedback(
                    pre_hook_result.messages(), permission_outcome.reason, True
                ),
                True,
            )

        self.record_tool_started(iterations, tool_name)
        try:
            output = self.tool_executor.execute(tool_name, effective_input)
            is_error = False
        except (ToolError, Exception) as error:  # noqa: BLE001
            # A tool failure is data for the model, not a crash.
            output = str(error)
            is_error = True

        output = merge_hook_feedback(pre_hook_result.messages(), output, False)

        if is_error:
            post_hook_result = self.run_post_tool_use_failure_hook(
                tool_name, effective_input, output
            )
        else:
            post_hook_result = self.run_post_tool_use_hook(
                tool_name, effective_input, output
            )

        post_rejected = (
            post_hook_result.is_denied()
            or post_hook_result.is_failed()
            or post_hook_result.is_cancelled()
        )
        if post_rejected:
            is_error = True
        output = merge_hook_feedback(post_hook_result.messages(), output, post_rejected)

        return ConversationMessage.tool_result(tool_use_id, tool_name, output, is_error)

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def run_pre_tool_use_hook(self, tool_name: str, input: dict[str, Any]) -> HookResult:
        return self.hook_registry.run(
            HookEvent.PRE_TOOL_USE, {"tool_name": tool_name, "input": input}
        )

    def run_post_tool_use_hook(
        self, tool_name: str, effective_input: dict[str, Any], output: str
    ) -> HookResult:
        return self.hook_registry.run(
            HookEvent.POST_TOOL_USE,
            {"tool_name": tool_name, "input": effective_input, "output": output},
        )

    def run_post_tool_use_failure_hook(
        self, tool_name: str, effective_input: dict[str, Any], output: str
    ) -> HookResult:
        return self.hook_registry.run(
            HookEvent.POST_TOOL_USE_FAILURE,
            {"tool_name": tool_name, "input": effective_input, "output": output},
        )

    # ------------------------------------------------------------------
    # context management
    # ------------------------------------------------------------------

    def maybe_auto_compact(self) -> Optional[CompactionRecord]:
        if not should_compact(self.session, self.compaction_config):
            return None
        result = compact_session(
            self.session, self.compaction_config, self.api_client.complete
        )
        if result.record is not None:
            self.session_tracer.emit(
                "auto_compaction",
                dropped_messages=result.record.dropped_messages,
                before_tokens=result.record.before_tokens,
                after_tokens=result.record.after_tokens,
            )
        return result.record

    def run_session_health_probe(self) -> None:
        """Cheapest possible check that the rewritten history is still coherent."""
        if not self.session.messages:
            raise ValueError("session is empty after compaction")
        for message in self.session.messages:
            if message.role not in {"user", "assistant", "tool"}:
                raise ValueError(f"unknown role `{message.role}` in history")
            message.to_wire()

    def estimated_tokens(self) -> int:
        return estimate_session_tokens(self.session)

    def usage(self) -> Any:
        return self.usage_tracker.cumulative_usage()

    def compact(self, config: CompactionConfig) -> Any:
        return compact_session(self.session, config, self.api_client.complete)

    def fork_session(self, branch_name: Optional[str] = None) -> Session:
        return self.session.fork_session(branch_name)

    # ------------------------------------------------------------------
    # trace events, emitted inline with execution
    # ------------------------------------------------------------------

    def record_turn_started(self, user_input: str) -> None:
        self.session_tracer.emit("turn_started", chars=len(user_input))

    def record_assistant_iteration(
        self, iterations: int, assistant_message: ConversationMessage, pending: int
    ) -> None:
        self.session_tracer.emit(
            "assistant_iteration",
            iteration=iterations,
            text_chars=len(assistant_message.text()),
            pending_tool_uses=pending,
        )

    def record_tool_started(self, iterations: int, tool_name: str) -> None:
        self.session_tracer.emit(
            "tool_started", iteration=iterations, tool_name=tool_name
        )

    def record_tool_finished(
        self, iterations: int, result_message: ConversationMessage
    ) -> None:
        self.session_tracer.emit(
            "tool_finished",
            iteration=iterations,
            tool_name=result_message.tool_name,
            is_error=result_message.is_error,
            output_chars=len(result_message.text()),
        )

    def record_turn_completed(self, summary: TurnSummary) -> None:
        self.session_tracer.emit(
            "turn_completed",
            iterations=summary.iterations,
            tool_results=len(summary.tool_results),
            input_tokens=summary.usage.input_tokens,
            output_tokens=summary.usage.output_tokens,
        )

    def record_turn_failed(self, iterations: int, error: Exception) -> None:
        self.session_tracer.emit(
            "turn_failed", iteration=iterations, error=str(error)
        )
