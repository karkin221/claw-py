"""The agentic loop.

Mirrors `runtime/src/conversation.rs`. This module is the whole architecture;
everything else is a collaborator it calls into. Read `run_turn` top to bottom
and you have the design.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from .tools import RISK_READ, ToolExecutor
from .types import (
    ApiRequest,
    CompactionRecord,
    ConversationMessage,
    RuntimeError,
    Session,
    TurnSummary,
    UsageTracker,
)


@dataclass
class ToolPlan:
    """One authorized (or denied) tool call, before execution."""

    tool_use_id: str
    tool_name: str
    effective_input: dict[str, Any]
    pre_hook_result: HookResult
    permission_outcome: PermissionOutcome


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
        parallel_tools: int = 1,
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
        self.parallel_tools = max(1, parallel_tools)
        self.usage_tracker = UsageTracker()
        self.on_text = on_text
        self.record_session_started()

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
        self.record_message_appended(self.session.messages[-1])

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
            self.record_message_appended(assistant_message)
            assistant_messages.append(assistant_message)

            # Runs before the next API call, including on the terminal
            # (no-tool) iteration, to prevent unbounded session growth.
            compaction = self.maybe_auto_compact()
            if compaction is not None:
                auto_compaction = compaction

            if not pending_tool_uses:
                break

            # Phase 1 (sequential, in order): hooks and permission decisions.
            # Kept serial so hook ordering is deterministic and only one
            # permission prompt can ever be in front of the user at a time.
            plans = [
                self._authorize_tool_call(tool_use_id, tool_name, input, prompter)
                for tool_use_id, tool_name, input in pending_tool_uses
            ]

            # Phase 2: execution. Runs of consecutive read-only tools go in
            # parallel; anything else runs in order.
            outcomes = self._dispatch_tool_plans(iterations, plans)

            # Phase 3 (sequential, in original order): post-hooks and results.
            for plan, (output, is_error) in zip(plans, outcomes):
                result_message = self._finalize_tool_call(plan, output, is_error)
                self.session.push_message(result_message)
                self.record_message_appended(result_message)
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

    def _authorize_tool_call(
        self,
        tool_use_id: str,
        tool_name: str,
        input: dict[str, Any],
        prompter: Optional[PermissionPrompter],
    ) -> ToolPlan:
        """Everything up to, but not including, running the tool."""
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

        return ToolPlan(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            effective_input=effective_input,
            pre_hook_result=pre_hook_result,
            permission_outcome=permission_outcome,
        )

    def _is_parallel_safe(self, plan: ToolPlan) -> bool:
        """Only idempotent reads run concurrently. Writes keep their order."""
        return self.tool_executor.risk_for(plan.tool_name) == RISK_READ

    def _dispatch_tool_plans(
        self, iterations: int, plans: list[ToolPlan]
    ) -> list[tuple[str, bool]]:
        """Execute authorized plans, batching consecutive read-only runs.

        Grouping into *runs* rather than partitioning by risk preserves the
        relative order of reads and writes: a write never overtakes a read that
        the model requested before it.
        """
        results: list[tuple[str, bool]] = [("", False)] * len(plans)
        batch: list[int] = []

        def flush() -> None:
            if not batch:
                return
            if len(batch) == 1:
                index = batch[0]
                results[index] = self._execute_tool_plan(iterations, plans[index])
            else:
                self.record_parallel_batch(iterations, [plans[i].tool_name for i in batch])
                workers = min(len(batch), self.parallel_tools)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(self._execute_tool_plan, iterations, plans[i]): i
                        for i in batch
                    }
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
            batch.clear()

        for index, plan in enumerate(plans):
            if not plan.permission_outcome.allowed:
                flush()
                results[index] = (
                    merge_hook_feedback(
                        plan.pre_hook_result.messages(),
                        plan.permission_outcome.reason,
                        True,
                    ),
                    True,
                )
                continue
            if self.parallel_tools > 1 and self._is_parallel_safe(plan):
                batch.append(index)
            else:
                flush()
                results[index] = self._execute_tool_plan(iterations, plan)
        flush()
        return results

    def _execute_tool_plan(self, iterations: int, plan: ToolPlan) -> tuple[str, bool]:
        """The parallel-safe part: run the tool, capture failure as data."""
        self.record_tool_started(iterations, plan.tool_name, plan.tool_use_id)
        try:
            output = self.tool_executor.execute(plan.tool_name, plan.effective_input)
            is_error = False
        except Exception as error:  # noqa: BLE001
            # A tool failure is data for the model, not a crash.
            output = str(error)
            is_error = True
        return merge_hook_feedback(plan.pre_hook_result.messages(), output, False), is_error

    def _finalize_tool_call(
        self, plan: ToolPlan, output: str, is_error: bool
    ) -> ConversationMessage:
        if not plan.permission_outcome.allowed:
            return ConversationMessage.tool_result(
                plan.tool_use_id, plan.tool_name, output, True
            )

        if is_error:
            post_hook_result = self.run_post_tool_use_failure_hook(
                plan.tool_name, plan.effective_input, output
            )
        else:
            post_hook_result = self.run_post_tool_use_hook(
                plan.tool_name, plan.effective_input, output
            )

        post_rejected = (
            post_hook_result.is_denied()
            or post_hook_result.is_failed()
            or post_hook_result.is_cancelled()
        )
        if post_rejected:
            is_error = True
        output = merge_hook_feedback(post_hook_result.messages(), output, post_rejected)

        return ConversationMessage.tool_result(
            plan.tool_use_id, plan.tool_name, output, is_error
        )

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
                summary=result.record.summary,
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

    def record_session_started(self) -> None:
        """Record everything the model saw that is not a message.

        The system prompt and tool list are sent on every request but never
        enter `session.messages`, so a trace without them cannot reproduce the
        conditions that produced its own history.
        """
        self.session_tracer.emit(
            "session_started",
            system_prompt=self.system_prompt,
            tool_names=self.offered_tool_names(),
            permission_mode=self.permission_policy.mode.as_str(),
            workspace_root=str(self.permission_policy.workspace_root),
            model=getattr(self.api_client, "model", ""),
            max_iterations=self.max_iterations,
        )

    def offered_tool_names(self) -> list[str]:
        """What this runtime could actually call.

        Not the same as the registry. A subagent shares the full executor and
        is narrowed by `allowed_tools`, so recording registry names would claim
        an explore agent had `write_file` when it was never offered it.
        """
        names = set(self.tool_executor.names())
        if self.permission_policy.allowed_tools is not None:
            names &= self.permission_policy.allowed_tools
        return sorted(names)

    def record_message_appended(self, message: ConversationMessage) -> None:
        """The event that makes the trace a complete, resumable log."""
        from .persistence import serialize_message

        self.session_tracer.emit(
            "message_appended", message=serialize_message(message)
        )

    def record_parallel_batch(self, iterations: int, tool_names: list[str]) -> None:
        self.session_tracer.emit(
            "parallel_batch", iteration=iterations, tools=sorted(tool_names)
        )

    def record_tool_started(
        self, iterations: int, tool_name: str, tool_use_id: str = ""
    ) -> None:
        self.session_tracer.emit(
            "tool_started",
            iteration=iterations,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
        )

    def record_tool_finished(
        self, iterations: int, result_message: ConversationMessage
    ) -> None:
        self.session_tracer.emit(
            "tool_finished",
            iteration=iterations,
            tool_name=result_message.tool_name,
            tool_use_id=result_message.tool_use_id,
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
