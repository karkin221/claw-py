"""A minimal reference implementation of the claw-code agentic loop."""

from .agents import (
    AGENT_SPECS,
    AgentConfig,
    AgentSpec,
    allowed_tools_for_subagent,
    build_agent_system_prompt,
    build_tool_executor,
    execute_agent,
    make_agent_tool,
    normalize_subagent_type,
)
from .api import ApiClient, build_assistant_message
from .compact import CompactionConfig, compact_session, should_compact
from .conversation import ConversationRuntime
from .hooks import HookEvent, HookRegistry, HookResult, merge_hook_feedback
from .permissions import (
    PermissionContext,
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
)
from .telemetry import SessionTracer
from .tools import ToolExecutor, ToolSpec, default_tool_executor
from .types import (
    ApiRequest,
    ContentBlock,
    ConversationMessage,
    Session,
    TurnSummary,
    Usage,
    UsageTracker,
)

__all__ = [
    "AGENT_SPECS",
    "AgentConfig",
    "AgentSpec",
    "ApiClient",
    "ApiRequest",
    "CompactionConfig",
    "ContentBlock",
    "ConversationMessage",
    "ConversationRuntime",
    "HookEvent",
    "HookRegistry",
    "HookResult",
    "PermissionContext",
    "PermissionMode",
    "PermissionOutcome",
    "PermissionPolicy",
    "Session",
    "SessionTracer",
    "ToolExecutor",
    "ToolSpec",
    "TurnSummary",
    "Usage",
    "UsageTracker",
    "allowed_tools_for_subagent",
    "build_agent_system_prompt",
    "build_assistant_message",
    "build_tool_executor",
    "compact_session",
    "default_tool_executor",
    "execute_agent",
    "make_agent_tool",
    "merge_hook_feedback",
    "normalize_subagent_type",
]
