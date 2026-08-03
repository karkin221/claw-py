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
from .mcp import (
    McpClient,
    McpError,
    McpServerConfig,
    McpServerManager,
    McpTool,
    bridge_mcp_tool,
    load_mcp_config,
)
from .persistence import (
    SessionInfo,
    deserialize_message,
    list_sessions,
    replay_session,
    serialize_message,
)
from .hooks import HookEvent, HookRegistry, HookResult, merge_hook_feedback
from .permissions import (
    PermissionContext,
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
)
from .telemetry import SessionTracer
from .tools import (
    RISK_ESCALATE,
    RISK_READ,
    RISK_WRITE,
    ToolExecutor,
    ToolSpec,
    default_tool_executor,
)
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
    "McpClient",
    "McpError",
    "McpServerConfig",
    "McpServerManager",
    "McpTool",
    "PermissionContext",
    "PermissionMode",
    "PermissionOutcome",
    "PermissionPolicy",
    "RISK_ESCALATE",
    "RISK_READ",
    "RISK_WRITE",
    "Session",
    "SessionInfo",
    "SessionTracer",
    "ToolExecutor",
    "ToolSpec",
    "TurnSummary",
    "Usage",
    "UsageTracker",
    "allowed_tools_for_subagent",
    "bridge_mcp_tool",
    "build_agent_system_prompt",
    "build_assistant_message",
    "build_tool_executor",
    "compact_session",
    "default_tool_executor",
    "deserialize_message",
    "execute_agent",
    "list_sessions",
    "load_mcp_config",
    "make_agent_tool",
    "merge_hook_feedback",
    "normalize_subagent_type",
    "replay_session",
    "serialize_message",
]
