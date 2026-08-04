"""Session persistence by replaying the trace.

The original serialises session state to its own `.jsonl` alongside a separate
telemetry stream. This does it the other way round: the trace *is* the store.

Because `SessionTracer` already emits an event at every point the session
changes, appending the message content to those events makes the trace a
complete, ordered log of everything that happened. Resuming is then a fold over
that log rather than a deserialisation of a snapshot — which means a resumed
session cannot silently disagree with its own trace, and a partially written
trace still resumes to its last consistent point.

Compaction replays as an *operation*, not as a rewritten history: the event
records how many messages were dropped and the summary that replaced them, and
`replay_session` applies exactly the transformation `compact_session` applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .compact import get_compact_continuation_message
from .types import CompactionRecord, ContentBlock, ConversationMessage, RuntimeError, Session


def serialize_message(message: ConversationMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "tool_use_id": message.tool_use_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
        "blocks": [
            {
                "kind": block.kind,
                "text": block.text,
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
            for block in message.blocks
        ],
    }


def deserialize_message(raw: dict[str, Any]) -> ConversationMessage:
    return ConversationMessage(
        role=raw.get("role", "user"),
        blocks=[
            ContentBlock(
                kind=block.get("kind", "text"),
                text=block.get("text", ""),
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input") or {},
            )
            for block in raw.get("blocks", [])
        ],
        tool_use_id=raw.get("tool_use_id", ""),
        tool_name=raw.get("tool_name", ""),
        is_error=bool(raw.get("is_error", False)),
    )


@dataclass
class SessionEnvironment:
    """The non-message conditions a session ran under.

    Recorded by `session_started`. Everything here is sent to the model on
    every request but lives outside `session.messages`, so without it a replay
    reproduces the conversation but not the conditions that produced it.
    """

    system_prompt: str = ""
    tool_names: list[str] = field(default_factory=list)
    permission_mode: str = ""
    workspace_root: str = ""
    model: str = ""
    max_iterations: int = 0


@dataclass
class SessionInfo:
    session_id: str
    turns: int = 0
    messages: int = 0
    first_prompt: str = ""
    last_ts: float = 0.0
    compactions: int = 0
    subagent_of: Optional[str] = None
    depth: int = 0

    @property
    def is_subagent(self) -> bool:
        return self.depth > 0


def read_events(path: Path) -> Iterator[dict[str, Any]]:
    """Stream events, skipping any truncated trailing line."""
    if not path.is_file():
        raise RuntimeError(f"no trace file at {path}")
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial final write; everything before it is intact


def list_sessions(path: Path, include_subagents: bool = False) -> list[SessionInfo]:
    """Summarise every session in a trace, newest last."""
    sessions: dict[str, SessionInfo] = {}
    order: list[str] = []

    for event in read_events(path):
        session_id = event.get("session_id")
        if not session_id:
            continue
        if session_id not in sessions:
            sessions[session_id] = SessionInfo(session_id=session_id)
            order.append(session_id)
        info = sessions[session_id]
        info.last_ts = max(info.last_ts, float(event.get("ts", 0)))

        kind = event.get("kind")
        if kind == "turn_started":
            info.turns += 1
        elif kind == "message_appended":
            info.messages += 1
            message = event.get("message") or {}
            if not info.first_prompt and message.get("role") == "user":
                info.first_prompt = "".join(
                    b.get("text", "") for b in message.get("blocks", [])
                )[:80]
        elif kind == "auto_compaction":
            info.compactions += 1
        elif kind == "subagent_started":
            info.depth = int(event.get("depth", 0))

    # subagent_started is emitted on the child's tracer and names its parent,
    # so a whole delegation tree reconstructs from one pass.
    for event in read_events(path):
        if event.get("kind") == "subagent_started":
            child = event.get("session_id")
            if child in sessions:
                sessions[child].subagent_of = event.get("parent")
                sessions[child].depth = int(event.get("depth", 1))

    result = [sessions[sid] for sid in order]
    if not include_subagents:
        result = [info for info in result if not info.is_subagent]
    return result


def replay_environment(
    path: Path, session_id: Optional[str] = None
) -> Optional[SessionEnvironment]:
    """The environment a session last ran under, or None if never recorded.

    Traces written before `session_started` existed return None; callers should
    fall back to rebuilding rather than failing.
    """
    if session_id is None:
        candidates = list_sessions(path)
        if not candidates:
            return None
        session_id = max(candidates, key=lambda info: info.last_ts).session_id

    environment: Optional[SessionEnvironment] = None
    for event in read_events(path):
        if event.get("session_id") != session_id or event.get("kind") != "session_started":
            continue
        # Last one wins: it is the environment the newest messages ran under.
        environment = SessionEnvironment(
            system_prompt=event.get("system_prompt") or "",
            tool_names=list(event.get("tool_names") or []),
            permission_mode=event.get("permission_mode") or "",
            workspace_root=event.get("workspace_root") or "",
            model=event.get("model") or "",
            max_iterations=int(event.get("max_iterations") or 0),
        )
    return environment


def describe_environment_drift(
    recorded: SessionEnvironment, current: SessionEnvironment
) -> list[str]:
    """Human-readable differences between a recorded and a current environment."""
    drift: list[str] = []
    if recorded.system_prompt and recorded.system_prompt != current.system_prompt:
        drift.append(
            f"system prompt changed ({len(recorded.system_prompt)} chars recorded, "
            f"{len(current.system_prompt)} now)"
        )
    if recorded.tool_names and set(recorded.tool_names) != set(current.tool_names):
        added = sorted(set(current.tool_names) - set(recorded.tool_names))
        removed = sorted(set(recorded.tool_names) - set(current.tool_names))
        parts = []
        if added:
            parts.append(f"added {', '.join(added)}")
        if removed:
            parts.append(f"removed {', '.join(removed)}")
        drift.append("tool list changed: " + "; ".join(parts))
    if recorded.permission_mode and recorded.permission_mode != current.permission_mode:
        drift.append(
            f"permission mode changed ({recorded.permission_mode} → {current.permission_mode})"
        )
    if recorded.workspace_root and recorded.workspace_root != current.workspace_root:
        drift.append(
            f"workspace changed ({recorded.workspace_root} → {current.workspace_root})"
        )
    if recorded.model and recorded.model != current.model:
        drift.append(f"model changed ({recorded.model} → {current.model})")
    return drift


def replay_session(path: Path, session_id: Optional[str] = None) -> Session:
    """Rebuild a Session by folding its trace events in order.

    With no `session_id`, the most recently active non-subagent session is used.
    """
    if session_id is None:
        candidates = list_sessions(path)
        if not candidates:
            raise RuntimeError(f"{path} contains no resumable sessions")
        session_id = max(candidates, key=lambda info: info.last_ts).session_id

    messages: list[ConversationMessage] = []
    compaction: Optional[CompactionRecord] = None
    system_prompt: Optional[str] = None
    seen = False

    for event in read_events(path):
        if event.get("session_id") != session_id:
            continue  # subagent traces share the file; they are separate sessions
        seen = True
        kind = event.get("kind")

        if kind == "session_started":
            system_prompt = event.get("system_prompt") or system_prompt

        elif kind == "message_appended":
            messages.append(deserialize_message(event.get("message") or {}))

        elif kind == "auto_compaction":
            # Apply the same operation compact_session applied, in the same order.
            dropped = int(event.get("dropped_messages", 0))
            summary = event.get("summary", "")
            if dropped and dropped <= len(messages):
                messages = [
                    get_compact_continuation_message(summary),
                    *messages[dropped:],
                ]
            compaction = CompactionRecord(
                summary=summary,
                dropped_messages=dropped,
                before_tokens=int(event.get("before_tokens", 0)),
                after_tokens=int(event.get("after_tokens", 0)),
            )

    if not seen:
        raise RuntimeError(f"no session `{session_id}` in {path}")

    return Session(
        session_id=session_id,
        messages=messages,
        compaction=compaction,
        system_prompt=system_prompt,
    )


def format_session_list(sessions: list[SessionInfo]) -> str:
    if not sessions:
        return "no resumable sessions"
    lines = [f"{'session':<14} {'turns':>5} {'msgs':>5} {'cmpt':>5}  first prompt"]
    for info in sorted(sessions, key=lambda i: i.last_ts):
        lines.append(
            f"{info.session_id:<14} {info.turns:>5} {info.messages:>5} "
            f"{info.compactions:>5}  {info.first_prompt}"
        )
    return "\n".join(lines)
