"""Context compaction.

Mirrors `runtime/src/compact.rs`, with four corrections that a real run made
necessary. Each is a failure mode observed in a trace, not a hypothetical:

1. **The original request is pinned.** Compaction used to drop the first
   messages, which meant it eventually dropped the user's actual requirements.
   The agent then worked from a paraphrase of them.

2. **Summaries are evidence-bounded.** Asked for "what was done", a model given
   a requirements list will happily report the requirements as accomplishments.
   The prompt now demands a tool result as evidence for any completion claim.

3. **Summaries are never re-summarised.** Previous notes are carried forward
   verbatim and extended; feeding a summary back through the summariser
   degrades it into nonsense within a few rounds.

4. **One pass gets under the threshold.** The split point is chosen by
   simulation before any model call, and oversized tool arguments in the
   retained tail are elided, so compaction cannot fire repeatedly without
   making progress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .types import CompactionRecord, ContentBlock, ConversationMessage, Session

CHARS_PER_TOKEN = 4
COMPACT_MARKER = "[compacted history]"
ELIDED_MARKER = "elided by compaction"
# A retained write_file payload is the single largest thing in most sessions,
# and it is the most recoverable: the file is on disk and read_file exists.
MAX_RETAINED_ARG_CHARS = 1500
MIN_KEEP_LAST = 2

SUMMARY_SYSTEM_PROMPT = (
    "You maintain running notes for a coding agent whose older messages are "
    "being discarded. Your notes are all it will remember.\n\n"
    "Rules, in order of importance:\n"
    "1. Record only what the transcript demonstrates. A requirement appearing "
    "in a request is NOT evidence that it was done.\n"
    "2. Treat something as done only if a tool result confirms it. Quote the "
    "confirming detail (a filename, a byte count, an exit status).\n"
    "3. Never invent open issues, next steps, or features. If the transcript "
    "does not mention it, it does not go in the notes.\n"
    "4. Record failures, including which tool failed and why. Repeated "
    "failures matter more than successes.\n"
    "5. If existing notes are supplied, extend them. Do not rewrite or "
    "re-summarise what is already there.\n\n"
    "Output exactly these sections, omitting any that are empty:\n"
    "CONFIRMED: things a tool result proves happened.\n"
    "FAILED: tool calls that errored, and the error.\n"
    "CONTEXT: facts learned that are not completions.\n\n"
    "Under 250 words. No preamble, no commentary, no speculation."
)


@dataclass
class CompactionConfig:
    # Half of api.DEFAULT_NUM_CTX. The two are coupled deliberately: a
    # threshold above the context window means the server truncates before
    # compaction ever fires, and the harness never learns it happened.
    threshold_tokens: int = 8192
    keep_last: int = 6


@dataclass
class CompactionResult:
    session: Session
    record: Optional[CompactionRecord]


def estimate_message_chars(message: ConversationMessage) -> int:
    """Characters this message contributes to the next request.

    Measured from `to_wire()`, not `text()`: a tool_use block carries its
    arguments in `block.input`, invisible to `text()` but serialised into
    `tool_calls` and re-sent on every later request.
    """
    return len(json.dumps(message.to_wire(), ensure_ascii=False))


def estimate_session_tokens(session: Session) -> int:
    """Cheap character-based estimate. Good enough to drive a threshold."""
    total = sum(estimate_message_chars(m) for m in session.messages)
    return total // CHARS_PER_TOKEN


def should_compact(session: Session, config: CompactionConfig) -> bool:
    if len(session.messages) <= config.keep_last + 1:
        return False
    return estimate_session_tokens(session) > config.threshold_tokens


def is_compaction_note(message: ConversationMessage) -> bool:
    return message.text().startswith(COMPACT_MARKER)


def format_compact_summary(summary: str) -> str:
    return f"{COMPACT_MARKER}\n{summary}"


def get_compact_continuation_message(summary: str) -> ConversationMessage:
    return ConversationMessage.user_text(
        format_compact_summary(summary)
        + "\n\nThe messages above were summarised to save context. The original "
        "request is still the first message in this conversation - re-read it "
        "before deciding what remains to be done."
    )


def elide_large_tool_args(message: ConversationMessage) -> ConversationMessage:
    """Replace oversized tool arguments with a placeholder.

    A 13KB write_file payload re-sent every iteration is the main driver of
    context growth, and the most recoverable thing in the history: the file is
    on disk. The call and its result stay; only the bulk goes.
    """
    if not any(b.kind == "tool_use" for b in message.blocks):
        return message

    blocks: list[ContentBlock] = []
    changed = False
    for block in message.blocks:
        if block.kind != "tool_use":
            blocks.append(block)
            continue
        trimmed: dict = {}
        for key, value in block.input.items():
            if isinstance(value, str) and len(value) > MAX_RETAINED_ARG_CHARS:
                trimmed[key] = (
                    f"[{len(value)} chars {ELIDED_MARKER}; re-read the file if needed]"
                )
                changed = True
            else:
                trimmed[key] = value
        blocks.append(ContentBlock.tool_use(block.id, block.name, trimmed))

    if not changed:
        return message
    return ConversationMessage(
        role=message.role,
        blocks=blocks,
        tool_use_id=message.tool_use_id,
        tool_name=message.tool_name,
        is_error=message.is_error,
    )


def pinned_count(messages: list[ConversationMessage]) -> int:
    """The original request is never compacted away."""
    return 1 if messages and messages[0].role == "user" else 0


def choose_split(
    messages: list[ConversationMessage], config: CompactionConfig, pinned: int
) -> int:
    """Pick how much to drop, by simulation, before spending a model call.

    Walks the split point forward until the projected result fits under the
    threshold, so one pass actually makes progress instead of firing again on
    the next iteration.
    """
    budget = config.threshold_tokens * CHARS_PER_TOKEN
    pinned_chars = sum(estimate_message_chars(m) for m in messages[:pinned])
    note_chars = 1200  # generous allowance for the continuation message

    latest = len(messages) - MIN_KEEP_LAST
    split = max(min(len(messages) - config.keep_last, latest), pinned)

    while split < latest:
        tail = [elide_large_tool_args(m) for m in messages[split:]]
        projected = pinned_chars + note_chars + sum(estimate_message_chars(m) for m in tail)
        if projected <= budget:
            break
        split += 1

    split = min(max(split, pinned), latest)
    # Never split so that a tool result is orphaned from its tool call.
    while split < len(messages) and messages[split].role == "tool":
        split += 1
    return split


def build_transcript(messages: list[ConversationMessage]) -> str:
    """Render messages for the summariser, skipping prior notes.

    Existing notes are passed separately so the model extends rather than
    re-summarises them.
    """
    lines: list[str] = []
    for message in messages:
        if is_compaction_note(message):
            continue
        if message.role == "assistant":
            for block in message.blocks:
                if block.kind == "tool_use":
                    args = json.dumps(block.input, ensure_ascii=False)[:300]
                    lines.append(f"assistant called {block.name}({args})")
            text = message.text().strip()
            if text:
                lines.append(f"assistant: {text[:600]}")
        elif message.role == "tool":
            status = "FAILED" if message.is_error else "ok"
            lines.append(f"tool {message.tool_name} [{status}]: {message.text()[:400]}")
        else:
            lines.append(f"user: {message.text()[:600]}")
    return "\n".join(lines)


def compact_session(
    session: Session,
    config: CompactionConfig,
    summarize: Callable[[str, str], str],
) -> CompactionResult:
    """Replace the middle of the history with notes, keeping the request."""
    before_tokens = estimate_session_tokens(session)
    messages = session.messages
    pinned = pinned_count(messages)
    split = choose_split(messages, config, pinned)

    head = messages[pinned:split]
    if not head:
        return CompactionResult(session=session, record=None)

    transcript = build_transcript(head)
    if not transcript.strip():
        return CompactionResult(session=session, record=None)

    prior_notes = session.compaction.summary if session.compaction else ""
    request = ""
    if prior_notes:
        request += f"EXISTING NOTES (extend these, do not rewrite):\n{prior_notes}\n\n"
    request += f"NEW TRANSCRIPT TO FOLD IN:\n{transcript}"

    summary = summarize(SUMMARY_SYSTEM_PROMPT, request).strip()
    if not summary:
        summary = prior_notes or "(summary unavailable)"

    tail = [elide_large_tool_args(m) for m in messages[split:]]
    session.messages = [
        *messages[:pinned],
        get_compact_continuation_message(summary),
        *tail,
    ]
    after_tokens = estimate_session_tokens(session)

    record = CompactionRecord(
        summary=summary,
        dropped_messages=len(head),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
    )
    session.compaction = record
    return CompactionResult(session=session, record=record)
