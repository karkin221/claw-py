"""Context compaction.

Mirrors `runtime/src/compact.rs`. Called from inside the turn loop, after the
assistant message lands and before the tool-use branch is evaluated — so it
also runs on the terminal no-tool iteration, which is what stops history from
growing without bound across turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .types import CompactionRecord, ConversationMessage, Session

CHARS_PER_TOKEN = 4

SUMMARY_SYSTEM_PROMPT = (
    "You compress conversation transcripts. Produce a terse factual summary of "
    "what was asked, what was done, what was learned, and what remains open. "
    "No preamble, no commentary. Under 200 words."
)


@dataclass
class CompactionConfig:
    threshold_tokens: int = 3000
    keep_last: int = 4


@dataclass
class CompactionResult:
    session: Session
    record: Optional[CompactionRecord]


def estimate_session_tokens(session: Session) -> int:
    """Cheap character-based estimate. Good enough to drive a threshold."""
    total = sum(len(message.text()) for message in session.messages)
    return total // CHARS_PER_TOKEN


def should_compact(session: Session, config: CompactionConfig) -> bool:
    if len(session.messages) <= config.keep_last + 1:
        return False
    return estimate_session_tokens(session) > config.threshold_tokens


def format_compact_summary(summary: str) -> str:
    return f"[compacted history]\n{summary}"


def get_compact_continuation_message(summary: str) -> ConversationMessage:
    return ConversationMessage.user_text(
        format_compact_summary(summary)
        + "\n\nContinue from here. The messages above were summarised to save context."
    )


def compact_session(
    session: Session,
    config: CompactionConfig,
    summarize: Callable[[str, str], str],
) -> CompactionResult:
    """Replace the head of the history with a generated summary.

    `summarize(system_prompt, user_text) -> str` is injected so this module
    stays independent of the provider.
    """
    before_tokens = estimate_session_tokens(session)

    split_at = max(0, len(session.messages) - config.keep_last)
    # Never split in the middle of an assistant/tool-result pair.
    while split_at < len(session.messages) and session.messages[split_at].role == "tool":
        split_at += 1

    head = session.messages[:split_at]
    tail = session.messages[split_at:]
    if not head:
        return CompactionResult(session=session, record=None)

    transcript = "\n".join(
        f"{message.role}: {message.text()[:800]}" for message in head
    )
    summary = summarize(SUMMARY_SYSTEM_PROMPT, transcript) or "(summary unavailable)"

    session.messages = [get_compact_continuation_message(summary), *tail]
    after_tokens = estimate_session_tokens(session)

    record = CompactionRecord(
        summary=summary,
        dropped_messages=len(head),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
    )
    session.compaction = record
    return CompactionResult(session=session, record=record)
