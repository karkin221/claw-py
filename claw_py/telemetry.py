"""Session tracing.

Mirrors `telemetry/src/lib.rs`. Events are emitted inline with execution rather
than reconstructed afterwards, so the resulting JSONL is sufficient on its own
to replay a turn.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, TextIO


class SessionTracer:
    def __init__(self, session_id: str, path: Optional[Path] = None, echo: bool = False) -> None:
        self.session_id = session_id
        self.echo = echo
        self._handle: Optional[TextIO] = None
        self._owns_handle = False
        self._lock = threading.Lock()  # workers emit concurrently
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a")
            self._owns_handle = True

    def child(self, session_id: str) -> "SessionTracer":
        """A tracer for a nested runtime, writing to the same sink."""
        clone = SessionTracer.__new__(SessionTracer)
        clone.session_id = session_id
        clone.echo = self.echo
        clone._handle = self._handle
        clone._owns_handle = False
        clone._lock = self._lock  # one lock guards the shared sink
        return clone

    def emit(self, kind: str, **fields: Any) -> None:
        event = {
            "ts": round(time.time(), 3),
            "session_id": self.session_id,
            "kind": kind,
            **fields,
        }
        line = json.dumps(event)
        with self._lock:
            if self._handle is not None:
                self._handle.write(line + "\n")
                self._handle.flush()
            if self.echo:
                print(f"    · {line}", file=sys.stderr)

    def close(self) -> None:
        if self._handle is not None and self._owns_handle:
            self._handle.close()
        self._handle = None
