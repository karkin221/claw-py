"""Session tracing.

Mirrors `telemetry/src/lib.rs`. Events are emitted inline with execution rather
than reconstructed afterwards, so the resulting JSONL is sufficient on its own
to replay a turn.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO


class SessionTracer:
    def __init__(self, session_id: str, path: Optional[Path] = None, echo: bool = False) -> None:
        self.session_id = session_id
        self.echo = echo
        self._handle: Optional[TextIO] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a")

    def emit(self, kind: str, **fields: Any) -> None:
        event = {
            "ts": round(time.time(), 3),
            "session_id": self.session_id,
            "kind": kind,
            **fields,
        }
        line = json.dumps(event)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()
        if self.echo:
            print(f"    · {line}", file=sys.stderr)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
