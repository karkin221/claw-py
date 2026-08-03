"""Built-in tools and the executor.

Mirrors `tools/src/lib.rs`. A `ToolSpec` carries both the wire schema sent to
the model and the handler the executor dispatches to. Tool failures are
returned to the runtime as errors, which the runtime converts into an
`is_error` tool result rather than aborting the turn.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT_CHARS = 4000


class ToolError(Exception):
    """Raised by a handler. Becomes an is_error tool result, not a crash."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolExecutor:
    """Name -> handler dispatch. MCP tools would register here identically."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def wire_specs(self) -> list[dict[str, Any]]:
        return [spec.to_wire() for spec in self._specs.values()]

    def execute(self, tool_name: str, effective_input: dict[str, Any]) -> str:
        spec = self._specs.get(tool_name)
        if spec is None:
            raise ToolError(f"unknown tool `{tool_name}`")
        output = spec.handler(effective_input)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return output


# --------------------------------------------------------------------------
# built-in handlers
# --------------------------------------------------------------------------


def _resolve(path_value: str) -> Path:
    if not path_value:
        raise ToolError("`path` is required")
    return Path(path_value).expanduser().resolve()


def read_file(input: dict[str, Any]) -> str:
    path = _resolve(input.get("path", ""))
    if not path.is_file():
        raise ToolError(f"no such file: {path}")
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(lines, 1)) or "(empty)"


def write_file(input: dict[str, Any]) -> str:
    path = _resolve(input.get("path", ""))
    content = input.get("content", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def edit_file(input: dict[str, Any]) -> str:
    path = _resolve(input.get("path", ""))
    old_str = input.get("old_str", "")
    new_str = input.get("new_str", "")
    if not path.is_file():
        raise ToolError(f"no such file: {path}")
    body = path.read_text()
    occurrences = body.count(old_str)
    if occurrences == 0:
        raise ToolError("old_str not found in file")
    if occurrences > 1:
        raise ToolError(f"old_str is not unique ({occurrences} matches)")
    path.write_text(body.replace(old_str, new_str, 1))
    return f"edited {path}"


def glob_search(input: dict[str, Any]) -> str:
    pattern = input.get("pattern", "*")
    root = Path(input.get("path", ".")).expanduser().resolve()
    hits = [
        str(p)
        for p in root.rglob("*")
        if p.is_file()
        and fnmatch.fnmatch(p.name, pattern)
        and ".git" not in p.parts
    ]
    return "\n".join(sorted(hits)[:200]) or "no matches"


def grep_search(input: dict[str, Any]) -> str:
    pattern = input.get("pattern", "")
    root = Path(input.get("path", ".")).expanduser().resolve()
    if not pattern:
        raise ToolError("`pattern` is required")
    try:
        regex = re.compile(pattern)
    except re.error as error:
        raise ToolError(f"bad regex: {error}") from error

    hits: list[str] = []
    targets = [root] if root.is_file() else list(root.rglob("*"))
    for candidate in targets:
        if not candidate.is_file() or ".git" in candidate.parts:
            continue
        try:
            for lineno, line in enumerate(candidate.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{candidate}:{lineno}: {line.strip()}")
                    if len(hits) >= 100:
                        return "\n".join(hits)
        except OSError:
            continue
    return "\n".join(hits) or "no matches"


def bash(input: dict[str, Any]) -> str:
    command = input.get("command", "")
    if not command:
        raise ToolError("`command` is required")
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=os.getcwd(),
    )
    parts = []
    if completed.stdout:
        parts.append(completed.stdout.rstrip())
    if completed.stderr:
        parts.append(f"[stderr] {completed.stderr.rstrip()}")
    if completed.returncode != 0:
        raise ToolError(
            f"exit {completed.returncode}\n" + "\n".join(parts)
        )
    return "\n".join(parts) or "(no output)"


def todo_write(input: dict[str, Any]) -> str:
    todos = input.get("todos", [])
    if not isinstance(todos, list):
        raise ToolError("`todos` must be a list")
    rendered = "\n".join(
        f"[{'x' if str(t.get('status', '')) == 'completed' else ' '}] {t.get('content', t)}"
        if isinstance(t, dict)
        else f"[ ] {t}"
        for t in todos
    )
    return f"todo list updated:\n{rendered}"


def default_tool_executor() -> ToolExecutor:
    executor = ToolExecutor()
    for spec in [
        ToolSpec(
            "read_file",
            "Read a file from disk with line numbers.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            read_file,
        ),
        ToolSpec(
            "write_file",
            "Write content to a file, creating it if needed.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            write_file,
        ),
        ToolSpec(
            "edit_file",
            "Replace a unique string in an existing file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            edit_file,
        ),
        ToolSpec(
            "glob_search",
            "Find files by filename glob pattern.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
            glob_search,
        ),
        ToolSpec(
            "grep_search",
            "Search file contents by regular expression.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
            grep_search,
        ),
        ToolSpec(
            "bash",
            "Run a shell command in the current working directory.",
            {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            bash,
        ),
        ToolSpec(
            "todo_write",
            "Record or update the working todo list.",
            {
                "type": "object",
                "properties": {"todos": {"type": "array", "items": {"type": "object"}}},
                "required": ["todos"],
            },
            todo_write,
        ),
    ]:
        executor.register(spec)
    return executor
