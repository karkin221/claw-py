"""System prompt assembly.

Mirrors `runtime/src/prompt.rs`. Walks from the working directory up to the git
root (or stops at cwd if there isn't one), collecting instruction files in
priority order. Every non-duplicate file found contributes to the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_INSTRUCTION_FILES = ("CLAUDE.md", "CLAW.md", "AGENTS.md")
SCOPED_DIRS = (".claw", ".claude")

BASE_PROMPT = """You are claw, a command-line coding agent.

Work by calling tools. Prefer reading before writing. Take one concrete step at
a time and check the result before continuing. When the task is done, reply with
a short plain-text answer and stop calling tools.

Tools available: {tool_names}
Working directory: {cwd}"""


@dataclass
class MemoryFile:
    path: Path
    source: str
    chars: int


def discover_memory_files(cwd: Path) -> list[MemoryFile]:
    cwd = cwd.resolve()
    root = _git_root(cwd) or cwd

    directories: list[Path] = []
    current = cwd
    while True:
        directories.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent

    found: list[MemoryFile] = []
    seen: set[Path] = set()
    for directory in reversed(directories):
        candidates = [directory / name for name in ROOT_INSTRUCTION_FILES]
        candidates += [
            directory / scoped / "CLAUDE.md" for scoped in SCOPED_DIRS
        ]
        for candidate in candidates:
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            body = candidate.read_text(errors="replace")
            found.append(
                MemoryFile(
                    path=candidate,
                    source=_source_label(candidate),
                    chars=len(body),
                )
            )
    return found


def build_system_prompt(cwd: Path, tool_names: list[str]) -> tuple[str, list[MemoryFile]]:
    memory_files = discover_memory_files(cwd)
    sections = [
        BASE_PROMPT.format(tool_names=", ".join(tool_names), cwd=cwd.resolve())
    ]
    for memory_file in memory_files:
        body = memory_file.path.read_text(errors="replace").strip()
        if body:
            sections.append(f"# Project instructions ({memory_file.path.name})\n{body}")
    return "\n\n".join(sections), memory_files


def _git_root(start: Path) -> Path | None:
    current = start
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _source_label(path: Path) -> str:
    parent = path.parent.name
    if parent in SCOPED_DIRS:
        return f"{parent.lstrip('.')}_claude_md"
    return path.name.lower().replace(".md", "_md")
