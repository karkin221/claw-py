"""MCP (Model Context Protocol) client over stdio.

Mirrors `runtime/src/mcp_tool_bridge.rs` and the MCP transport crate.

An MCP server is a subprocess speaking JSON-RPC 2.0 over newline-delimited
stdin/stdout. This module spawns it, negotiates a session, lists its tools, and
wraps each one in an ordinary `ToolSpec`.

That last step is the whole point. Once a remote tool is a `ToolSpec`, it enters
the same registry as `read_file` and inherits the identical pipeline: PreToolUse
hook, permission check, execute, PostToolUse hook. There is no separate path for
remote tools, so there is no way for one to skip a gate.

Bridged tools are namespaced `mcp__{server}__{tool}` and default to
`RISK_ESCALATE`, so under `workspace-write` they require explicit approval. A
remote tool is the last thing that should be auto-allowed just because its name
is unfamiliar.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .tools import RISK_ESCALATE, ToolError, ToolSpec

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "claw-py", "version": "0.2.0"}
DEFAULT_TIMEOUT = 30.0


class McpError(Exception):
    """Transport or protocol failure. Surfaced to the model as a tool error."""


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    risk: str = RISK_ESCALATE

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "McpServerConfig":
        if "command" not in raw:
            raise McpError(f"mcp server `{name}` is missing `command`")
        return cls(
            name=name,
            command=str(raw["command"]),
            args=[str(a) for a in raw.get("args", [])],
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            cwd=raw.get("cwd"),
            timeout=float(raw.get("timeout", DEFAULT_TIMEOUT)),
            risk=str(raw.get("risk", RISK_ESCALATE)),
        )


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"mcp__{self.server}__{self.name}"


class McpClient:
    """One server subprocess. Synchronous request/response over stdio."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.server_info: dict[str, Any] = {}
        self._next_id = 0
        self._lock = threading.Lock()  # tools may be dispatched in parallel
        self._stderr: list[str] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        env = {**os.environ, **self.config.env}
        try:
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=self.config.cwd,
            )
        except (OSError, ValueError) as error:
            raise McpError(f"could not start mcp server `{self.config.name}`: {error}") from error

        threading.Thread(target=self._drain_stderr, daemon=True).start()

        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            },
        )
        self.server_info = result.get("serverInfo", {})
        self.notify("notifications/initialized", {})

    def stop(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
        finally:
            # stdin is closed above; the read pipes leak handles without this.
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-50]  # keep the tail only

    # ------------------------------------------------------------------
    # json-rpc
    # ------------------------------------------------------------------

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._request_locked(method, params)

    def _request_locked(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise McpError(f"mcp server `{self.config.name}` is not running")

        self._next_id += 1
        request_id = self._next_id
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        # Skip notifications and responses to other ids until ours arrives.
        while True:
            message = self._read()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                raise McpError(
                    f"{self.config.name}/{method} failed: "
                    f"{error.get('message', error)} (code {error.get('code')})"
                )
            return message.get("result") or {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise McpError(f"mcp server `{self.config.name}` is not running")
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise McpError(f"mcp server `{self.config.name}` closed its input: {error}") from error

    def _read(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise McpError(f"mcp server `{self.config.name}` is not running")
        line = process.stdout.readline()
        if not line:
            tail = "; ".join(self._stderr[-3:])
            raise McpError(
                f"mcp server `{self.config.name}` exited unexpectedly"
                + (f" (stderr: {tail})" if tail else "")
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise McpError(
                f"mcp server `{self.config.name}` sent malformed json: {line[:120]!r}"
            ) from error

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------

    def list_tools(self) -> list[McpTool]:
        result = self.request("tools/list", {})
        tools: list[McpTool] = []
        for raw in result.get("tools", []):
            name = raw.get("name")
            if not name:
                continue
            tools.append(
                McpTool(
                    server=self.config.name,
                    name=name,
                    description=raw.get("description", f"{name} (via {self.config.name})"),
                    input_schema=raw.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        result = self.request("tools/call", {"name": tool_name, "arguments": arguments})
        text = flatten_content(result.get("content", []))
        if result.get("isError"):
            # The server reported failure; make it a tool error so the runtime
            # marks the result is_error and the model can react.
            raise ToolError(text or f"{tool_name} reported an error")
        return text


def flatten_content(content: Any) -> str:
    """MCP returns a list of typed content blocks. Render them as plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(resource.get("text") or f"[resource {resource.get('uri', '?')}]")
        elif kind == "image":
            parts.append(f"[image {block.get('mimeType', 'unknown')}]")
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p)


class McpServerManager:
    """Starts a set of servers and bridges every tool they expose."""

    def __init__(self, configs: list[McpServerConfig]) -> None:
        self.configs = configs
        self.clients: dict[str, McpClient] = {}
        self.failures: dict[str, str] = {}

    def start_all(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for config in self.configs:
            client = McpClient(config)
            try:
                client.start()
                tools = client.list_tools()
            except (McpError, ToolError) as error:
                # A broken server degrades the tool list; it does not stop startup.
                self.failures[config.name] = str(error)
                client.stop()
                continue
            self.clients[config.name] = client
            specs.extend(bridge_mcp_tool(client, tool, config.risk) for tool in tools)
        return specs

    def stop_all(self) -> None:
        for client in self.clients.values():
            client.stop()
        self.clients.clear()

    def __enter__(self) -> "McpServerManager":
        self.start_all()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop_all()


def bridge_mcp_tool(client: McpClient, tool: McpTool, risk: str = RISK_ESCALATE) -> ToolSpec:
    """Wrap one remote tool as an ordinary ToolSpec."""

    def handler(input: dict[str, Any]) -> str:
        try:
            return client.call_tool(tool.name, input)
        except McpError as error:
            raise ToolError(str(error)) from error

    return ToolSpec(
        name=tool.qualified_name,
        description=f"[{tool.server}] {tool.description}",
        input_schema=normalize_schema(tool.input_schema),
        handler=handler,
        risk=risk,
    )


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """MCP schemas are JSON Schema, but providers are picky. Fill in the gaps."""
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def load_mcp_config(path: Path) -> list[McpServerConfig]:
    """Read a `.mcp.json`-style file: {"mcpServers": {"name": {...}}}."""
    raw = json.loads(path.read_text())
    servers = raw.get("mcpServers", raw)
    if not isinstance(servers, dict):
        raise McpError(f"{path}: expected an object of server definitions")
    return [McpServerConfig.from_dict(name, cfg) for name, cfg in servers.items()]
