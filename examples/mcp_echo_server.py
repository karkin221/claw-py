#!/usr/bin/env python3
"""A minimal but real MCP server over stdio. Stdlib only.

Not a mock — it speaks actual JSON-RPC 2.0 and implements `initialize`,
`tools/list`, and `tools/call`. The client in `claw_py/mcp.py` is tested against
this, so the transport is exercised for real.

Exposes three tools:
  word_count   — count words in a string
  reverse      — reverse a string
  always_fails — returns isError, to exercise the failure path

Run it directly to poke at it:
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \\
      | python examples/mcp_echo_server.py
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "word_count",
        "description": "Count the words in a piece of text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "reverse",
        "description": "Reverse a string.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "always_fails",
        "description": "Always returns an error. For testing the failure path.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, arguments: dict) -> dict:
    if name == "word_count":
        count = len(str(arguments.get("text", "")).split())
        return {"content": [{"type": "text", "text": f"{count} words"}]}
    if name == "reverse":
        return {"content": [{"type": "text", "text": str(arguments.get("text", ""))[::-1]}]}
    if name == "always_fails":
        return {
            "content": [{"type": "text", "text": "this tool always fails"}],
            "isError": True,
        }
    return {
        "content": [{"type": "text", "text": f"unknown tool `{name}`"}],
        "isError": True,
    }


def handle(message: dict) -> dict | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "echo-server", "version": "0.1.0"},
        }
    elif method == "notifications/initialized":
        return None  # notifications get no reply
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = call_tool(params.get("name", ""), params.get("arguments") or {})
    elif method == "ping":
        result = {}
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
