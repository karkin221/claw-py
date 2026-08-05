"""Provider client.

Mirrors `api/src/providers/` — an SSE-style event stream that the runtime folds
into a single assistant message. Targets Ollama so the whole thing runs against
a small open-weights model with no API key.

Stdlib only: Ollama's /api/chat is newline-delimited JSON over plain HTTP.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from .types import ApiRequest, ContentBlock, ConversationMessage, RuntimeError, Usage

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b"
# Wall-clock ceiling for one request. A reasoning model generating a long file
# can hold a socket open for many minutes; the previous 300s was reached in
# practice, so this is both larger and configurable.
DEFAULT_REQUEST_TIMEOUT = 900.0


class ApiClient:
    """Streaming chat client. `tool_specs` are sent as native Ollama tools."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        tool_specs: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.tool_specs = tool_specs or []
        self.temperature = temperature
        self._call_counter = 0

    def next_call_id(self) -> str:
        """Monotonic across the client's lifetime, not per-stream.

        A per-stream counter restarts at 1 every iteration, so two tool calls in
        one turn can share `call_1`. That makes tool_use ids useless for joining
        a result back to its request in the trace, and would break any provider
        that correlates by id.
        """
        self._call_counter = getattr(self, "_call_counter", 0) + 1
        return f"call_{self._call_counter}"

    def stream(self, request: ApiRequest) -> Iterator[dict[str, Any]]:
        """Yield provider events.

        Event kinds: text_delta, tool_use, usage, message_stop.
        """
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(m.to_wire() for m in request.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        if self.tool_specs:
            payload["tools"] = self.tool_specs

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                yield from self._decode(response)
        except TimeoutError as error:
            # A read-phase timeout raises bare TimeoutError, not URLError, so it
            # used to escape every handler and crash the CLI with a traceback.
            raise RuntimeError(
                f"provider stopped sending after {time.monotonic() - started:.0f}s "
                f"(timeout {self.request_timeout:.0f}s). The model may still be "
                f"generating. Raise it with --request-timeout, or use a smaller "
                f"model for long outputs."
            ) from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, TimeoutError):
                raise RuntimeError(
                    f"could not reach {self.base_url} within "
                    f"{self.request_timeout:.0f}s. Is `ollama serve` running?"
                ) from error
            raise RuntimeError(
                f"provider request failed ({error}). Is `ollama serve` running "
                f"at {self.base_url} and `{self.model}` pulled?"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"connection to {self.base_url} broke mid-stream ({error})."
            ) from error

    def _decode(self, response: Any) -> Iterator[dict[str, Any]]:
        for raw in response:
            line = raw.decode().strip()
            if not line:
                continue
            chunk = json.loads(line)

            message = chunk.get("message") or {}
            content = message.get("content") or ""
            if content:
                yield {"type": "text_delta", "text": content}

            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw": arguments}
                yield {
                    "type": "tool_use",
                    "id": self.next_call_id(),
                    "name": function.get("name", ""),
                    "input": arguments,
                }

            if chunk.get("done"):
                yield {
                    "type": "usage",
                    "input_tokens": chunk.get("prompt_eval_count", 0),
                    "output_tokens": chunk.get("eval_count", 0),
                }
                yield {"type": "message_stop"}

    def complete(self, system_prompt: str, user_text: str) -> str:
        """Non-streaming one-shot. Used by compaction to write its summary."""
        request = ApiRequest(
            system_prompt=system_prompt,
            messages=[ConversationMessage.user_text(user_text)],
        )
        saved, self.tool_specs = self.tool_specs, []
        try:
            parts = [
                event["text"]
                for event in self.stream(request)
                if event["type"] == "text_delta"
            ]
        finally:
            self.tool_specs = saved
        return strip_reasoning("".join(parts)).strip()


def strip_reasoning(text: str) -> str:
    """Small reasoning models emit <think> blocks. Drop them before display."""
    while "<think>" in text and "</think>" in text:
        head, _, rest = text.partition("<think>")
        _, _, tail = rest.partition("</think>")
        text = head + tail
    return text


def build_assistant_message(
    events: Iterator[dict[str, Any]],
    on_text: Any = None,
) -> tuple[ConversationMessage, Usage | None]:
    """Fold a provider event stream into one assistant message plus usage.

    `on_text` is an optional callback for live terminal rendering.
    """
    blocks: list[ContentBlock] = []
    text_parts: list[str] = []
    usage: Usage | None = None

    try:
        for event in events:
            kind = event["type"]
            if kind == "text_delta":
                text_parts.append(event["text"])
                if on_text is not None:
                    on_text(event["text"])
            elif kind == "tool_use":
                blocks.append(
                    ContentBlock.tool_use(
                        id=event["id"], name=event["name"], input=event["input"]
                    )
                )
            elif kind == "usage":
                usage = Usage(
                    input_tokens=event["input_tokens"],
                    output_tokens=event["output_tokens"],
                )
    except RuntimeError as error:
        # Carry what arrived before the break. A turn that dies 200 seconds into
        # generation should not throw that output away silently.
        error.partial_text = strip_reasoning("".join(text_parts)).strip()
        raise

    text = strip_reasoning("".join(text_parts)).strip()
    if text:
        blocks.insert(0, ContentBlock.text_block(text))

    return ConversationMessage(role="assistant", blocks=blocks), usage
