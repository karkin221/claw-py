"""Multi-provider routing.

`RoutedApiClient` is a drop-in replacement for `ApiClient`. It implements the
same `stream()` contract — yielding `text_delta` / `tool_use` / `usage` /
`message_stop` events — so nothing above it in the stack knows or cares which
provider answered.

LiteLLM is an optional dependency:

    pip install litellm

The routing table maps short role names to concrete models, so a caller asks
for "the cheap local one" rather than naming a model. Each route carries an
ordered fallback chain; the client walks it on failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .api import ApiClient
from .types import ApiRequest, RuntimeError


@dataclass
class Route:
    """One routing target. `model` is a LiteLLM model string."""

    model: str
    api_base: Optional[str] = None
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# Role names, not model names. Rebind these to whatever you actually run.
DEFAULT_ROUTES: dict[str, list[Route]] = {
    "local-fast": [
        Route("ollama/qwen3:4b", api_base="http://localhost:11434"),
        Route("ollama/llama3.1:8b", api_base="http://localhost:11434"),
    ],
    "local-deep": [
        Route("ollama/qwen3:14b", api_base="http://localhost:11434"),
        Route("ollama/llama3.1:8b", api_base="http://localhost:11434"),
    ],
    "frontier": [
        Route("anthropic/claude-sonnet-4-6"),
        Route("openai/gpt-4.1"),
    ],
}

DEFAULT_ROLE = "local-fast"


class RoutedApiClient(ApiClient):
    """Same interface as ApiClient, backed by LiteLLM across many providers."""

    def __init__(
        self,
        role: str = DEFAULT_ROLE,
        routes: Optional[dict[str, list[Route]]] = None,
        tool_specs: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.0,
        on_route: Any = None,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "RoutedApiClient needs litellm. Install it with `pip install litellm`, "
                "or use the stdlib ApiClient against Ollama directly."
            ) from error

        self.routes = routes or dict(DEFAULT_ROUTES)
        if role not in self.routes:
            raise RuntimeError(
                f"unknown route `{role}`. Known roles: {', '.join(sorted(self.routes))}"
            )
        self.role = role
        self.tool_specs = tool_specs or []
        self.temperature = temperature
        self.on_route = on_route
        # `model` is exposed for display and for anything that reads it.
        self.model = self.routes[role][0].model

    # ------------------------------------------------------------------

    def stream(self, request: ApiRequest) -> Iterator[dict[str, Any]]:
        import litellm

        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(m.to_wire() for m in request.messages)

        chain = self.routes[self.role]
        failures: list[str] = []

        for attempt, route in enumerate(chain):
            kwargs: dict[str, Any] = {
                "model": route.model,
                "messages": messages,
                "stream": True,
                "temperature": route.temperature or self.temperature,
                **route.extra,
            }
            if route.api_base:
                kwargs["api_base"] = route.api_base
            if self.tool_specs:
                kwargs["tools"] = self.tool_specs

            try:
                response = litellm.completion(**kwargs)
                if self.on_route is not None:
                    self.on_route(self.role, route.model, attempt)
                self.model = route.model
                yield from self._decode(response)
                return
            except Exception as error:  # noqa: BLE001 - try the next route
                failures.append(f"{route.model}: {error}")

        raise RuntimeError(
            f"every route for `{self.role}` failed:\n  " + "\n  ".join(failures)
        )

    # ------------------------------------------------------------------

    def _decode(self, response: Any) -> Iterator[dict[str, Any]]:
        """Fold OpenAI-style streaming deltas into our event vocabulary.

        Tool calls arrive fragmented — name in one chunk, arguments split across
        several — so they are accumulated by index and emitted once complete.
        """
        pending: dict[int, dict[str, Any]] = {}
        usage: Any = None

        for chunk in response:
            usage = getattr(chunk, "usage", None) or usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                yield {"type": "text_delta", "text": content}

            for fragment in getattr(delta, "tool_calls", None) or []:
                index = getattr(fragment, "index", 0) or 0
                slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if getattr(fragment, "id", None):
                    slot["id"] = fragment.id
                function = getattr(fragment, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        slot["name"] = function.name
                    if getattr(function, "arguments", None):
                        slot["arguments"] += function.arguments

        for index in sorted(pending):
            slot = pending[index]
            if not slot["name"]:
                continue
            yield {
                "type": "tool_use",
                "id": slot["id"] or f"call_{index + 1}",
                "name": slot["name"],
                "input": _parse_arguments(slot["arguments"]),
            }

        yield {
            "type": "usage",
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        yield {"type": "message_stop"}


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def routed_client_factory(
    role: str,
    routes: Optional[dict[str, list[Route]]] = None,
    on_route: Any = None,
):
    """A `client_factory` for AgentConfig that routes subagents by role.

    Lets a cheap local model do the exploring while the parent stays on a
    stronger one — the usual reason to reach for routing in the first place.
    """

    def factory(requested: str) -> RoutedApiClient:
        return RoutedApiClient(role=requested or role, routes=routes, on_route=on_route)

    return factory
