# Architecture

What each stage of the agentic loop does, in execution order.

Every section names the Python file in this repo and the Rust file in
`ultraworkers/claw-code` it was derived from, so you can read either side by side.

<p align="center">
  <img src="docs/agent-loop.svg" alt="The claw-py agentic turn loop" width="680">
</p>

**Contents**

- [The shape of the thing](#the-shape-of-the-thing)
- [Stage 1 — Turn start](#stage-1--turn-start)
- [Stage 2 — Assemble request](#stage-2--assemble-request)
- [Stage 3 — Stream from provider](#stage-3--stream-from-provider)
- [Stage 4 — Auto-compact check](#stage-4--auto-compact-check)
- [Stage 5 — Any tool calls pending?](#stage-5--any-tool-calls-pending)
- [Stage 6 — PreToolUse hook](#stage-6--pretooluse-hook)
- [Stage 7 — Permission check](#stage-7--permission-check)
- [Stage 8 — Execute tool](#stage-8--execute-tool)
- [Stage 9 — PostToolUse hook](#stage-9--posttooluse-hook)
- [Stage 10 — Loop back](#stage-10--loop-back)
- [Stage 11 — Turn summary](#stage-11--turn-summary)
- [Cross-cutting: tracing and replay](#cross-cutting-tracing-and-replay)
- [Three design decisions worth stealing](#three-design-decisions-worth-stealing)
- [Extending it](#extending-it)

---

## The shape of the thing

An agent harness is, structurally, one `while` loop with a gate in the middle.

The loop asks a model what to do, does it, tells the model what happened, and
asks again — until the model stops asking for things. Everything interesting is
in the gate: what gets to run, who can veto it, and what the model is told when
something is refused.

Two properties are worth holding in mind as you read:

1. **The model never sees the machinery.** It sees a system prompt, a message
   history, and tool results. It does not know a hook rewrote its arguments or
   that a policy denied a call — it only sees the resulting text.
2. **Nothing in the tool pipeline raises.** Denials and failures both come back
   as tool results with `is_error: true`. The model reads them and adapts. The
   only thing that aborts a turn is the iteration cap or a provider failure.

---

## Stage 1 — Turn start

**Python:** `claw_py/conversation.py` → `run_turn`, opening block
**Rust:** `runtime/src/conversation.rs`

Before anything else, a guard runs — but only if this session has been compacted
before:

```python
if self.session.compaction is not None:
    try:
        self.run_session_health_probe()
    except Exception as error:
        raise RuntimeError(
            f"Session health probe failed after compaction: {error}. "
            "The session may be in an inconsistent state. "
            "Consider starting a fresh session."
        )
```

The Rust source calls this the **session-health canary**. The reasoning: compaction
rewrites message history in place. A malformed rewrite doesn't fail loudly — it
quietly poisons every subsequent request in the session. It's far cheaper to
detect that once, at the start of the next turn, than to debug it later.

The probe itself is deliberately cheap: confirm the history isn't empty, confirm
every role is one of `user` / `assistant` / `tool`, and confirm every message
still serialises to the wire format.

Then two bookkeeping steps:

```python
self.record_turn_started(user_input)
self.session.push_user_text(user_input)
```

The first emits a trace event. The second appends the user's message to
`session.messages`.

---

## Stage 2 — Assemble request

**Python:** `claw_py/conversation.py`, `claw_py/prompt.py`
**Rust:** `runtime/src/conversation.rs`, `runtime/src/prompt.rs`

Request construction is deliberately thin:

```python
request = ApiRequest(
    system_prompt=self.system_prompt,
    messages=list(self.session.messages),
)
```

**The entire history is copied on every iteration.** There is no incremental
append to a wire buffer. This is the single biggest cost driver in the loop, and
it is exactly why Stage 4 exists.

The `system_prompt` was built once at startup, not per iteration.
`prompt.build_system_prompt` walks from the working directory up to the git root
(or stops at cwd if there isn't one) and collects instruction files in priority
order:

1. `CLAUDE.md`
2. `CLAW.md`
3. `AGENTS.md`
4. scoped variants under `.claw/` and `.claude/`

Every non-duplicate file found contributes to the rendered prompt. Files closer
to the working directory are appended later, so they read as refinements of the
broader project instructions.

---

## Stage 3 — Stream from provider

**Python:** `claw_py/api.py` → `ApiClient.stream`, `build_assistant_message`
**Rust:** `api/src/providers/`

Two functions, cleanly split:

`ApiClient.stream()` opens the HTTP connection and yields normalised events:

| Event | Meaning |
|---|---|
| `text_delta` | A chunk of assistant prose |
| `tool_use` | The model wants to call a tool |
| `usage` | Token counts, emitted once at the end |
| `message_stop` | Stream complete |

`build_assistant_message()` folds that stream into a single
`ConversationMessage` plus an optional `Usage` record. It also strips `<think>`
blocks, which small reasoning models emit and which shouldn't reach the user.

The fold produces a list of `ContentBlock`s, and this is the branch point of the
entire loop:

```python
pending_tool_uses = [
    (block.id, block.name, block.input)
    for block in assistant_message.tool_uses()
]
```

The model's decision to keep working or to stop is *just whether that list is
empty*. There is no separate "am I done" signal, no stop token to interpret, no
completion classifier. The absence of tool calls is the completion signal.

The assistant message is pushed to the session **immediately**, before any tool
runs. If a tool later fails, the model's request to call it is already in the
history, so the failure has context.

---

## Stage 4 — Auto-compact check

**Python:** `claw_py/compact.py`, `conversation.maybe_auto_compact`
**Rust:** `runtime/src/compact.rs`

This runs *after* the assistant message is appended and *before* the tool-use
branch is evaluated. That placement is deliberate and load-bearing. The Rust
source carries a comment explaining it: compaction must also run on the terminal
no-tool iteration, or history grows without bound **across** turns rather than
within them.

Three functions:

```python
estimate_session_tokens(session)   # len(chars) // 4 — cheap, good enough
should_compact(session, config)    # compare against threshold
compact_session(session, config, summarize)
```

`compact_session` splits the history, asks the model to summarise the older
portion, and replaces it with that summary plus a continuation message. It never
splits in the middle of an assistant/tool-result pair — that would orphan a tool
result and confuse the model on the next request.

The `summarize` callable is injected rather than imported, so `compact.py` has no
dependency on the provider. That makes it trivially testable, and the test suite
does exactly that.

If compaction fires, a `CompactionRecord` is stored on the session (which is what
arms the Stage 1 health probe on the next turn) and carried into the turn summary
so the caller can surface it.

---

## Stage 5 — Any tool calls pending?

**Python:** `claw_py/conversation.py`
**Rust:** `runtime/src/conversation.rs`

Literally one line:

```python
if not pending_tool_uses:
    break
```

No tools means the model produced its final answer. The loop exits and Stage 11
builds the summary.

Separately, at the *top* of each iteration, the runaway guard:

```python
iterations += 1
if iterations > self.max_iterations:
    error = RuntimeError("conversation loop exceeded the maximum number of iterations")
    self.record_turn_failed(iterations, error)
    raise error
```

This is the circuit breaker. Small open-weights models hit it regularly — they'll
happily call `read_file` on the same path twenty times. The cap turns an infinite
loop and an unbounded bill into a clean, traced failure.

---

## Stage 6 — PreToolUse hook

**Python:** `claw_py/hooks.py`, `conversation._run_tool_call`
**Rust:** `runtime/src/hooks.rs`

**This is the most powerful stage in the system, and the one most worth
understanding.** A `PreToolUse` hook can do three distinct things:

**1. Rewrite the arguments.**

```python
effective_input = pre_hook_result.updated_input()
if effective_input is None:
    effective_input = dict(input)
```

Whatever the hook returns becomes what actually executes. The model is never told
its input was changed. Useful for injecting defaults, scoping paths, or
normalising sloppy arguments from a small model.

**2. Inject a permission override.**

```python
permission_context = PermissionContext(
    permission_override=pre_hook_result.permission_override(),
    permission_reason=pre_hook_result.permission_reason(),
)
```

The policy in Stage 7 checks this *first*, before its own mode logic. A hook can
therefore authorise something the configured mode forbids.

**3. Short-circuit entirely.**

```python
if pre_hook_result.is_cancelled():
    permission_outcome = PermissionOutcome.Deny(...)
elif pre_hook_result.is_failed():
    permission_outcome = PermissionOutcome.Deny(...)
elif pre_hook_result.is_denied():
    permission_outcome = PermissionOutcome.Deny(...)
else:
    permission_outcome = self.permission_policy.authorize_with_context(...)
```

Note the `else`. If a hook short-circuits, **the permission policy never runs at
all**.

> **The consequence:** hooks outrank permission modes in both directions. A hook
> can veto what the mode allows *and* allow what the mode forbids. If you are
> hardening a system built on this design, hooks are the real trust boundary —
> the modes are a convenience layer on top of them.

---

## Stage 7 — Permission check

**Python:** `claw_py/permissions.py` → `PermissionPolicy.authorize_with_context`
**Rust:** `runtime/src/permissions.rs`

Five modes, checked in this order: hook override → allowlist → mode.

| Mode | Behaviour |
|---|---|
| `read-only` | Denies `write_file`, `edit_file`, `bash`. Everything else allowed. |
| `workspace-write` | **Default.** Writes allowed inside the workspace root only; `bash` requires approval. |
| `danger-full-access` | Allows everything, no questions. |
| `prompt` | Every tool call requires approval. |
| `allow` | Allows everything. |

The escalation path is the interesting part:

```python
def _escalate(self, tool_name, effective_input, prompter):
    # No prompter means non-interactive: deny rather than silently allow.
    if prompter is None:
        return PermissionOutcome.Deny(
            f"`{tool_name}` needs approval but no prompter is attached"
        )
    if prompter.confirm(tool_name, effective_input):
        return PermissionOutcome.Allow()
    return PermissionOutcome.Deny(f"user declined `{tool_name}`")
```

`prompter` is `None` in one-shot and JSON-output modes and present in the
interactive REPL. **When there is nobody to ask, the answer is no.** That's the
mechanism behind flags like `--dangerously-skip-permissions` in the original: the
dangerous thing isn't skipping the prompt, it's changing the mode so the prompt is
never reached.

A denial is not an exception. It becomes a tool result:

```python
return ConversationMessage.tool_result(
    tool_use_id, tool_name,
    merge_hook_feedback(pre_hook_result.messages(), permission_outcome.reason, True),
    True,
)
```

The model reads *"`/etc/passwd` is outside the workspace root"* on its next
iteration and can try something else. This is what makes a restrictive policy
usable rather than merely obstructive.

---

## Stage 8 — Execute tool

**Python:** `claw_py/tools.py` → `ToolExecutor.execute`
**Rust:** `tools/src/lib.rs`

Dispatch by name over a registry. This repo ships seven tools: `read_file`,
`write_file`, `edit_file`, `glob_search`, `grep_search`, `bash`, `todo_write`.

The error handling is the design point:

```python
try:
    output = self.tool_executor.execute(tool_name, effective_input)
    is_error = False
except Exception as error:
    output = str(error)
    is_error = True
```

**A tool failure is data, not an exception.** The error text becomes the tool's
output with `is_error: true`, goes into the history, and the model sees it next
iteration. A model that tries to read a nonexistent file gets told so and can
`glob_search` instead — no human intervention, no crash.

`ToolExecutor.register()` is also the extension seam. Anything that produces a
`ToolSpec` — an MCP-bridged tool, a plugin, a subagent — registers here and
inherits the identical hook and permission path as the built-ins. That single
choke point is the property that makes the whole design worth copying.

---

## Stage 9 — PostToolUse hook

**Python:** `claw_py/hooks.py`, `conversation._run_tool_call`
**Rust:** `runtime/src/hooks.rs`

Two different hooks fire depending on outcome:

```python
if is_error:
    post_hook_result = self.run_post_tool_use_failure_hook(...)
else:
    post_hook_result = self.run_post_tool_use_hook(...)
```

That split lets you attach a formatter to successful writes and a diagnostic to
failed ones without either checking the other's condition.

A post-hook can also **flip a successful tool into an error**:

```python
post_rejected = (
    post_hook_result.is_denied()
    or post_hook_result.is_failed()
    or post_hook_result.is_cancelled()
)
if post_rejected:
    is_error = True
```

This is how you build a post-hoc validation gate — run a linter on written code,
reject output that fails schema validation, catch a `bash` command that succeeded
but produced something unacceptable.

Finally `merge_hook_feedback` splices hook commentary into the output the model
sees, prefixed as `[hook note]` or `[hook error]`, and the result is packed into a
`tool_result` message and appended to the session.

---

## Stage 10 — Loop back

Control returns to Stage 2. The session now contains the user message, the
assistant message, and one tool result per call — all of which get copied into
the next request.

This is why Stage 4's placement matters. A ten-iteration tool-heavy turn adds a
lot of history before the turn ever ends, and the compaction check is the only
thing standing between that and an unbounded context.

---

## Stage 11 — Turn summary

```python
summary = TurnSummary(
    assistant_messages=assistant_messages,
    tool_results=tool_results,
    iterations=iterations,
    usage=self.usage_tracker.cumulative_usage(),
    auto_compaction=auto_compaction,
)
self.record_turn_completed(summary)
```

Everything the caller needs to render the turn, bill for it, or decide whether to
warn the user that history was compacted.

---

## Cross-cutting: tracing and replay

**Python:** `claw_py/telemetry.py`
**Rust:** `telemetry/src/lib.rs`

Every stage has a paired `record_*` call:

```
record_turn_started
record_assistant_iteration
record_tool_started
record_tool_finished
record_turn_completed
record_turn_failed
```

The important property is that these are **emitted inline with execution**, not
reconstructed afterwards from logs. A session transcript is therefore sufficient
on its own to replay a turn deterministically — you don't need to correlate
across sources or infer ordering from timestamps.

```bash
python -m claw_py.cli --trace trace.jsonl "find every TODO comment"
```

```jsonl
{"ts": 1234.5, "session_id": "a1b2", "kind": "turn_started", "chars": 24}
{"ts": 1236.1, "session_id": "a1b2", "kind": "assistant_iteration", "iteration": 1, "pending_tool_uses": 1}
{"ts": 1236.2, "session_id": "a1b2", "kind": "tool_started", "iteration": 1, "tool_name": "grep_search"}
{"ts": 1236.4, "session_id": "a1b2", "kind": "tool_finished", "iteration": 1, "is_error": false}
```

Of everything in this codebase, this is the piece most worth lifting wholesale
into other systems. It costs almost nothing and it's very hard to retrofit.

---

## Three design decisions worth stealing

**1. Failure is data, not control flow.** Tool errors and permission denials both
become tool results with `is_error: true`. The model reads them and adapts. Only
the iteration cap and provider failures abort a turn. This single choice is what
lets an agent operate under a restrictive policy without a human babysitting it.

**2. One choke point for every tool.** Built-ins, plugins, MCP-bridged tools, and
subagents all pass through the same hook → permission → execute → hook pipeline.
Adding a capability cannot accidentally add a bypass, because there is only one
path.

**3. Trace at the point of action.** Emitting structured events inline — rather
than logging and reconstructing later — makes replay a property of the system
rather than a project.

And one to be careful about: **hooks outrank the permission policy in both
directions.** That's a powerful escape hatch and a real risk surface. Whatever
configures hooks is as trusted as the harness itself.

---

## Extending it

Both of these are small precisely because Stage 8 is a single choke point.

### Subagents (~40 lines)

Register an `agent` tool whose handler constructs a fresh `ConversationRuntime`:

```python
def execute_agent(input):
    subagent_type = input.get("subagent_type", "general-purpose")
    runtime = ConversationRuntime(
        api_client=ApiClient(model=...),
        tool_executor=default_tool_executor(),
        permission_policy=PermissionPolicy(
            mode=PermissionMode.READ_ONLY,
            allowed_tools=allowed_tools_for_subagent(subagent_type),
        ),
        system_prompt=build_agent_system_prompt(subagent_type),
        max_iterations=8,
    )
    return runtime.run_turn(input["prompt"]).assistant_messages[-1].text()
```

The original ships five types — general-purpose, explore, plan, verification, and
clawguide — each with its own prompt and restricted `allowed_tools` set. Depth
limiting is the only real subtlety: without it, an agent can spawn agents that
spawn agents.

### MCP tools

Anything producing a `ToolSpec` can be handed to `ToolExecutor.register()`.
Bridged tools then inherit the identical hook and permission path as the
built-ins — which is exactly the property you want, since remote tools are the
ones you least want bypassing your gates.

### Multi-provider routing

`ApiClient` is one class with one `stream()` method. Swapping in a router that
fans across local and hosted models is a drop-in replacement — nothing above it
in the stack knows or cares which model answered.
