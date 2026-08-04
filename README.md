# claw-py

A minimal, readable reference implementation of the agentic loop found in
[`ultraworkers/claw-code`](https://github.com/ultraworkers/claw-code) — rewritten
in Python, running against a small open-weights model via Ollama.

The Rust original is roughly 35,000 lines across 11 crates. Most of that is
parity harnesses, MCP transport, provider-compatibility shims, and config
validation. **This repository strips all of it out and keeps only the
architecture**: the turn loop, the three-layer tool-gating pipeline, context
compaction, and inline tracing.

About 4,200 lines. Standard library only — no `pip install` required, including
the MCP client. Optional: `litellm`, if you want multi-provider routing.

---

## The architecture in one picture

<p align="center">
  <img src="docs/agent-loop.svg" alt="The claw-py agentic turn loop" width="680">
</p>

Purple stages are the model turn. Coral stages are the tool pipeline, which runs
once per pending tool call. Everything loops back to *assemble request* until the
model stops calling tools.

Same thing as text, if you prefer:

```
turn start
   └─> assemble request ──> stream from provider ──> auto-compact check
                                                            │
                                              any tool calls pending?
                                                     │             │
                                                    yes            no ──> turn summary
                                                     │
                                     ┌───────────────┴───────────────┐
                                     │  for each tool call:          │
                                     │    1. PreToolUse hook         │
                                     │    2. permission check        │
                                     │    3. execute tool            │
                                     │    4. PostToolUse hook        │
                                     └───────────────┬───────────────┘
                                                     │
                                        (loop back to assemble request)
```

**[→ Read `ARCHITECTURE.md` for what each stage does, line by line.](ARCHITECTURE.md)**

---

## Quickstart

### 1. Run it with no model at all

The fastest way to see the whole thing work. A scripted provider replays a fixed
sequence of tool calls, exercising every branch of the pipeline in a single turn.
No network, no Ollama, no API key.

```bash
git clone <this-repo> && cd claw-py
python examples/demo_offline.py
```

Expected output:

```
iterations       : 6
assistant msgs   : 6
tool results     : 5
usage            : Usage(input_tokens=720, output_tokens=240)
session messages : 12
estimated tokens : 178

ok  read_file        1  alpha ⏎  2  beta ⏎  3  gamma
ok  grep_search  [hook note] defaulted search path to cwd ⏎ ...
ERR bash         [hook error] command contains blocked fragment `rm -rf /`
ERR write_file   `/etc/claw-should-not-write` is outside the workspace root
ERR read_file    no such file: /tmp/claw-py-.../does-not-exist.txt
```

Those five lines are the whole gating design in miniature:

| Line | What it proves |
|---|---|
| `read_file` ok | Normal path: policy allows, tool succeeds |
| `grep_search` ok | A **PreToolUse hook rewrote the arguments** before execution |
| `bash` ERR | A **hook vetoed the call before the permission policy ever ran** |
| `write_file` ERR | The **policy** denied a path outside the workspace root |
| `read_file` ERR | A **tool failure became data for the model**, not a crash |

### 2. Run it against a real model

```bash
ollama serve
ollama pull qwen3:4b

python -m claw_py.cli "list the python files here and summarise what each does"
```

Other modes:

```bash
python -m claw_py.cli                                    # interactive REPL
python -m claw_py.cli --permission-mode read-only "..."  # no writes, no shell
python -m claw_py.cli --trace trace.jsonl -v "..."       # emit a replay trace
python -m claw_py.cli --model llama3.1:8b "..."          # a different model
```

`llama3.1:8b` is a more reliable tool-caller than `qwen3:4b` if you have the
memory. Anything below ~4B tends to emit malformed tool arguments or never stop
calling tools — which is itself a live demonstration of why `max_iterations`
exists.

### 3. Watch a subagent get delegated to

```bash
python examples/demo_subagents.py
```

Also offline. A parent agent delegates twice to an `explore` subagent:

```
PARENT
  iterations       : 3
  tool results     : 2
  session messages : 6

ok  agent   [explore-1a4d6d] map the package (4 iteration(s), 3 tool call(s)) ...

INSIDE THE SUBAGENTS (from the trace)
  explore-1a4d6d  spawned at depth 1  tools=glob_search,grep_search,read_file
  explore-1a4d6d  ok     glob_search
  explore-1a4d6d  ok     read_file
  explore-1a4d6d  ok     grep_search
  explore-6f4c6e  DENIED write_file

DEPTH LIMIT
  tools at depth 0 : True
  tools at depth 1 : False   (max_depth=1)
```

Four properties in one run: the subagent's three tool results **never entered the
parent's context** (only its final paragraph did), `write_file` was **denied even
though the parent runs in `workspace-write`**, the parent's hooks **still gated
the subagent**, and the `agent` tool **was not registered at all** at the depth
ceiling.

### 4. Route across providers

```bash
pip install litellm
python -m claw_py.cli --router litellm --model local-fast --subagent-model local-fast "..."
```

You ask for a *role* (`local-fast`, `local-deep`, `frontier`), not a model. Each
role carries an ordered fallback chain. See
[ARCHITECTURE.md](ARCHITECTURE.md#multi-provider-routing).

### 5. MCP tools, parallel dispatch, and resume

```bash
python examples/demo_advanced.py
```

Offline, but the MCP part spawns a **real server subprocess** speaking actual
JSON-RPC (`examples/mcp_echo_server.py`, ~110 lines of stdlib).

```
1. MCP — a real subprocess, bridged into the same registry
  server     : echo-server v0.1.0
  bridged    : mcp__echo__word_count, mcp__echo__reverse, mcp__echo__always_fails
  risk class : escalate (remote tools are never auto-allowed)
  no prompter: `mcp__echo__word_count` needs approval but no prompter is attached
  with prompter:
  ok  mcp__echo__word_count    5 words
  ERR mcp__echo__always_fails  this tool always fails

2. PARALLEL — same four reads, sequential vs concurrent
  sequential       1.60s   results in order: file1 → file2 → file3 → file4
  parallel (4)     0.40s   results in order: file1 → file2 → file3 → file4

3. RESUME — rebuild the session by replaying the trace
  replayed   : c816560ac05e → 8 messages
  roles      : user assi tool assi tool assi user assi
```

Using them for real:

```bash
python -m claw_py.cli --mcp-config .mcp.json "..."          # bridge MCP servers
python -m claw_py.cli --parallel-tools 4 "..."              # concurrent reads
python -m claw_py.cli --trace t.jsonl --list-sessions       # what can I resume?
python -m claw_py.cli --trace t.jsonl --resume "..."        # resume the latest
python -m claw_py.cli --trace t.jsonl --resume a1b2c3 "..." # resume by id
python -m claw_py.cli --trace t.jsonl --resume --rebuild-prompt "..."  # ignore
                                                            # the recorded prompt
```

### 6. Run the tests

```bash
python -m unittest discover -s tests -v
```

118 tests, no network, no model. They cover the loop, the iteration cap, tool
failures, all five permission modes, hook rewrite/deny/override, post-hook error
flipping, compaction, the session health probe, subagent tool restriction, mode
narrowing, depth limiting, hook inheritance, context isolation, the router's
fragmented-tool-call reassembly, MCP handshake/dispatch/concurrency against a
real subprocess, trace replay including compaction and truncated writes, and
parallel dispatch ordering guarantees, every permission-decision source, and three regressions
found by reading a real production trace (see the commit log).

---

## Repository layout

```
claw-py/
├── README.md               you are here
├── ARCHITECTURE.md         stage-by-stage explanation of the loop
├── docs/
│   └── agent-loop.svg      the diagram above
├── claw_py/
│   ├── conversation.py     ← the whole architecture lives here
│   ├── types.py            messages, blocks, session, usage
│   ├── api.py              Ollama client + event folding
│   ├── tools.py            tool specs and executor
│   ├── permissions.py      modes, policy, outcomes, prompter
│   ├── hooks.py            lifecycle hooks and feedback merging
│   ├── compact.py          token estimation and history rewriting
│   ├── agents.py           subagents: the agent tool, restriction, depth
│   ├── mcp.py              MCP stdio client and tool bridging
│   ├── persistence.py      resume by replaying the trace
│   ├── routing.py          optional LiteLLM multi-provider router
│   ├── prompt.py           system prompt assembly
│   ├── telemetry.py        inline trace events
│   └── cli.py              REPL, one-shot mode, demo hooks
├── examples/
│   ├── demo_offline.py     the gating pipeline, no model needed
│   ├── demo_subagents.py   delegation and isolation, no model needed
│   ├── demo_advanced.py    MCP, parallel dispatch, resume
│   └── mcp_echo_server.py  a real stdio MCP server, for testing
└── tests/
    ├── test_loop.py        the loop, gating, compaction
    ├── test_agents.py      subagents and routing
    └── test_infra.py       MCP, persistence, parallel dispatch
```

**If you read one file, read `claw_py/conversation.py`.** Everything else is a
collaborator it calls into. `run_turn` maps one-to-one onto the diagram.

---

## Where things came from

Names match the Rust originals wherever Python allows it.

| This repo | claw-code |
|---|---|
| `conversation.py` → `ConversationRuntime.run_turn` | `runtime/src/conversation.rs` |
| `types.py` | `runtime/src/` + `api/src/types.rs` |
| `api.py` → `ApiClient.stream`, `build_assistant_message` | `api/src/providers/` |
| `tools.py` → `ToolExecutor.execute` | `tools/src/lib.rs` |
| `permissions.py` → `PermissionPolicy.authorize_with_context` | `runtime/src/permissions.rs` |
| `hooks.py` → `HookEvent`, `merge_hook_feedback` | `runtime/src/hooks.rs` |
| `compact.py` → `should_compact`, `compact_session` | `runtime/src/compact.rs` |
| `prompt.py` → `build_system_prompt` | `runtime/src/prompt.rs` |
| `agents.py` → `execute_agent`, `allowed_tools_for_subagent` | `tools/src/lib.rs` |
| `mcp.py` → `McpClient`, `bridge_mcp_tool` | `runtime/src/mcp_tool_bridge.rs` |
| `persistence.py` → `replay_session` | (diverges — see below) |
| `routing.py` → `RoutedApiClient` | (no direct analogue — replaces `api/src/providers/`) |
| `telemetry.py` → `SessionTracer` | `telemetry/src/lib.rs` |
| `cli.py` | `rusty-claude-cli` |

Preserved verbatim: `run_turn`, `pending_tool_uses`, `effective_input`,
`PermissionOutcome.Allow/Deny`, `authorize_with_context`, `permission_override`,
`merge_hook_feedback`, `format_hook_message`, `maybe_auto_compact`,
`build_assistant_message`, `ConversationMessage.tool_result`, `TurnSummary`,
`execute_agent`, `normalize_subagent_type`, `allowed_tools_for_subagent`,
`build_agent_system_prompt`, and the full `record_*` trace family.
`PermissionMode` and `HookEvent` keep their exact string values
(`workspace-write`, `PreToolUse`).

---

## Deliberate simplifications

These are cuts, not oversights. Each is a place the original does more.

- **Tool calls run sequentially.** So does the Rust original — but a production
  harness would dispatch independent calls concurrently.
- **Token estimation is `chars // 4`** of the serialised wire payload, not a
  real tokeniser. The shape is right; the precision is not.
- **No session persistence, resume, or fork-to-disk.** `Session.fork_session`
  exists but only copies in memory.
- **Subagents run sequentially and share one workspace.** A production harness
  would sandbox each one and may run them in parallel.
- **Only stdio MCP transport.** No HTTP/SSE servers, no resources or prompts —
  just `initialize`, `tools/list`, `tools/call`.
- **Parallelism is per-batch, not global.** Consecutive read-only calls batch;
  a write ends the batch. Safe, but leaves throughput on the table.
- **Persistence diverges from the original deliberately.** claw-code serialises
  session state to its own file. Here the trace *is* the store, and resuming is
  a fold over it — so a resumed session cannot silently disagree with its own
  trace. See [ARCHITECTURE.md](ARCHITECTURE.md#session-persistence-and-resume).
- **`types.RuntimeError` deliberately shadows the builtin** inside this package
  to match the Rust name. It is imported explicitly everywhere it is used.

---

## A note on the upstream repository

`ultraworkers/claw-code` describes itself as an agent-maintained museum exhibit
rather than a product, its README warns that `cargo install claw-code` installs a
deprecated stub, and its star and fork counts sit oddly against its commit
history. It is a coherent architecture reference; treat it with more caution as a
dependency. This repository reimplements the design from scratch and shares no
code with it.

## License

MIT. See [LICENSE](LICENSE).
