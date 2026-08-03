# claw-py

A minimal, readable reference implementation of the agentic loop found in
[`ultraworkers/claw-code`](https://github.com/ultraworkers/claw-code) — rewritten
in Python, running against a small open-weights model via Ollama.

The Rust original is roughly 35,000 lines across 11 crates. Most of that is
parity harnesses, MCP transport, provider-compatibility shims, and config
validation. **This repository strips all of it out and keeps only the
architecture**: the turn loop, the three-layer tool-gating pipeline, context
compaction, and inline tracing.

About 1,800 lines. Standard library only — no `pip install` required.

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

### 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

27 tests, no network. They cover the loop, the iteration cap, tool failures,
all five permission modes, hook rewrite/deny/override, post-hook error flipping,
compaction, and the session health probe.

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
│   ├── prompt.py           system prompt assembly
│   ├── telemetry.py        inline trace events
│   └── cli.py              REPL, one-shot mode, demo hooks
├── examples/
│   └── demo_offline.py     scripted run, no model needed
└── tests/
    └── test_loop.py        27 tests
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
| `telemetry.py` → `SessionTracer` | `telemetry/src/lib.rs` |
| `cli.py` | `rusty-claude-cli` |

Preserved verbatim: `run_turn`, `pending_tool_uses`, `effective_input`,
`PermissionOutcome.Allow/Deny`, `authorize_with_context`, `permission_override`,
`merge_hook_feedback`, `format_hook_message`, `maybe_auto_compact`,
`build_assistant_message`, `ConversationMessage.tool_result`, `TurnSummary`, and
the full `record_*` trace family. `PermissionMode` and `HookEvent` keep their
exact string values (`workspace-write`, `PreToolUse`).

---

## Deliberate simplifications

These are cuts, not oversights. Each is a place the original does more.

- **Tool calls run sequentially.** So does the Rust original — but a production
  harness would dispatch independent calls concurrently.
- **Token estimation is `len(chars) // 4`**, not a real tokeniser.
- **No session persistence, resume, or fork-to-disk.** `Session.fork_session`
  exists but only copies in memory.
- **No MCP client and no subagents.** Both are sketched in
  [ARCHITECTURE.md](ARCHITECTURE.md#extending-it) — each is a small addition
  precisely because the tool pipeline is the single choke point.
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
