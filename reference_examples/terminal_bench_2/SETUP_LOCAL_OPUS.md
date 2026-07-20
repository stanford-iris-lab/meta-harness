# Running the TB2 Meta-Harness against a LOCAL Opus deployment

This is a runbook for pointing **both** the proposer (Claude Code CLI) **and** the
inner-loop solver (Harbor/litellm) at a locally-deployed Opus endpoint instead of
api.anthropic.com, and running a cheap **8-task** evolve loop.

All paths below are relative to `reference_examples/terminal_bench_2/`.

---

## 0. The one thing to confirm first

There are two model consumers and they have different requirements:

| Consumer | What it is | API it speaks | Flexible? |
|---|---|---|---|
| **Proposer** | Claude Code CLI (`claude -p`), writes new agents | **Anthropic Messages API only** | ❌ no |
| **Inner solver** | Harbor → litellm → your model, solves the tasks | anything litellm supports | ✅ yes |

➡️ **The local endpoint MUST expose an Anthropic-compatible (`/v1/messages`) API**, otherwise Claude Code cannot use it.

- If your deployment is already Anthropic-compatible (e.g. an internal proxy, or a **LiteLLM proxy** in front of vLLM/Bedrock/Vertex) → you're good.
- If it is **OpenAI-compatible only** (raw vLLM/SGLang) → put a **LiteLLM proxy** in front of it to translate. The proxy URL becomes your `ANTHROPIC_BASE_URL`. (Note: "Opus" weights aren't public, so a "local Opus" is almost always a gateway/proxy to Anthropic, Bedrock, or Vertex — which is exactly the Anthropic-compatible case.)

Ask your infra owner for: **(a) base URL, (b) auth token, (c) the model id** the endpoint expects.

---

## 1. Prerequisites

- Python 3.12 + [`uv`](https://docs.astral.sh/uv/)
- A recent **Claude Code** CLI on PATH (`claude --version`) — must support `ANTHROPIC_BASE_URL` and `ANTHROPIC_DEFAULT_OPUS_MODEL`.
- A sandbox backend for Harbor — either local **Docker** (free) or **Runloop** (cloud, needs a key). See §4.

```bash
cd reference_examples/terminal_bench_2
uv sync
```

---

## 2. Configure the endpoint (no code changes needed)

Copy the template and fill in the 3 `<...>` values:

```bash
cp .env.local-opus.example .env
$EDITOR .env
```

The important variables (full annotations in the template):

```bash
ANTHROPIC_API_BASE=<LOCAL_BASE_URL>     # inner solver (litellm)
ANTHROPIC_BASE_URL=<LOCAL_BASE_URL>     # proposer (Claude Code) — same URL
ANTHROPIC_API_KEY=<LOCAL_TOKEN>         # inner solver auth (x-api-key)
ANTHROPIC_AUTH_TOKEN=<LOCAL_TOKEN>      # proposer auth (Bearer) — survives the key-strip
ANTHROPIC_DEFAULT_OPUS_MODEL=<LOCAL_MODEL_ID>   # maps Claude Code's `--model opus`
ANTHROPIC_DEFAULT_SONNET_MODEL=<LOCAL_MODEL_ID>
ANTHROPIC_DEFAULT_HAIKU_MODEL=<LOCAL_MODEL_ID>
```

**Why two URL names and two token names?** litellm reads `ANTHROPIC_API_BASE` + `ANTHROPIC_API_KEY`; Claude Code reads `ANTHROPIC_BASE_URL` + (because the harness strips `ANTHROPIC_API_KEY` before launching the proposer to force non-API auth) `ANTHROPIC_AUTH_TOKEN`. Set each pair to the same value.

### Inner-solver model id

`meta_harness.py` forces `HARBOR_MODEL` to its `MODEL` constant
(`meta_harness.py:99`, default `anthropic/claude-opus-4-6`).

- If your gateway accepts the literal name **`claude-opus-4-6`** → no edit needed.
- Otherwise edit that one line to `anthropic/<your-model-id>`. Keep the
  `anthropic/` prefix so litellm uses the Anthropic transport (which honors `ANTHROPIC_API_BASE`).

### Auth fallback (only if your gateway rejects Bearer)

If the proposer gets 401s, your gateway probably wants `x-api-key`, not Bearer.
The harness strips `ANTHROPIC_API_KEY` at `meta_harness.py:413` (restored at `:426`).
Comment out those two lines so the key passes through to Claude Code, and keep
`ANTHROPIC_API_KEY` set in `.env`.

---

## 3. Smoke-test the wiring (one task, cheapest)

This goes through Harbor + your local solver, but not the proposer:

```bash
# local Docker sandbox:
uv run harbor run --agent-import-path agents.baseline_kira:AgentHarness \
  -d terminal-bench@2.0 -m "${HARBOR_MODEL:-anthropic/claude-opus-4-6}" \
  -e docker -n 1 --n-attempts 1 -i extract-elf
```

Then smoke-test the **proposer** path separately (cheap, no Harbor):

```bash
uv run python -c "
import claude_wrapper
r = claude_wrapper.run(prompt='Reply with exactly: PONG', model='opus', progress=False)
print('exit', r.exit_code); r.show()"
```

If that prints `PONG` and exit 0, Claude Code is talking to your local endpoint.

---

## 4. Choose the sandbox backend

The shipped scripts use Runloop (`-e runloop`, `scripts/run_eval.sh:69`).

- **Local Docker (free):** change `-e runloop` → `-e docker` in `scripts/run_eval.sh:69`, and you can leave `RUNLOOP_API_KEY` empty. Requires Docker on the box and works only at low concurrency.
- **Runloop (cloud, high concurrency):** keep `-e runloop`, set `RUNLOOP_API_KEY` in `.env` (sign up at runloop.ai; new accounts get $50 free credit). Billed per CPU/GB-hour.

Note: the sandbox only runs the *task environment*. Your local Opus still serves all LLM calls either way.

---

## 5. Run the 8-task evolve loop

The `mini` split (8 tasks) is already wired up. With `MH_TASK_SET=mini` and
`MH_N_TASKS=8` in `.env`:

```bash
# 1 iteration, 1 trial/task, low concurrency (local endpoints have limited throughput)
uv run python meta_harness.py --iterations 1 --trials 1 --concurrent 4 \
  --run-name local-mini --fresh
```

What happens each iteration:
`Phase 0` runs both baselines on the 8 tasks → `Propose` (Claude Code on local Opus writes 1 new agent) → `Validate` (import + `extract-elf` smoke) → `Benchmark` (8 tasks × 1 trial on local Opus) → updates `logs/local-mini/frontier_val.json` + `evolution_summary.jsonl`.

Scale up once it's working: drop `--trials` back to 2, raise `--concurrent` to what
your endpoint can sustain, add more iterations, and (optionally) set
`MH_TASK_SET=hard` (30 tasks) or `full` (89) for real signal.

To change which 8 tasks run, set `MH_MINI_TASKS="taskA taskB ..."` in `.env`
(any valid TB2 task names; the default 8 are a mix from the hard subset).

---

## 6. Where the output lands

Under `logs/local-mini/` and `jobs/local-mini/` (see `ARCHITECTURE_TB2.md` §2.8):
- `evolution_summary.jsonl` — per-candidate scores, hypotheses, cost/token metrics
- `frontier_val.json` — best agent per task + overall best
- `claude_sessions/` — proposer transcripts (check here if the proposer misbehaves)
- `jobs/.../<task>__<n>/result.json` — per-trial reward + token/cost

---

## 7. Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All candidate scores 0 | solver can't reach endpoint / wrong model id | check `ANTHROPIC_API_BASE`, `ANTHROPIC_API_KEY`, `MODEL` in meta_harness.py; read `jobs/.../result.json` |
| Proposer exits non-zero / 401 | Claude Code auth/base wrong, or gateway wants x-api-key | set `ANTHROPIC_BASE_URL`+`ANTHROPIC_AUTH_TOKEN`; if still 401, apply the §2 key-strip fallback |
| Proposer uses wrong model | alias not mapped | set `ANTHROPIC_DEFAULT_OPUS_MODEL` (and sonnet/haiku) |
| Iteration "silently lost" | proposer timed out before writing `pending_eval.json` | raise `--propose-timeout`; check `logs/.../claude_sessions/` |
| Many benchmark timeouts | endpoint throughput too low for the concurrency | lower `--concurrent` |
| `.env` ignored | scripts source `.env` from THIS dir | keep `.env` in `reference_examples/terminal_bench_2/` |
