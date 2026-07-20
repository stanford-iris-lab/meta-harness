# Setup on a new machine

This experiment is **API-bound and light**: it calls a remote LLM (the solver) and runs
short-lived Python subprocesses on small `.xlsx` files. No GPU, little CPU. The two things
that actually matter are **network reachability to your solver endpoint** and **memory
headroom for concurrent workers**.

## 1. Requirements

| Need | Why | Check |
|---|---|---|
| `uv` | project/venv manager | `uv --version` |
| Python ≥3.11 | `uv sync` provisions it | (automatic) |
| Network to the solver endpoint | inner-loop LLM calls | `curl -sI https://api.qnaigc.com/v1/models` |
| `claude` CLI, authenticated | the **proposer** is a nested `claude -p` subprocess | `claude --version` |
| ~2 GB RAM per concurrent worker | each worker imports litellm+numpy+pandas | see §6 |

The `claude` CLI is only needed for `meta_harness.py` (the evolution loop). Benchmarking and
test eval work without it. The outer loop strips `ANTHROPIC_API_KEY` before launching the
proposer so it uses subscription auth, decoupled from the solver key.

## 2. Install

```bash
git clone <your fork or the upstream repo> meta-harness
cd meta-harness/reference_examples/spreadsheet_bench
uv sync                       # creates .venv
```

Prefer `uv run <cmd>`; or `source .venv/bin/activate` and drop the prefix.

## 3. Data

```bash
bash data/fetch_data.sh                                   # sample_data_200  (200 tasks, ~19MB dl)
bash data/fetch_data.sh spreadsheetbench_912_v0.1         # full benchmark   (912 tasks, ~526MB extracted)
bash data/fetch_data.sh spreadsheetbench_verified_400     # verified subset  (~394 tasks)
```

Data is gitignored — always re-fetch on a new machine. Pick the set in `config.yaml`
(`dataset.source`). Use the 912 set if you need a val subset and a disjoint test split.

## 4. Keys

```bash
export QNAIGC_API_KEY=sk-...        # solver; .env is NOT auto-loaded by Python
# or, for an OpenRouter-routed solver:
export OPENROUTER_API_KEY=sk-or-... # + set model.name: openrouter/openai/gpt-oss-120b, api_base: null
```

`scripts/run_eval.sh` *does* source `.env`; the Python entrypoints do not. Never commit keys.

## 5. Verify, in this order

```bash
uv run python executor.py                                        # no API needed
uv run python -m data.loader --source <source> --list | head     # data resolves
uv run python inner_loop.py --agent agents/baseline_react.py --task-id <id>   # 1 task, real API
MH_N_TASKS=3 uv run bash scripts/run_eval.sh agents.baseline_single           # small sweep
uv run python meta_harness.py --iterations 1 --fresh --run-name smoke         # proposer works
```

If step 3 returns real `tokens=` you have a working solver. If step 5 proposes a candidate,
the nested-`claude` proposer is authenticated.

## 6. Concurrency and memory (read this before a long run)

Each worker resident set is ~1–2 GB. **Concurrency × 2 GB must fit your memory budget**,
including any per-user cgroup cap:

```bash
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max      # cap, if any
grep -E '^oom' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.events
```

If `oom_kill` climbs during a run, lower `--concurrent`. On a 16 GiB cap, **3–4 is safe; 6
caused OOM kills**. The harness now retries workers killed by a signal and prints a loud
`WARNING: N/M task records MISSING` — treat any such warning as an invalid measurement, not
a low score.

Keep caches off slow/quota'd home dirs:

```bash
export SPREADSHEET_BENCH_LLM_CACHE_DIR=$SCRATCH/.cache/sb-llm
export XDG_CACHE_HOME=$SCRATCH/.cache
```

If the venv lives on a network filesystem (Lustre/NFS), imports get slow under concurrency;
a local-disk venv is noticeably better.

## 7. Running the loop

```bash
uv run python meta_harness.py --iterations 10 --fresh --run-name run1 --concurrent 4
uv run python test_eval.py --run-dir logs/run1 --test-size 200 --concurrency 3 \
    --agents agents.baseline_single:AgentHarness agents.baseline_react:AgentHarness \
             agents.<winner>:AgentHarness
uv run python plot_report.py --run-dir logs/run1        # curve.png + report.md
```

Long runs: launch detached (`setsid nohup … &`) so they survive a dropped shell.

## 8. Cluster note (Vulcan, and clusters like it)

Compute nodes reach the internet only through a proxy (`http_proxy=http://squid:3128`) whose
allowlist **permits OpenRouter/pypi/github but blocks `api.qnaigc.com`**; the login node
reaches QNAIGC directly. So: run against QNAIGC where it's reachable, or switch the solver to
OpenRouter to run under Slurm. Confirm before a long run:

```bash
curl -sI https://api.qnaigc.com/v1/models    # expect 200 wherever you intend to run
```

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `InternalServerError … Connection error` | endpoint unreachable from this host | §8 — check the proxy/allowlist |
| `WARNING: N/M task records MISSING` | workers OOM-killed | lower `--concurrent` (§6) |
| `RuntimeError: can't start new thread` | thread/proc limits under load | lower `--concurrent` |
| `blas_thread_init: pthread_create failed` | numpy/OpenBLAS thread spawn | already fixed — `executor.py` pins math backends to 1 thread |
| Empty model reply / `finish_reason: length` | gpt-oss spent `max_tokens` on `reasoning_content` | raise `max_tokens` in `llm.py` |
| Proposer produces no candidate | `claude` CLI missing or unauthenticated | `claude --version`, log in |
