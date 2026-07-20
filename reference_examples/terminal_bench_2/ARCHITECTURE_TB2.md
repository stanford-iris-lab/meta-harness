# Terminal-Bench 2 Meta-Harness 架构详解

> 代码位置:`reference_examples/terminal_bench_2/`。下文行号对应仓库 `Jun 21` 版本。
> 一句话区分:**外层 = `meta_harness.py`(进化驱动) ;内层 = Harbor(`scripts/run_eval.sh` → `uv run harbor run`)做真正的评测**。外层只负责"提议候选 + 派发评测 + 更新前沿",不在 proposer 会话里跑 benchmark。

---

## Part 1. 两个初始化的 Harness(baselines)

### 1.1 它们在哪里定义

`meta_harness.py:86`

```python
BASELINES = [
    ("kira-baseline",      "agents.baseline_kira:AgentHarness"),
    ("terminus2-baseline", "agents.baseline_terminus2:AgentHarness"),
]
BASELINE_AGENT_NAME = BASELINES[0][0]   # kira-baseline = 前沿对比的主基线
```

进化开始的 **Phase 0** 会先把这两个 baseline 跑一遍,用结果给 `frontier_val.json` 播种(`run_evolve` 第 571–643 行)。`kira-baseline` 是"主基线",前沿(frontier)的 `_best` 初始就是它。

| Harness | 文件 | 本质 | 行数 |
|---|---|---|---|
| **kira-baseline** | `agents/baseline_kira.py` | `class AgentHarness(Terminus2)`,KIRA 增强版,**进化的父代** | 1214 |
| **terminus2-baseline** | `agents/baseline_terminus2.py` | `AgentHarness = Terminus2`,**原版** Terminus-2,零改动 | 5 |

`baseline_terminus2.py` 全文就是:

```python
from harbor.agents.terminus_2 import Terminus2
AgentHarness = Terminus2
```

它存在的意义是**对照组**:证明 KIRA 的改动相对原版 Terminus-2 有没有用。论文里 KIRA 基线起点 28.1%,进化到 46.5%(见 SKILL.md:30)。

### 1.2 这里的 "harness/agent" 是什么

一个 agentic scaffold:驱动 LLM(默认 `claude-opus-4-6`)在沙箱里的 **tmux 会话**中解一个终端任务。**单 episode,无跨任务持久记忆**——这点和 text_classification 的 `MemorySystem`(有 `get_state/set_state` 的跨 batch 记忆)**完全不同**,见 §1.4。

### 1.3 Tools(`TOOLS` 常量,`baseline_kira.py:141`)

KIRA 与原版 Terminus2 最大的区别:**用 Claude 原生 tool calling**(结构化 `tools` 参数),而不是让模型吐 JSON/XML 再正则解析(类 docstring `baseline_kira.py:217`)。共 3 个工具:

| Tool | 参数 | 作用 | 定义行 |
|---|---|---|---|
| **`execute_commands`** | `analysis`、`plan`、`commands[]`(每条含 `keystrokes` + `duration`) | 把 keystrokes 逐条发到 tmux 终端执行;`analysis`/`plan` 强制模型先想再做 | 142 |
| **`task_complete`** | 无 | 声明任务完成;会触发"are you sure?"二次确认(`_get_completion_confirmation_message`, 316) | 181 |
| **`image_read`** | `file_path`、`image_read_instruction` | 多模态:读图片文件 → 送给视觉模型 → 下一轮拿回文字描述。原版 Terminus2 没有这个 | 193 |

> `_KEYSTROKES_DESC`(99)/`_DURATION_DESC`(109)是给模型的详细使用说明:keystrokes 完全 verbatim 发送,bash 命令要带 `\n`,特殊键用 tmux 风格(`C-c`/`C-d`);duration 是等命令完成的秒数(`cd/ls` 用 0.1,`make` 等长命令调大,但**永不超过 60s,宁可轮询**)。

### 1.4 "Memory" 在这里指什么

**没有持久化的跨任务记忆库。** 这里的 "memory" = **单个 episode 内的工作上下文管理**:

- **chat 历史**:`Chat` 对象只用于消息历史 / token 计数(SKILL.md:126),LLM 调用本身走 `litellm.acompletion`(`_call_llm_with_tools`, 603)。
- **上下文溢出摘要**:`_summarize_context` 在历史过长时压缩(可 override)。
- **输出截断**:`_limit_output_length` 把终端输出截到 **30KB**(`baseline_kira.py:333`),原版 Terminus2 是 10KB。这是 KIRA 让 agent "记住"更多终端输出的关键调整。
- **block 超时保护**:`BlockError` + `_with_block_timeout`(229),基础设施 API 卡死 600s 抛错,避免一个 episode 永久挂住。

所以"改 memory"在 TB2 语境 = 改上下文/历史/摘要/截断策略,**不是**存一个跨任务知识库。

### 1.5 可 override 的关键方法(进化的搜索空间)

SKILL.md 明确:**搜索空间是任意 Python 代码**,可重写任何方法。常用入口:

| 方法 | 行 | 用途 |
|---|---|---|
| `_call_llm_with_tools` | 603 | litellm 调用;改 tools / 参数 / 重试 |
| `_parse_tool_calls` | 377 | 原始 tool call → 命令;加新工具 |
| `_execute_commands` | 236 | tmux 执行行为 |
| `_run_agent_loop` | 872 | 主 episode 循环(结构性改动) |
| `_get_completion_confirmation_message` | 316 | "确认完成"那一步问什么 |
| `_get_prompt_template_path` | 308 | 指向 system prompt(默认 `prompt-templates/terminus-kira.txt`) |
| `_summarize_context` | — | 上下文溢出摘要 |
| `_execute_image_read` | 501 | 多模态读图行为 |

### 1.6 Prompt 模板(`prompt-templates/terminus-kira.txt`)

system prompt,含 `{instruction}` 和 `{terminal_state}` 两个 `.format()` 占位符。两个值得注意的硬规则:
- "你没有眼睛和耳朵,必须用程序/AI 工具理解多媒体文件"(→ 配合 `image_read`)。
- 调 `task_complete` 前必须**核对最小状态变更**:只改必要文件,不留任何多余文件/配置/副作用。

### 1.7 KIRA vs 原版 Terminus2 小结

| 维度 | kira-baseline | terminus2-baseline |
|---|---|---|
| 结构化输出 | 原生 tool calling | 文本(JSON/XML)解析 |
| 输出截断 | 30KB | 10KB |
| 多模态 | 有 `image_read` | 无 |
| 基础设施卡死保护 | `BlockError` 600s | 无 |
| 完成确认 | 自定义 confirmation message | 原版 |

---

## Part 2. Evolve 方法 / Workflow / Validation / Logs

### 2.1 总体流程(`run_evolve`, `meta_harness.py:545`)

```
Phase 0  跑两个 baseline → 播种 frontier
每次 Iteration:
  1) Propose   propose_claude() 起一个 Claude Code 子进程提议候选
  2) Parse     读 pending_eval.json,强制类名规范成 AgentHarness
  3) Validate  import-check + smoke_test(1 任务×1 trial)
  4) Benchmark harbor_run() 每个有效候选:89 任务 × N trials
  5) Score     compute_pass_rates + parse/summarize_trial_metrics
  6) Update    update_frontier + update_evolution_summary
Phase Final  (--full-eval) 最优非基线 agent 再跑 5 trials
```

运行级隔离:所有输出落在 `logs/<run_name>/` 和 `jobs/<run_name>/`(`run_evolve:548–560`),`<run_name>` 默认是时间戳或 `--run-name`。`--fresh` 清空旧 agents 和日志(`fresh_start`, 520)。

### 2.2 Step 1 — Propose(`propose_claude`, 409)

- 起一个 **Claude Code 子进程**(`claude_wrapper.run`),`model="opus"`、`effort="max"`,只挂载本实验的 skill `meta-harness-terminal-bench-2`。
- **关键**:`os.environ.pop("ANTHROPIC_API_KEY")`(413)——剥掉 API key 让 CLI 走订阅认证,把 API key 省给内层 solver(避免限流)。跑完再恢复(426)。
- 受限工具集 `PROPOSER_ALLOWED_TOOLS`(`meta_harness.py:~100`)。
- proposer **不跑 benchmark**(SKILL.md:10)。它按 SKILL.md 的 workflow:
  1. **Analyze(1 subagent)**:读 `frontier_val.json` + `evolution_summary.jsonl`(含 `rollout_metrics`),**深读 `jobs/`、`logs/` 里成功+失败的轨迹**,读 `agents/*.py`;产出 `STATE / HYPOTHESIS / CANDIDATE`。
  2. **Implement(1 subagent)**:复制某个 `agents/` 文件为起点 → 定向改动 → import 冒烟 `uv run python -c "from agents.<name> import *"`。
  3. **写 `pending_eval.json`**(候选含 `name / import_path / hypothesis / changes / expected_efficiency`)。
- 反过拟合硬规则:每轮至少 1 个新候选;禁止任务名/任务专属 hardcode;一个候选只测一个机制(SKILL.md:12–66)。
- 返回 `True` 当且仅当 `pending_eval.json` 落盘(434)。
- 给 proposer 的任务说明由 `render_task_prompt`(502)生成,里面直接告诉它各日志文件的绝对路径。

### 2.3 Step 2 — Validate(`validate_candidate` 437 + `smoke_test` 453)

两道闸,任一不过就丢弃该候选(`run_evolve:691–720`):

1. **import-check**:`uv run python -c "from <module> import *; print('OK')"`,30s 超时。
2. **smoke_test**(除非 `--skip-smoke`):在单任务 `extract-elf`(`SMOKE_TEST_TASK`, 96)上跑 1 trial,检查 ① 退出码 0 ② 有 `result.json` ③ `stats.n_errors == 0`。捕运行时崩溃。

### 2.4 Step 3 — Benchmark(`harbor_run` 139)

- 调 `scripts/run_eval.sh <import_path> full <trials> <concurrent> --job-name <job> --jobs-dir <JOBS_DIR>`,内部即 `uv run harbor run -d terminal-bench@2.0 -m claude-opus-4-6 -e runloop ...`。
- 默认 89 任务 × `--trials`(smoke/默认路径 2 trials),并发默认 50。单次 harbor 4 小时超时(`14400s`, 163);超时仍读 partial 结果。
- job 名约定:baseline `{name}-t{trials}` · smoke `smoke-{name}` · evolve `evolve-{name}-t{trials}` · final `final-{agent}-t5`。

### 2.5 Step 4 — 打分与指标

- **`parse_job_results`(187)**:遍历每个 trial 子目录 `<task>__<n>/result.json`,取 `verifier_result.rewards.reward`。**缺文件 / 损坏 / 报错一律记 reward=0**(对齐 harbor 的 `total_passes/total_trials` 口径)。
- **`compute_pass_rates`(233)**:`per_task[task] = reward>0 的比例`;`avg = 总通过数 / 总 trial 数`(**扁平平均**,不是各任务率的均值)。
- **`parse_trial_metrics`(251)+`summarize_trial_metrics`(287)**:从 `agent_result` 抽 `n_input_tokens / n_output_tokens / n_cache_tokens / cost_usd / metadata.n_episodes(turns) / api_calls`,汇总成 `rollout_metrics`(总/均 cost、tokens、mean turns、per_task mean_cost/mean_turns)。

### 2.6 Step 5 — 更新前沿与汇总

- **`update_frontier`(350)→ `frontier_val.json`**:逐任务记 `pass_rate` 更高的 `best_agent`;`_best` 记 `avg_pass_rate` 最高的 agent。**只升不降**(per-task best-of)。
- **`update_evolution_summary`(374)→ `evolution_summary.jsonl`**:append-only,每候选一行。

### 2.7 Phase Final(`--full-eval`, 830)

取当前 `_best` 且非基线的 agent,从 `evolution_summary.jsonl` 找回其 `import_path`,**再跑 5 trials** 全量,作为 winner 的更稳估计。

---

### 2.8 落盘的 Logs / 状态文件清单

所有路径在 `logs/<run_name>/` 与 `jobs/<run_name>/` 下(`run_evolve:553–557`)。

| 文件/目录 | 写入者 | 内容 |
|---|---|---|
| `logs/<run>/pending_eval.json` | proposer 写,iter 开头 unlink(663) | 本轮候选:`name / import_path / hypothesis / changes / expected_efficiency` |
| `logs/<run>/frontier_val.json` | `update_frontier` | `{task: {best_agent, pass_rate}, _best: {agent, avg_pass_rate}}` |
| `logs/<run>/evolution_summary.jsonl` | `update_evolution_summary` | 每候选一行:`iteration / agent / import_path / avg_pass_rate / per_task / hypothesis / changes / delta / outcome / timing_s / rollout_metrics`。也用于断点续跑(`count_iterations`, 335) |
| `logs/<run>/claude_sessions/` | `claude_wrapper`(`log_dir`) | proposer 的 Claude Code 会话日志 |
| `logs/<run>/reports/` | proposer | 评测后报告(任务 prompt 里引用) |
| `jobs/<run>/<job>/` | Harbor | 评测产物根目录;含 job 级 `result.json`(`stats.n_errors`)、`config.json`(`model`/`n_attempts`,用于 baseline 缓存校验 588) |
| `jobs/<run>/<job>/<task>__<n>/result.json` | Harbor | 单 trial:`verifier_result.rewards.reward` + `agent_result`(tokens / cost_usd / `metadata.n_episodes` / `api_request_times_msec`) |

### 2.9 一行流转图

```
proposer(Claude Code, 订阅认证)
  └─写→ agents/<name>.py + pending_eval.json
outer loop
  ├─validate→ import-check + smoke(extract-elf×1)
  ├─benchmark→ harbor run (89×trials, -e runloop, API key 认证)
  │    └─产→ jobs/<run>/evolve-<name>-t<trials>/<task>__<n>/result.json
  ├─score→ pass_rate + rollout_metrics
  └─update→ frontier_val.json (+) / evolution_summary.jsonl (append)
```

---

## 附:与 text_classification 的对照(便于迁移直觉)

| | terminal_bench_2 | text_classification |
|---|---|---|
| 候选基类 | `Terminus2`(agent scaffold) | `MemorySystem`(predict/learn_from_batch/get_state/set_state) |
| "记忆" | 单 episode 上下文(无跨任务持久化) | 真·跨 batch 持久记忆库 |
| 内层评测 | Harbor `harbor run`(沙箱内 tmux) | `benchmark.py` + `inner_loop.py` |
| 前沿文件 | `frontier_val.json` / `evolution_summary.jsonl` | 同名同构 |
| proposer | 同一 `claude_wrapper.py` + 各自 SKILL.md | 同 |
