# Meta-Harness (text_classification) 系统讲解

本文档面向"想从 0 理解这个仓库怎么跑"的读者，聚焦 `reference_examples/text_classification/`。所有文件路径以这个目录为根。

---

## 0. 全景图：三层角色

在 evolve 一轮里同时活跃着**三种不同的"主体"**，初学时最容易混淆。先把它们对上号：

| 角色 | 是什么 | 模型 | 干什么 |
|---|---|---|---|
| **Proposer**（提议者） | 一个 Claude Code 子进程 | **Claude Opus 4.7 @ max effort** | 看历史结果，**写出 3 个新的 `agents/<name>.py`** |
| **Memory System**（候选 harness 本身） | proposer 写出来的 Python 类，继承 `MemorySystem` | （没有自己的模型，它**用 Task LLM 工作**） | 决定怎么记、怎么检索、怎么拼 prompt |
| **Task LLM / Solver**（解题模型） | litellm 后面的任意 OpenAI 兼容模型 | **默认 `openrouter/openai/gpt-oss-120b`**（`config.yaml`） | 真正去回答每道分类题 |

一句话：**Proposer 用 Opus 4.7 写代码，Solver 用 gpt-oss-120b 答题，Memory System 是写出来的那段代码、决定怎么调 Solver**。

---

## 1. 初始 harness 是什么样子的？

### 1.1 接口 = `MemorySystem`（`memory_system.py:61`）

每个候选 harness 都是一个 Python 类，继承下面这个抽象基类：

```python
class MemorySystem(ABC):
    def __init__(self, llm: LLMCallable): ...
    def predict(self, input: str) -> tuple[str, dict]: ...         # 在线推理
    def learn_from_batch(self, batch_results: list[dict]) -> None: ...  # 离线学习
    def get_state(self) -> str: ...                                  # 序列化
    def set_state(self, state: str) -> None: ...
```

只有四个方法，**没有任何工具接口**（no `Bash`、no `Read`、no `WebSearch`）。它能做的就是：
- 在 `predict()` 里调一次/多次 `self.call_llm(prompt)` 让 Solver 答题
- 在 `learn_from_batch()` 里看 ground-truth 更新自己的内部 state

`call_llm` 本身只是一层薄封装（`memory_system.py:72`），记录每次调用的 prompt 长度/hash 用于日志，**真正打到 Solver 的就是一次普通的 chat completion**。

### 1.2 仓库里现成的两个 baseline

`agents/` 下被 `BASELINE_FILES` 保护、evolve 不会覆盖的就两个：

#### `no_memory`（`agents/no_memory.py`）

零学习、零 memory。`predict()` 把 input 塞进一个固定 JSON 输出模板里，调一次 Solver，结束：

```python
PROMPT = """Answer the following question.

{input}

**Answer in this exact JSON format:**
{{ "reasoning": "...", "final_answer": "..." }}
"""

class NoMemory(MemorySystem):
    def predict(self, input):
        response = self.call_llm(PROMPT.format(input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response}

    def learn_from_batch(self, batch_results): pass   # 完全 no-op
```

#### `fewshot_all`（继承自 `fewshot_memory.py`）

经典 in-context learning：累积所有训练例子，predict 时**按 input 的 hash 当种子**随机采样最多 9999 条（实际被 `MAX_CHARS=30000` 截断）拼成 few-shot demo。

```python
class FewShotMemory(MemorySystem):
    def learn_from_batch(self, batch_results):
        for r in batch_results:
            self.examples.append({
                "input": r["input"],
                "target": r["ground_truth"],
                "raw_question": r.get("raw_question"),
            })

    def predict(self, input):
        seed = hash(input) & 0xFFFFFFFF
        examples_section = self._format_examples_section(seed=seed)  # 拼 demo
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {...}
```

**注意**：拼 demo 用 `raw_question`（光秃秃的题目），**不是 `input`（带任务说明的完整 prompt）**，避免把"你是医学诊断专家..."这种任务说明在每个 demo 里重复几十遍。这一点对 LawBench 尤其关键（案件事实本身就很长）。

`learn_from_batch` 在这里**不调 LLM**——只是把 (question, answer) 对追加到 list 里。所以 `fewshot_all` 整个 evolve 期间每条 val 例子只触发 1 次 Solver 调用。

### 1.3 一个具体例子：Symptom2Disease 用 `fewshot_all` 是怎么解一道题的

走一遍完整链路：

**Step 1 — Loader 把原始 JSON 包装成 prompt**（`data/loaders.py:132` `_load_symptom2disease`）

原始数据：
```json
{"question": "I get heartburn and indigestion a lot. It's worse when I eat spicy or fatty foods...",
 "answer": "gastroesophageal reflux disease"}
```

Loader 把它变成：
```python
{
  "input": "You are an expert medical diagnostician. Based on the patient's symptoms...\n"
           "Possible diagnoses include: drug reaction, allergy, chicken pox, diabetes,...\n"
           "Please analyze the symptoms step by step, then provide your final diagnosis in the format:\n"
           "[DIAGNOSIS]diagnosis_name[/DIAGNOSIS]\n\n"
           "## Patient Symptoms\n"
           "I get heartburn and indigestion a lot...",
  "target": "gastroesophageal reflux disease",
  "raw_question": "I get heartburn and indigestion a lot..."   # 仅原文，无任务说明
}
```

**Step 2 — Train 阶段，memory 累积 200 条 (raw_question, answer)**（`inner_loop.py:_run_offline_loop`）

```python
for ex in train_examples:    # 200 条 Symptom2Disease train
    memory.learn_from_batch([{
        "input": ex["input"],
        "ground_truth": ex["target"],
        "raw_question": ex["raw_question"],
        ...
    }])
```

`fewshot_all` 在这步**不调 Solver**，只是 `self.examples.append(...)`。

**Step 3 — Val 阶段，对 50 道 val 题各调一次 Solver**

对一道 val 题（比如 "I've been having severe headaches with nausea and sensitivity to light"）：

1. `predict(input)` 被调用
2. memory 用 `hash(input)` 做种子，从已累积的 200 条里随机抽 N 条（N 受 `MAX_CHARS=30000` 截断，实际可能是 ~50-100 条），拼成 demo block:
   ```
   Q: I get heartburn and indigestion a lot...
   A: gastroesophageal reflux disease

   Q: I've been feeling really thirsty and hungry lately...
   A: diabetes

   ... （几十条）
   ```
3. 拼出完整 prompt：
   ```
   Solve the problem below based on the examples provided.

   [上面那一大坨 demo]

   **Problem:**
   You are an expert medical diagnostician. ... [22 类标签] ...
   ## Patient Symptoms
   I've been having severe headaches with nausea and sensitivity to light...

   **Instructions:**
   - Follow the patterns shown in the examples above
   - Respond in JSON format

   {"reasoning": "...", "final_answer": "..."}
   ```
4. `self.call_llm(prompt)` → 触发 `ProviderLLM._call_completion()`（`llm.py:204`）→ `litellm.completion(model="openrouter/openai/gpt-oss-120b", messages=[...])`
5. Solver 返回类似：
   ```json
   {"reasoning": "Severe headaches with nausea and photophobia are classic triad of migraine...",
    "final_answer": "[DIAGNOSIS]migraine[/DIAGNOSIS]"}
   ```
6. `extract_json_field(response, "final_answer")` → `"[DIAGNOSIS]migraine[/DIAGNOSIS]"`
7. Evaluator `eval_symptom2disease`（`data/evaluators.py:79`）用正则抽出 `migraine`，与 target 比对，正确 → `was_correct=True`

50 道 val 题做完，accuracy 写到 `logs/<run>/Symptom2Disease/fewshot_all/<model>/val.json`。

**关键观察**：整个流程里**唯一与 Solver 交互的地方就是 `predict` 中的那一次 `call_llm`**——这就是 memory system 的全部"动作空间"。它能"做的事"=如何构造那一次 prompt，仅此而已。这就是为什么 evolve 在搜的本质上是 **prompt 构造策略 + 状态累积策略**，不是 agent loop。

### 1.4 两个任务的差异（速查）

| | Symptom2Disease | LawBench |
|---|---|---|
| 语言 | 英文 | 中文 |
| 标签 | **22 类封闭集**写在 prompt 里 | **《刑法》全部罪名，prompt 不列** |
| 标签数 | 单标签 | **多标签**（`;` 分隔） |
| 输出 marker | `[DIAGNOSIS]...[/DIAGNOSIS]` | `[罪名]...<eoa>` |
| 评测 | `bool`（normalize 后精确相等） | set-based F1（tp/fp/fn）|
| Partial credit | 无 | 有（多标签部分匹配）|

**Harness 代码本身在两个任务上完全相同**——同一份 `MemorySystem`、同一份 `predict()`。差异**全部由 loader 注入到 `input` 字段里**，再由 evaluator 在外面打分。这是 Meta-Harness 的设计核心：**harness 必须是 task-agnostic 的**（skill 文件明确禁止 dataset 名出现在候选代码里）。

---

## 2. 怎么 Evolve？

### 2.1 入口和文件

外层循环全部在 **`meta_harness.py`** 里（539 行，挺薄）。命令：

```bash
uv run python meta_harness.py --iterations 10 --fresh --run-name my-run
```

一轮 `run_evolve()` 的主流程（`meta_harness.py:273`）：

```
Phase 0: 跑 baseline（no_memory + fewshot_all）→ 初始化 frontier
Phase 1..N (iteration loop):
    1. propose_claude()       # 调 Claude Code 写出 3 个新候选
    2. validate_candidates()  # python -c "from ... import *" 静态校验
    3. 对每个候选跑 benchmark.py --memory <name>  # val 评估
    4. benchmark.py --frontier 更新 frontier_val.json
    5. update_evolution_summary()  # 一行 JSON 一个候选写 evolution_summary.jsonl
Phase Final: test eval（只在最后一次，对 frontier 上的系统跑 test split）
```

每轮的产物全部落到 `logs/<run_name>/` 下，按 `<dataset>/<memory>/<model>/val.json` 的层级组织。

### 2.2 Proposer 看到什么？输出什么？

#### Proposer 是怎么被调起来的（`meta_harness.py:151` `propose_claude`）

```python
def propose_claude(task_prompt, iteration, timeout=2400):
    os.environ.pop("CLAUDECODE", None)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)   # ← 关键：剥掉 API key
    result = claude_wrapper.run(
        prompt=task_prompt,
        model="opus",                # = Claude Opus 4.7
        allowed_tools=["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"],
        skills=[".claude/skills/meta-harness"],   # 喂给它的"先验"
        cwd=str(EVOLVE_DIR),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        effort="max",
        timeout_seconds=timeout,
    )
    if saved_key: os.environ["ANTHROPIC_API_KEY"] = saved_key
    return PENDING_EVAL.exists()
```

剥 `ANTHROPIC_API_KEY` 是为了让 Claude Code CLI 走**订阅认证**（不烧 API 费），同时把 API key 留给 Solver 用。

#### 它看到的 prompt（`render_task_prompt`，`meta_harness.py:123`）

非常短，**只是告诉它"去哪里读历史、把结果写到哪里"**：

```
Run iteration 5 of the evolution loop. There are 3 datasets.

## Run directories
All logs and results for this run are under `logs/my-run/`.
- `logs/my-run/evolution_summary.jsonl` — past results
- `logs/my-run/frontier_val.json` — frontier
- `logs/my-run/reports/` — post-eval reports
- Write pending_eval.json to: `logs/my-run/pending_eval.json`
```

#### 它的"先验" = `.claude/skills/meta-harness/SKILL.md`

这是把 evolve 的**所有约束和工作流**告诉 proposer 的地方。要点摘录：

- **硬性约束**：每轮**必须**写 3 个新 memory system；不准说"frontier 已最优、停止"；不准只是改 baseline 的常数（pool size、retrieval count 这类）；不准在代码/prompt/注释里出现 dataset 名。
- **工作流强制顺序**：
  1. **Step 0**：给过去没写报告的 iteration 补写 ≤30 行的复盘
  2. **Step 1**：读 `evolution_summary.jsonl` + `frontier_val.json` + 最近的 `log.jsonl` traces；形成 3 个 falsifiable hypothesis
  3. **Step 2（强制）**：在 `/tmp/` 写小脚本**先 prototype**，拿真实 log 里的例子试 2-3 个变体，再选最好的
  4. **Step 3**：把表现最好的 base system 复制成 `agents/<name>.py`，做改造；自我批判一遍（这是真的新机制还是只是参数变体？）；`uv run python -c "from text_classification.agents.<name> import *"` 跑通
  5. **Step 4**：写 `pending_eval.json`
- **MemorySystem 接口规范**（前面已经讲过）
- **方向轴**：A=Prompt 模板 / B=Memory 内容 / C=选择算法 / D=Memory 大小 / E=学习触发 / F=学习时是否用 LLM——最近 3 轮如果都在同一个轴上，要换别的轴。

#### 它输出什么

两个东西：

1. **3 个文件：`agents/<name>.py`**（直接 Write 到磁盘）
2. **一个 `pending_eval.json`** 告诉外层循环这 3 个候选叫啥、假设是什么：
   ```json
   {
     "iteration": 5,
     "candidates": [
       {
         "name": "contrastive_retrieval_memory",
         "file": "agents/contrastive_retrieval_memory.py",
         "hypothesis": "Retrieving pairs of (positive, hard-negative) examples improves label discrimination on confusable classes",
         "axis": "exploration",
         "base_system": "fewshot_all",
         "components": ["contrastive", "retrieval", "diversity"]
       },
       ...
     ]
   }
   ```

外层循环在 `propose_claude` 返回后就读 `PENDING_EVAL`，把 candidates 列表喂给 `validate_candidates()` → `benchmark.py`。

### 2.3 Validate 和 Benchmark

**Validate**（`meta_harness.py:179`）：纯 import 检查，确保候选文件能被 Python 加载，不抓行为上的 bug：

```python
uv run python -c "from text_classification.agents.<name> import *; print('OK')"
```

**Benchmark**：对每个 valid 候选调一次 `benchmark.py --memory <name>`，它内部用 asyncio 并发跑 `dataset × seed` 的所有 (dataset, memory, model, seed) 组合，每个 subprocess 是一次 `inner_loop.py` 的 train+val 流程。结果写 `val.json`。**Test 集在 evolve 期间从来不被读**——只在 Phase Final 用一次。

### 2.4 Frontier 更新

`benchmark.py --frontier` 重算 Pareto 前沿：横轴是平均 ctx token 数（小好），纵轴是平均 val accuracy（大好）。Frontier 上的系统**不是单一最优**——是一组在 (准确率, 上下文成本) 平面上互不被支配的系统。这影响 proposer 下一轮的"该 exploit 谁"。

---

## 3. Task LLM (Solver) 详解

### 3.1 默认是什么模型

`config.yaml`：
```yaml
models:
  - model: openrouter/openai/gpt-oss-120b
  # 也可以换：
  # - model: gpt-oss-120b
  #   api_base: https://your-api-host/v1
```

paper 原跑法是**本地 vLLM 部署的 gpt-oss-120b（MXFP4 量化，`max-model-len=32768`）**。README 写明 OpenRouter 路径"可能在质量上和 paper setup 有差异，也可能更好"。

### 3.2 调用链（`llm.py:134` `ProviderLLM`）

memory system 里 `self.call_llm(prompt)` 的真实落点：

```
MemorySystem.call_llm(prompt)
  └─► self._llm(prompt)                        # 实例化时注入的 LLMCallable
        └─► ProviderLLM.__call__               # llm.py
              ├─► 计算 sha256(prompt) 做磁盘 cache key
              ├─► 命中 → 直接返回 cached content（节省费用）
              └─► 不命中 → _call_completion()
                    ├─► litellm.completion(model=..., messages=[...], timeout=600)
                    ├─► retry：tenacity 在 429 / timeout / rate-limit 上指数退避（最多 6 次）
                    ├─► parse_harmony_response(content)   # 仅 GPT-OSS 专用，剥 harmony 包装
                    └─► 写 cache、累加 token/cost、返回 string
```

几个不显眼但很重要的点：

- **磁盘缓存默认开**：`CACHE_DIR = ~/.cache/text-classification/litellm/`，按 `(model, api_base, system_prompt, prompt, kwargs)` 的 sha256 做 key。**同一个 candidate + 同一份训练序列 ≈ 完全免费跑第二次**。这对反复调试候选非常重要。
- **`MAX_PROMPT_CHARS = 224_000`**：硬上限，防止 fewshot 类系统拼出爆炸长 prompt 把 API 打死。
- **GPT-OSS 特殊处理**：harmony format 是 OpenAI 开源 gpt-oss 模型用的 channel-based 输出协议（reasoning 和 final answer 分通道），`parse_harmony_response` 抽 `final` channel 的内容。换非 gpt-oss 模型时这一段是 no-op fallback（返回原文），不会出问题。
- **并发**：`config.yaml` 的 `benchmark.concurrency: 16` 是数据集×memory×seed 这种**外层**任务的并发；val 评估时**单个候选内部**还有 `inner_loop.py` 的 `ThreadPoolExecutor(max_workers=32)`（`inner_loop.py:231`）。所以总打到 Solver 的并发可能高达 16×32 = 512，受 API tier 限制。

### 3.3 换 Solver 模型（比如 DeepSeek）

只需要改 `config.yaml`：
```yaml
models:
  - model: deepseek/deepseek-chat        # litellm 自动认 DEEPSEEK_API_KEY
```
或本地 OpenAI 兼容端点：
```yaml
models:
  - model: my-llama-3
    api_base: http://localhost:8000/v1
```

`_normalized_model()`（`llm.py:160`）会自动给"有 api_base 但没 provider 前缀"的 model 补 `openai/` 前缀，让 litellm 当 OpenAI 兼容协议调。

### 3.4 Proposer 模型 ≠ Solver 模型（澄清）

两个模型在代码里完全隔离：
- **Solver** 用 `litellm` + `config.yaml` 里指定的模型，烧 API key（OpenRouter / OpenAI / DeepSeek / ...）
- **Proposer** 走 `claude_wrapper.py` → `claude -p` CLI，硬编码 `model="opus"`（`meta_harness.py:159`），认证方式被 `propose_claude` 主动切到订阅（剥掉 ANTHROPIC_API_KEY）

唯一会让它们撞车的场景：你用 ANTHROPIC 模型当 Solver 又没有 Claude 订阅。这时 Solver 和 Proposer 会争同一个 API key 的 rate limit。

---

## 附：一轮 evolve 的端到端时间线（速查）

```
t=0       meta_harness.py 启动
          │
          ├─ Phase 0 (仅首次)：跑 no_memory + fewshot_all → val.json
          │
          ├─ Iteration 1
          │   ├─ propose_claude()  →  spawn Claude Code CLI (Opus 4.7, 订阅认证)
          │   │     │
          │   │     ├─ proposer 读: evolution_summary.jsonl / frontier_val.json / log.jsonl
          │   │     ├─ proposer 在 /tmp/ 跑 prototype 脚本
          │   │     ├─ proposer Write: agents/cand_a.py, cand_b.py, cand_c.py
          │   │     └─ proposer Write: logs/<run>/pending_eval.json
          │   │
          │   ├─ validate_candidates()  → 3 个 import 检查
          │   ├─ 串行跑 benchmark.py --memory <each>
          │   │     │
          │   │     └─ inner_loop._run_offline_loop()
          │   │           ├─ for ex in train (450 条): memory.learn_from_batch([ex])
          │   │           └─ ThreadPool: for ex in val (130 条): memory.predict(ex)
          │   │                              │
          │   │                              └─ self.call_llm() → litellm → gpt-oss-120b
          │   │
          │   ├─ benchmark.py --frontier  → 更新 frontier_val.json
          │   └─ update_evolution_summary()  → 追加 3 行
          │
          ├─ Iteration 2..N (同上)
          │
          └─ Phase Final：对 frontier 上的系统跑 test split → results/
```

---

## 进一步阅读路径

想深入哪个部分，按下面顺序读：

- **Memory 接口** → `memory_system.py` （短，120 行）
- **两个 baseline** → `agents/no_memory.py`, `agents/fewshot_memory.py`
- **训练/评估循环** → `inner_loop.py` 的 `_run_offline_loop`
- **外层 evolve 循环** → `meta_harness.py` 的 `run_evolve`
- **Proposer 约束** → `.claude/skills/meta-harness/SKILL.md`
- **数据怎么被包装成 prompt** → `data/loaders.py`（每个 `_load_*` 函数就是一个任务的"任务说明 + 输出格式"）
- **评测怎么打分** → `data/evaluators.py`
- **Solver 调用细节** → `llm.py` 的 `ProviderLLM._call_completion`
