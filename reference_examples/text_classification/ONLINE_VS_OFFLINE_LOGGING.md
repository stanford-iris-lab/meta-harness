# online vs offline：各自产生什么日志 / proposer 能看到什么

> 全部对照 `inner_loop.py`（**当前 = committed 原始版本，`val_step` patch 已回退**）。结论先放最前：
>
> **online → proposer 看到的是 train-set 逐样本轨迹，看不到 val 逐样本。**
> **offline（当前，无 patch）→ proposer 既看不到 train 逐样本，也看不到 val 逐样本（仅聚合 `val_epoch`）。**
>
> 逐样本轨迹只能在真正发生预测的地方产生：online 在 train 预测，offline 在 train 喂 label 不预测、val 早停 eval 不写逐样本。所以现状下 **只有 online 有逐样本轨迹（且是 train）**。
>
> ℹ️ 历史：曾短暂加过一个 `val_step` patch 给 offline 补 val 逐样本轨迹，已按要求**回退**。本文保留它作为对照（标注 "已回退 patch"），方便日后判断是否重做。
>
> ⚠️ 更正：我之前口头说"online 暴露 val、隐藏 train"，那是**反的**。代码事实是 online 暴露 train。下面每条都标了行号，可逐条核对。

---

## 0. proposer（claude code）到底读什么

`meta_harness.py:render_task_prompt` 只把 `LOGS_DIR/` 指给 proposer。具体读哪些文件由 skill 决定：

- `.claude/skills/meta-harness/SKILL.md:56,67,125` → 让 proposer 读
  `logs/<dataset>/<memory>/<model>/log.jsonl`（skill 里叫 "training logs / training traces"）。
- `val.json`（`SKILL.md:124`）只有 **聚合** accuracy，无逐样本。

所以 **"proposer 看到的轨迹" == `log.jsonl` 里的逐样本条目**。问题归结为：每个 mode 往 `log.jsonl` 里写了哪些逐样本条目。

---

## 1. 入口分发：`__main__`

```python
if args.mode == "offline":                       # inner_loop.py:713
    run_inner_loop(..., mode="offline",
                   val_examples=val_examples ...,  # ← 传 val 进去
                   skip_train_eval=not eval_test)  # :724
else:                                             # :731  online
    for chunk in ...:
        run_inner_loop(..., mode="online")        # ← 不传 val_examples
```

两点关键差异从这里就定了：
1. **online 根本不把 `val_examples` 传进训练循环**（`:734-742`）。
2. offline 把 val 传进去，并用 `skip_train_eval=not eval_test` 控制是否跑最终 eval。

> evolve 的 val 阶段：只请求 `--val-output`，所以 `eval_test=False` → `skip_train_eval=True`。

---

## 2. online 产生什么（`run_inner_loop` online 分支，`:336+`）

流程：对每个 **train** 样本 `predict` → 判对错 → `learn_from_batch`。

逐样本日志 `step`（`:387-397`）：

```python
logger.log("step", step=global_idx,
           input_preview=inp[:200], pred=pred, tgt=tgt, ok=ok,
           prompt_len=..., prompt_hash=...)
```

一条 online `log.jsonl` 的内容：

```
meta → step×N_train → learn_batch×B → checkpoint → done
```

| 条目 | 来源 | 是哪个 split | 有 pred 吗 |
|---|---|---|---|
| `step` | `:388` | **train** | ✅ pred/tgt/ok/prompt_len/prompt_hash |
| `learn_batch` | `:419` | — | 只有耗时 |
| `done` | `:785` | — | 聚合 `val_acc`/`test_acc`（无逐样本）|

**val 在 online 里：** 只在训练后那段共享 eval（见 §4）跑一次，产出聚合 `val.json`，**不写任何逐样本日志**。

➡️ **online 的 proposer 只能看 train 逐样本轨迹，看不到 val 逐样本。**

---

## 3. offline 产生什么（`_run_offline_loop`，`:108`）

流程：train 阶段**直接喂 ground truth** 给 `learn_from_batch`（`:147-161`，**不预测**）→ 每个 epoch 后对 val 跑早停 eval。

### 3a. train 阶段日志 `train_batch`（`:164-172`）

```python
logger.log("train_batch", step=global_idx, epoch=epoch,
           batch_size=len(batch), train_ms=train_ms)
```

注意：**没有 pred**。因为 offline 训练根本不预测，标签是直接喂进去的。所以 train 在 offline 里**天然没有逐样本轨迹**。

### 3b. val 早停 eval（当前 `:178-190`）

```python
val_result = evaluate_memory(memory, val_examples, ...)   # 算了逐样本预测
logger.log("val_epoch", epoch=epoch, val_acc=..., ...)    # 但只记聚合
```

注意：`evaluate_memory` 内部其实算出了每条 val 的 `prediction`，但当前代码**只取聚合数写 `val_epoch`，逐样本预测被丢弃**。所以 offline 现状下 val 没有逐样本日志。

> 〔已回退 patch〕曾在这里加过一个循环，把 `val_result["predictions"]` 逐条写成 `val_step`
> （`input_preview/pred/tgt/ok/prompt_len/metrics`）。它是 offline 唯一能拿到逐样本 pred 的地方——
> 因为 train 不预测、最终 eval 又被跳过。现已移除，offline 回到"只有聚合"的原始行为。

### 3c. 最终 train-eval（`:226-266`）—— evolve 时被跳过

```python
if skip_train_eval:        # :228  evolve val 阶段 = True
    return ...             # 直接返回，下面整段不执行
...
logger.log("eval_step", ...)  # :257  仅当 skip_train_eval=False（即跑 test 时）
```

一条 offline `log.jsonl`（evolve val 阶段，**当前无 patch**）的内容：

```
meta → train_batch×N_train → checkpoint → val_epoch → done
```

| 条目 | 来源 | 是哪个 split | 有 pred 吗 |
|---|---|---|---|
| `train_batch` | `:166` | train | ❌ 只有耗时 |
| `val_epoch` | `:184` | val | 聚合 |
| `eval_step` | 最终 eval（仅 `skip_train_eval=False` 时）| train | ✅，但 evolve 时被跳过 |
| `done` | `:785` | — | 聚合 |
| ~~`val_step`~~ | ~~val（已回退 patch）~~ | ~~val~~ | ~~曾经 ✅，现已移除~~ |

➡️ **offline 现状：train 不预测、val 只记聚合、最终 eval 被跳过 → proposer 一条逐样本轨迹都看不到。**

---

## 4. 两个 mode 共享、与 mode 无关的部分

### val.json 的产出（`:757-774`，在 mode 分支之外）

```python
elif eval_val:
    result = evaluate_memory(memory, val_examples, evaluator)  # :768
    val_preds = result["predictions"]
# → make_result → _build_output → 写 val.json
```

无论 online/offline，训练结束都走这里产出 `val.json`，**只有聚合 accuracy/correct/total + 用量元数据，无逐样本**。这一段才是你直觉里"online/offline 不该影响 validation 怎么存"——**对，val.json 确实 mode-independent**。会变的只是 `log.jsonl` 里的逐样本轨迹。

### `done` 条目（`:785`）

两个 mode 都写，含聚合 `val_acc`/`test_acc`。

---

## 5. 对照总表

| 维度 | online | offline（当前，无 patch） |
|---|---|---|
| train 阶段对 train | predict→learn（`:340-414`） | 喂 label→learn（`:147-161`，不预测） |
| **train 逐样本轨迹** | ✅ `step`（`:388`） | ❌ 只有 `train_batch` |
| 训练中跑 val 吗 | ❌（不传 val_examples，`:734`） | ✅ 每 epoch 早停 eval（`:178`），但只记聚合 |
| **val 逐样本轨迹** | ❌ 永远没有 | ❌ 无（曾有 `val_step` patch，已回退） |
| val.json（聚合） | ✅ 共享路径 `:768` | ✅ 共享路径 `:768` |
| **proposer 实际能看到的逐样本** | **train** | **无** |

---

## 6. 一句话总结

- 逐样本轨迹**跟着"哪里发生预测"走，不跟着 split 走**：online 在 train 预测，offline 在 train 喂 label 不预测、val 早停 eval 只记聚合。
- 所以**当前**：online 给 proposer 喂 train 逐样本轨迹；offline 给 proposer 的逐样本轨迹是**零**（只有聚合 val_acc）。
- `val.json` 的聚合结果两个 mode 完全一致，与 mode 无关。
- 已回退的 `val_step` patch 曾让 offline 暴露 **val** 逐样本（不含 train）。若日后认为"该让 proposer 看 val 失败、又不泄露 train 标注"，重做那个 patch 是唯一能在 offline 下达成此意图的位置（train 不预测、最终 eval 被跳过）。当前已按要求回到无此能力的原始状态。
