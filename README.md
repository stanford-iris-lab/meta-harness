# Meta-Harness

![Meta-Harness](assets/repo.png)

Meta-Harness is a framework for automated search over task-specific model harnesses: the code around a fixed base model that decides what to store, retrieve, and show while the model works. This repo contains the framework and two reference experiments from the paper.

[![Paper](https://img.shields.io/badge/arXiv-2603.28052-b31b1b)](https://arxiv.org/abs/2603.28052)
[![Website](https://img.shields.io/badge/Project-Website-3569a8)](https://yoonholee.com/meta-harness/)
[![Slides](https://img.shields.io/badge/Slides-PDF-777777)](https://yoonholee.com/assets/pdf/meta-harness-slides.pdf)
[![TB2 Harness](https://img.shields.io/badge/TB2-Optimized_Harness-3569a8)](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)
[![License](https://img.shields.io/badge/License-MIT-777777)](LICENSE)

If you build with Meta-Harness, please send us a link.
We may feature your work here.

## Contents

- The reusable framework and onboarding flow for applying Meta-Harness to a new domain.
- Two paper reference experiments under `reference_examples/`:
  - [`reference_examples/text_classification/`](reference_examples/text_classification/README.md): memory-system search for text classification.
  - [`reference_examples/terminal_bench_2/`](reference_examples/terminal_bench_2/README.md): scaffold evolution for Terminal-Bench 2.0.
- The optimized Terminal-Bench 2 harness from the paper lives in the separate artifact repo: [stanford-iris-lab/meta-harness-tbench2-artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact).

## Built on Meta-Harness

### Projects

- [Don't Train the Model, Evolve the Harness](https://huggingface.co/spaces/joelniklaus/harness-optimization): Applies Meta-Harness to Harvey's Legal Agent Benchmark.
- [SuperagenticAI/metaharness](https://github.com/SuperagenticAI/metaharness): Python library and CLI for harness optimization with Codex.
- [Harness Forge](https://github.com/001TMF/harness-forge): Reimplements Meta-Harness as a native Claude Code skill.
- [dkhanal/meta-harness](https://huggingface.co/dkhanal/meta-harness): Independent implementation with a Symptom2Disease demo.
- [meta-harness-on-islo](https://github.com/zozo123/meta-harness-on-islo): Applies the Meta-Harness optimization loop to Islo sandboxes.
- [VideoHarness-RSI](https://github.com/Tencent/VideoHarness-RSI): Applies Meta-Harness to long-video retrieval and context construction around frozen vision-language models.

### Research

- [RRSI](https://arxiv.org/abs/2609.24972) ([code](https://github.com/google-research/rrsi)): Regularizes harness evolution by constraining candidate edits and filtering benchmark-specific or unnecessary changes.
- [WHALE](https://arxiv.org/abs/2609.00196) ([code](https://github.com/krafton-ai/WHALE)): Alternates model fine-tuning with harness search using Meta-Harness.
- [AutoRef](https://arxiv.org/abs/2609.35530) ([code](https://github.com/KuOnoda/AutoRef)): Applies harness optimization to multi-reference image generation.
- [MetaBench-Harness](https://arxiv.org/abs/2609.33411): Applies harness search over accumulated execution history to benchmark-generation pipelines.
- [Mixture of Self-Improving Branches](https://arxiv.org/abs/2609.37834): Extends Meta-Harness with adaptive search branches and a router that selects a harness for each input.
- [TTHE](https://arxiv.org/abs/2607.08124) ([code](https://github.com/junnie00/TTHE)): Builds on Meta-Harness to evolve harnesses at test time using unlabeled execution traces, adapting to shifting task distributions.

## Quick Start

Text classification:

```bash
cd reference_examples/text_classification
uv sync
uv run python meta_harness.py --iterations 1
```

Terminal-Bench 2 single-task check:

```bash
cd reference_examples/terminal_bench_2
uv sync
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

Use the subdir READMEs for setup details, expected runtime, and additional commands.

## Applying Meta-Harness To A New Domain

Start by pointing your coding assistant to [`ONBOARDING.md`](ONBOARDING.md) and having a conversation with it.
The conversation should produce a `domain_spec.md` with an implementation and evaluation plan for your domain.

The examples use Claude Code as the proposer agent. To use a different proposer agent, adapt the example `claude_wrapper.py` scripts in [`reference_examples/text_classification/claude_wrapper.py`](reference_examples/text_classification/claude_wrapper.py) or [`reference_examples/terminal_bench_2/claude_wrapper.py`](reference_examples/terminal_bench_2/claude_wrapper.py). The wrapper must log proposer interactions.

## Release Note

This is a cleaned-up version of the code we used for the paper. It has not been tested beyond verifying that it runs. Please let us know if anything goes wrong.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@misc{lee2026metaharnessendtoendoptimizationmodel,
      title={Meta-Harness: End-to-End Optimization of Model Harnesses},
      author={Yoonho Lee and Roshen Nair and Qizheng Zhang and Kangwook Lee and Omar Khattab and Chelsea Finn},
      year={2026},
      eprint={2603.28052},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.28052},
}
```
