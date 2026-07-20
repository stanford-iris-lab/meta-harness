"""Build the val-vs-iteration curve (PNG) + a factual markdown report from a run's logs.

    uv run python plot_report.py --run-dir logs/overnight

Reads Phase-0 baseline summaries, evolution_summary.jsonl, frontier_val.json, and (if
present) test/test_summary.json. Writes curve.png, report.md, and report_data.json under
the run dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

EVOLVE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((EVOLVE_DIR / "config.yaml").read_text())
BASELINES = ["baseline_react", "baseline_single"]


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def baseline_val(run_dir: Path) -> dict:
    out = {}
    for b in BASELINES:
        f = run_dir / b / "summary.json"
        if f.exists():
            out[b] = json.loads(f.read_text()).get("pass_rate", 0.0)
    return out


def build(run_dir: Path) -> dict:
    rows = load_jsonl(run_dir / "evolution_summary.jsonl")
    bval = baseline_val(run_dir)
    baseline_best = max(bval.values()) if bval else 0.0

    # Per-iteration candidates + cumulative frontier best-so-far.
    iters = sorted({r["iteration"] for r in rows})
    per_iter = {}
    for r in rows:
        per_iter.setdefault(r["iteration"], []).append(r)

    running = baseline_best
    curve = [{"iter": 0, "frontier": baseline_best, "best_candidate": baseline_best,
              "label": "baselines"}]
    for it in iters:
        cands = per_iter[it]
        best_this = max((c.get("pass_rate", 0.0) for c in cands), default=0.0)
        running = max(running, best_this)
        curve.append({"iter": it, "frontier": running, "best_candidate": best_this,
                      "candidates": [c["agent"] for c in cands]})

    frontier = {}
    fpath = run_dir / "frontier_val.json"
    if fpath.exists():
        frontier = json.loads(fpath.read_text())
    test = {}
    tpath = run_dir / "test" / "test_summary.json"
    if tpath.exists():
        test = json.loads(tpath.read_text())

    return {
        "baseline_val": bval,
        "baseline_best": baseline_best,
        "rows": rows,
        "curve": curve,
        "frontier": frontier,
        "test": test,
    }


def plot(data: dict, run_dir: Path, run_name: str) -> Path:
    curve = data["curve"]
    xs = [c["iter"] for c in curve]
    ys = [c["frontier"] for c in curve]
    cand_x = [c["iter"] for c in curve[1:]]
    cand_y = [c["best_candidate"] for c in curve[1:]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [y * 100 for y in ys], "-o", color="#2563eb", lw=2, label="frontier best val (running max)")
    if cand_x:
        ax.scatter(cand_x, [y * 100 for y in cand_y], color="#f59e0b", zorder=5, label="iteration candidate val")
    for b, v in data["baseline_val"].items():
        ax.axhline(v * 100, ls="--", lw=1, alpha=0.6, label=f"{b} (val)")
    ax.set_xlabel("iteration (0 = baselines)")
    ax.set_ylabel("val pass_rate %  (8 tasks, OJ hard)")
    ax.set_title(f"SpreadsheetBench evolution — {run_name}")
    ax.set_ylim(-3, 103)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = run_dir / "curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def write_report(data: dict, run_dir: Path, run_name: str) -> Path:
    L = []
    L.append(f"# SpreadsheetBench evolution report — `{run_name}`\n")
    src = CONFIG["dataset"]["source"]
    L.append(f"- model: `{CONFIG['model']['name']}` (QNAIGC)  ·  source: `{src}`  ·  "
             f"val={CONFIG['dataset']['dev_size']} tasks  ·  metric: OJ hard (all test cases pass)\n")
    bv = data["baseline_val"]
    L.append("## Baselines (val)\n")
    for b, v in bv.items():
        L.append(f"- `{b}`: {v:.1%}")
    L.append(f"\n**Baseline best val: {data['baseline_best']:.1%}**\n")

    best = data["frontier"].get("_best", {})
    L.append(f"## Frontier\n\n- best agent: `{best.get('agent','?')}`  ·  "
             f"val {best.get('pass_rate', 0):.1%}\n")

    L.append("## Per-iteration\n")
    L.append("| iter | agent | val | delta | turns | tokens | outcome | hypothesis |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in data["rows"]:
        rm = r.get("rollout_metrics") or {}
        L.append(
            f"| {r.get('iteration')} | `{r.get('agent')}` | {r.get('pass_rate',0):.1%} | "
            f"{(r.get('delta') if r.get('delta') is not None else 0):+.3f} | "
            f"{rm.get('mean_turns','')} | {rm.get('mean_tokens','')} | {r.get('outcome','')} | "
            f"{(r.get('hypothesis','') or '')[:80].replace(chr(10),' ')} |"
        )

    if data["test"]:
        L.append("\n## Held-out test (200 tasks, disjoint from val)\n")
        L.append("| agent | val | test | mean_turns | mean_tokens |")
        L.append("|---|---|---|---|---|")
        # val lookup: baselines from baseline_val; others from frontier per-agent isn't stored,
        # so pull val from evolution rows (last occurrence).
        val_by_agent = dict(data["baseline_val"])
        for r in data["rows"]:
            val_by_agent[r["agent"]] = r.get("pass_rate", 0.0)
        for name, t in data["test"].items():
            v = val_by_agent.get(name)
            vstr = f"{v:.1%}" if v is not None else "—"
            L.append(f"| `{name}` | {vstr} | {t['pass_rate']:.1%} ({t['n_pass']}/{t['n_tasks']}) | "
                     f"{t.get('mean_turns','')} | {t.get('mean_tokens','')} |")

    L.append("\n## Analysis\n\n_(filled in at delivery)_\n")
    out = run_dir / "report.md"
    out.write_text("\n".join(L))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_name = args.run_name or run_dir.name
    data = build(run_dir)
    (run_dir / "report_data.json").write_text(json.dumps(data, indent=2, default=str))
    png = plot(data, run_dir, run_name)
    rep = write_report(data, run_dir, run_name)
    print("wrote:", png, rep, run_dir / "report_data.json")


if __name__ == "__main__":
    main()
