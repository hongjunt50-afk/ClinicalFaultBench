"""生成论文主图:注入严重度 → 性能下降曲线 + 各故障效应条形图。

读 results/baseline_<model>.jsonl(干净基线)与 results/matrix_<model>.jsonl(故障矩阵),
重判分后出图。不调用任何模型,可反复重跑。

用法:
  python src/figures.py --model deepseek-v4.1-flash
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as task_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "tasks.jsonl"
RESULTS = REPO / "results"

FAULT_NAMES = {"F1": "F1 关键化验缺失", "F2": "F2 单位错误", "F3": "F3 矛盾记录",
               "F4": "F4 陈旧数据", "F5": "F5 结果截断", "F6": "F6 跨患者污染",
               "F7": "F7 时间线漂移"}
SEV_ORDER = ["clean", "mild", "severe"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(model: str) -> tuple[dict, dict, dict]:
    tasks = {json.loads(l)["task_id"]: json.loads(l)
             for l in CONFIG.read_text(encoding="utf-8").splitlines()}
    clean = {}
    for line in (RESULTS / f"baseline_{model}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        clean[r["task_id"]] = task_lib.score(tasks[r["task_id"]], r.get("parsed"))["score"] \
            if r.get("parsed") else 0.0
    faulted = []
    for line in (RESULTS / f"matrix_{model}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = tasks[r["task_id"]]
        m = task_lib.score(t, r.get("parsed")) if r.get("parsed") else {"score": 0.0}
        faulted.append({"task_id": r["task_id"], "task_type": t["task_type"],
                        "fault": r["fault"], "severity": r["severity"],
                        "score": m["score"], "clean": clean.get(r["task_id"])})
    return tasks, clean, {"faulted": faulted}


def fig_severity_curve(faulted: list[dict], clean: dict, model: str, out: Path) -> None:
    """每条曲线 = 一个故障类型;x 轴 clean → mild → severe;y 轴平均得分。"""
    by = defaultdict(lambda: defaultdict(list))
    for r in faulted:
        by[r["fault"]][r["severity"]].append(r["score"])
    clean_mean = statistics.mean(clean.values())

    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = [0, 1, 2]
    for fault in sorted(by):
        ys = [clean_mean]
        for sev in ("mild", "severe"):
            vals = by[fault].get(sev, [])
            ys.append(statistics.mean(vals) if vals else None)
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=FAULT_NAMES[fault])
    ax.set_xticks(xs)
    ax.set_xticklabels(["干净基线", "mild", "severe"])
    ax.set_ylabel("任务平均得分")
    ax.set_title(f"注入严重度 → 性能变化({model},n={len(clean)} 任务)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_drop_bars(faulted: list[dict], model: str, out: Path) -> None:
    """配对下降幅度条形图(干净分 - 故障分),按故障与严重度分组。"""
    acc = defaultdict(list)
    for r in faulted:
        if r["clean"] is None:
            continue
        acc[(r["fault"], r["severity"])].append(r["clean"] - r["score"])
    faults = sorted({f for f, _ in acc})
    mild = [statistics.mean(acc[(f, "mild")]) if acc.get((f, "mild")) else 0 for f in faults]
    sev = [statistics.mean(acc[(f, "severe")]) if acc.get((f, "severe")) else 0 for f in faults]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.38
    idx = range(len(faults))
    ax.bar([i - w / 2 for i in idx], mild, w, label="mild")
    ax.bar([i + w / 2 for i in idx], sev, w, label="severe")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels([FAULT_NAMES[f] for f in faults], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("配对性能下降(干净分 − 故障分)")
    ax.set_title(f"各类故障造成的性能损失({model})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="出图:严重度曲线与故障效应")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    args = ap.parse_args()

    tasks, clean, data = load(args.model)
    faulted = data["faulted"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    p1 = RESULTS / f"fig_severity_curve_{args.model}.png"
    p2 = RESULTS / f"fig_fault_drop_{args.model}.png"
    fig_severity_curve(faulted, clean, args.model, p1)
    fig_drop_bars(faulted, args.model, p2)
    print(f"{p1}\n{p2}")

    # 打印表格(论文 Table 用)
    by = defaultdict(lambda: defaultdict(list))
    for r in faulted:
        by[r["fault"]][r["severity"]].append(r["score"])
    print(f"\n干净基线均值 {statistics.mean(clean.values()):.3f} (n={len(clean)})")
    print(f"{'故障':<6}{'mild n':>8}{'mild 均值':>10}{'severe n':>10}{'severe 均值':>12}")
    for fault in sorted(by):
        m, s = by[fault].get("mild", []), by[fault].get("severe", [])
        print(f"{fault:<6}{len(m):>8}{(statistics.mean(m) if m else 0):>10.3f}"
              f"{len(s):>10}{(statistics.mean(s) if s else 0):>12.3f}")


if __name__ == "__main__":
    main()
