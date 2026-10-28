"""重测噪声下限:同一任务、同一 prompt、temperature=0 再跑一遍,两次干净分之差。

为什么必须测:kimi-k3.1-pro 这类模型在 temperature=0 下**仍然不是确定性的**(推理路径有
采样)。配对差值 = 干净分 − 故障分,若模型自身重测就能差出 ±0.2,那 0.05 量级的故障效应
根本不可分辨,负效应(F7 −0.040)也只是噪声而不是"故障反而帮忙"。

不看这一层,就会把"模型自己抖了一下"写成"模型对这类故障免疫"或"故障有好处"。

用法(仓库根目录):
  python src/retest.py --model kimi-k3.1-pro --per-type 2      # 每类重测 2 个,共 6 次调用
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"
RESULTS_DIR = REPO / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description="重测噪声下限(temperature=0 的可复现性)")
    ap.add_argument("--model", default="kimi-k3.1-pro")
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    secrets = runner.load_secrets()
    tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    base_path = RESULTS_DIR / f"baseline_{args.model}.jsonl"
    base = {r["task_id"]: r for r in
            (json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines()
             if line.strip())}

    picked, seen = [], {}
    for t in tasks:
        if t["task_id"] not in base:
            continue
        if seen.get(t["task_type"], 0) < args.per_type:
            seen[t["task_type"]] = seen.get(t["task_type"], 0) + 1
            picked.append(t)

    from concurrent.futures import ThreadPoolExecutor
    out = []
    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        running = []
        for t in picked:
            if spent >= args.budget:
                break
            running.append((t, pool.submit(runner.run_one, secrets["RELAY_BASE_URL"],
                                           secrets["RELAY_API_KEY"], args.model, t)))
            if len(running) >= args.workers:
                for tt, fut in running:
                    r = fut.result()
                    spent += r.get("credit", 0)
                    out.append((tt, r))
                running = []
        for tt, fut in running:
            r = fut.result()
            spent += r.get("credit", 0)
            out.append((tt, r))

    rows, deltas, by_type = [], [], {}
    for t, r in out:
        first = base[t["task_id"]]["metrics"]["score"]
        second = r["metrics"]["score"]
        d = second - first
        rows.append({"task_id": t["task_id"], "task_type": t["task_type"],
                     "run1": first, "run2": second, "delta": round(d, 4),
                     "credit": r.get("credit", 0)})
        deltas.append(d)
        by_type.setdefault(t["task_type"], []).append(d)
        print(f"  {t['task_id'][:26]:26s} {t['task_type']} "
              f"run1={first:.3f} run2={second:.3f} Δ={d:+.3f}")

    absd = [abs(d) for d in deltas]
    summary = {
        "model": args.model, "n": len(rows), "credit": round(spent, 2),
        "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
        "mean_abs_delta": round(statistics.mean(absd), 4) if absd else None,
        "max_abs_delta": round(max(absd), 4) if absd else None,
        "sd_delta": round(statistics.stdev(deltas), 4) if len(deltas) > 1 else None,
        "by_type": {k: round(statistics.mean(v), 4) for k, v in sorted(by_type.items())},
    }
    print(f"\n{args.model} 重测噪声:平均绝对差 {summary['mean_abs_delta']},"
          f"最大 {summary['max_abs_delta']},样本标准差 {summary['sd_delta']}(n={len(rows)},"
          f"{summary['credit']} credit)")
    print("注:配对故障效应低于这个量级的一律不可信——包括负效应。")
    path = RESULTS_DIR / f"retest_{args.model}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
