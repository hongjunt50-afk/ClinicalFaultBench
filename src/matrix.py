"""故障矩阵跑批:对每个任务逐个注入 F1–F7 × mild/severe,记录扣分。

与干净基线的配对关系:同一 task_id 的干净分 - 故障分 = 该故障造成的性能损失。
产物:results/matrix_<model>.jsonl,每行一个 (task, fault, severity) 结果。

成本闸:--budget 为 credit 上限;----limit-per-fault 限制每种组合跑几个任务。

用法:
  python src/matrix.py --model deepseek-v4.1-flash --limit-per-fault 2      # 冒烟 84 次调用
  python src/matrix.py --model deepseek-v4.1-flash                        # 全量
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faults as fault_lib  # noqa: E402
import run as runner  # noqa: E402
import tasks as task_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"
RESULTS_DIR = REPO / "results"


def recompute_key(task: dict, corrupted: dict) -> dict:
    """**不改答案键**——这是本基准最关键的一条设计。

    患者的真实病情不会因为数据被注入故障而改变。故障注入只是让 agent 看到的世界变坏,
    正确答案仍然是**干净数据上的答案**。所以:
      - 若 agent 被故障带偏,输出偏移 → 配对错误率上升(这就是要度量的失效);
      - 若 agent 识别出数据有问题、拒绝作答或明确标注不确定 → 它的答案可能仍然对齐干净真值,
        或者至少不会给出错误结论。
    早期版本在这里按注入后数据重算答案键,效果等于"奖励盲信坏数据",已废弃。
    """
    return json.loads(json.dumps(task["answer_key"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="故障矩阵跑批")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--limit-per-fault", type=int, default=0,
                    help="每种 (fault,severity) 只跑前 N 个任务,0=全量")
    ap.add_argument("--task-type", choices=["T1", "T2", "T3"])
    ap.add_argument("--budget", type=float, default=500.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rescore", type=Path, help="对既有矩阵结果重判分")
    args = ap.parse_args()

    tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    if args.task_type:
        tasks = [t for t in tasks if t["task_type"] == args.task_type]

    if args.rescore:
        records = [json.loads(line) for line in args.rescore.read_text(encoding="utf-8").splitlines()]
        clean = {}
        cpath = RESULTS_DIR / f"baseline_{args.model}.jsonl"
        if cpath.exists():
            for line in cpath.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                clean[r["task_id"]] = r["metrics"]["score"]
        for r in records:
            parsed = r.get("parsed")
            r["metrics"] = task_lib.score(r["_task"], parsed) if parsed else {"score": 0.0}
        summarize(records, clean, args.model)
        return

    secrets = runner.load_secrets()
    # 干净基线分,用于配对相减
    clean: dict[str, float] = {}
    cpath = RESULTS_DIR / f"baseline_{args.model}.jsonl"
    if cpath.exists():
        for line in cpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            clean[r["task_id"]] = r["metrics"]["score"]

    # 供体化验(F6 用):取另一个任务的前几条化验
    donor = [e for t in tasks[-1:] for e in t["payload"]["events"]
             if e["category"] == "lab"][:5]

    jobs = []
    for fault in fault_lib.FAULTS:
        for severity in fault_lib.SEVERITIES:
            chosen = tasks if not args.limit_per_fault else tasks[:args.limit_per_fault]
            for t in chosen:
                corrupted, rec = fault_lib.inject(
                    t["payload"], {"fault": fault, "severity": severity,
                                   "seed": t["task_id"]}, donor)
                if not rec["applied"]:
                    continue
                jobs.append((t, fault, severity, corrupted, rec))

    print(f"待跑 {len(jobs)} 次调用(任务 {len(tasks)} × 故障 7 × 严重度 2,扣除不适用组合)")

    from concurrent.futures import ThreadPoolExecutor
    results: list[dict] = []
    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batch = []

        def settle(batch):
            nonlocal spent
            for fut in batch:
                t, fault, severity, corrupted, rec = fut[0]
                r = fut[1].result()
                r["_task"] = {**t, "answer_key": fut[2]}
                r.update({"fault": fault, "severity": severity, "fault_record": rec})
                results.append(r)
                spent += r.get("credit", 0)
                print(f"  {t['task_id']:16s} {fault}/{severity:6s} "
                      f"score={r['metrics']['score']:.3f} credit={r.get('credit', 0):.3f}")

        for t, fault, severity, corrupted, rec in jobs:
            if spent >= args.budget:
                print(f"预算用尽 {spent:.1f} >= {args.budget}")
                break
            key = recompute_key(t, corrupted)
            probe = {**t, "payload": corrupted, "answer_key": key}
            batch.append(((t, fault, severity, corrupted, rec),
                          pool.submit(runner.run_one, secrets["RELAY_BASE_URL"],
                                      secrets["RELAY_API_KEY"], args.model, probe), key))
            if len(batch) >= args.workers:
                settle(batch)
                batch = []
        if batch:
            settle(batch)

    out = RESULTS_DIR / f"matrix_{args.model}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(results, clean, args.model)
    print(f"明细 → {out}")


def summarize(records: list[dict], clean: dict[str, float], model: str) -> None:
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["fault"]][r["severity"]].append(
            (clean.get(r["task_id"]), r["metrics"]["score"], r["task_type"],
             r["metrics"].get("flagged")))
    print(f"\n{model} 故障效应(配对差值 = 干净分 - 故障分,正值表示性能下降)")
    header = f"{'故障':<8}{'mild n':>8}{'mild 分':>9}{'mild 降':>9}{'mild 告警':>10}" \
             f"{'sev n':>8}{'sev 分':>9}{'sev 降':>9}{'sev 告警':>10}"
    print(header)
    table = {}
    for fault in sorted(by):
        row = {}
        cells = f"{fault:<8}"
        for severity in ("mild", "severe"):
            vals = by[fault].get(severity, [])
            paired = [(c, s) for c, s, _, _ in vals if c is not None]
            drop = (sum(c - s for c, s in paired) / len(paired)) if paired else None
            flags = [f for *_, f in vals if f is not None]
            row[severity] = {
                "n": len(vals),
                "mean_score": round(sum(s for _, s, _, _ in vals) / len(vals), 4) if vals else None,
                "paired_drop": round(drop, 4) if drop is not None else None,
                "flag_rate": round(sum(flags) / len(flags), 3) if flags else None,
            }
            cells += (f"{len(vals):>8}{(row[severity]['mean_score'] or 0):>9.3f}"
                      f"{(row[severity]['paired_drop'] or 0):>9.3f}"
                      f"{(row[severity]['flag_rate'] if row[severity]['flag_rate'] is not None else 0):>10.2f}")
        print(cells)
        table[fault] = row
    out = RESULTS_DIR / f"matrix_summary_{model}.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总 → {out}")
    print("注:告警率=模型主动指出'数据有问题'的比例(T1)。干净基线告警率可作为假阳校准。")


if __name__ == "__main__":
    main()
