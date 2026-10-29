"""噪声下限对照:把 payload 换成空白/无关内容,量模型的"什么都不看"基线分。

为什么必须有:F2/F4 这类故障只是改数值、不改结构,模型若本来就不读数据、
靠提示词里的常识瞎猜,也能拿到不错的分——那不是"抗故障",是"没用数据"。
这道对照把"不看数据也能得的分"扣掉,剩下的才能算故障效应。

用法:
  python src/control.py --model deepseek-v4.1-flash --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repopaths  # noqa: E402
import run as runner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"


def blank_payload(payload: dict, kind: str) -> dict:
    if kind == "empty":
        return {"events": [], "diagnoses": []}
    # kind == "noise":结构合法但内容与临床无关(干扰项)
    return {
        "hadm_id": payload.get("hadm_id"), "admittime": payload.get("admittime"),
        "dischtime": payload.get("dischtime"), "age_at_admission": None,
        "diagnoses": [{"seq_num": 1, "icd_code": "000000", "icd_version": 0,
                       "title": "Routine administrative encounter"}],
        "events": [
            {"time": payload.get("admittime"), "category": "lab", "itemid": "0000-0",
             "label": "Administrative note", "fluid": "", "lab_category": "survey",
             "value": "n/a", "valuenum": None, "valueuom": None,
             "ref_range_lower": None, "ref_range_upper": None, "flag": None, "priority": None},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="噪声下限对照(不看数据能得多少分)")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--kind", choices=["empty", "noise"], default="empty")
    args = ap.parse_args()

    secrets = runner.load_secrets()
    tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    # 每类各取前 limit 个,便于分类型看
    picked, seen = [], {}
    for t in tasks:
        if seen.get(t["task_type"], 0) < args.limit:
            seen[t["task_type"]] = seen.get(t["task_type"], 0) + 1
            picked.append(t)

    total = 0.0
    by_type: dict[str, list[float]] = {}
    for t in picked:
        probe = dict(t)
        probe["payload"] = blank_payload(t["payload"], args.kind)
        rec = runner.run_one(secrets["RELAY_BASE_URL"], secrets["RELAY_API_KEY"],
                             args.model, probe)
        total += rec["metrics"]["score"]
        by_type.setdefault(t["task_type"], []).append(rec["metrics"]["score"])
        print(f"  {t['task_id']} [{args.kind}] score={rec['metrics']['score']:.3f} "
              f"credit={rec.get('credit', 0):.2f}")
    mean = total / len(picked)
    print(f"\n{args.model} 在「{args.kind}」payload 上平均分 {mean:.3f}"
          f"(n={len(picked)})——这是不看数据也能拿到的基础分,故障效应须扣除它")
    per_type = {tt: round(sum(v) / len(v), 4) for tt, v in sorted(by_type.items())}
    print("按任务类型:" + "  ".join(f"{tt}={v:.3f}(n={len(by_type[tt])})"
                                    for tt, v in per_type.items()))
    # T2 的干净对照上"无冲突"就是正确答案,空 payload 交空数组即满分——这不是模型鲁棒,
    # 是任务本身在白送分。必须单独点出来,否则会被当成"基线很高"读
    if per_type.get("T2", 0) >= 0.99:
        print("注意:T2 在空白 payload 上得 1.0——干净对照任务交空数组即满分,"
              "该子集不提供信息(见 RESULTS.md 局限)")
    out = REPO / "results" / f"control_{args.kind}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "kind": args.kind, "n": len(picked),
                               "mean_score": round(mean, 4), "by_type": per_type},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {repopaths.rel(out)}")


if __name__ == "__main__":
    main()
