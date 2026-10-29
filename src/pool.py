"""任务池构建:从 Synthea 住院里筛出"三要素齐全"的实例,并按病种分层抽样。

为什么需要分层:Synthea 的 lung_cancer 模块化疗流程天然产大量住院,实测占可用住院 61%。
不分层会让整个基准退化成"肺癌化疗场景基准"。

用法(仓库根目录):
  python src/pool.py --build                     # 生成 configs/task_pool.json
  python src/pool.py --build --max-per-disease 20
  python src/pool.py --stats                     # 只看统计,不写文件
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repopaths  # noqa: E402
import synthea  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "data" / "synthea" / "csv"
OUT_PATH = REPO / "configs" / "task_pool.json"

# 三要素门槛:化验够多才谈得上"找出最异常",医嘱够多才有冲突可查
MIN_LABS = 10
MIN_RX = 3


def _encounter_reasons(csv_dir: Path) -> dict[str, str]:
    with open(csv_dir / "encounters.csv", encoding="utf-8-sig", newline="") as f:
        return {r["Id"]: (r.get("REASONDESCRIPTION") or "unknown") for r in csv.DictReader(f)}


def collect(csv_dir: Path = CSV_DIR) -> list[dict]:
    reasons = _encounter_reasons(csv_dir)
    idx = synthea.Index(csv_dir)
    out = []
    for pid, gender, birth, n_adm in synthea.list_patients(csv_dir):
        if n_adm == 0:
            continue
        tl = synthea.build_timeline(csv_dir, pid, idx)
        for adm in tl["admissions"]:
            n_lab = sum(1 for e in adm["events"] if e["category"] == "lab")
            n_rx = sum(1 for e in adm["events"] if e["category"] == "prescription")
            if n_lab < MIN_LABS or n_rx < MIN_RX or not adm["diagnoses"]:
                continue
            out.append({
                "subject_id": pid, "hadm_id": adm["hadm_id"], "gender": gender,
                "admittime": adm["admittime"], "dischtime": adm["dischtime"],
                "reason": reasons.get(adm["hadm_id"], "unknown"),
                "n_labs": n_lab, "n_rx": n_rx, "n_dx": len(adm["diagnoses"]),
            })
    return out


def stratify(instances: list[dict], max_per_disease: int = 20) -> list[dict]:
    """每病种最多取 max_per_disease 个,超出部分按化验数(信息量)降序保留。"""
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        by_reason[inst["reason"]].append(inst)
    picked = []
    for reason, group in sorted(by_reason.items()):
        group.sort(key=lambda x: (-x["n_labs"], -x["n_rx"], x["hadm_id"]))
        picked.extend(group[:max_per_disease])
    return sorted(picked, key=lambda x: x["hadm_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="任务池构建与病种分层抽样")
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--max-per-disease", type=int, default=20)
    args = ap.parse_args()

    instances = collect(args.csv_dir)
    counts = Counter(x["reason"] for x in instances)
    print(f"三要素齐全住院 {len(instances)} 次,病种 {len(counts)} 类")
    for reason, n in counts.most_common(12):
        print(f"  {n:4d}  {reason[:60]}")

    if args.build:
        picked = stratify(instances, args.max_per_disease)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "source": "synthea", "seed": 20261026,
            "criteria": {"min_labs": MIN_LABS, "min_rx": MIN_RX,
                         "max_per_disease": args.max_per_disease},
            "n_instances": len(picked),
            "instances": picked,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        after = Counter(x["reason"] for x in picked)
        print(f"\n分层后 {len(picked)} 次(病种 {len(after)} 类)→ {repopaths.rel(OUT_PATH)}")
        top = after.most_common(1)[0]
        print(f"最大病种占比 {top[1] / len(picked):.1%} ({top[0][:40]})")
        assert len(picked) >= 90, f"分层后不足 90 个实例(实得 {len(picked)})"
        print("BUILD OK")


if __name__ == "__main__":
    main()
