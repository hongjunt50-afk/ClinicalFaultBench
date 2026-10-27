"""Synthea 适配器:把 Synthea 的 CSV 导出聚合成与 timeline.py 同构的"病程时间线" JSON。

定位:ClinicalFaultBench v2 的 T1 主实验数据源(纯合成、Apache-2.0、零合规约束)。
输出 schema 与 src/timeline.py(MIMIC demo)完全一致,FaultProxy 与判分器只认 schema,不认数据源。

与 MIMIC 版的已知差异(设计取舍,写进论文 Limitations):
  - diagnoses 用 SNOMED-CT 编码(icd_version=0),而非 ICD-9/10;中文映射需另配 SNOMED 表
  - 过敏记录极少(Synthea 默认只产环境过敏原),F3 矛盾记录改由"诊断特征 vs 用药禁忌"承载
  - 时间戳是 UTC 带 T/Z,统一裁成 'YYYY-MM-DD HH:MM:SS'(与 MIMIC 版可比)

用法(仓库根目录):
  python src/synthea.py --list                    # 列出患者与可用住院
  python src/synthea.py --patient-id <uuid>       # 输出该患者时间线 JSON
  python src/synthea.py --patient-id <uuid> --out results/examples/s.json
  python src/synthea.py --check                   # 自检
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = REPO / "data" / "synthea" / "csv"

REQUIRED_FILES = ["patients.csv", "encounters.csv", "observations.csv", "medications.csv",
                  "conditions.csv", "allergies.csv"]


def _rows(csv_dir: Path, name: str) -> list[dict]:
    with open(csv_dir / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _ts(value: str | None) -> str | None:
    """Synthea 的 '2017-06-22T03:07:24Z' / '2017-06-22' → 'YYYY-MM-DD HH:MM:SS'。"""
    if not value:
        return None
    v = value.strip().replace("Z", "").replace("T", " ")
    return v if len(v) == 19 else (v + " 00:00:00" if len(v) == 10 else v)


def _num(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fluid_of(description: str) -> str:
    """LOINC 长名 'Potassium [Moles/volume] in Serum or Plasma' → 'Serum or Plasma'。"""
    marker = " in "
    return description.split(marker, 1)[1].strip() if marker in description else ""


def _overlaps(start: str, stop: str | None, enc_start: str, enc_stop: str) -> bool:
    """区间相交:无 stop 视为持续至今。日期字符串定宽,可直接字典序比较。"""
    s = _ts(start) or ""
    e = _ts(stop) or "9999-12-31 23:59:59"
    return s <= enc_stop and e >= enc_start


def list_patients(csv_dir: Path = DEFAULT_CSV_DIR) -> list[tuple]:
    """只读 patients + encounters 两张小表即可算住院数,不需要全量索引。"""
    pts = {r["Id"]: r for r in _rows(csv_dir, "patients.csv")}
    counts: dict[str, int] = {}
    for e in _rows(csv_dir, "encounters.csv"):
        if e["ENCOUNTERCLASS"] == "inpatient":
            counts[e["PATIENT"]] = counts.get(e["PATIENT"], 0) + 1
    return [(pid, p["GENDER"], p["BIRTHDATE"], counts.get(pid, 0))
            for pid, p in sorted(pts.items())]


class Index:
    """一次性把 Synthea CSV 读进内存并按患者建索引。

    build_timeline 早期版本每次调用都重读全部 CSV,1378 患者时是 O(n²)(实测跑不完)。
    批量场景必须传同一个 Index 实例。
    """

    def __init__(self, csv_dir: Path = DEFAULT_CSV_DIR):
        for name in REQUIRED_FILES:
            if not (csv_dir / name).exists():
                raise FileNotFoundError(f"缺文件 {csv_dir / name}")
        self.patients = {r["Id"]: r for r in _rows(csv_dir, "patients.csv")}
        self.encounters: dict[str, list[dict]] = {}
        self.raw_encounters: dict[str, list[dict]] = {}   # 含门诊,随访间隔判分要用
        for e in _rows(csv_dir, "encounters.csv"):
            self.raw_encounters.setdefault(e["PATIENT"], []).append(e)
            if e["ENCOUNTERCLASS"] == "inpatient":
                self.encounters.setdefault(e["PATIENT"], []).append(e)
        for group in self.encounters.values():
            group.sort(key=lambda e: e["START"])
        self.obs: dict[str, list[dict]] = {}
        for o in _rows(csv_dir, "observations.csv"):
            if o["CATEGORY"] in ("laboratory", "vital-signs"):
                self.obs.setdefault(o["PATIENT"], []).append(o)
        self.meds: dict[str, list[dict]] = {}
        for m in _rows(csv_dir, "medications.csv"):
            self.meds.setdefault(m["PATIENT"], []).append(m)
        self.dx: dict[str, list[dict]] = {}
        for d in _rows(csv_dir, "conditions.csv"):
            self.dx.setdefault(d["PATIENT"], []).append(d)
        self.allergies: dict[str, list[dict]] = {}
        for a in _rows(csv_dir, "allergies.csv"):
            self.allergies.setdefault(a["PATIENT"], []).append(a)


def build_timeline(csv_dir: Path, patient_id: str, index: Index | None = None) -> dict:
    idx = index if index is not None else Index(csv_dir)
    pat = idx.patients.get(patient_id)
    if pat is None:
        raise KeyError(f"患者 {patient_id} 不存在")

    birthdate = _ts(pat["BIRTHDATE"])
    anchor_age = datetime.fromisoformat(birthdate).year if birthdate else None

    encs = idx.encounters.get(patient_id, [])
    obs = idx.obs.get(patient_id, [])
    meds = idx.meds.get(patient_id, [])
    dxs = idx.dx.get(patient_id, [])
    algs = idx.allergies.get(patient_id, [])

    admissions = []
    for enc in encs:
        enc_start, enc_stop = _ts(enc["START"]), _ts(enc["STOP"])
        enc_id = enc["Id"]
        birth_year = int(birthdate[:4]) if birthdate else 0
        adm_year = int(enc_start[:4]) if enc_start else 0

        # 按 ENCOUNTER 外键直连,不用时间窗——时间窗会把门诊化验误记到住院里
        lab_events = [{
            "time": _ts(o["DATE"]), "category": "lab", "itemid": o["CODE"],
            "label": o["DESCRIPTION"].split(" [")[0], "fluid": _fluid_of(o["DESCRIPTION"]),
            "lab_category": o["CATEGORY"], "value": o["VALUE"], "valuenum": _num(o["VALUE"]),
            "valueuom": o.get("UNITS") or None, "ref_range_lower": None,
            "ref_range_upper": None, "flag": None, "priority": None,
        } for o in obs if o.get("ENCOUNTER") == enc_id]

        rx_events = [{
            "time": _ts(m["START"]), "category": "prescription",
            "drug": m["DESCRIPTION"], "drug_type": "MAIN",
            "dose_val_rx": None, "dose_unit_rx": None, "route": None,
            "stoptime": _ts(m["STOP"]),
        } for m in meds if m.get("ENCOUNTER") == enc_id]

        # Synthea 没有给药执行记录(emar),prescription 即用药事件的终点粒度
        events = sorted(lab_events + rx_events, key=lambda e: (e["time"] is None, e["time"] or ""))

        # 诊断:Synthea 只有 18/81 次住院把 condition 挂到 ENCOUNTER 外键上,
        # 其余靠"活动期与住院窗口重叠"归属(实测 81/81 全覆盖,中位 17 条)。
        # 两种口径取并集,按 seq_num 排序;icd_version=0 表示 SNOMED-CT。
        linked = [d for d in dxs if d.get("ENCOUNTER") == enc_id]
        linked_ids = {d["CODE"] for d in linked}
        overlapping = [d for d in dxs
                       if d["CODE"] not in linked_ids
                       and _overlaps(d["START"], d["STOP"], enc_start, enc_stop)]
        admission_dx = [{"seq_num": i + 1, "icd_code": d["CODE"], "icd_version": 0,
                         "title": d["DESCRIPTION"]}
                        for i, d in enumerate(linked + overlapping)]

        admissions.append({
            "hadm_id": enc["Id"], "admittime": enc_start, "dischtime": enc_stop,
            "admission_type": enc.get("REASONCODE") or None,
            "admission_location": None,
            # Synthea 用出院指向区分死亡(encounters 的 STOP 后接 death);此处保守留空
            "discharge_location": None,
            "hospital_expire_flag": 0,
            "age_at_admission": (adm_year - birth_year) if adm_year else None,
            "diagnoses": admission_dx,
            "events": events,
        })

    return {
        "subject_id": patient_id, "source": "synthea", "gender": pat["GENDER"],
        "anchor_age": anchor_age, "dod": _ts(pat.get("DEATHDATE")),
        "n_admissions": len(admissions),
        "allergies": [{"description": a.get("DESCRIPTION"), "category": a.get("CATEGORY"),
                       "severity": a.get("SEVERITY1")} for a in algs],
        "admissions": admissions,
    }


def check(csv_dir: Path = DEFAULT_CSV_DIR) -> None:
    idx = Index(csv_dir)
    patients = [(pid, p["GENDER"], p["BIRTHDATE"], len(idx.encounters.get(pid, [])))
                for pid, p in sorted(idx.patients.items())]
    with_inpatient = [(pid, n) for pid, _, _, n in patients if n > 0]
    print(f"患者 {len(patients)} 人,其中有住院的患者 {len(with_inpatient)} 人")

    usable = rich = 0
    total_adm = 0
    all_labs = []
    for pid, _ in with_inpatient:
        tl = build_timeline(csv_dir, pid, idx)
        for adm in tl["admissions"]:
            total_adm += 1
            n_lab = sum(1 for e in adm["events"] if e["category"] == "lab")
            n_rx = sum(1 for e in adm["events"] if e["category"] == "prescription")
            usable += n_lab >= 10 and n_rx >= 3
            rich += n_lab >= 30 and n_rx >= 8
            times = [e["time"] for e in adm["events"] if e["time"]]
            assert times == sorted(times), f"{pid}/{adm['hadm_id']} 事件未按时间排序"
            assert adm["admittime"] <= adm["dischtime"], f"{pid}/{adm['hadm_id']} 时间倒挂"
            all_labs.extend(e for e in adm["events"]
                            if e["category"] == "lab" and e["valuenum"] is not None)
    print(f"住院 {total_adm} 次,可用(化验>=10且医嘱>=3){usable} 次,富集{rich} 次")
    assert total_adm > 0 and usable > 0, "无可用住院"

    uom = sum(1 for e in all_labs if e["valueuom"])
    print(f"数值型化验 {len(all_labs)} 条,带单位 {uom} 条({uom / len(all_labs):.1%})")
    assert all_labs and uom / len(all_labs) >= 0.9, "数值型化验带单位比例过低,F2 依赖此列"
    print("CHECK OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthea 病程时间线适配器")
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--patient-id")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        check(args.csv_dir)
    elif args.list:
        for pid, gender, birth, n in list_patients(args.csv_dir):
            print(f"{pid}  {gender}  {birth}  住院 {n} 次")
    elif args.patient_id:
        text = json.dumps(build_timeline(args.csv_dir, args.patient_id),
                          ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"{args.out} 写入完成")
        else:
            print(text)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
