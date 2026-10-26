"""M1.1 病程时间线加载器:把 MIMIC-IV demo 散表聚合成结构化"病程时间线" JSON。

用法(仓库根目录):
  python src/timeline.py --list                     # 列出全部患者与住院数
  python src/timeline.py --patient-id 10000032      # 输出该患者时间线 JSON(stdout)
  python src/timeline.py --patient-id 10000032 --out results/examples/t.json
  python src/timeline.py --check                    # 自检:规模/排序/完整性断言
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO / "data" / "mimic-iv-demo-hf-mirror" / "data"


def _s(t):
    # 镜像的时间列是 'YYYY-MM-DD HH:MM:SS' 定宽字符串(字典序=时间序);官方 zip 换源后可能是
    # TIMESTAMP,这里统一归一成字符串,两种来源兼容
    if t is None:
        return None
    return t if isinstance(t, str) else t.isoformat(sep=" ")


def _num(v):
    # 镜像的数值列同为 VARCHAR,且 NULL 被转成了 '-' 之类占位符;解析失败一律归 None
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

# 镜像 parquet 的最低列要求(footer 实测;缺列直接报错,防静默错位)
REQUIRED_COLUMNS = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                   "admission_location", "discharge_location", "hospital_expire_flag"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum",
                  "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority"],
    "d_labitems": ["itemid", "label", "fluid", "category"],
    "prescriptions": ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "drug_type",
                      "dose_val_rx", "dose_unit_rx", "route"],
    "emar": ["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
    "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "d_icd_diagnoses": ["icd_code", "icd_version", "long_title"],
}


def _t(data_dir: Path, table: str) -> str:
    return f"read_parquet('{(data_dir / f'{table}.parquet').as_posix()}')"


def connect(data_dir: Path = DEFAULT_DATA_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for table, cols in REQUIRED_COLUMNS.items():
        path = data_dir / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺表: {path}")
        actual = {row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM {_t(data_dir, table)}").fetchall()}
        missing = set(cols) - actual
        if missing:
            raise ValueError(f"{table} 缺列: {sorted(missing)}(镜像与预期 schema 不符)")
    return con


def list_patients(con: duckdb.DuckDBPyConnection, data_dir: Path) -> list[tuple]:
    return con.execute(f"""
        SELECT p.subject_id, p.gender, p.anchor_age, count(DISTINCT a.hadm_id) AS n_admissions
        FROM {_t(data_dir, 'patients')} p
        LEFT JOIN {_t(data_dir, 'admissions')} a USING (subject_id)
        GROUP BY 1, 2, 3 ORDER BY 1
    """).fetchall()


def build_timeline(con: duckdb.DuckDBPyConnection, data_dir: Path, subject_id: int) -> dict:
    pat = con.execute(f"""
        SELECT subject_id, gender, anchor_age, anchor_year, dod
        FROM {_t(data_dir, 'patients')} WHERE subject_id = ?
    """, [subject_id]).fetchone()
    if pat is None:
        raise KeyError(f"患者 {subject_id} 不在数据集中")
    sid, gender, anchor_age, anchor_year, dod = pat
    # 镜像把整数列也存成了 VARCHAR(偷懒的 CSV 直转),显式归一
    anchor_age, anchor_year = int(anchor_age), int(anchor_year)

    admits = con.execute(f"""
        SELECT hadm_id, admittime, dischtime, admission_type, admission_location,
               discharge_location, hospital_expire_flag
        FROM {_t(data_dir, 'admissions')} WHERE subject_id = ? ORDER BY admittime
    """, [subject_id]).fetchall()

    admissions = []
    for hadm, admittime, dischtime, atype, aloc, dloc, expire in admits:
        dx = con.execute(f"""
            SELECT d.seq_num, d.icd_code, d.icd_version, coalesce(i.long_title, '') AS title
            FROM {_t(data_dir, 'diagnoses_icd')} d
            LEFT JOIN {_t(data_dir, 'd_icd_diagnoses')} i USING (icd_code, icd_version)
            WHERE d.subject_id = ? AND d.hadm_id = ? ORDER BY d.seq_num
        """, [sid, hadm]).fetchall()

        labs = con.execute(f"""
            SELECT l.charttime, l.itemid, i.label, i.fluid, i.category,
                   l.value, l.valuenum, l.valueuom,
                   l.ref_range_lower, l.ref_range_upper, l.flag, l.priority
            FROM {_t(data_dir, 'labevents')} l
            LEFT JOIN {_t(data_dir, 'd_labitems')} i USING (itemid)
            WHERE l.subject_id = ? AND l.hadm_id = ?
        """, [sid, hadm]).fetchall()
        lab_events = [{
            "time": _s(t), "category": "lab", "itemid": itemid, "label": label,
            "fluid": fluid, "lab_category": cat, "value": value, "valuenum": _num(valuenum),
            "valueuom": uom, "ref_range_lower": _num(lo), "ref_range_upper": _num(hi),
            "flag": flag, "priority": prio,
        } for t, itemid, label, fluid, cat, value, valuenum, uom, lo, hi, flag, prio in labs]

        rx = con.execute(f"""
            SELECT starttime, stoptime, drug, drug_type, dose_val_rx, dose_unit_rx, route
            FROM {_t(data_dir, 'prescriptions')}
            WHERE subject_id = ? AND hadm_id = ?
        """, [sid, hadm]).fetchall()
        rx_events = [{
            "time": _s(start), "category": "prescription", "drug": drug, "drug_type": dtype,
            "dose_val_rx": dose, "dose_unit_rx": unit, "route": route, "stoptime": _s(stop),
        } for start, stop, drug, dtype, dose, unit, route in rx]

        emars = con.execute(f"""
            SELECT charttime, medication, event_txt
            FROM {_t(data_dir, 'emar')} WHERE subject_id = ? AND hadm_id = ?
        """, [sid, hadm]).fetchall()
        emar_events = [{
            "time": _s(t), "category": "med_admin", "medication": med, "event_txt": txt,
        } for t, med, txt in emars]

        events = sorted(lab_events + rx_events + emar_events,
                        key=lambda e: (e["time"] is None, e["time"] or ""))

        admissions.append({
            "hadm_id": hadm,
            "admittime": _s(admittime),
            "dischtime": _s(dischtime),
            "admission_type": atype,
            "admission_location": aloc,
            "discharge_location": dloc,
            "hospital_expire_flag": expire,
            "age_at_admission": anchor_age + int(str(admittime)[:4]) - anchor_year,
            "diagnoses": [{"seq_num": s, "icd_code": c, "icd_version": v, "title": t}
                          for s, c, v, t in dx],
            "events": events,
        })

    return {
        "subject_id": sid, "gender": gender, "anchor_age": anchor_age,
        "dod": _s(dod),
        "n_admissions": len(admissions),
        "admissions": admissions,
    }


def check(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    n_pat = con.execute(f"SELECT count(*) FROM {_t(data_dir, 'patients')}").fetchone()[0]
    n_adm = con.execute(f"SELECT count(*) FROM {_t(data_dir, 'admissions')}").fetchone()[0]
    print(f"患者 {n_pat} 人,住院 {n_adm} 次")
    assert 50 <= n_pat <= 200, f"demo 规模异常: {n_pat}"

    patients = list_patients(con, data_dir)
    with_labs = with_meds = 0
    for sid, *_ in patients[:10]:
        tl = build_timeline(con, data_dir, sid)
        for adm in tl["admissions"]:
            times = [e["time"] for e in adm["events"] if e["time"]]
            assert times == sorted(times), f"{sid}/{adm['hadm_id']} 事件未按时间排序"
            cats = {e["category"] for e in adm["events"]}
            with_labs += "lab" in cats
            with_meds += bool(cats & {"prescription", "med_admin"})
    print(f"前 10 名患者抽样: {with_labs}/次住院有化验事件, {with_meds}/次住院有用药事件")
    assert with_labs > 0 and with_meds > 0, "抽样住院缺化验或用药事件"

    ratio = con.execute(f"""
        SELECT avg(CASE WHEN valuenum IS NOT NULL AND valueuom IS NOT NULL THEN 1.0 ELSE 0 END)
        FROM {_t(data_dir, 'labevents')} WHERE valuenum IS NOT NULL
    """).fetchone()[0]
    print(f"数值型化验带单位比例: {ratio:.1%}(F2 单位错误故障依赖此列)")
    print("CHECK OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC-IV demo 病程时间线加载器")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--list", action="store_true", help="列出全部患者")
    ap.add_argument("--patient-id", type=int, help="输出该患者的时间线 JSON")
    ap.add_argument("--out", type=Path, help="JSON 写入该文件(默认打印 stdout)")
    ap.add_argument("--check", action="store_true", help="运行自检断言")
    args = ap.parse_args()

    con = connect(args.data_dir)
    if args.check:
        check(con, args.data_dir)
    elif args.list:
        for sid, gender, age, n in list_patients(con, args.data_dir):
            print(f"{sid}  {gender}  anchor_age={age}  住院 {n} 次")
    elif args.patient_id:
        tl = build_timeline(con, args.data_dir, args.patient_id)
        text = json.dumps(tl, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            n_events = sum(len(a["events"]) for a in tl["admissions"])
            print(f"{args.out} 写入完成:{tl['n_admissions']} 次住院,{n_events} 个事件")
        else:
            print(text)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
