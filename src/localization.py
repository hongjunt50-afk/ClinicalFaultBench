"""中文化适配层:把 MIMIC-IV(美国 ICU 数据)的术语与单位对齐到国内报告习惯。

定位(2026-10-25 用户拍板):v1 数据源保持 MIMIC(国内无公开纵向结构化 EHR),
"适配国内"落在术语层——检验项目/药品的中文通用名 + 诊断 ICD 章节中文名 + 中美
单位制换算;数据源级泛化(真实国内 EHR)是 v2 路线,见 README「v2 路线图」。

合规:映射表的键是通用医学术语(英文标准名/INN 通用名),不含 MIMIC itemid 或
任何原始数据;--coverage 在用户本地读取自己的 MIMIC 字典做 join,repo 只随仓库
分发对照表本身,不触 DUA 再分发红线。

匹配约定:键统一归一化(lower + 去逗号 + 合并空白),因此 CSV 里 "Calcium, Total"
写作 "Calcium Total",且药名大小写变体(LORazepam/Lorazepam)天然归并。

用法(仓库根目录):
  python src/localization.py --coverage             # 实际数据上的术语覆盖率报告
  python src/localization.py --patient-id 10000032  # 输出带中文名的时间线 JSON
  python src/localization.py --patient-id 10000032 --out results/examples/t_zh.json
  python src/localization.py --check                # 自检:映射加载/换算系数/覆盖率下限
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import sys

import duckdb

REPO = Path(__file__).resolve().parents[1]
TERM_DIR = REPO / "configs" / "terminology_zh"
DEFAULT_DATA_DIR = REPO / "data" / "mimic-iv-demo-hf-mirror" / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline  # noqa: E402


def norm(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.replace(",", "").strip().lower())


def load_lab_map() -> dict[tuple[str, str], str]:
    with open(TERM_DIR / "lab_items_zh.csv", encoding="utf-8-sig") as f:
        return {(norm(r["label"]), norm(r["fluid"])): r["name_zh"].strip()
                for r in csv.DictReader(f) if r["label"].strip()}


def load_drug_map() -> dict[str, str]:
    with open(TERM_DIR / "drugs_zh.csv", encoding="utf-8-sig") as f:
        return {norm(r["drug"]): r["name_zh"].strip()
                for r in csv.DictReader(f) if r["drug"].strip()}


def load_icd_chapters() -> list[tuple[int, str, int, int, str]]:
    with open(TERM_DIR / "icd_chapters_zh.csv", encoding="utf-8-sig") as f:
        return [(int(r["version"]), r["letter"].strip().upper(),
                 int(r["num_from"]), int(r["num_to"]), r["name_zh"].strip())
                for r in csv.DictReader(f) if r["version"].strip()]


def load_conversions() -> dict[tuple[str, str], tuple[float, str]]:
    out = {}
    with open(TERM_DIR / "unit_conversions.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r["label"].strip():
                continue
            out[(norm(r["label"]), norm(r["uom_from"]))] = (
                float(r["factor"]), r["uom_to"].strip())
    return out


def load_loinc_ref() -> dict[str, dict]:
    """LOINC 码 → 中文名 + 参考区间(Synthea 数据缺参考区间,判分需要)。"""
    path = REPO / "configs" / "ref_ranges_loinc.csv"
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            code = r["loinc"].strip()
            if not code:
                continue
            out[code] = {
                "label_zh": r["label_zh"].strip(), "label_en": r["label_en"].strip(),
                "ref_low": float(r["ref_low"]) if r["ref_low"].strip() else None,
                "ref_high": float(r["ref_high"]) if r["ref_high"].strip() else None,
                "critical_low": float(r["critical_low"]) if r["critical_low"].strip() else None,
                "critical_high": float(r["critical_high"]) if r["critical_high"].strip() else None,
            }
    return out


def deviation(value: float | None, ref: dict | None) -> float | None:
    """把偏离参考区间的程度归一成"超出区间宽度的倍数",用于跨检验项比较异常程度。

    value 落在区间内 → 0;超出 → 超出的距离 / 区间宽度 (宽度为 0 或缺失时用 1 兜底)。
    """
    if value is None or not ref or ref.get("ref_low") is None or ref.get("ref_high") is None:
        return None
    low, high = ref["ref_low"], ref["ref_high"]
    width = (high - low) or 1.0
    if value < low:
        return (low - value) / width
    if value > high:
        return (value - high) / width
    return 0.0


def icd_category(code: str, version: int) -> tuple[str, int] | None:
    # 4019→("",401);42731→("",427);I2510→("I",25);Z87891→("Z",87);
    # ICD-9 V 码类目两位(V45),E 码类目三位(E93)
    code = code.strip().upper()
    if version == 9:
        if code[0].isalpha():
            digits = re.sub(r"\D", "", code[1:])
            return code[0], int(digits[:2] if code[0] == "V" else digits[:3] or 0)
        return "", int(code[:3])
    if version == 10:
        return code[0], int(re.sub(r"\D", "", code[1:])[:2] or 0)
    return None


def icd_chapter_zh(code: str, version: int,
                   chapters: list[tuple[int, str, int, int, str]]) -> str | None:
    version = int(version)  # parquet 出来的 icd_version 是字符串 '10'/'9'
    cat = icd_category(code, version)
    if cat is None:
        return None
    letter, num = cat
    for v, ltr, lo, hi, name in chapters:
        if v == version and ltr == letter and lo <= num <= hi:
            return name
    return None


def convert(value: float, label: str, uom: str,
            conversions: dict[tuple[str, str], tuple[float, str]]) -> tuple[float, str] | None:
    hit = conversions.get((norm(label), norm(uom)))
    return (value * hit[0], hit[1]) if hit else None


def annotate(timeline_obj: dict, lab_map: dict | None = None,
             drug_map: dict | None = None,
             chapters: list | None = None) -> dict:
    """给时间线补中文字段(纯函数,不改入参):lab→label_zh,rx→drug_zh,dx→chapter_zh。"""
    lab_map = lab_map if lab_map is not None else load_lab_map()
    drug_map = drug_map if drug_map is not None else load_drug_map()
    chapters = chapters if chapters is not None else load_icd_chapters()

    out = copy.deepcopy(timeline_obj)
    for adm in out.get("admissions", []):
        for d in adm.get("diagnoses", []):
            d["chapter_zh"] = icd_chapter_zh(d.get("icd_code", ""), int(d.get("icd_version", 0)),
                                             chapters)
        for e in adm.get("events", []):
            if e.get("category") == "lab":
                e["label_zh"] = lab_map.get((norm(e.get("label")), norm(e.get("fluid"))), "")
            elif e.get("category") == "prescription":
                e["drug_zh"] = drug_map.get(norm(e.get("drug")), "")
    return out


def coverage(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """事件加权覆盖率:实际数据里有多少比例的术语/诊断能对上中文映射;未覆盖 top 也列出。"""
    t = lambda x: timeline._t(data_dir, x)  # noqa: E731
    lab_map, drug_map, chapters = load_lab_map(), load_drug_map(), load_icd_chapters()

    print("== 检验项(按事件数加权) ==")
    rows = con.execute(f"""
        SELECT i.label, i.fluid, count(*) n FROM {t('labevents')} l
        JOIN {t('d_labitems')} i USING (itemid)
        GROUP BY 1, 2 ORDER BY n DESC
    """).fetchall()
    total = sum(r[2] for r in rows)
    hit = sum(r[2] for r in rows if (norm(r[0]), norm(r[1])) in lab_map)
    print(f"覆盖率 {hit}/{total} = {hit/total:.1%}(字典条目 {len(rows)} 个)")
    print("未覆盖 top15:")
    for lab, fluid, n in [r for r in rows if (norm(r[0]), norm(r[1])) not in lab_map][:15]:
        print(f"  {n:6d}  [{fluid}] {lab}")

    print("== 药品(按医嘱数加权) ==")
    rows = con.execute(f"SELECT drug, count(*) n FROM {t('prescriptions')} GROUP BY 1 ORDER BY n DESC").fetchall()
    total = sum(r[1] for r in rows)
    hit = sum(n for drug, n in rows if norm(drug) in drug_map)
    print(f"覆盖率 {hit}/{total} = {hit/total:.1%}(不同药名 {len(rows)} 个)")
    print("未覆盖 top15:")
    for drug, n in [r for r in rows if norm(r[0]) not in drug_map][:15]:
        print(f"  {n:6d}  {drug!r}")

    print("== 诊断章节(按诊断条数加权) ==")
    rows = con.execute(f"SELECT icd_code, icd_version, count(*) n FROM {t('diagnoses_icd')} GROUP BY 1, 2").fetchall()
    total = sum(r[2] for r in rows)
    hit = sum(n for code, ver, n in rows if icd_chapter_zh(code, ver, chapters))
    print(f"覆盖率 {hit}/{total} = {hit/total:.1%}")


def check(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    lab_map, drug_map = load_lab_map(), load_drug_map()
    chapters, conversions = load_icd_chapters(), load_conversions()
    assert len(lab_map) >= 70 and len(drug_map) >= 150 and len(chapters) >= 40, "映射表行数异常"

    # 映射抽查(写死这批是可靠标准知识;若改表导致失败,先人工复核对应条目)
    assert lab_map[("potassium", "blood")] == "血钾"
    assert lab_map[("creatinine", "blood")] == "血肌酐"
    assert drug_map["lorazepam"] == "劳拉西泮"
    assert drug_map["0.9% sodium chloride"].startswith("0.9%氯化钠")

    # 单位换算抽查:中美单位制差异的四个代表
    assert convert(180.0, "Glucose", "mg/dL", conversions) == (180.0 * 0.0555, "mmol/L")
    assert convert(1.0, "Creatinine", "mg/dL", conversions) == (88.4, "µmol/L")
    assert convert(9.5, "Hemoglobin", "g/dL", conversions) == (95.0, "g/L")
    assert convert(4.2, "Potassium", "mEq/L", conversions) == (4.2, "mmol/L")
    assert convert(7.0, "Hematocrit", "%", conversions) is None, "血细胞计数中美同制,不应有换算"

    # ICD 章节抽查(两版各三例,覆盖字母章/数字章/V码)
    assert icd_chapter_zh("I2510", 10, chapters) == "循环系统疾病"
    assert icd_chapter_zh("E785", 10, chapters) == "内分泌、营养和代谢疾病"
    assert icd_chapter_zh("Z794", 10, chapters) == "影响健康状态与保健机构接触的因素"
    assert icd_chapter_zh("4019", 9, chapters) == "循环系统疾病"
    assert icd_chapter_zh("42731", 9, chapters) == "循环系统疾病"
    assert icd_chapter_zh("V4501", 9, chapters) == "影响健康状态与保健机构接触的因素"
    assert icd_chapter_zh("E9331", 9, chapters) == "疾病和死亡的外因"

    # annotate 不改入参、字段正确附加
    src = {"admissions": [{"diagnoses": [{"icd_code": "I2510", "icd_version": "10"}],
                           "events": [
        {"category": "lab", "label": "Potassium", "fluid": "Blood"},
        {"category": "prescription", "drug": "LORazepam"},
    ]}]}
    snapshot = json.dumps(src, ensure_ascii=False)
    out = annotate(src, lab_map, drug_map, chapters)
    assert json.dumps(src, ensure_ascii=False) == snapshot, "annotate 修改了入参(应纯函数)"
    assert out["admissions"][0]["diagnoses"][0]["chapter_zh"] == "循环系统疾病"
    assert out["admissions"][0]["events"][0]["label_zh"] == "血钾"
    assert out["admissions"][0]["events"][1]["drug_zh"] == "劳拉西泮"

    # 实际数据覆盖率下限(高把握映射应吃掉绝大部分事件量)
    t = lambda x: timeline._t(data_dir, x)  # noqa: E731
    rows = con.execute(f"""
        SELECT i.label, i.fluid, count(*) n FROM {t('labevents')} l
        JOIN {t('d_labitems')} i USING (itemid) GROUP BY 1, 2
    """).fetchall()
    total = sum(r[2] for r in rows)
    hit = sum(r[2] for r in rows if (norm(r[0]), norm(r[1])) in lab_map)
    print(f"检验项覆盖率(事件加权): {hit/total:.1%}")
    assert hit / total >= 0.80, f"检验项覆盖率过低: {hit/total:.1%}(预期 ≥80%)"
    print("CHECK OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC-IV 术语中文化适配层")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--patient-id", type=int, help="输出该患者带中文名的时间线 JSON")
    ap.add_argument("--out", type=Path, help="JSON 写入该文件(默认打印 stdout)")
    ap.add_argument("--coverage", action="store_true", help="术语映射覆盖率报告")
    ap.add_argument("--check", action="store_true", help="运行自检断言")
    args = ap.parse_args()

    if args.check:
        check(timeline.connect(args.data_dir), args.data_dir)
    elif args.coverage:
        coverage(timeline.connect(args.data_dir), args.data_dir)
    elif args.patient_id:
        tl = timeline.build_timeline(timeline.connect(args.data_dir), args.data_dir, args.patient_id)
        text = json.dumps(annotate(tl), ensure_ascii=False, indent=2)
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
