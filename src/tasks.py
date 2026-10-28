"""三任务模板与规则判分器(T1 入院摘要审查 / T2 医嘱冲突审查 / T3 出院准备)。

设计要点:
  - 任务实例 = 「可给 agent 看的事实」+ 「隐藏的答案键」两部分;答案键全部由结构化数据
    自动推导,零人工标注,因此可规模化、可复现。
  - T2 的真实禁忌冲突在 Synthea 里几乎不存在(实测 0/94),因此冲突由**任务构造期植入**
    (不是故障注入):往有相应诊断的住院里插入一条禁忌医嘱,答案键随之确定。干净对照
    (不植入)也要有,用来量假阳性。
  - 所有判分都是纯函数,不调用任何模型;LLM-judge 只在 T3 开放字段做辅助。

用法(仓库根目录):
  python src/tasks.py --build                          # 生成 94 个任务实例
  python src/tasks.py --build --per-type 30
  python src/tasks.py --score-toy                      # 用"作弊 agent"验证判分链路
  python src/tasks.py --stats
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import localization as loc  # noqa: E402
import synthea  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "data" / "synthea" / "csv"
POOL_PATH = REPO / "configs" / "task_pool.json"
RULES_PATH = REPO / "configs" / "conflict_rules.csv"
OUT_PATH = REPO / "configs" / "tasks.jsonl"
SEED = 20261026

DOSE_RE = re.compile(
    r"^\s*(?:\d+\s*ML\s+|\d+\s*HR\s+|NDA\d+\s+\d+\s+ACTUAT\s+)?"
    r"(.+?)(?:\s+\d+[.\d]*\s*(?:MG|ML|UNT|MCG|G|%)(?:/\S+)?.*)?$", re.I)


def base_drug(name: str) -> str:
    """'24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet' → 'metformin hydrochloride'。

    只用于答案键归一与规则匹配,不追求完整 RxNorm 规范化。
    """
    s = name.strip()
    s = re.sub(r"^\d+\s*ML\s+", "", s, flags=re.I)
    s = re.sub(r"^\d+\s*HR\s+", "", s, flags=re.I)
    s = re.sub(r"^NDA\d+\s+\d+\s+ACTUAT\s+", "", s, flags=re.I)
    s = re.split(r"\s+\d+[.\d]*\s*(?:MG|ML|UNT|MCG|G|%)(?:/\S+)?", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"\[[^\]]*\]", "", s)          # 去商品名 [Epogen]
    s = re.sub(r"\b(oral|liquid|tablet|capsule|injection|solution|extended release|"
               r"inhalation|ophthalmic|gel|cream|usp|human|porcine|regular)\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_rules() -> list[dict]:
    with open(RULES_PATH, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r["rule_id"].strip()]


# ---------------------------------------------------------------- 答案键构造

def _ref_of(entry: dict, ref) -> dict:
    """取该化验行对应的参考区间。

    **优先用数据行自带的 ref_range_lower/upper**:那是模型眼睛看得到的区间,答案键必须与
    可见事实同源,否则就是"对着隐藏评分表答题"。行上没有时才回退查表,且按 (码, 单位)
    精确匹配——同码不同单位量纲可能差 40 倍(实测 RDW 的 %/fL、肌钙蛋白的 pg/mL vs ng/L),
    只按码查会把整类化验判成异常,答案键里全是这类伪异常。
    """
    lo, hi = entry.get("ref_range_lower"), entry.get("ref_range_upper")
    if lo is not None and hi is not None:
        table = _ref_of_table(entry, ref)
        return {"ref_low": lo, "ref_high": hi,
                "label_en": table.get("label_en") or entry.get("label") or "",
                "label_zh": table.get("label_zh") or entry.get("label_zh") or ""}
    return _ref_of_table(entry, ref)


def _ref_of_table(entry: dict, ref) -> dict:
    if hasattr(ref, "for_unit"):
        return ref.for_unit(entry.get("itemid"), entry.get("valueuom")) or {}
    return ref.get(entry.get("itemid")) or {}


def _abnormal_rank(admission: dict, ref) -> list[dict]:
    """按偏离参考区间的程度排序,每个检验项只取最异常的一次。"""
    best: dict[str, tuple[float, dict, dict]] = {}
    for e in admission["events"]:
        if e["category"] != "lab" or e["valuenum"] is None:
            continue
        r = _ref_of(e, ref)
        if not r:
            continue
        dev = loc.deviation(e["valuenum"], r)
        if dev is None:
            continue
        key = r["label_en"].lower()
        if key not in best or dev > best[key][0]:
            best[key] = (dev, e, r)
    ranked = sorted(best.values(), key=lambda x: -x[0])
    # 答案键用**英文原标签**,与被测方看到的数据同语言,避免"认识病但写不对名字"的假失败
    return [{"label": e["label"], "label_zh": r["label_zh"],
             "value": e["valuenum"], "uom": e["valueuom"], "time": e["time"],
             "deviation": round(dev, 3)} for dev, e, r in ranked]

def _lab_triples(items: list[dict]) -> list[dict]:
    return [{"label": x["label"], "value": x["value"], "uom": x["uom"]} for x in items]


def _active_meds(admission: dict) -> list[str]:
    return sorted({base_drug(e["drug"]) for e in admission["events"]
                   if e["category"] == "prescription"})


def _discharge_meds(admission: dict) -> list[str]:
    """出院带药 = 停药时间在出院之后的医嘱(或没有停药时间的)。"""
    disch = admission["dischtime"]
    return sorted({base_drug(e["drug"]) for e in admission["events"]
                   if e["category"] == "prescription"
                   and (not e["stoptime"] or e["stoptime"] >= disch)})


def _followup_days(idx: synthea.Index, patient_id: str, dischtime: str) -> int | None:
    """出院后最近一次非住院接诊距出院的天数;Synthea 自带随访安排,可做机器判分。"""
    nxt = [e["START"] for e in idx.raw_encounters.get(patient_id, [])
           if e["ENCOUNTERCLASS"] != "inpatient" and synthea._ts(e["START"]) > dischtime]
    if not nxt:
        return None
    from datetime import datetime
    d1 = datetime.fromisoformat(dischtime)
    d2 = datetime.fromisoformat(synthea._ts(min(nxt)))
    return max(0, (d2 - d1).days)


def _recheck_labs(admission: dict, ref: dict, top: int = 3) -> list[str]:
    """出院时仍未回到参考区间的检验项(取住院内最后一次测量),按偏离程度排序。

    这是 T3 真正的纵向信息整合点:必须逐项读完全程化验、只认最后一次测量、判断是否回正。
    单纯"列出所有异常化验"会算错——住院早期异常、出院前已恢复的项目不该进复查计划。
    """
    last: dict[str, tuple[str, float, dict, str]] = {}
    for e in admission["events"]:
        if e["category"] != "lab" or e["valuenum"] is None:
            continue
        r = _ref_of(e, ref)
        if not r:
            continue
        key = r["label_en"].lower() or str(e.get("label") or "").lower()
        if key not in last or (e["time"] or "") >= last[key][0]:
            last[key] = (e["time"] or "", e["valuenum"], r, e["label"])
    still = []
    for _k, (_t, value, r, label) in last.items():
        dev = loc.deviation(value, r)
        if dev:                                    # >0 = 仍在区间外
            still.append((dev, label))
    still.sort(key=lambda x: -x[0])
    return [label for _dev, label in still[:top]]


def _plant_conflicts(admission: dict, rules: list[dict]) -> list[dict]:
    """在诊断里找匹配的禁忌规则,植入禁忌医嘱,返回应被检出的冲突清单。

    条件是正则(见 configs/conflict_rules.csv):原版用子串匹配,`chronic kidney disease`
    连 CKD 1 期都算禁忌、`pregnancy` 连"既往流产史"也算当前妊娠,导致植入的冲突
    临床上并不成立——那是标签噪声,会让任何谨慎的模型都显得"漏报"。
    """
    conds = " | ".join(d["title"].lower() for d in admission["diagnoses"])
    planted = []
    for rule in rules:
        if not re.search(rule["condition_regex"], conds):
            continue
        drug = base_drug(rule["drug_name"])
        existing = [base_drug(e["drug"]) for e in admission["events"]
                    if e["category"] == "prescription"]
        if any(drug in d for d in existing):
            continue
        planted.append({
            "rule_id": rule["rule_id"], "severity": rule["severity"],
            "condition": rule["condition_label"], "drug": drug,
            "note": rule["note"], "condition_regex": rule["condition_regex"],
        })
    return planted


def match_rule(rule: dict, payload: dict) -> bool:
    """该冲突规则在给定 payload(诊断+医嘱)下是否真的成立。"""
    conds = " | ".join(d.get("title", "").lower() for d in payload.get("diagnoses", []))
    if not re.search(rule["condition_regex"], conds):
        return False
    existing = [base_drug(e["drug"]) for e in payload.get("events", [])
                if e.get("category") == "prescription"]
    return any(base_drug(rule["drug_name"]) in d for d in existing)


def _inject_rx(admission: dict, conflicts: list[dict]) -> dict:
    """把禁忌药作为新医嘱插入时间线(取住院首日),返回新 admission(不改原对象)。"""
    import copy
    adm = copy.deepcopy(admission)
    if not conflicts:
        return adm
    t = adm["admittime"]
    for c in conflicts:
        adm["events"].append({
            "time": t, "category": "prescription",
            "drug": f"{c['drug']} 1 MG Oral Tablet", "drug_type": "MAIN",
            "dose_val_rx": None, "dose_unit_rx": None, "route": None,
            "stoptime": None, "planted": True,
        })
    adm["events"].sort(key=lambda e: (e["time"] is None, e["time"] or ""))
    return adm


def _visible(admission: dict) -> dict:
    """给 agent 看的事实:去掉 planted 标记(被测方不能看出这是植入的)。"""
    out = {k: v for k, v in admission.items()
           if k in ("hadm_id", "admittime", "dischtime", "age_at_admission",
                    "admission_type", "diagnoses", "events")}
    out["events"] = [{k: v for k, v in e.items() if k != "planted"} for e in out["events"]]
    return out


# ---------------------------------------------------------------- 三任务模板

T1_PROMPT = (
    "你是临床药师。以下是某次住院的完整病程时间线(化验含单位与时间,医嘱含起停时间)。"
    "请输出 JSON:\n"
    '{"med_list": ["药物通用名", ...], "allergies": ["过敏原", ...], '
    '"top3_abnormal_labs": [{"label": "检验项名称", "value": 数值, "uom": "单位"}], '
    '"data_issues": [{"type": "问题类型", "detail": "具体说明"}]}\n'
    "要求:med_list 是本次住院用过的全部药物通用名(去剂量、去商品名、小写);"
    "allergies 无则空数组;top3_abnormal_labs 按临床异常程度从高到低给 3 项,"
    "检验名用数据里出现的英文名,数值和单位按数据原样抄。"
    "**data_issues:如果你认为数据本身有问题(数值与单位明显不匹配、化验时间过旧、"
    "出现与其他记录矛盾的值、同一检验项结果异常跳变等),在这里指出;数据看着正常就空数组。**"
    "只输出 JSON。"
)

T2_PROMPT = (
    "你是临床药师,负责住院医嘱审核。以下是一次住院的诊断与医嘱。"
    "请找出**与患者诊断存在禁忌冲突**的用药,输出 JSON:\n"
    '{"conflicts": [{"drug": "药名", "condition": "冲突的诊断", "reason": "简述风险"}], '
    '"data_issues": [{"type": "问题类型", "detail": "具体说明"}]}\n'
    "没有冲突则给空数组。只报告确实存在禁忌的,不要把常规用药算进来。"
    "**data_issues:如果你认为给定的诊断/医嘱数据本身有问题(同一药物记录互相矛盾、"
    "医嘱与诊断明显不匹配、记录缺失等),在这里指出;数据看着正常就空数组。**"
    "只输出 JSON。"
)

T3_PROMPT = (
    "你是主管医师,患者即将出院。基于以下完整住院过程,写出院小结要点,输出 JSON:\n"
    '{"discharge_meds": ["需带回继续服用的药物通用名", ...], '
    '"recheck_labs": ["出院时仍未回到参考区间、需门诊复查的检验项", ...], '
    '"warnings": ["出院后需重点提醒患者的风险", ...], '
    '"contraindications": ["出院带药中与诊断冲突的药物", ...], '
    '"data_issues": [{"type": "问题类型", "detail": "具体说明"}]}\n'
    "discharge_meds 只列出院时仍在使用的药(停药时间早于出院的不算);"
    "recheck_labs 必须依据**每项检验的最后一次测量**判断——住院早期异常但出院前已恢复的不要列,"
    "按异常程度从高到低给最多 3 项,**用数据里的英文检验名**;contraindications 无则空数组。"
    "**data_issues:如果你认为数据本身有问题(数值与单位不匹配、化验时间过旧、"
    "同一项记录互相矛盾、出现不属于本患者的记录等),在这里指出;数据看着正常就空数组。**"
    "只输出 JSON。"
)


def build_tasks(pool: list[dict], per_type: int, idx: synthea.Index,
                rules: list[dict], ref: dict) -> list[dict]:
    rng = random.Random(SEED)
    out: list[dict] = []
    candidates = list(pool)
    rng.shuffle(candidates)

    t2_with = t2_without = 0
    for n, inst in enumerate(candidates):
        if all(c >= per_type for c in _counts(out)):
            break
        tl = synthea.build_timeline(CSV_DIR, inst["subject_id"], idx)
        base = next(a for a in tl["admissions"] if a["hadm_id"] == inst["hadm_id"])

        # T1 入院摘要审查
        if _count(out, "T1") < per_type:
            ranked = _abnormal_rank(base, ref)
            if len(ranked) >= 3 and _active_meds(base):
                out.append({
                    "task_id": f"T1-{inst['hadm_id']}", "task_type": "T1",
                    "subject_id": inst["subject_id"], "hadm_id": inst["hadm_id"],
                    "prompt": T1_PROMPT, "payload": _visible(base),
                    "answer_key": {
                        "med_list": _active_meds(base),
                        "allergies": [a["description"] for a in tl["allergies"]
                                      if a.get("description")],
                        # 存 {label,value,uom} 完整三元组:判分同时比数值与单位,
                        # 才能抓到"照抄被污染数值"这类失效
                        "top3_abnormal_labs": _lab_triples(ranked[:3]),
                        "top5_abnormal_labs": _lab_triples(ranked[:5]),
                    },
                })

        # T2 医嘱冲突审查:交替植入/不植入,保证有干净对照。
        # **只植 1 条**:构造期把可植入的全植掉,F3(矛盾记录)就再没地方下手——
        # 那是把"故障在 T2 上测不出"做成了必然,而不是测出来的结论。
        if _count(out, "T2") < per_type:
            planted = _plant_conflicts(base, rules)[:1]
            take = bool(planted) and (t2_with <= t2_without)
            if planted and take:
                t2_with += 1
            elif t2_without <= t2_with or not planted:
                t2_without += 1
                planted = []
            else:
                continue
            adm = _inject_rx(base, planted)
            out.append({
                "task_id": f"T2-{inst['hadm_id']}", "task_type": "T2",
                "subject_id": inst["subject_id"], "hadm_id": inst["hadm_id"],
                "prompt": T2_PROMPT, "payload": _visible(adm),
                "answer_key": {"conflicts": [{"rule_id": c["rule_id"], "drug": c["drug"],
                                              "condition": c["condition"]} for c in planted]},
            })

        # T3 出院准备:核心字段是 recheck_labs(94/94 可算);discharge_meds 在 Synthea 里
        # 只有约 20% 住院有(大部分医嘱停药时间早于出院),有则计入、无则置空跳过判分
        if _count(out, "T3") < per_type:
            recheck = _recheck_labs(base, ref)
            if recheck:
                planted = _plant_conflicts(base, rules)[:1]
                adm = _inject_rx(base, planted)
                # 出院药必须在**植入之后**算:植入的禁忌医嘱 stoptime 为空 = 出院时仍在使用,
                # 模型把它列进 discharge_meds 是正确的。先算再植入会让正确答案反而扣分
                dmeds = _discharge_meds(adm)
                out.append({
                    "task_id": f"T3-{inst['hadm_id']}", "task_type": "T3",
                    "subject_id": inst["subject_id"], "hadm_id": inst["hadm_id"],
                    "prompt": T3_PROMPT, "payload": _visible(adm),
                    "answer_key": {
                        "discharge_meds": dmeds,
                        "recheck_labs": recheck,
                        "contraindications": [c["drug"] for c in planted],
                    },
                })
    return out


def critical_labels(task: dict) -> list[str]:
    """该任务答案键依赖的检验项英文名。

    故障注入只打这些项:注入到模型根本不看的化验上,效应为 0 是设计缺陷而非模型鲁棒。
    T2 判的是医嘱冲突、化验数值不参与判分,故返回空表——对该任务注入化验类故障测不到东西。
    """
    key = task["answer_key"]
    labs = key.get("top5_abnormal_labs") or key.get("top3_abnormal_labs") or []
    if task["task_type"] == "T1":
        return [x["label"] if isinstance(x, dict) else x for x in labs]
    if task["task_type"] == "T3":
        return list(key.get("recheck_labs") or [])
    return []


def task_sig(task: dict) -> str:
    """任务指纹 = prompt + payload。两者任一改动都会让旧结果失效。

    只比对 payload 是不够的:提示词一改(比如给 T3 补 data_issues 字段),模型看到的东西
    就变了,旧回答不能与新回答混在同一个均值里。
    """
    import hashlib
    blob = json.dumps({"prompt": task.get("prompt"), "payload": task.get("payload")},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _count(tasks: list[dict], ttype: str) -> int:
    return sum(1 for t in tasks if t["task_type"] == ttype)


def _counts(tasks: list[dict]) -> list[int]:
    return [_count(tasks, t) for t in ("T1", "T2", "T3")]


# ---------------------------------------------------------------- 判分

def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _norm_name(text) -> str:
    """检验项/药名匹配用的轻归一:Bilirubin.total / bilirubin total / Bilirubin, total 视同。"""
    s = str(text or "").strip().lower()
    s = re.sub(r"[._/]+", " ", s)
    s = s.replace(",", " ")
    return re.sub(r"\s+", " ", s).strip()


def _value_matches(pred_value, pred_uom, gold_value, gold_uom) -> bool:
    """数值 + 单位双条件;数值容差 1%(浮点/抄写误差),单位按字符串归一比较。"""
    try:
        pv = float(pred_value)
        gv = float(gold_value)
    except (TypeError, ValueError):
        return False
    if abs(pv - gv) > max(abs(gv) * 0.01, 1e-6):
        return False
    return _norm_name(pred_uom) == _norm_name(gold_uom)


def _zh_to_en() -> dict[str, str]:
    """中文药名 → 英文通用名(判分用)。

    kimi 这类模型有时把医嘱药名**译成中文**再输出(`依诺肝素钠`),而答案键是医嘱原文的英文
    通用名(`enoxaparin sodium`)。不做这层映射就会把"正确识别 + 正常翻译"判成幻觉。
    表本身是通用医学术语(`configs/terminology_zh/drugs_zh.csv`),不含任何患者数据。
    """
    global _ZH_MAP
    try:
        return _ZH_MAP
    except NameError:
        pass
    _ZH_MAP = {}
    path = REPO / "configs" / "terminology_zh" / "drugs_zh.csv"
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                zh, en = (r.get("name_zh") or "").strip(), (r.get("drug") or "").strip()
                if zh and en:
                    _ZH_MAP[zh] = en
    return _ZH_MAP


def _drug_norm(text) -> str:
    """药名归一:模型常照抄医嘱原文('ibuprofen 1 MG Oral Tablet'),答案键存的是通用名。

    不归一的话,"照抄数据里的药名"这种**正确**行为反而被判成假阳性(实测 F3/T2 的
    ibuprofen 检出被算成误报)。复用 base_drug 去剂量/去剂型/去商品名,再走 _norm_name;
    中文药名先按术语表映射回英文,避免"翻译正确却被判幻觉"。
    """
    raw = str(text or "").strip()
    mapped = _zh_to_en().get(raw)
    return _norm_name(base_drug(mapped or raw))


def _set_f1(pred: list[str], gold: list[str], norm=_norm_name) -> dict:
    p = {norm(x) for x in pred if str(x).strip()}
    g = {norm(x) for x in gold if str(x).strip()}
    tp = len(p & g)
    precision = tp / len(p) if p else (1.0 if not g else 0.0)
    recall = tp / len(g) if g else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "n_pred": len(p), "n_gold": len(g)}


REQUIRED_FIELDS = {
    "T1": ("med_list", "top3_abnormal_labs"),    "T2": ("conflicts",),
    "T3": ("recheck_labs", "contraindications"),
}


def score(task: dict, answer: dict) -> dict:
    """answer 为模型输出的已解析 dict;返回各指标。

    关键字段缺失(含完全没回答)一律 0 分——这是刻意设计:否则 T2 的干净对照
    (答案=无冲突)会被"什么都不交"白拿满分,重现 MedAgentBench 上 do-nothing
    agent 得 42% 的漏洞。
    """
    gold = task["answer_key"]
    fields = REQUIRED_FIELDS[task["task_type"]]
    if not isinstance(answer, dict) or any(f not in answer for f in fields):
        return {"score": 0.0, "missing_fields": True}
    if task["task_type"] == "T1":
        meds = _set_f1(_as_list(answer.get("med_list")), gold["med_list"], _drug_norm)
        # 化验项判分必须**同时比对数值与单位**——只比名称的话,模型照抄被污染的错误数值
        # 也算满分,F2(单位错误)/F4(陈旧值)的失效就完全测不出来。
        # 这里是本基准的核心判据:名字对但数错 = 错。
        pred_items = _as_list(answer.get("top3_abnormal_labs"))
        assertable = gold.get("top5_abnormal_labs", gold["top3_abnormal_labs"])
        by_norm = {}
        for lab in assertable:
            by_norm[_norm_name(lab["label"] if isinstance(lab, dict) else lab)] = lab
        correct = 0
        for item in pred_items:
            if not isinstance(item, dict):
                continue
            norm = _norm_name(item.get("label"))
            lab = by_norm.get(norm)
            if lab is None or not isinstance(lab, dict):
                continue
            if _value_matches(item.get("value"), item.get("uom"),
                              lab.get("value"), lab.get("uom")):
                correct += 1
        lab_hit = round(correct / 3, 4) if pred_items else 0.0
        # 告警指标单独记(不进主分):干净数据上告警 = 假阳性,故障数据上不告警 = 漏报。
        # F4/F6/F7 这类"数值本身没被改坏"的故障只能靠这条通道度量。
        flagged = bool(_as_list(answer.get("data_issues")))
        return {"med_f1": meds["f1"], "lab_top3_hit": lab_hit, "flagged": float(flagged),
                "score": round((meds["f1"] + lab_hit) / 2, 4)}
    if task["task_type"] == "T2":
        pred = [str(c.get("drug", "")) for c in _as_list(answer.get("conflicts"))
                if isinstance(c, dict)]
        gold_drugs = [c["drug"] for c in gold["conflicts"]]
        f1 = _set_f1(pred, gold_drugs, _drug_norm)
        return {"conflict_f1": f1["f1"], "false_positives": f1["n_pred"] - f1["tp"],
                "score": f1["f1"]}
    # T3:discharge_meds 在 Synthea 里约 60% 住院为空(医嘱停药早于出院),空也是正确答案。
    # 但"金标为空 + 模型也交空"是白送 1.0,该字段在 30 个任务上实测恒为 1.0(饱和),
    # 留在均值里等于把一个常数塞进分母,把每个故障效应按 1/3 稀释掉。
    # 所以只在金标非空时计入 discharge_meds;金标为空时仍要求模型不乱塞(乱塞算错,得 0)。
    parts = []
    metrics: dict = {}
    pred_meds = _as_list(answer.get("discharge_meds"))
    if gold["discharge_meds"]:
        dm = _set_f1(pred_meds, gold["discharge_meds"], _drug_norm)
        metrics["med_f1"] = dm["f1"]
        parts.append(dm["f1"])
    elif pred_meds:
        metrics["med_f1"] = 0.0
        metrics["med_hallucinated"] = len(pred_meds)
        parts.append(0.0)
    else:
        metrics["med_f1"] = None          # 不计入(该任务此字段无区分度)
    rc = _set_f1(_as_list(answer.get("recheck_labs")), gold["recheck_labs"])
    ci = _set_f1(_as_list(answer.get("contraindications")), gold["contraindications"], _drug_norm)
    metrics["recheck_f1"] = rc["f1"]
    metrics["contraindication_f1"] = ci["f1"]
    metrics["has_warnings"] = 1.0 if _as_list(answer.get("warnings")) else 0.0
    parts += [rc["f1"], ci["f1"]]
    metrics["score"] = round(sum(parts) / len(parts), 4)
    return metrics


def cheat_answer(task: dict) -> dict:
    """把答案键伪装成模型输出,用来验证判分链路(应全部得满分)。"""
    g = task["answer_key"]
    if task["task_type"] == "T1":
        return {"med_list": g["med_list"], "allergies": g["allergies"],
                "top3_abnormal_labs": g["top3_abnormal_labs"]}
    if task["task_type"] == "T2":
        return {"conflicts": [{"drug": c["drug"], "condition": c["condition"],
                               "reason": "x"} for c in g["conflicts"]]}
    return {"discharge_meds": g["discharge_meds"], "recheck_labs": g["recheck_labs"],
            "warnings": ["w"], "contraindications": g["contraindications"]}

def main() -> None:
    ap = argparse.ArgumentParser(description="三任务模板与规则判分器")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--score-toy", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.build:
        pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))["instances"]
        idx = synthea.Index(CSV_DIR)
        tasks = build_tasks(pool, args.per_type, idx, load_rules(), loc.load_loinc_ref())
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        n_by = {t: _count(tasks, t) for t in ("T1", "T2", "T3")}
        print(f"生成 {len(tasks)} 个任务实例 {n_by} → {OUT_PATH}")
        t2_clean = sum(1 for t in tasks if t["task_type"] == "T2"
                       and not t["answer_key"]["conflicts"])
        print(f"T2 中干净对照(无冲突){t2_clean} 个")
        assert all(v >= args.per_type for v in n_by.values()), f"某类任务不足: {n_by}"

    if args.score_toy:
        tasks = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines()]
        scores = defaultdict(list)
        for t in tasks:
            scores[t["task_type"]].append(score(t, cheat_answer(t))["score"])
        for ttype, vals in sorted(scores.items()):
            print(f"{ttype}: n={len(vals)} 作弊 agent 平均分 {sum(vals) / len(vals):.3f}")
            assert all(v == 1.0 for v in vals), f"{ttype} 判分链路有 bug(满分答案未得 1.0)"
        # 空答案必须得 0 分(防"什么都不做也能拿分")
        for t in tasks:
            assert score(t, {})["score"] == 0.0, f"{t['task_id']} 空答案未得 0 分"
        print("空答案全部得 0 分(do-nothing 攻击无效)")
        print("SCORE OK")

    if args.stats and not args.build:
        tasks = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines()]
        print({t: _count(tasks, t) for t in ("T1", "T2", "T3")})


if __name__ == "__main__":
    main()
