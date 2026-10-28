"""FaultProxy:临床语义故障注入(七类 × 两档严重度)。

设计原则:
  - **纯函数**:`inject(payload, spec) -> (corrupted_payload, record)`。同一 (payload, spec)
    永远得到同一结果(用 payload 内部 id 派生随机数,不依赖全局 random 状态),可复现、可单测。
  - **对被测方透明**:注入后的 payload 与干净 payload 结构完全一致,schema 不变,
    被测 agent 无法从字段形状判断数据有问题。
  - **注入的是"看起来合理但错"的临床语义故障**,不是随机噪声——F2 是真单位换算错位,
    F4 是真陈旧值,F3 是真禁忌组合。

七类故障:
  F1 关键化验置空      | data   | 该测的检验缺失(NaN/None)
  F2 单位错误          | data   | mg/dL ↔ mmol/L 换算错位(数值与单位不匹配)
  F3 矛盾记录          | data   | 诊断存在禁忌但医嘱照开(植入禁忌药)
  F4 陈旧数据          | data   | 关键化验时间戳被前移到数天前(用旧值做当前决策)
  F5 工具截断          | tool   | 化验列表被截断到前 N 条(时间靠后的关键项消失)
  F6 跨患者污染        | tool   | 混入另一位患者的化验行
  F7 时间线乱序        | time   | 事件时间戳小范围漂移 + 顺序打乱

用法:
  python src/faults.py --selftest          # 七类 × 两档 全部单测
  python src/faults.py --demo T1-xxxx      # 对单个任务演示注入效果
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"

# 单位换算:把"数值换成另一种单位的数、单位标签却保持不变"制造真实的数据集成错误。
# 换算因子必须**真的改变数值**:肌酐原来写的 1.0(mg/dL 与 mg/dL 之间"换算")等于什么都没做,
# 记录里却记成 applied,于是 F2 在肌酐为主的 T1 上测到的是一堆无声无息的样本,
# 效应被稀释到 0.006。mg/dL → µmol/L 是 ×88.4。
UNIT_SWAP = {
    ("mg/dL", "mg/dL→mmol/L"): [("Glucose", 0.0555), ("Creatinine", 88.4),
                                ("Urea nitrogen", 0.357), ("Calcium", 0.2495)],
}

# 关键化验:临床上"看不见就会做错决定"的那些
CRITICAL_LABS = ("Potassium", "Sodium", "Creatinine", "Glucose", "Hemoglobin",
                 "Leukocytes", "Platelets", "Glomerular filtration rate")

SEVERITIES = ("mild", "severe")


def _rng(*parts) -> float:
    """由内容派生 [0,1) 的稳定伪随机数(不依赖全局状态,保证可复现)。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def _lab_events(payload: dict) -> list[dict]:
    return [e for e in payload.get("events", []) if e.get("category") == "lab"]


def _norm_label(text) -> str:
    """检验项名归一(与 tasks._norm_name 同口径,避免两处判字面差)。"""
    import re
    s = str(text or "").strip().lower()
    s = re.sub(r"[._/]+", " ", s).replace(",", " ")
    return re.sub(r"\s+", " ", s).strip()


def donor_pool(task: dict, all_tasks: list[dict]) -> dict[str, list[dict]]:
    """为 F6 备料:按"检验项名 → 另一位患者的同名测量行"建表。

    排除同一患者(同一患者的另一住院不算跨患者),也排除同一条 answered 化验本身就有的重复。
    实测 60 个化验类任务(30 T1 + 30 T3)全部能找到同名供体,故不会因缺料而漏跑。
    """
    pid = task.get("subject_id")
    pool: dict[str, list[dict]] = {}
    for other in all_tasks:
        if other is task or other.get("subject_id") == pid:
            continue
        for e in _lab_events(other["payload"]):
            if e.get("valuenum") is None or not e.get("time") or "planted" in e:
                continue
            pool.setdefault(_norm_label(e.get("label")), []).append(e)
    return pool


def _key_labels(critical: list[str] | None) -> set[str]:
    """任务答案键依赖的检验项。故障必须打在这些项上才算"与任务相关"——
    否则注入到任务根本不看的化验上,效应为 0 是设计问题而非模型鲁棒。"""
    return {str(x).lower() for x in (critical or [])}


def _pick_targets(labs: list[dict], critical: list[str] | None) -> list[dict]:
    """优先挑答案键依赖的检验项;没有则退回全量关键化验。"""
    keys = _key_labels(critical)
    if keys:
        hit = [e for e in labs if str(e.get("label", "")).lower() in keys]
        if hit:
            return hit
    return [e for e in labs if e.get("label") in CRITICAL_LABS]


def _shift_time(ts: str, days: float) -> str:
    from datetime import datetime, timedelta
    dt = datetime.fromisoformat(ts)
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 各类故障

def f1_drop_critical(payload: dict, severity: str, seed: str,
                     critical: list[str] | None = None) -> tuple[dict, dict]:
    """F1 关键化验置空:数值与单位一起抹掉,行还在(看起来像"没测出结果")。"""
    p = copy.deepcopy(payload)
    labs = _pick_targets(_lab_events(p), critical)
    if not labs:
        return p, {"applied": False, "reason": "无关键化验"}
    labs.sort(key=lambda e: (e.get("time") or ""))
    n = 1 if severity == "mild" else min(len(labs), max(3, len(labs) // 2))
    # mild 抹最近一次(最容易影响当前判断);severe 抹掉一大片
    targets = labs[-n:] if severity == "mild" else labs[:n]
    for e in targets:
        e["valuenum"] = None
        e["value"] = None
        e["valueuom"] = None
    return p, {"applied": True, "n_affected": len(targets),
               "labels": sorted({e["label"] for e in targets})}


def f2_unit_error(payload: dict, severity: str, seed: str,
                  critical: list[str] | None = None) -> tuple[dict, dict]:
    """F2 单位错误:数值按另一单位制换算,但单位标签保持原样。

    这是真实的数据集成错误(mg/dL 值被当 mmol/L 用),不是随机扰动:
    血糖 180 mg/dL 被写成 10 mg/dL,看上去"低血糖",会导致完全错误的处置。
    """
    p = copy.deepcopy(payload)
    candidates = []
    keys = _key_labels(critical)
    for e in _lab_events(p):
        if e.get("valuenum") is None or not e.get("valueuom"):
            continue
        if keys and str(e.get("label", "")).lower() not in keys:
            continue        # 只打答案键依赖的项,否则模型不看它也照样对
        for _label, factor in UNIT_SWAP[("mg/dL", "mg/dL→mmol/L")]:
            if e["label"].startswith(_label) and abs(factor - 1.0) > 1e-9:
                candidates.append((e, factor))
                break
    if not candidates:
        return p, {"applied": False, "reason": "无可换算检验"}

    n = 1 if severity == "mild" else len(candidates)
    hit = []
    for i, (e, factor) in enumerate(candidates[:n]):
        old = e["valuenum"]
        e["valuenum"] = round(old * factor, 3)
        e["value"] = str(e["valuenum"])
        hit.append({"label": e["label"], "from": old, "to": e["valuenum"],
                    "uom_kept": e["valueuom"]})
    # severe 再随机挑一条把单位标签改成另一个制式(双重错位)
    if severity == "severe" and candidates:
        e0 = candidates[0][0]
        e0["valuenum"] = round(candidates[0][0]["valuenum"] / candidates[0][1], 3)
        e0["value"] = str(e0["valuenum"])
        e0["valueuom"] = "mmol/L" if e0["valueuom"] == "mg/dL" else "mg/dL"
        hit.append({"label": e0["label"], "unit_flipped_to": e0["valueuom"]})
    return p, {"applied": True, "n_affected": len(hit), "changes": hit}


def f3_contradiction(payload: dict, severity: str, seed: str) -> tuple[dict, dict]:
    """F3 矛盾记录:诊断显示禁忌,医嘱里却出现该药(植入一条禁忌医嘱)。

    mild = 植入 1 条;severe = 植入全部可植入的禁忌药。
    规则用正则匹配(见 conflict_rules.csv):子串匹配会把 CKD 1 期、既往孕史也算成禁忌,
    植入的"矛盾"临床上不成立,任何谨慎的模型都会被判成漏报。
    """
    import tasks as task_lib
    p = copy.deepcopy(payload)
    rules = task_lib.load_rules()
    conds = " | ".join(d.get("title", "").lower() for d in p.get("diagnoses", []))
    existing = [task_lib.base_drug(e["drug"]) for e in p.get("events", [])
                if e.get("category") == "prescription"]
    plantable = [r for r in rules
                 if re.search(r["condition_regex"], conds)
                 and not any(task_lib.base_drug(r["drug_name"]) in d for d in existing)]
    if not plantable:
        return p, {"applied": False, "reason": "该住院无匹配禁忌规则"}
    n = 1 if severity == "mild" else min(len(plantable), 3)
    planted = plantable[:n]
    t = p.get("admittime") or "2000-01-01 00:00:00"
    for c in planted:
        p["events"].append({
            "time": t, "category": "prescription",
            "drug": f"{c['drug_name']} 1 MG Oral Tablet", "drug_type": "MAIN",
            "dose_val_rx": None, "dose_unit_rx": None, "route": None,
            "stoptime": None,
        })
    p["events"].sort(key=lambda e: (e["time"] is None, e["time"] or ""))
    return p, {"applied": True, "n_affected": len(planted),
               "rules": [{"rule_id": c["rule_id"], "drug": task_lib.base_drug(c["drug_name"]),
                          "condition": c["condition_label"]} for c in planted]}


def f4_stale_data(payload: dict, severity: str, seed: str,
                  critical: list[str] | None = None) -> tuple[dict, dict]:
    """F4 陈旧数据:关键化验的时间戳被前移(mild 2 天 / severe 7 天),数值保持当时的值。

    效果:agent 看到的是"几天前的化验",却要用来做当前的临床判断。
    """
    p = copy.deepcopy(payload)
    labs = [e for e in _pick_targets(_lab_events(p), critical) if e.get("time")]
    if not labs:
        return p, {"applied": False, "reason": "无关键化验"}
    labs.sort(key=lambda e: e["time"])
    days = 2 if severity == "mild" else 7
    n = 1 if severity == "mild" else min(len(labs), max(2, len(labs) // 3))
    targets = labs[-n:]
    for e in targets:
        e["time"] = _shift_time(e["time"], -days)
    p["events"].sort(key=lambda e: (e["time"] is None, e["time"] or ""))
    return p, {"applied": True, "n_affected": len(targets), "shift_days": -days,
               "labels": sorted({e["label"] for e in targets})}


def f5_tool_truncation(payload: dict, severity: str, seed: str,
                       critical: list[str] | None = None) -> tuple[dict, dict]:
    """F5 工具截断:化验接口只返回前 N 条(时间靠后的、往往最关键的被切掉)。

    截断必须**切到答案键依赖的检验项**,否则模型拿到的信息完好,效应自然为 0。
    实现:按时间排序后保留前面一段,并确保目标项落在被切掉的尾部。
    """
    p = copy.deepcopy(payload)
    labs = sorted([e for e in _lab_events(p) if e.get("time")], key=lambda e: e["time"])
    if len(labs) <= 3:
        return p, {"applied": False, "reason": "化验太少"}
    keys = _key_labels(critical)
    if keys:
        # 目标项挪到最尾部,保证被截断掉
        target = [e for e in labs if str(e.get("label", "")).lower() in keys]
        rest = [e for e in labs if str(e.get("label", "")).lower() not in keys]
        labs = rest + target
    keep = max(3, int(len(labs) * (0.7 if severity == "mild" else 0.3)))
    kept = {id(e) for e in labs[:keep]}
    dropped = [e for e in labs[keep:]]
    p["events"] = [e for e in p["events"]
                   if e.get("category") != "lab" or e.get("time") is None or id(e) in kept]
    return p, {"applied": True, "n_affected": len(dropped), "kept": keep,
               "dropped_tail_labels": sorted({e["label"] for e in dropped})[:8]}


def f6_cross_patient(payload: dict, severity: str, seed: str,
                     donor_labs: dict | list | None = None,
                     critical: list[str] | None = None) -> tuple[dict, dict]:
    """F6 跨患者污染:混入另一位患者对**同一检验项**的测量行(标签真实、数值真实,但属于别人)。

    2026-10-27 修:早期版本从固定供体取前 5 行,标签与任务无关,等于往 payload 里撒噪声——
    在 T2(判医嘱冲突、不看化验数值)上效应为纯随机,T2 的方差就是 F6 严重度曲线反向的全部来源。
    现在与 F1/F2/F4/F5 一致按答案键定向:只打该任务判分依赖的检验项,且供体必须同名同量纲、
    数值与本患者不同——否则要么模型根本看不到,要么两条数值一致而无声。
    """
    p = copy.deepcopy(payload)
    keys = [str(k) for k in (critical or []) if str(k).strip()]
    if not keys:
        return p, {"applied": False, "reason": "该任务不依赖化验数值(T2 等),F6 不适用"}
    own = [e for e in _lab_events(p) if e.get("time")]
    if not own:
        return p, {"applied": False, "reason": "本住院无化验"}

    donors: dict[str, list[dict]] = {}
    if isinstance(donor_labs, dict):
        donors = {k: list(v) for k, v in donor_labs.items() if v}
    elif donor_labs:
        for row in donor_labs:
            donors.setdefault(_norm_label(row.get("label")), []).append(row)
    if not donors:
        return p, {"applied": False, "reason": "无供体患者数据"}

    want = {_norm_label(k) for k in keys}
    targets = []
    seen: set[str] = set()
    for e in sorted(own, key=lambda x: (x.get("time") or ""), reverse=True):
        lbl = _norm_label(e.get("label"))
        if lbl not in want or lbl in seen or e.get("valuenum") is None:
            continue
        pool = [r for r in donors.get(lbl, [])
                if r.get("valuenum") is not None
                and abs(float(r["valuenum"]) - float(e["valuenum"])) > 1e-9]
        if not pool:
            continue
        seen.add(lbl)
        targets.append((e, pool))
        if len(targets) >= 5:
            break
    if not targets:
        return p, {"applied": False, "reason": "答案键检验项找不到同名且数值不同的供体行"}

    # mild 只污染最靠近出院的那一项(当前决策最依赖它);severe 覆盖全部可污染项
    n = 1 if severity == "mild" else len(targets)
    t = max((e.get("time") or "") for e in own) or p.get("admittime")
    injected = []
    for own_row, pool in targets[:n]:
        j = int(_rng(seed, "f6", own_row.get("label"), own_row.get("time")) * len(pool)) % len(pool)
        row = copy.deepcopy(pool[j])
        row.pop("planted", None)
        row["time"] = t                     # 伪装成本住院时间点的"另一次测量",标签不动
        p["events"].append(row)
        injected.append({"label": row.get("label"), "value": row.get("valuenum"),
                         "uom": row.get("valueuom"),
                         "patient_own_value": own_row.get("valuenum")})
    p["events"].sort(key=lambda e: (e["time"] is None, e["time"] or ""))
    return p, {"applied": True, "n_affected": len(injected), "injected": injected}


def f7_timeline_drift(payload: dict, severity: str, seed: str,
                      critical: list[str] | None = None) -> tuple[dict, dict]:
    """F7 时间线漂移:事件时间戳随机小幅前后漂移,并打乱顺序(病程先后关系被破坏)。

    critical 只用于**记录**受影响的判分相关检验项:不记的话定向告警率恒为 0,
    那不是"模型从不点名",而是"判据无从匹配"——把测量缺口写成模型缺陷。
    """
    p = copy.deepcopy(payload)
    events = [e for e in p.get("events", []) if e.get("time")]
    if len(events) < 5:
        return p, {"applied": False, "reason": "事件太少"}
    hours = 6 if severity == "mild" else 48
    keys = {_norm_label(k) for k in (critical or [])}
    touched: set[str] = set()
    for i, e in enumerate(events):
        r = _rng(seed, "f7", e.get("time"), e.get("label") or e.get("drug"), i)
        offset = (r - 0.5) * 2 * hours / 24.0
        e["time"] = _shift_time(e["time"], offset)
        if keys and _norm_label(e.get("label")) in keys:
            touched.add(str(e.get("label")))
    # 打乱顺序(severity 越高打得越狠)
    if severity == "severe":
        events_sorted = sorted(events, key=lambda e: e["time"])
        k = len(events_sorted) // 2
        head, tail = events_sorted[:k], events_sorted[k:]
        events[:] = (tail + head) if head and tail else events_sorted
    return p, {"applied": True, "n_affected": len(events), "drift_hours": hours,
               "reordered": severity == "severe", "labels": sorted(touched)}


FAULTS = {
    "F1": ("数据层", "关键化验置空", f1_drop_critical),
    "F2": ("数据层", "单位错误", f2_unit_error),
    "F3": ("数据层", "矛盾记录", f3_contradiction),
    "F4": ("数据层", "陈旧数据", f4_stale_data),
    "F5": ("工具层", "结果截断", f5_tool_truncation),
    "F6": ("工具层", "跨患者污染", f6_cross_patient),
    "F7": ("时序层", "时间线漂移", f7_timeline_drift),
}

# 每个故障只在**有判分通道**的任务上测(2026-10-27 定):
#   化验数值类(F1/F2/F4/F5/F6)要靠 top3_abnormal_labs / recheck_labs 才能体现,T2 不判化验;
#   矛盾记录类(F3)要靠 conflicts / contraindications 才能体现,T1 没有冲突字段;
#   时间线类(F7)要靠"取最后一次数值"或"当前异常项"体现,同样与 T2 无关。
# 不设这道闸时,注入到无通道任务上得到的是纯方差,却被平均进"故障效应"里稀释一切
# (实测 F5 整体 0.256 vs 只看 T1/T3 的 0.417;F6 的严重度曲线反向也全部来自 T2)。
APPLICABLE = {
    "F1": ("T1", "T3"),
    "F2": ("T1", "T3"),
    "F3": ("T2", "T3"),
    "F4": ("T1", "T3"),
    "F5": ("T1", "T3"),
    "F6": ("T1", "T3"),
    "F7": ("T1", "T3"),
}


def applicable(fault: str, critical: list[str] | None = None,
               task_type: str | None = None) -> bool:
    """该故障在该任务上是否存在判分通道。task_type 未知时按 critical 是否为空判。"""
    if fault in APPLICABLE:
        if task_type:
            return task_type in APPLICABLE[fault]
        return bool(critical) if fault != "F3" else True
    return True


def _not_applicable_reason(fault: str) -> str:
    return f"{fault} 在该任务类型上无判分通道(适用任务:{'/'.join(APPLICABLE.get(fault, ()))}),已跳过"


def inject(payload: dict, spec: dict, donor_labs: dict | list | None = None,
           critical: list[str] | None = None, task_type: str | None = None) -> tuple[dict, dict]:
    """按 spec 注入故障。spec = {"fault": "F2", "severity": "mild"}。

    critical:该任务答案键依赖的检验项名;传了就只打这些项(否则注入到模型根本不看的
    化验上,效应为 0 只是设计缺陷,不是模型鲁棒)。
    donor_labs:F6 的供体表(见 donor_pool);其他故障忽略。
    task_type:用于适用性判定(见 applicable)。

    适用性不成立的组合一律返回 applied=False,由调用方跳过——不能让它以"零效应"混进
    故障效应表:那会把"这项故障与该任务无关"读成"模型对这项故障免疫"。
    """
    fault, severity = spec["fault"], spec["severity"]
    if fault not in FAULTS:
        raise ValueError(f"未知故障类型 {fault}")
    if severity not in SEVERITIES:
        raise ValueError(f"未知严重度 {severity}")
    if not applicable(fault, critical, task_type):
        return copy.deepcopy(payload), {"applied": False, "reason": _not_applicable_reason(fault)}
    fn = FAULTS[fault][2]
    seed = spec.get("seed", "")
    if fault == "F6":
        out, rec = fn(payload, severity, seed, donor_labs, critical)
    elif fault in ("F1", "F2", "F4", "F5", "F7"):
        out, rec = fn(payload, severity, seed, critical)
    else:
        out, rec = fn(payload, severity, seed)
    rec.update({"fault": fault, "severity": severity,
                "layer": FAULTS[fault][0], "name": FAULTS[fault][1]})
    return out, rec


def selftest() -> None:
    tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    import tasks as task_lib
    rules = task_lib.load_rules()

    def crit(t):
        return task_lib.critical_labels(t)

    # 每个故障各自挑一个"能打得进去"的基底:同一基底不可能对所有故障都可注入
    # (如 F2 只能换算 4 种化验,F3 只能在不含该药的住院里植入)
    subjects: dict[str, dict] = {}
    for fault in FAULTS:
        for t in tasks:
            if not applicable(fault, crit(t), t["task_type"]):
                continue
            _out, rec = inject(t["payload"], {"fault": fault, "severity": "severe",
                                              "seed": "probe"}, donor_pool(t, tasks),
                               crit(t), t["task_type"])
            if rec["applied"]:
                subjects[fault] = t
                break
        assert fault in subjects, f"{fault} 在所有适用任务上都注不进去——数据与规则表脱节"
    base = subjects["F1"]
    t2 = next(t for t in tasks if t["task_type"] == "T2")
    print(f"基底任务 {base['task_id']}({len(base['payload']['events'])} 事件,"
          f"答案键检验项 {len(crit(base))} 个);F3 基底 {subjects['F3']['task_id']}"
          f"(规则 {len(rules)} 条)")
    original = json.dumps(base["payload"], ensure_ascii=False, sort_keys=True)

    for fault, (layer, name, _fn) in FAULTS.items():
        for severity in SEVERITIES:
            subject = subjects[fault]
            donors = donor_pool(subject, tasks) if fault == "F6" else None
            spec = {"fault": fault, "severity": severity, "seed": "st"}
            before = json.dumps(subject["payload"], ensure_ascii=False, sort_keys=True)
            out, rec = inject(subject["payload"], spec, donors, crit(subject), subject["task_type"])
            assert json.dumps(subject["payload"], ensure_ascii=False, sort_keys=True) == before, \
                f"{fault} 修改了入参"
            assert set(out.keys()) == set(subject["payload"].keys()), f"{fault} 改变了 schema 顶层键"
            out2, _ = inject(subject["payload"], spec, donors, crit(subject), subject["task_type"])
            assert json.dumps(out, ensure_ascii=False, sort_keys=True) == \
                   json.dumps(out2, ensure_ascii=False, sort_keys=True), f"{fault}/{severity} 不可复现"
            assert rec["applied"], f"{fault}/{severity} 未生效: {rec.get('reason')}"
            if fault == "F2":
                # 换算必须真的改数值:因子为 1.0 时"注入"了等于没注入,却仍记 applied=True,
                # 会让该故障的效应被无声样本稀释(肌酐踩过这个坑,实测降到 0.006)
                real = [c for c in rec.get("changes") or [] if "from" in c]
                assert real and any(c["from"] != c["to"] for c in real), \
                    f"F2/{severity} 换算前后数值相同,故障无声"
            if fault == "F6":
                # 定向性:注入行的标签必须在答案键依赖项内,否则等于撒噪声
                assert all(_norm_label(x["label"]) in {_norm_label(k) for k in crit(subject)}
                           for x in rec["injected"]), "F6 注入到了非判分项上"
                assert all(x.get("patient_own_value") != x.get("value")
                           for x in rec["injected"]), "F6 供体数值与本患者相同,故障无声"
            print(f"  OK {fault}/{severity:6s} [{layer}] {name}: "
                  f"{rec.get('n_affected')} 项受影响")

    # 适用性闸:T2 只判医嘱冲突,化验类故障在该任务上必须判为不适用。
    # 否则注入到无判分通道的任务上得到的是纯方差,却被平均进"故障效应"。
    for fault in ("F1", "F2", "F4", "F5", "F6", "F7"):
        _out, rec = inject(t2["payload"], {"fault": fault, "severity": "severe", "seed": "st"},
                           donor_pool(t2, tasks), crit(t2), "T2")
        assert not rec["applied"], f"{fault} 在 T2 上仍然生效——会把无判分通道的噪声算进效应"
    _out, rec = inject(base["payload"], {"fault": "F3", "severity": "mild", "seed": "st"},
                       None, crit(base), "T1")
    assert not rec["applied"], "F3 在 T1 上仍然生效——T1 没有冲突字段可判"
    print("  OK 适用性闸:化验类故障在 T2 全跳过、F3 在 T1 跳过")

    # 纯函数:基底对象在所有注入后应逐字节不变
    assert json.dumps(base["payload"], ensure_ascii=False, sort_keys=True) == original, \
        "注入修改了入参"
    print("SELFTEST OK(七类 × 两档 = 14 组全部生效,纯函数、schema 不变、可复现;"
          "F6 定向且数值异于本患者;适用性闸生效)")


def main() -> None:
    ap = argparse.ArgumentParser(description="FaultProxy 故障注入")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", help="对指定 task_id 演示七类故障")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.demo:
        tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
        task = next(t for t in tasks if t["task_id"] == args.demo)
        for fault in FAULTS:
            out, rec = inject(task["payload"], {"fault": fault, "severity": "severe",
                                                "seed": args.demo})
            print(f"{fault}: {json.dumps(rec, ensure_ascii=False)[:220]}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
