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
import repopaths  # noqa: E402
import run as runner  # noqa: E402
import tasks as rule_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"
RESULTS_DIR = REPO / "results"


def _norm(text) -> str:
    return rule_lib._norm_name(text)


def recompute_key(task: dict, corrupted: dict, fault_record: dict | None = None) -> dict:
    """**除 F3 外不改答案键**——这是本基准最关键的一条设计。

    患者的真实病情不会因为数据被注入故障而改变。故障注入只是让 agent 看到的世界变坏,
    正确答案仍然是**干净数据上的答案**。所以:
      - 若 agent 被故障带偏,输出偏移 → 配对错误率上升(这就是要度量的失效);
      - 若 agent 识别出数据有问题、拒绝作答或明确标注不确定 → 它的答案可能仍然对齐干净真值,
        或者至少不会给出错误结论。
    早期版本在这里按注入后数据重算答案键,效果等于"奖励盲信坏数据",已废弃。

    **唯一例外是 F3(矛盾记录)**:它注入的是一条真实存在的禁忌医嘱——世界里真的多了一张
    开错的处方,不是记录在说谎。原答案键里不含这条,模型正确检出反而被判成假阳性
    (实测 F3/T2 因此恒为"零效应",纯属判分缺陷)。所以 F3 要把注入项并入答案键,
    度量的是"新增的这条矛盾有没有被检出"。
    """
    key = json.loads(json.dumps(task["answer_key"]))
    if not fault_record or fault_record.get("fault") != "F3":
        return key
    drugs = [c for c in fault_record.get("rules") or [] if c.get("drug")]
    if not drugs:
        return key
    if task["task_type"] == "T2":
        have = {str(c.get("drug", "")).lower() for c in key.get("conflicts") or []}
        for c in drugs:
            if c["drug"].lower() not in have:
                key.setdefault("conflicts", []).append(
                    {"rule_id": c.get("rule_id"), "drug": c["drug"],
                     "condition": c.get("condition")})
    elif task["task_type"] == "T3":
        have = {str(x).lower() for x in key.get("contraindications") or []}
        for c in drugs:
            if c["drug"].lower() not in have:
                key.setdefault("contraindications", []).append(c["drug"])
        # 注入的医嘱 stoptime 为空 = 出院时仍在用,所以它**也是**出院带药之一。
        # 不并进去的话,模型照数据如实列出这条药反而被 med_f1 扣分——F3 的"效应"
        # 全是这个判分伪影(实测 10/10 条都是这个模式),不是模型出了问题。
        dmeds = key.get("discharge_meds") or []
        for c in drugs:
            if c["drug"].lower() not in {str(x).lower() for x in dmeds}:
                dmeds.append(c["drug"])
        key["discharge_meds"] = dmeds
    return key


def targeted_flag(task: dict, answer: dict, fault_record: dict) -> float:
    """告警是否**指向被注入的那一项**——而不是"有没有随便说点什么"。

    实测发现:模型在干净数据上也会报告"检验值异常跳变/疑似重复数据"之类
    (假告警率 100%),所以二元"有没有告警"不能区分故障。这里要求告警文本里
    出现被注入故障影响的检验项/药名才算命中。

    F3 的检出通道不是 data_issues 而是答案字段本身(conflicts / contraindications):
    被注入的禁忌药如果出现在答案里,就是"识别到了",与告警文本无关。
    """
    if not isinstance(answer, dict):
        return 0.0
    injected_drugs = [c["drug"] for c in fault_record.get("rules") or [] if c.get("drug")]
    if fault_record.get("fault") == "F3" and injected_drugs:
        reported = {_norm(rule_lib.base_drug(str(c.get("drug", ""))))
                    for c in answer.get("conflicts") or [] if isinstance(c, dict)}
        reported |= {_norm(rule_lib.base_drug(str(x))) for x in answer.get("contraindications") or []}
        return 1.0 if any(_norm(rule_lib.base_drug(d)) in reported for d in injected_drugs) else 0.0
    issues = answer.get("data_issues")
    if not issues:
        return 0.0
    text = json.dumps(issues, ensure_ascii=False).lower()
    targets = set()
    for key in ("labels", "dropped_tail_labels"):
        for x in fault_record.get(key) or []:
            targets.add(str(x).lower())
    for c in fault_record.get("changes") or []:
        if c.get("label"):
            targets.add(str(c["label"]).lower())
    for c in fault_record.get("injected") or []:
        if c.get("label"):
            targets.add(str(c["label"]).lower())
    if not targets and fault_record.get("fault") == "F7":
        # 旧记录没记受影响项(F7 早期版本不接收 critical),补上:漂移覆盖全部事件,
        # 判分相关项就是任务答案键依赖的那几项。不补的话定向告警率恒为 0 是**测量缺口**,
        # 会被读成"模型对时序故障从不告警"。
        targets |= {str(x).lower() for x in rule_lib.critical_labels(task)}
    if not targets:
        return 0.0
    return 1.0 if any(t and t in text for t in targets) else 0.0


def critical_labels(task: dict) -> list[str]:
    """该任务答案键依赖的检验项英文名(实现见 tasks.critical_labels)。"""
    return rule_lib.critical_labels(task)


def _stale(record: dict, by_id: dict[str, dict]) -> bool:
    """该矩阵记录是否已失效(任务重建后 prompt/payload 变了,或故障对该任务类型不适用)。

    两种失效都必须剔除:留着会让"旧任务集上的故障效应"和"新任务集上的"混在一起平均,
    前者甚至不是同一个临床病例。
    """
    if not fault_lib.applicable(record["fault"], None, record["task_type"]):
        return True
    task = by_id.get(record["task_id"])
    old = record.get("_task")
    if task is None or old is None:
        return True
    # 只换 answer_key 的改动不算失效(F3 在运行期并入注入项属于正常),要看 prompt+payload
    return rule_lib.task_sig(old) != rule_lib.task_sig(task)


def main() -> None:
    ap = argparse.ArgumentParser(description="故障矩阵跑批")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--limit-per-fault", type=int, default=0,
                    help="每种 (fault,severity) 只跑前 N 个任务,0=全量")
    ap.add_argument("--task-type", choices=["T1", "T2", "T3"])
    ap.add_argument("--budget", type=float, default=500.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rescore", type=Path, help="对既有矩阵结果重判分")
    ap.add_argument("--append", action="store_true",
                    help="追加到既有结果(按 task×fault×severity 去重),分次扩样本用")
    ap.add_argument("--target-per-fault", type=int, default=0,
                    help="每个 (fault,severity) 的目标样本数;已完成的会先扣掉,只补差额")
    ap.add_argument("--faults", default="", help="只跑指定故障,逗号分隔(如 F6 或 F3,F6)")
    args = ap.parse_args()

    tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    if args.task_type:
        tasks = [t for t in tasks if t["task_type"] == args.task_type]
    by_id = {t["task_id"]: t for t in tasks}

    if args.rescore:
        records = [json.loads(line) for line in args.rescore.read_text(encoding="utf-8").splitlines()]
        clean = {}
        cpath = RESULTS_DIR / f"baseline_{args.model}.jsonl"
        if cpath.exists():
            for line in cpath.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                clean[r["task_id"]] = r["metrics"]["score"]
        by_id = {t["task_id"]: t for t in tasks}
        for r in records:
            parsed = r.get("parsed")
            # F3 的答案键在运行当时才并入注入项,只能从记录里取(见 recompute_key)
            task = by_id.get(r["task_id"], r.get("_task"))
            key = recompute_key(task, None, r.get("fault_record")) if task else None
            probe = {**task, "answer_key": key} if task else None
            r["metrics"] = rule_lib.score(probe, parsed) if (parsed and probe) else {"score": 0.0}
            # 重判必须连定向告警一起重算:只重算分数会让告警率退化成"有没有说点什么"
            r["metrics"]["targeted_flag"] = targeted_flag(task, parsed,
                                                          r.get("fault_record") or {})
            if probe:
                r["_task"] = probe
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

    # F6 供体按任务现建(F6 只打答案键依赖项,供体必须同名同量纲、且来自别的患者)
    donor_cache: dict[str, dict] = {}

    def donors_for(task: dict) -> dict:
        if task["task_id"] not in donor_cache:
            donor_cache[task["task_id"]] = fault_lib.donor_pool(task, tasks)
        return donor_cache[task["task_id"]]

    # 分次扩样本:先读已完成的 (task, fault, severity),只补缺口,不重复花钱
    out = RESULTS_DIR / f"matrix_{args.model}.jsonl"
    done: dict[tuple[str, str, str], dict] = {}
    stale = 0
    if args.append and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if _stale(r, by_id):
                    stale += 1
                    continue
                done[(r["task_id"], r["fault"], r["severity"])] = r
        print(f"已有结果 {len(done)} 条,分次跑模式:只补缺口"
              + (f"(作废 {stale} 条:故障不适用或任务已重建)" if stale else ""))
    done_per = defaultdict(int)
    for (_tid, f, s) in done:
        done_per[(f, s)] += 1

    wanted = set(args.faults.split(",")) if args.faults else set(fault_lib.FAULTS)
    jobs = []
    for fault in fault_lib.FAULTS:
        if fault not in wanted:
            continue
        for severity in fault_lib.SEVERITIES:
            # 先剔掉该组合已跑过的任务,再按缺口数取新任务(否则会反复跳过、补不满)
            candidates = [t for t in tasks
                          if (t["task_id"], fault, severity) not in done]
            if args.limit_per_fault:
                candidates = candidates[:args.limit_per_fault]
            quota = (max(0, args.target_per_fault - done_per[(fault, severity)])
                     if args.target_per_fault else None)
            picked = 0
            # 注入失败(如 F3 找不到可植入禁忌)不计入配额,继续往后找
            for t in candidates:
                if quota is not None and picked >= quota:
                    break
                corrupted, rec = fault_lib.inject(
                    t["payload"], {"fault": fault, "severity": severity,
                                   "seed": t["task_id"]}, donors_for(t),
                    critical=critical_labels(t), task_type=t["task_type"])
                if not rec["applied"]:
                    continue
                jobs.append((t, fault, severity, corrupted, rec))
                picked += 1

    print(f"待跑 {len(jobs)} 次调用(任务 {len(tasks)} × 故障 7 × 严重度 2,扣除已完成与不适用组合)")

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
                # 定向告警:是否点名了被注入的那一项(区分"真识别"与"泛泛而谈")
                r["metrics"]["targeted_flag"] = targeted_flag(t, r.get("parsed"), rec)
                results.append(r)
                spent += r.get("credit", 0)
                print(f"  {t['task_id'][:20]:20s} {fault}/{severity:6s} "
                      f"score={r['metrics']['score']:.3f} "
                      f"targeted={r['metrics']['targeted_flag']:.0f} "
                      f"credit={r.get('credit', 0):.3f}")

        for t, fault, severity, corrupted, rec in jobs:
            if spent >= args.budget:
                print(f"预算用尽 {spent:.1f} >= {args.budget}")
                break
            key = recompute_key(t, corrupted, rec)
            probe = {**t, "payload": corrupted, "answer_key": key}
            batch.append(((t, fault, severity, corrupted, rec),
                          pool.submit(runner.run_one, secrets["RELAY_BASE_URL"],
                                      secrets["RELAY_API_KEY"], args.model, probe), key))
            if len(batch) >= args.workers:
                settle(batch)
                batch = []
        if batch:
            settle(batch)

    for r in results:
        done.setdefault((r["task_id"], r["fault"], r["severity"]), None)
        done[(r["task_id"], r["fault"], r["severity"])] = r
    merged = [r for r in done.values() if r is not None]
    with open(out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(merged, clean, args.model)
    print(f"明细 → {repopaths.rel(out)}(本次 {len(results)} 条,累计 {len(merged)} 条)")


def summarize(records: list[dict], clean: dict[str, float], model: str) -> None:
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["fault"]][r["severity"]].append(
            (clean.get(r["task_id"]), r["metrics"]["score"], r["task_type"],
             r["metrics"].get("targeted_flag", r["metrics"].get("flagged"))))
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
                # 按任务类型拆开:T3 分数是三字段均值,单个字段的伤害会被摊薄 1/3,
                # 只看总分会把"F3 打不出效果"读错。这张分表必须和总表一起看
                "by_type": {},
            }
            for ttype in ("T1", "T2", "T3"):
                sub = [(c, s) for c, s, t, _ in vals if t == ttype and c is not None]
                if not sub:
                    continue
                row[severity]["by_type"][ttype] = {
                    "n": len(sub),
                    "drop": round(sum(c - s for c, s in sub) / len(sub), 4)}
            cells += (f"{len(vals):>8}{(row[severity]['mean_score'] or 0):>9.3f}"
                      f"{(row[severity]['paired_drop'] or 0):>9.3f}"
                      f"{(row[severity]['flag_rate'] if row[severity]['flag_rate'] is not None else 0):>10.2f}")
        print(cells)
        table[fault] = row
    print("\n按任务类型拆解(配对下降):")
    for fault in sorted(table):
        parts = []
        for severity in ("mild", "severe"):
            bt = table[fault][severity].get("by_type") or {}
            if bt:
                parts.append(severity + " " + "/".join(
                    f"{t}:{v['drop']:+.3f}(n={v['n']})" for t, v in sorted(bt.items())))
        print(f"  {fault}  " + "  ".join(parts))
    out = RESULTS_DIR / f"matrix_summary_{model}.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总 → {repopaths.rel(out)}")
    print("注:告警率=模型主动点名被注入项的比例(定向)。干净基线上该值为假阳基线。")


if __name__ == "__main__":
    main()
