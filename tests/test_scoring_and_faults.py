"""判分与故障注入的回归测试(标准库 unittest,不需要 API key,不花钱)。

这套测试守的是 2026-10-27/28 那轮"结果全是判分伪影"的六个坑,每个都用一个失败过
的具体现象做断言——它们都曾经让故障矩阵读出完全错误的结论,而且**没有任何报错**:

  1. T3 出院药在植入禁忌医嘱之前算 → 模型如实列出那条药反而被扣分
  2. 化验判分只比项目名 → 照抄被污染的错误数值也算满分,F2/F4 效应为 0
  3. T2 判冲突时药名照抄医嘱原文('ibuprofen 1 MG Oral Tablet')→ 正确检出算假阳性
  4. F6 从固定供体取前 5 行 → 注入的项与任务无关,T2 上效应纯噪声
  5. F3 答案键不含被注入的禁忌药 → 正确检出反而算误报,恒为"零效应"
  6. 故障注入到无判分通道的任务类型(T2 上注化验故障)→ 效应为 0 被读成"模型免疫"

运行(仓库根目录): python -m unittest discover tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faults as fault_lib  # noqa: E402
import matrix as matrix_lib  # noqa: E402
import tasks as task_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_tasks() -> list[dict]:
    path = REPO / "configs" / "tasks.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class TestScoringDefects(unittest.TestCase):
    """判分器:每个断言都对应一次"结果看起来合理但完全错"的实测事故。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = load_tasks()

    def test_cheat_answer_scores_full_marks(self):
        # 判分链路自检:把答案键原样交回去必须 1.0,否则后面所有数字都不可信
        for t in self.tasks:
            self.assertAlmostEqual(task_lib.score(t, task_lib.cheat_answer(t))["score"], 1.0,
                                   places=6, msg=f"{t['task_id']} 满分答案未得 1.0")

    def test_empty_answer_scores_zero(self):
        # 防 do-nothing:尤其 T2 干净对照(答案=无冲突)不能被"什么都不交"白拿满分
        for t in self.tasks:
            self.assertEqual(task_lib.score(t, {})["score"], 0.0, t["task_id"])

    def test_t1_lab_scoring_requires_value_and_unit(self):
        """只比项目名会让"照抄被污染数值"拿满分——F2/F4 的效应就会恒为 0。"""
        t = next(x for x in self.tasks
                 if x["task_type"] == "T1" and x["answer_key"]["top3_abnormal_labs"])
        gold = t["answer_key"]["top3_abnormal_labs"][0]
        base = {"med_list": t["answer_key"]["med_list"], "allergies": []}
        exact = {**base, "top3_abnormal_labs": [dict(gold)]}
        self.assertGreater(task_lib.score(t, exact)["lab_top3_hit"], 0)
        wrong_value = {**base, "top3_abnormal_labs": [{**gold, "value": (gold["value"] or 0) * 2}]}
        self.assertEqual(task_lib.score(t, wrong_value)["lab_top3_hit"], 0.0)
        wrong_unit = {**base, "top3_abnormal_labs": [{**gold, "uom": "mmol/L"}]}
        if (gold.get("uom") or "").lower() != "mmol/l":
            self.assertEqual(task_lib.score(t, wrong_unit)["lab_top3_hit"], 0.0)

    def test_t2_conflict_accepts_drug_name_copied_from_orders(self):
        """模型照抄医嘱原文的药名('ibuprofen 1 MG Oral Tablet')是在正确做事,不能算误报。"""
        t = next(x for x in self.tasks
                 if x["task_type"] == "T2" and x["answer_key"]["conflicts"])
        c = t["answer_key"]["conflicts"][0]
        verbose = {"conflicts": [{"drug": f"{c['drug']} 1 MG Oral Tablet",
                                  "condition": c["condition"], "reason": "x"}]}
        self.assertAlmostEqual(task_lib.score(t, verbose)["score"], 1.0, places=6)

    def test_t3_discharge_meds_include_planted_drug(self):
        """T3 的禁忌医嘱 stoptime 为空 = 出院时仍在用,所以它也是出院带药。

        先算 dmeds 再植入,会让模型如实列出这条药反而被 med_f1 扣分(F3 的"效应"
        全是这个伪影,实测 10/10 条)。
        """
        for t in (x for x in self.tasks if x["task_type"] == "T3"):
            for drug in t["answer_key"]["contraindications"]:
                self.assertIn(drug.lower(), [d.lower() for d in t["answer_key"]["discharge_meds"]],
                              f"{t['task_id']} 植入的 {drug} 未计入出院带药")

    def test_t3_empty_discharge_meds_not_counted(self):
        """金标为空且模型也交空,是白送分——该字段在饱和时必须退出均值。"""
        t = next((x for x in self.tasks if x["task_type"] == "T3"
                  and not x["answer_key"]["discharge_meds"]), None)
        if t is None:
            self.skipTest("本任务集里没有空白出院药的 T3")
        ans = {"discharge_meds": [], "recheck_labs": t["answer_key"]["recheck_labs"],
               "contraindications": t["answer_key"]["contraindications"], "warnings": ["w"]}
        self.assertIsNone(task_lib.score(t, ans)["med_f1"])
        hallucinated = {**ans, "discharge_meds": ["furosemide"]}
        self.assertEqual(task_lib.score(t, hallucinated)["med_f1"], 0.0)


class TestFaultInjection(unittest.TestCase):
    """故障注入:定向性、适用性、纯函数。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = load_tasks()
        cls.by_id = {t["task_id"]: t for t in cls.tasks}

    def test_selftest_passes(self):
        fault_lib.selftest()

    def test_f6_targets_answer_key_items_and_differs_from_own_value(self):
        """F6 早期从固定供体取前 5 行,注入项与任务无关 = 往 payload 里撒噪声。"""
        checked = 0
        for t in self.tasks:
            critics = {fault_lib._norm_label(x) for x in task_lib.critical_labels(t)}
            if not critics:
                continue
            _out, rec = fault_lib.inject(
                t["payload"], {"fault": "F6", "severity": "severe", "seed": t["task_id"]},
                fault_lib.donor_pool(t, self.tasks), task_lib.critical_labels(t), t["task_type"])
            if not rec["applied"]:
                continue
            for row in rec["injected"]:
                self.assertIn(fault_lib._norm_label(row["label"]), critics, t["task_id"])
                self.assertNotEqual(row["value"], row.get("patient_own_value"),
                                    f"{t['task_id']} 供体数值与本患者相同,故障无声")
            checked += 1
        self.assertGreater(checked, 20, "可注入 F6 的任务太少,供体池有问题")

    def test_no_test_channel_faults_are_skipped(self):
        """故障注入到无判分通道的任务上,效应为 0 会被读成"模型免疫"——必须判不适用。"""
        t2 = next(t for t in self.tasks if t["task_type"] == "T2")
        for fault in ("F1", "F2", "F4", "F5", "F6", "F7"):
            _out, rec = fault_lib.inject(t2["payload"],
                                         {"fault": fault, "severity": "severe", "seed": "t"},
                                         fault_lib.donor_pool(t2, self.tasks),
                                         task_lib.critical_labels(t2), "T2")
            self.assertFalse(rec["applied"], f"{fault} 在 T2 上仍然生效")
        t1 = next(t for t in self.tasks if t["task_type"] == "T1")
        _out, rec = fault_lib.inject(t1["payload"],
                                     {"fault": "F3", "severity": "mild", "seed": "t"},
                                     None, task_lib.critical_labels(t1), "T1")
        self.assertFalse(rec["applied"], "F3 在 T1 上仍然生效(无冲突字段可判)")

    def test_rule_table_matches_only_clinically_valid_conditions(self):
        """子串匹配把 CKD 1 期、既往流产史都算成禁忌,植入的冲突临床上不成立。"""
        rules = {r["rule_id"]: r for r in task_lib.load_rules()}
        self.assertNotIn("CKD_METFORMIN", rules)     # 旧的 CKD 通用规则已被分期规则取代
        import re
        for rid in ("CKD4_METFORMIN", "PREG_ACEI"):
            rule = rules[rid]
            self.assertFalse(re.search(rule["condition_regex"],
                                       "past pregnancy history of miscarriage (situation)")
                             and rule["condition_regex"].startswith("(normal"),
                             f"{rid} 把既往孕史误判为当前妊娠")
        self.assertFalse(re.search(rules["CKD4_METFORMIN"]["condition_regex"],
                                   "chronic kidney disease stage 1 (disorder)"),
                         "CKD_METFORMIN 把 1 期算作禁忌")


class TestAnswerKeyUnderFault(unittest.TestCase):
    """注入故障时答案键的处理:只有 F3 并入注入项,其余一律保持干净真值。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = load_tasks()

    def test_non_f3_faults_keep_clean_answer_key(self):
        t = next(x for x in self.tasks if x["task_type"] == "T3")
        rec = {"fault": "F5", "rules": [{"drug": "ibuprofen"}]}
        self.assertEqual(matrix_lib.recompute_key(t, None, rec), t["answer_key"])

    def test_f3_key_includes_injected_drug_and_discharge_med(self):
        t = next(x for x in self.tasks if x["task_type"] == "T3")
        rec = {"fault": "F3", "rules": [{"rule_id": "CKD3_IBUPROFEN", "drug": "ibuprofen",
                                        "condition": "ckd"}]}
        key = matrix_lib.recompute_key(t, None, rec)
        self.assertIn("ibuprofen", key["contraindications"])
        self.assertIn("ibuprofen", key["discharge_meds"])

    def test_f3_detection_measured_through_answer_fields(self):
        """F3 的检出通道是 conflicts/contraindications,不是 data_issues 告警文本。"""
        t = next(x for x in self.tasks if x["task_type"] == "T3")
        rec = {"fault": "F3", "rules": [{"drug": "ibuprofen"}]}
        self.assertEqual(matrix_lib.targeted_flag(
            t, {"contraindications": ["ibuprofen 1 MG Oral Tablet"]}, rec), 1.0)
        self.assertEqual(matrix_lib.targeted_flag(
            t, {"contraindications": ["metformin"]}, rec), 0.0)


if __name__ == "__main__":
    unittest.main()
