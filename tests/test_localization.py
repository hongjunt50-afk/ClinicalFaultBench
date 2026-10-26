"""localization 中文化适配层单元测试(标准库 unittest,不依赖 MIMIC 数据)。

运行(仓库根目录): python -m unittest discover tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import localization as loc  # noqa: E402


class TestLoadMaps(unittest.TestCase):
    def test_lab_map_has_core_entries(self):
        m = loc.load_lab_map()
        self.assertEqual(m[("potassium", "blood")], "血钾")
        self.assertEqual(m[("creatinine", "blood")], "血肌酐")
        self.assertEqual(m[("specific gravity", "urine")], "尿比重")
        # 同名不同标本是不同条目
        self.assertEqual(m[("glucose", "urine")], "尿糖")
        self.assertNotEqual(m[("glucose", "urine")], m[("glucose", "blood")])

    def test_drug_map_normalizes_case_and_spaces(self):
        m = loc.load_drug_map()
        # MIMIC 药名大小写混乱:LORazepam / Lorazepam / CefTRIAXone 必须同归一
        self.assertEqual(m[loc.norm("LORazepam")], "劳拉西泮")
        self.assertEqual(m[loc.norm("OxycoDONE (Immediate Release) ")],
                         m[loc.norm("OxyCODONE (Immediate Release)")])
        self.assertEqual(m[loc.norm("Sodium Chloride 0.9%  Flush")], "0.9%氯化钠冲管液")


class TestUnitConversions(unittest.TestCase):
    def test_bi_chemistry_conversions(self):
        c = loc.load_conversions()
        val, uom = loc.convert(180.0, "Glucose", "mg/dL", c)
        self.assertAlmostEqual(val, 9.99, places=6)
        self.assertEqual(uom, "mmol/L")
        self.assertEqual(loc.convert(1.0, "Creatinine", "mg/dL", c), (88.4, "µmol/L"))
        val, uom = loc.convert(2.0, "Bilirubin Total", "mg/dL", c)
        self.assertAlmostEqual(val, 34.2, places=6)
        self.assertEqual(uom, "µmol/L")

    def test_hematology_counts_need_no_conversion(self):
        c = loc.load_conversions()
        self.assertIsNone(loc.convert(7.0, "Hematocrit", "%", c))
        self.assertIsNone(loc.convert(150.0, "PTT", "sec", c))
        self.assertIsNone(loc.convert(4.2, "Sodium", "mmol/L", c))  # 已是 SI 不转换

    def test_unknown_label_returns_none(self):
        c = loc.load_conversions()
        self.assertIsNone(loc.convert(1.0, "不存在的项目", "mg/dL", c))


class TestIcdChapters(unittest.TestCase):
    def test_icd10(self):
        ch = loc.load_icd_chapters()
        self.assertEqual(loc.icd_chapter_zh("I2510", 10, ch), "循环系统疾病")
        self.assertEqual(loc.icd_chapter_zh("E785", 10, ch), "内分泌、营养和代谢疾病")
        self.assertEqual(loc.icd_chapter_zh("Z794", 10, ch), "影响健康状态与保健机构接触的因素")
        # D 码按数值分章:D48 肿瘤,D50 血液
        self.assertEqual(loc.icd_chapter_zh("D485", 10, ch), "肿瘤")
        self.assertEqual(loc.icd_chapter_zh("D649", 10, ch), "血液及造血器官疾病和某些涉及免疫机制的疾患")

    def test_icd9(self):
        ch = loc.load_icd_chapters()
        self.assertEqual(loc.icd_chapter_zh("4019", 9, ch), "循环系统疾病")
        self.assertEqual(loc.icd_chapter_zh("42731", 9, ch), "循环系统疾病")
        self.assertEqual(loc.icd_chapter_zh("V4501", 9, ch), "影响健康状态与保健机构接触的因素")
        self.assertEqual(loc.icd_chapter_zh("E9331", 9, ch), "疾病和死亡的外因")

    def test_bad_version(self):
        self.assertIsNone(loc.icd_chapter_zh("I10", 11, loc.load_icd_chapters()))


class TestAnnotate(unittest.TestCase):
    SAMPLE = {
        "admissions": [{
            "diagnoses": [{"icd_code": "I2510", "icd_version": "10"}],
            "events": [
                {"category": "lab", "label": "Potassium", "fluid": "Blood"},
                {"category": "prescription", "drug": "LORazepam"},
                {"category": "med_admin", "medication": "NS"},
            ],
        }],
    }

    def test_annotates_without_touching_input(self):
        import copy
        src = copy.deepcopy(self.SAMPLE)
        out = loc.annotate(src)
        self.assertEqual(src, self.SAMPLE)  # 入参不动(纯函数)
        dx = out["admissions"][0]["diagnoses"][0]
        self.assertEqual(dx["chapter_zh"], "循环系统疾病")
        self.assertEqual(out["admissions"][0]["events"][0]["label_zh"], "血钾")
        self.assertEqual(out["admissions"][0]["events"][1]["drug_zh"], "劳拉西泮")
        # med_admin 类事件不被标注
        self.assertNotIn("drug_zh", out["admissions"][0]["events"][2])
        self.assertNotIn("label_zh", out["admissions"][0]["events"][2])

    def test_unmapped_terms_get_empty_string(self):
        out = loc.annotate({"admissions": [{"diagnoses": [], "events": [
            {"category": "lab", "label": "某个没映射的项目", "fluid": "Blood"},
            {"category": "prescription", "drug": "Bag"},
        ]}]})
        self.assertEqual(out["admissions"][0]["events"][0]["label_zh"], "")
        self.assertEqual(out["admissions"][0]["events"][1]["drug_zh"], "")


if __name__ == "__main__":
    unittest.main()
