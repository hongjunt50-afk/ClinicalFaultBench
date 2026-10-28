"""离线重判分:用当前的 tasks.jsonl 答案键 + 当前 tasks.score 重算既有结果。

用途(不花一分 API 额度):
  - 判分器修了(T3 出院药漏算植入禁忌、数值单位双判据、T2 空答案闸…),
    已记录在 jsonl 里的模型输出(parsed/raw)可直接重判,不必重跑模型;
  - 任务答案键重建后(configs/tasks.jsonl 更新),基线分与故障矩阵必须同步重判,
    否则配对差值把"判分器改动"混进了"故障效应"。

用法:
  python src/rescore.py                       # 重判基线与矩阵,就地覆盖(先备份)
  python src/rescore.py --dry-run             # 只看变化,不写回
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faults as fault_lib  # noqa: E402
import matrix as matrix_lib  # noqa: E402
import tasks as task_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"
RESULTS_DIR = REPO / "results"


def load_tasks() -> dict[str, dict]:
    return {t["task_id"]: t
            for t in (json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines())}


def rescore_file(path: Path, tasks: dict[str, dict], is_matrix: bool) -> list[tuple[str, float, float]]:
    """就地重判;返回 [(task_id, 旧分, 新分)]。答案键缺失的任务只报不改。"""
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    deltas = []
    for r in lines:
        task = tasks.get(r["task_id"])
        if task is None:
            deltas.append((r["task_id"], r["metrics"].get("score"), float("nan")))
            continue
        parsed = r.get("parsed")
        if not parsed:
            new = {"score": 0.0}
        elif is_matrix:
            # 矩阵记录只能用"运行当时的故障记录"重建答案键(见 matrix.recompute_key);
            # 适用性闸已作废的记录在这一步被丢在外面,不会进来
            key = matrix_lib.recompute_key(task, None, r.get("fault_record"))
            new = task_lib.score({**task, "answer_key": key}, parsed)
        else:
            new = task_lib.score(task, parsed)
        old = r["metrics"].get("score")
        r["metrics"] = new
        if is_matrix:
            r["metrics"]["targeted_flag"] = matrix_lib.targeted_flag(
                task, parsed, r.get("fault_record") or {})
            r["_task"] = {**task,
                          "answer_key": matrix_lib.recompute_key(task, None, r.get("fault_record"))}
        deltas.append((r["task_id"], old, new.get("score")))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines),
                    encoding="utf-8")
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser(description="离线重判分(不调用模型)")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    files = [RESULTS_DIR / f"baseline_{args.model}.jsonl",
             RESULTS_DIR / f"matrix_{args.model}.jsonl"]
    for path in files:
        if not path.exists():
            print(f"跳过(不存在){path.name}")
            continue
        is_matrix = path.name.startswith("matrix_")
        if args.dry_run:
            # dry-run 不写文件:复制到临时目录再跑一遍
            tmp = path.with_suffix(".dryrun.jsonl")
            tmp.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            deltas = rescore_file(tmp, tasks, is_matrix)
            tmp.unlink()
        else:
            bak = path.with_suffix(path.suffix + ".prerescore")
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            deltas = rescore_file(path, tasks, is_matrix)
        changed = [(t, o, n) for t, o, n in deltas
                   if o is not None and n == n and abs(o - n) > 1e-9]
        mean = (sum(n for _, _, n in deltas if n == n) / max(1, len(deltas)))
        print(f"{path.name}: {len(deltas)} 条,重判后平均 {mean:.3f},"
              f"分数变化 {len(changed)} 条"
              + (f"(最大 {max((n - o for _, o, n in changed), key=abs):+.3f})" if changed else ""))
        for t, o, n in changed[:8]:
            print(f"  {t[:26]:26s} {o:.3f} → {n:.3f}")
        if len(changed) > 8:
            print(f"  … 另 {len(changed) - 8} 条")


if __name__ == "__main__":
    main()
