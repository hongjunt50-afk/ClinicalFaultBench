"""干净基线跑批:对 tasks.jsonl 里的任务逐个调用被测模型,用规则判分器打分。

成本闸:每次调用前累计 credit,超过 --budget 立即停(默认 20,单位同中转站返回的 credit)。
并发:默认 4,中转站对并发不敏感;失败重试 2 次。

用法(仓库根目录):
  python src/run.py --model deepseek-v4.1-flash --limit 3           # 冒烟
  python src/run.py --model deepseek-v4.1-flash                     # 全量 90
  python src/run.py --model kimi-k3.1-pro --budget 20
  python src/run.py --rescore results/baseline_deepseek-v4.1-flash.jsonl  # 只重判分
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as task_lib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "configs" / "tasks.jsonl"
RESULTS_DIR = REPO / "results"
SECRETS = Path(os.environ.get("RELAY_ENV") or REPO / ".env")


def load_secrets() -> dict[str, str]:
    out = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    for key in ("RELAY_BASE_URL", "RELAY_API_KEY"):
        if not out.get(key):
            raise SystemExit(f"{SECRETS} 缺 {key}")
    return out


def call_model(base_url: str, api_key: str, model: str, prompt: str,
               payload: dict, timeout: int = 600) -> tuple[str, float]:
    """返回 (原始回复文本, credit 消耗)。

    max_tokens 必须给足:deepseek-v4.1-flash 这类思考模型会把 budget 全烧在
    reasoning 上(T3 实测 3000/8000 都 finish=length 且 content 为空,16000 才出结果),
    预算给不够的表现是"空回复",会被误判成模型不会做。
    """
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": 20000,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"].get("content") or ""
    credit = float(data.get("usage", {}).get("credit", 0) or 0)
    return text, credit


def parse_json(text: str) -> dict | None:
    """模型常包 ```json ... ``` 或前后带解释,尽量抠出第一个 JSON 对象。"""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if start >= 0 and end > start else text
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_one(base_url: str, api_key: str, model: str, task: dict,
            retries: int = 2) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            text, credit = call_model(base_url, api_key, model, task["prompt"], task["payload"])
            parsed = parse_json(text)
            metrics = task_lib.score(task, parsed) if parsed is not None else {"score": 0.0,
                                                                              "parse_error": True}
            return {"task_id": task["task_id"], "task_type": task["task_type"],
                    "model": model, "credit": credit, "raw": text[:4000],
                    "parsed": parsed, "metrics": metrics}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            time.sleep(2 * (attempt + 1))
    return {"task_id": task["task_id"], "task_type": task["task_type"], "model": model,
            "credit": 0.0, "raw": "", "parsed": None,
            "metrics": {"score": 0.0, "error": last_err}}


def summarize(records: list[dict]) -> dict:
    by_type: dict[str, list[float]] = {}
    for r in records:
        by_type.setdefault(r["task_type"], []).append(r["metrics"]["score"])
    total_credit = sum(r.get("credit", 0) for r in records)
    return {
        "n": len(records),
        "credit": round(total_credit, 3),
        "by_type": {t: {"n": len(v), "mean_score": round(sum(v) / len(v), 4)}
                    for t, v in sorted(by_type.items())},
        "overall": round(sum(r["metrics"]["score"] for r in records) / len(records), 4)
        if records else 0.0,
        "errors": sum(1 for r in records if r["metrics"].get("error")),
        "parse_failures": sum(1 for r in records if r["metrics"].get("parse_error")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="干净基线跑批")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个(冒烟用)")
    ap.add_argument("--task-type", choices=["T1", "T2", "T3"])
    ap.add_argument("--budget", type=float, default=20.0, help="credit 上限,超过即停")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rescore", type=Path, help="只对既有结果重判分")
    ap.add_argument("--append", action="store_true",
                    help="追加到既有结果文件(分任务类型分批跑时用),并按 task_id 去重")
    args = ap.parse_args()

    if args.rescore:
        records = [json.loads(line) for line in args.rescore.read_text(encoding="utf-8").splitlines()]
        tasks = {t["task_id"]: t for t in
                 (json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines())}
        for r in records:
            parsed = r.get("parsed")
            r["metrics"] = task_lib.score(tasks[r["task_id"]], parsed) if parsed else {"score": 0.0}
        print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
        return

    secrets = load_secrets()
    all_tasks = [json.loads(x) for x in TASKS_PATH.read_text(encoding="utf-8").splitlines()]
    if args.task_type:
        all_tasks = [t for t in all_tasks if t["task_type"] == args.task_type]
    if args.limit:
        all_tasks = all_tasks[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{args.model}.jsonl"
    records: list[dict] = []
    spent = 0.0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = []
        for task in all_tasks:
            if spent >= args.budget:
                print(f"预算用尽({spent:.2f} >= {args.budget}),停止派发")
                break
            pending.append(pool.submit(run_one, secrets["RELAY_BASE_URL"],
                                       secrets["RELAY_API_KEY"], args.model, task))
            # 每满一批就等结果、结算真实 credit——否则一批派发完才发现超预算
            if len(pending) >= args.workers:
                for fut in pending:
                    rec = fut.result()
                    records.append(rec)
                    spent += rec.get("credit", 0)
                    print(f"  {rec['task_id']} score={rec['metrics']['score']:.3f} "
                          f"credit={rec.get('credit', 0):.3f} 累计={spent:.2f}")
                pending = []
        for fut in pending:
            rec = fut.result()
            records.append(rec)
            spent += rec.get("credit", 0)
            print(f"  {rec['task_id']} score={rec['metrics']['score']:.3f} "
                  f"credit={rec.get('credit', 0):.3f} 累计={spent:.2f}")

    # 先读旧结果再打开写(顺序反了会把文件截断成空)
    merged: dict[str, dict] = {}
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old = json.loads(line)
                merged[old["task_id"]] = old
    for r in records:
        merged[r["task_id"]] = r
    order = {t["task_id"]: i for i, t in enumerate(all_tasks)}
    with open(out_path, "w", encoding="utf-8") as f:
        for tid in sorted(merged, key=lambda x: order.get(x, 10 ** 6)):
            f.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")
    records = list(merged.values())

    summary = summarize(records)
    summary["model"] = args.model
    summary["elapsed_s"] = round(time.time() - started, 1)
    (RESULTS_DIR / f"summary_{args.model}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细 → {out_path}")


if __name__ == "__main__":
    main()
