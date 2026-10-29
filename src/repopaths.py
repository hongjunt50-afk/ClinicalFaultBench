"""仓库路径小工具:日志与终端输出只打相对路径。

绝对路径会把本机目录结构(用户名、盘符、项目摆放)带进公开日志,发布前清理过一轮,
这里做成唯一入口,以后新脚本照着用即可。
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rel(p: str | Path) -> str:
    """仓库内的路径转相对路径;仓库外的原样返回,不泄露也不隐藏。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)
