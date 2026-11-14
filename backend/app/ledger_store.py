"""预测执行台账的持久化存储：artifact、run_ledger、evaluation_record、分析读取记录。

通用模块的列表数据放在内存 store 里即可；但执行单元的产物必须跨刷新、
跨重启保持一致（模型版本、置信水平、预测结果、重算范围不能对不上），
所以这里用 JSON 文件落盘，写入采用「临时文件 + 原子替换」，避免半截文件。
真实项目里可以把这层替换成数据库实现，接口保持不变。
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "forecast_ledger.json"

TABLES = ("artifacts", "runs", "evaluations", "readings")


class LedgerStore:
    """带落盘能力的键值台账：每张表以主键为索引，写即持久化。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path or os.environ.get("FORECAST_LEDGER_PATH") or DEFAULT_PATH)
        self._state: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in TABLES}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def configure(self, path: str | Path) -> None:
        """重新指向落盘文件并加载（测试隔离、环境切换时用）。"""
        with self._lock:
            self._path = Path(path)
            self._state = {name: {} for name in TABLES}
            self._load()

    def reset(self) -> None:
        """清空全部台账并落盘。"""
        with self._lock:
            self._state = {name: {} for name in TABLES}
            self._save()

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for name in TABLES:
                self._state[name] = dict(data.get(name) or {})

    def _save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._path)

    def all(self, table: str) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(list(self._state[table].values()))

    def get(self, table: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._state[table].get(key)
            return copy.deepcopy(row) if row is not None else None

    def put(self, table: str, key: str, row: dict[str, Any]) -> None:
        with self._lock:
            self._state[table][key] = copy.deepcopy(row)
            self._save()


ledger = LedgerStore()
