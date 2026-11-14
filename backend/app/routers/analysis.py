"""分析方案接口：维护分析方案，覆盖提交方案评审、启用方案、归档方案等动作。

分析读取接口（/readings）登记在 /{entry_id} 之前，避免被单条明细路由抢占。
读取记录把训练产物、参数复评版本与分析输出绑定，三方互相定位。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.ledger_store import ledger
from app.schemas import ActionResult, EntryPayload, PageResult, ReadingPayload
from app.services.analysis import AnalysisService
from app.services.analysis_reading import AnalysisReadingService
from app.store import store

router = APIRouter(prefix="/api/analysis", tags=["分析方案"])

service = AnalysisService()
reading_service = AnalysisReadingService(ledger_store=ledger, module_store=store)

LIST_FIELDS = ["方案编号", "分析主题", "数据集范围", "分析方法", "对比基准", "显著性阈值", "输出图表类型", "调度周期"]
STATUSES = ["草稿", "方案评审中", "已启用", "已归档"]


@router.post("/readings", response_model=ActionResult)
def create_reading(payload: ReadingPayload) -> ActionResult:
    """分析方案读取预测产物：快照模型版本、窗口、置信水平与预测中值。"""
    reading, message = reading_service.create_reading(payload.分析方案编号, payload.预测方案编号)
    if reading is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=reading)


@router.get("/readings", response_model=PageResult[dict])
def list_readings(
    分析方案编号: str | None = Query(default=None, description="按分析方案编号过滤读取记录"),
    page: int = 1,
    size: int = 50,
) -> PageResult[dict]:
    """读取记录列表：每条都实时与训练产物对账，刷新/重进后口径一致。"""
    readings = reading_service.list_readings(analysis_code=分析方案编号)
    start = max(page - 1, 0) * size
    return PageResult(items=readings[start:start + size], total=len(readings), page=page, size=size)


@router.get("", response_model=PageResult[dict])
def list_entries(
    keyword: str | None = Query(default=None, description="按方案编号检索"),
    status: str | None = Query(default=None, description="草稿、方案评审中、已启用、已归档"),
    page: int = 1,
    size: int = 20,
) -> PageResult[dict]:
    """按方案编号与状态过滤分析方案列表；没有数据时返回空页，不报错。"""
    if size > 200:
        raise HTTPException(status_code=400, detail="每页最多 200 条，请缩小分页范围")
    items, total = service.list_entries(keyword=keyword, status=status, page=page, size=size)
    return PageResult(items=items, total=total, page=page, size=size)


@router.get("/{entry_id}", response_model=dict)
def get_entry(entry_id: int) -> dict:
    """读取单条分析方案明细；不存在时给出可读的错误说明。"""
    entry = service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"分析方案 {entry_id} 不存在或已归档")
    return entry


@router.post("", response_model=ActionResult)
def create_entry(payload: EntryPayload) -> ActionResult:
    """登记一条分析方案，缺字段时说明原因而不是静默丢弃。"""
    entry, missing = service.create_entry(payload.values)
    if missing:
        return ActionResult(ok=False, message=f"缺少必填字段：{'、'.join(missing)}")
    return ActionResult(ok=True, message="分析方案已登记", entry=entry)


@router.post("/{entry_id}/actions", response_model=ActionResult)
def run_action(entry_id: int, payload: EntryPayload) -> ActionResult:
    """对单条分析方案执行提交方案评审、启用方案、归档方案；不允许的动作会被拦下并说明原因。"""
    action = str(payload.values.get("action") or "").strip()
    entry, message = service.run_action(entry_id, action)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.get("/export")
def export_entries() -> dict[str, Any]:
    """导出分析方案清单：返回当前过滤条件下的全量数据。"""
    items, total = service.list_entries(page=1, size=10000)
    return {"module": "analysis", "total": total, "items": items}
