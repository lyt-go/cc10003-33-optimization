"""预测模型接口：维护预测方案，覆盖提交模型评估、部署模型、下线模型等动作。

执行单元相关接口（/executions、/ledger、/artifacts）登记在 /{entry_id} 之前，
避免被单条明细路由抢占。执行单元的复用与版本策略见 services/forecast_exec.py。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.ledger_store import ledger
from app.schemas import ActionResult, EntryPayload, ExecutionPayload, PageResult
from app.services.forecast import ForecastService
from app.services.forecast_exec import ForecastExecutionEngine, parse_span_text
from app.store import store

router = APIRouter(prefix="/api/forecast", tags=["预测模型"])

service = ForecastService()
engine = ForecastExecutionEngine(ledger_store=ledger, module_store=store)

LIST_FIELDS = ["方案编号", "预测目标", "算法名称", "特征字段", "训练周期", "预测窗口天", "置信水平", "模型版本"]
STATUSES = ["草稿", "模型评估中", "已部署", "已下线"]


def _execute_result(result: dict[str, Any]) -> ActionResult:
    entry: dict[str, Any] = {"exec_mode": result["exec_mode"], "run": result["run"]}
    if result.get("artifact"):
        entry["artifact"] = result["artifact"]
    return ActionResult(ok=result["ok"], message=result["message"], entry=entry)


@router.post("/executions", response_model=ActionResult)
def submit_execution(payload: ExecutionPayload) -> ActionResult:
    """提交一次执行单元：按 identity 决定 cache_reuse / train_new / parameter_replay / resumed_slice。"""
    try:
        result = engine.submit(payload.model_dump(exclude_none=True))
    except ValueError as exc:
        return ActionResult(ok=False, message=str(exc))
    return _execute_result(result)


@router.get("/ledger", response_model=PageResult[dict])
def list_ledger(
    target_key: str | None = Query(default=None, description="按预测方案编号过滤台账"),
    page: int = 1,
    size: int = 50,
) -> PageResult[dict]:
    """run_ledger 台账：每次执行的执行方式、checkpoint_seq、输入快照与产物定位。"""
    runs = engine.list_runs(target_key=target_key)
    start = max(page - 1, 0) * size
    return PageResult(items=runs[start:start + size], total=len(runs), page=page, size=size)


@router.get("/artifacts", response_model=PageResult[dict])
def list_artifacts(
    target_key: str | None = Query(default=None, description="按预测方案编号过滤产物"),
    page: int = 1,
    size: int = 50,
) -> PageResult[dict]:
    """训练产物列表：同一 artifact_sha256 只有一份，模型版本随 identity 变化递增。"""
    artifacts = engine.list_artifacts(target_key=target_key)
    start = max(page - 1, 0) * size
    return PageResult(items=artifacts[start:start + size], total=len(artifacts), page=page, size=size)


@router.get("/artifacts/{sha}", response_model=dict)
def get_artifact(sha: str) -> dict:
    """产物明细：连同参数复评记录与分析读取记录一起返回，三方互相定位。"""
    artifact = engine.get_artifact(sha)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"训练产物 {sha} 不存在")
    return artifact


@router.get("", response_model=PageResult[dict])
def list_entries(
    keyword: str | None = Query(default=None, description="按方案编号检索"),
    status: str | None = Query(default=None, description="草稿、模型评估中、已部署、已下线"),
    page: int = 1,
    size: int = 20,
) -> PageResult[dict]:
    """按方案编号与状态过滤预测模型列表；没有数据时返回空页，不报错。"""
    if size > 200:
        raise HTTPException(status_code=400, detail="每页最多 200 条，请缩小分页范围")
    items, total = service.list_entries(keyword=keyword, status=status, page=page, size=size)
    return PageResult(items=items, total=total, page=page, size=size)


@router.get("/{entry_id}", response_model=dict)
def get_entry(entry_id: int) -> dict:
    """读取单条预测方案明细；不存在时给出可读的错误说明。"""
    entry = service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"预测方案 {entry_id} 不存在或已归档")
    return entry


@router.post("", response_model=ActionResult)
def create_entry(payload: EntryPayload) -> ActionResult:
    """登记一条预测方案，缺字段时说明原因而不是静默丢弃。"""
    entry, missing = service.create_entry(payload.values)
    if missing:
        return ActionResult(ok=False, message=f"缺少必填字段：{'、'.join(missing)}")
    return ActionResult(ok=True, message="预测方案已登记", entry=entry)


@router.post("/{entry_id}/actions", response_model=ActionResult)
def run_action(entry_id: int, payload: EntryPayload) -> ActionResult:
    """对单条预测方案执行提交模型评估、部署模型、下线模型；不允许的动作会被拦下并说明原因。"""
    action = str(payload.values.get("action") or "").strip()
    entry, message = service.run_action(entry_id, action)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.post("/{entry_id}/execute", response_model=ActionResult)
def execute_entry(entry_id: int, payload: EntryPayload) -> ActionResult:
    """按方案行派生 identity 提交执行；values 里可覆盖 run_id、parameters、fail_at_slices 等。"""
    entry = service.get_entry(entry_id)
    if entry is None:
        return ActionResult(ok=False, message=f"预测方案 {entry_id} 不存在或已归档")
    overrides = payload.values or {}
    train_span = overrides.get("train_span") or parse_span_text(entry.get("训练周期"))
    if train_span is None:
        return ActionResult(ok=False, message=f"方案 {entry.get('方案编号')} 的训练周期无法解析：{entry.get('训练周期')}")
    forecast_span = overrides.get("forecast_span") or _derive_forecast_span(train_span, entry.get("预测窗口天"))
    fields = [f for f in str(entry.get("特征字段") or "").replace("，", ",").replace("、", ",").split(",") if f.strip()]
    request = {
        "target_key": entry.get("方案编号"),
        "predicted_target": entry.get("预测目标"),
        "feature_snapshot": overrides.get("feature_snapshot") or {
            "fields": [f.strip() for f in fields],
            "snapshot_id": entry.get("特征快照") or f"snap-{entry.get('方案编号')}",
        },
        "train_span": train_span,
        "forecast_span": forecast_span,
        "parameters": overrides.get("parameters") or _entry_parameters(entry),
    }
    for key in ("run_id", "task_id", "input_snapshot", "fail_at_slices"):
        if overrides.get(key) is not None:
            request[key] = overrides[key]
    try:
        result = engine.submit(request)
    except ValueError as exc:
        return ActionResult(ok=False, message=str(exc))
    return _execute_result(result)


def _derive_forecast_span(train_span: dict[str, str], window_days: Any) -> dict[str, str]:
    try:
        days = int(str(window_days).strip())
    except (TypeError, ValueError):
        days = 7
    days = max(days, 1)
    start = date.fromisoformat(train_span["end"]) + timedelta(days=1)
    end = start + timedelta(days=days - 1)
    return {"start": start.isoformat(), "end": end.isoformat()}


def _entry_parameters(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        confidence = float(str(entry.get("置信水平") or "").strip())
    except ValueError:
        confidence = 0.8
    return {"置信水平": confidence, "算法名称": entry.get("算法名称")}


@router.get("/export")
def export_entries() -> dict[str, Any]:
    """导出预测模型清单：返回当前过滤条件下的全量数据。"""
    items, total = service.list_entries(page=1, size=10000)
    return {"module": "forecast", "total": total, "items": items}
