"""分析侧读取预测产物：把 artifact 的模型版本、窗口与预测中值固化成读取记录。

分析方案（如 ANAL-STOCKOUT_RISK）消费预测方案（如 FORE-INVENTORY_WEEKLY）的产物时，
按 target_key 定位最新 artifact，读取时快照模型版本、参数复评版本、训练/预测窗口、
置信水平与预测中值；之后每次展示都重新与产物对账，保证刷新、返回、重进后
模型版本、置信水平、预测结果和重算范围都能对上。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.ledger_store import LedgerStore, ledger as default_ledger
from app.services.forecast_exec import ForecastExecutionEngine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnalysisReadingService:
    """分析读取：训练产物、参数复评与分析记录三方互相定位。"""

    def __init__(self, ledger_store: LedgerStore | None = None, module_store: Any = None) -> None:
        self.ledger = ledger_store or default_ledger
        self.modules = module_store
        self.engine = ForecastExecutionEngine(ledger_store=self.ledger, module_store=module_store)

    def create_reading(self, analysis_code: str, forecast_code: str) -> tuple[dict[str, Any] | None, str]:
        analysis_code = str(analysis_code or "").strip()
        forecast_code = str(forecast_code or "").strip()
        if not analysis_code or not forecast_code:
            return None, "分析方案编号与预测方案编号都不能为空"
        if self.modules is not None and not any(
            row.get("方案编号") == analysis_code for row in self.modules.rows("analysis")
        ):
            return None, f"分析方案 {analysis_code} 不存在，无法登记读取记录"
        artifact = self.engine.latest_artifact(forecast_code)
        if artifact is None:
            return None, f"预测方案 {forecast_code} 尚未产出训练产物，请先执行训练"
        latest_eval = self.ledger.get("evaluations", artifact.get("latest_eval_version") or "")
        seq = 1 + sum(1 for r in self.ledger.all("readings") if r.get("分析方案编号") == analysis_code)
        median = list((artifact.get("prediction") or {}).get("median") or [])
        reading = {
            "reading_id": f"READ-{analysis_code}-{seq:03d}",
            "分析方案编号": analysis_code,
            "预测方案编号": forecast_code,
            "artifact_sha256": artifact["artifact_sha256"],
            "model_version": artifact["model_version"],
            "eval_version": artifact.get("latest_eval_version"),
            "predicted_target": artifact.get("predicted_target"),
            "train_span": artifact.get("train_span"),
            "forecast_span": artifact.get("forecast_span"),
            "confidence_level": (latest_eval or {}).get("confidence_level"),
            "median_series": median,
            "median_total": sum(int(p.get("median", 0)) for p in median),
            "recompute_scope": artifact.get("recompute_scope"),
            "created_at": _now(),
        }
        self.ledger.put("readings", reading["reading_id"], reading)
        reading["consistency"] = self.consistency(reading)
        return reading, f"已读取预测产物 {artifact['model_version']}（参数复评版本 {reading['eval_version']}）"

    def list_readings(self, analysis_code: str | None = None) -> list[dict[str, Any]]:
        readings = self.ledger.all("readings")
        if analysis_code:
            readings = [r for r in readings if r.get("分析方案编号") == analysis_code]
        readings.sort(key=lambda r: (r.get("created_at"), r.get("reading_id")), reverse=True)
        for reading in readings:
            reading["consistency"] = self.consistency(reading)
        return readings

    def consistency(self, reading: dict[str, Any]) -> dict[str, Any]:
        """读取记录与当前产物对账：模型版本、窗口、复评版本、置信水平逐项核对。"""
        issues: list[str] = []
        artifact = self.ledger.get("artifacts", str(reading.get("artifact_sha256") or ""))
        if artifact is None:
            return {"consistent": False, "issues": ["训练产物缺失，无法按 artifact_sha256 定位"]}
        if artifact.get("model_version") != reading.get("model_version"):
            issues.append(f"模型版本不一致：读取 {reading.get('model_version')} / 产物 {artifact.get('model_version')}")
        if artifact.get("train_span") != reading.get("train_span") or artifact.get("forecast_span") != reading.get("forecast_span"):
            issues.append("训练/预测窗口与产物不一致")
        if artifact.get("latest_eval_version") != reading.get("eval_version"):
            issues.append(f"存在更新的参数复评版本：{artifact.get('latest_eval_version')}（当前读取 {reading.get('eval_version')}）")
        latest_eval = self.ledger.get("evaluations", artifact.get("latest_eval_version") or "")
        current_confidence = (latest_eval or {}).get("confidence_level")
        if current_confidence != reading.get("confidence_level"):
            issues.append(f"置信水平已更新：{current_confidence}（当前读取 {reading.get('confidence_level')}）")
        return {"consistent": not issues, "issues": issues}
