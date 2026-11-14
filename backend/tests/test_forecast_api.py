"""接口层验收：执行提交、台账查询、分析读取与刷新后的一致性。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ledger_store import ledger
from app.main import app

INVENTORY_REQ = {
    "target_key": "FORE-INVENTORY_WEEKLY",
    "predicted_target": "stockout_units",
    "feature_snapshot": {"fields": ["sku", "warehouse", "promo"], "snapshot_id": "inv-weekly-snap-20260901"},
    "train_span": {"start": "2026-09-01", "end": "2026-10-01"},
    "forecast_span": {"start": "2026-10-02", "end": "2026-10-15"},
    "parameters": {"置信水平": 0.8, "算法名称": "lightgbm-quantile"},
}


@pytest.fixture()
def client(tmp_path):
    ledger.configure(tmp_path / "forecast_ledger.json")
    yield TestClient(app)


def test_full_chain_over_api(client):
    # 09-08 切片失败 -> checkpoint_seq=7
    resp = client.post("/api/forecast/executions", json={**INVENTORY_REQ, "run_id": "RUN-INV-1", "fail_at_slices": ["2026-09-08"]})
    body = resp.json()
    assert resp.status_code == 200 and not body["ok"]
    assert body["entry"]["run"]["checkpoint_seq"] == 7

    # 同一 run_id 续跑 -> resumed_slice，只落一份产物
    resp = client.post("/api/forecast/executions", json={**INVENTORY_REQ, "run_id": "RUN-INV-1"})
    body = resp.json()
    assert body["ok"] and body["entry"]["exec_mode"] == "resumed_slice"
    sha = body["entry"]["artifact"]["artifact_sha256"]

    # 参数复评 -> 只追加 evaluation_record
    resp = client.post("/api/forecast/executions", json={**INVENTORY_REQ, "run_id": "RUN-INV-2", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    assert resp.json()["entry"]["exec_mode"] == "parameter_replay"

    # 台账与产物互相定位
    ledger_page = client.get("/api/forecast/ledger", params={"target_key": "FORE-INVENTORY_WEEKLY"}).json()
    assert ledger_page["total"] == 2
    modes = {item["run_id"]: item["exec_mode"] for item in ledger_page["items"]}
    assert modes == {"RUN-INV-1": "resumed_slice", "RUN-INV-2": "parameter_replay"}
    artifacts = client.get("/api/forecast/artifacts").json()
    assert artifacts["total"] == 1
    detail = client.get(f"/api/forecast/artifacts/{sha}").json()
    assert len(detail["evaluation_records"]) == 2

    # 分析读取：模型版本与窗口一致，预测中值落账
    resp = client.post("/api/analysis/readings", json={"分析方案编号": "ANAL-STOCKOUT_RISK", "预测方案编号": "FORE-INVENTORY_WEEKLY"})
    reading = resp.json()["entry"]
    assert resp.json()["ok"]
    assert reading["model_version"] == "FORE-INVENTORY_WEEKLY#v1"
    assert reading["eval_version"] == detail["latest_eval_version"]
    assert reading["confidence_level"] == 0.9
    assert reading["train_span"] == {"start": "2026-09-01", "end": "2026-10-01"}
    assert reading["consistency"]["consistent"]

    # 刷新/重进：重新 GET 列表，口径不变
    again = client.get("/api/analysis/readings", params={"分析方案编号": "ANAL-STOCKOUT_RISK"}).json()
    assert again["total"] == 1
    assert again["items"][0]["median_total"] == reading["median_total"]
    assert again["items"][0]["consistency"]["consistent"]


def test_execute_from_entry_row(client):
    # 从方案行派生 identity 执行（FORE-INVENTORY_WEEKLY 种子行）
    resp = client.post("/api/forecast/4/execute", json={"values": {"run_id": "RUN-ROW-1"}})
    body = resp.json()
    assert body["ok"] and body["entry"]["exec_mode"] == "train_new"
    artifact = body["entry"]["artifact"]
    assert artifact["identity"]["feature_snapshot"]["fields"] == ["promo", "sku", "warehouse"]
    assert artifact["forecast_span"] == {"start": "2026-10-02", "end": "2026-10-15"}
    # 方案行模型版本已回写
    entry = client.get("/api/forecast/4").json()
    assert entry["模型版本"] == "FORE-INVENTORY_WEEKLY#v1"


def test_reading_without_artifact_returns_message(client):
    resp = client.post("/api/analysis/readings", json={"分析方案编号": "ANAL-STOCKOUT_RISK", "预测方案编号": "FORE-NONE"})
    assert not resp.json()["ok"]
    assert "尚未产出训练产物" in resp.json()["message"]


def test_reading_requires_existing_analysis_entry(client):
    client.post("/api/forecast/executions", json={**INVENTORY_REQ, "run_id": "RUN-X"})
    resp = client.post("/api/analysis/readings", json={"分析方案编号": "ANAL-NOT_EXISTS", "预测方案编号": "FORE-INVENTORY_WEEKLY"})
    assert not resp.json()["ok"]
    assert "不存在" in resp.json()["message"]
