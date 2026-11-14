"""库存预测场景验收：执行单元复用策略、断点续跑、参数复评与分析读取一致性。"""
from __future__ import annotations

import pytest

from app.ledger_store import LedgerStore, ledger
from app.services.analysis_reading import AnalysisReadingService
from app.services.forecast_exec import ForecastExecutionEngine
from app.store import store

# 库存预测场景：FORE-INVENTORY_WEEKLY，特征 sku/warehouse/promo，训练区间 2026-09-01 ~ 10-01
INVENTORY_REQ = {
    "target_key": "FORE-INVENTORY_WEEKLY",
    "predicted_target": "stockout_units",
    "feature_snapshot": {"fields": ["sku", "warehouse", "promo"], "snapshot_id": "inv-weekly-snap-20260901"},
    "train_span": {"start": "2026-09-01", "end": "2026-10-01"},
    "forecast_span": {"start": "2026-10-02", "end": "2026-10-15"},
    "parameters": {"置信水平": 0.8, "算法名称": "lightgbm-quantile"},
}


@pytest.fixture()
def engine(tmp_path):
    ledger.configure(tmp_path / "forecast_ledger.json")
    yield ForecastExecutionEngine(ledger_store=ledger, module_store=store)


@pytest.fixture()
def readings(engine):
    return AnalysisReadingService(ledger_store=ledger, module_store=store)


def test_train_new_creates_single_artifact(engine):
    result = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    assert result["ok"] and result["exec_mode"] == "train_new"
    artifact = result["artifact"]
    assert artifact["model_version"] == "FORE-INVENTORY_WEEKLY#v1"
    assert artifact["train_span"] == {"start": "2026-09-01", "end": "2026-10-01"}
    assert len(artifact["slices"]) == 31  # 09-01 ~ 10-01 按日切片
    assert artifact["recompute_scope"] == {"from": "2026-09-01", "to": "2026-10-01", "mode": "train_new"}
    assert len(engine.list_artifacts()) == 1
    # 训练产出即落第一条 evaluation_record
    assert artifact["latest_eval_version"] == f"EV-{artifact['artifact_sha256'][:8]}-01"


def test_cache_reuse_reads_existing_prediction(engine):
    first = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    again = engine.submit({**INVENTORY_REQ, "run_id": "RUN-B"})
    assert again["exec_mode"] == "cache_reuse"
    assert again["ok"] and again["run"]["recompute_scope"] is None
    assert again["artifact"]["artifact_sha256"] == first["artifact"]["artifact_sha256"]
    assert again["artifact"]["prediction"] == first["artifact"]["prediction"]
    assert len(engine.list_artifacts()) == 1  # 没有新产物


def test_parameter_replay_only_appends_evaluation(engine):
    trained = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    artifact = trained["artifact"]
    replay = engine.submit({
        **INVENTORY_REQ,
        "run_id": "RUN-B",
        "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"},
    })
    assert replay["exec_mode"] == "parameter_replay"
    assert replay["run"]["recompute_scope"] is None  # 参数区变化不重算
    after = engine.get_artifact(artifact["artifact_sha256"])
    assert after["model_version"] == artifact["model_version"]  # 模型版本不变
    assert len(engine.list_artifacts()) == 1  # 同一 artifact_sha256 不生成第二份产物
    assert after["evaluations"] == [f"EV-{artifact['artifact_sha256'][:8]}-0{i}" for i in (1, 2)]
    assert replay["run"]["eval_version"] == after["latest_eval_version"]
    assert after["confidence_level"] == 0.9
    # 相同参数再次提交：与当前生效参数区一致，复用而不重复复评
    dup = engine.submit({**INVENTORY_REQ, "run_id": "RUN-C", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    assert dup["exec_mode"] == "cache_reuse"
    assert len(engine.get_artifact(artifact["artifact_sha256"])["evaluations"]) == 2


def test_snapshot_or_span_change_triggers_train_new(engine):
    first = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    changed_snapshot = engine.submit({
        **INVENTORY_REQ,
        "run_id": "RUN-B",
        "feature_snapshot": {"fields": ["sku", "warehouse", "promo"], "snapshot_id": "inv-weekly-snap-20261001"},
    })
    assert changed_snapshot["exec_mode"] == "train_new"
    assert changed_snapshot["artifact"]["artifact_sha256"] != first["artifact"]["artifact_sha256"]
    assert changed_snapshot["artifact"]["model_version"] == "FORE-INVENTORY_WEEKLY#v2"
    changed_span = engine.submit({
        **INVENTORY_REQ,
        "run_id": "RUN-C",
        "train_span": {"start": "2026-09-08", "end": "2026-10-01"},
    })
    assert changed_span["exec_mode"] == "train_new"
    assert changed_span["artifact"]["model_version"] == "FORE-INVENTORY_WEEKLY#v3"
    assert len(engine.list_artifacts()) == 3


def test_failed_slice_resumes_from_checkpoint(engine):
    failed = engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1", "fail_at_slices": ["2026-09-08"]})
    assert not failed["ok"]
    assert failed["run"]["status"] == "failed"
    assert failed["run"]["checkpoint_seq"] == 7  # 09-08 是第 7 号切片（09-01 起算）
    assert failed["run"]["completed_slices"] == list(range(7))
    assert len(engine.list_artifacts()) == 0  # 失败不产出 artifact

    resumed = engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1"})
    assert resumed["ok"] and resumed["exec_mode"] == "resumed_slice"
    artifact = resumed["artifact"]
    assert len(engine.list_artifacts()) == 1  # 续跑不生成第二份产物
    # 成功切片不重训：0~6 来自第 1 次尝试，7~30 来自第 2 次
    attempts = {s["seq"]: s["trained_in_attempt"] for s in artifact["slices"]}
    assert all(attempts[i] == 1 for i in range(7))
    assert all(attempts[i] == 2 for i in range(7, 31))
    # 重算范围只覆盖失败切片及之后
    assert artifact["recompute_scope"]["from"] == "2026-09-08"
    assert artifact["recompute_scope"]["to"] == "2026-10-01"
    modes = [a["mode"] for a in resumed["run"]["attempts"]]
    assert modes == ["train_new", "resumed_slice"]


def test_repeated_run_id_is_idempotent(engine):
    first = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    dup = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    assert dup["ok"] and "幂等" in dup["message"]
    assert len(engine.list_runs()) == 1  # 台账不重复
    assert len(engine.list_artifacts()) == 1  # 不产生重复 artifact
    assert dup["run"]["updated_at"] == first["run"]["updated_at"]


def test_run_id_bound_to_single_identity(engine):
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    with pytest.raises(ValueError, match="已绑定其他执行单元"):
        engine.submit({**INVENTORY_REQ, "run_id": "RUN-A", "train_span": {"start": "2026-09-02", "end": "2026-10-01"}})


def test_parallel_tasks_share_artifact_keep_own_snapshots(engine):
    task_a = engine.submit({**INVENTORY_REQ, "run_id": "RUN-A", "task_id": "TASK-A", "input_snapshot": {"source": "scheduler-a", "rows": 100}})
    task_b = engine.submit({**INVENTORY_REQ, "run_id": "RUN-B", "task_id": "TASK-B", "input_snapshot": {"source": "scheduler-b", "rows": 100}})
    assert task_a["artifact"]["artifact_sha256"] == task_b["artifact"]["artifact_sha256"]
    assert task_b["exec_mode"] == "cache_reuse"
    runs = {r["run_id"]: r for r in engine.list_runs()}
    assert runs["RUN-A"]["input_snapshot"] == {"source": "scheduler-a", "rows": 100}
    assert runs["RUN-B"]["input_snapshot"] == {"source": "scheduler-b", "rows": 100}
    assert len(engine.list_artifacts()) == 1


def test_analysis_reading_distinguishes_model_and_param_versions(engine, readings):
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1", "fail_at_slices": ["2026-09-08"]})
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1"})

    reading1, _ = readings.create_reading("ANAL-STOCKOUT_RISK", "FORE-INVENTORY_WEEKLY")
    assert reading1["model_version"] == "FORE-INVENTORY_WEEKLY#v1"
    assert reading1["eval_version"].endswith("-01")
    assert reading1["confidence_level"] == 0.8
    assert reading1["predicted_target"] == "stockout_units"
    assert reading1["train_span"] == {"start": "2026-09-01", "end": "2026-10-01"}
    assert reading1["forecast_span"] == {"start": "2026-10-02", "end": "2026-10-15"}
    assert reading1["median_total"] == sum(p["median"] for p in reading1["median_series"])
    assert reading1["recompute_scope"]["from"] == "2026-09-08"
    assert reading1["consistency"]["consistent"]

    # 参数复评：同一模型的新参数结果
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-2", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    reading2, _ = readings.create_reading("ANAL-STOCKOUT_RISK", "FORE-INVENTORY_WEEKLY")
    assert reading2["model_version"] == reading1["model_version"]  # 模型版本一致
    assert reading2["eval_version"] != reading1["eval_version"]  # 复评版本不同
    assert reading2["confidence_level"] == 0.9
    assert reading2["consistency"]["consistent"]

    # 旧读取记录对账：能指出复评版本与置信水平已更新
    stale = readings.list_readings("ANAL-STOCKOUT_RISK")
    old = next(r for r in stale if r["reading_id"] == reading1["reading_id"])
    assert not old["consistency"]["consistent"]
    assert any("参数复评版本" in issue for issue in old["consistency"]["issues"])
    assert any("置信水平" in issue for issue in old["consistency"]["issues"])


def test_persistence_roundtrip(engine, readings, tmp_path):
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1", "fail_at_slices": ["2026-09-08"]})
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-1"})
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-INV-2", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    readings.create_reading("ANAL-STOCKOUT_RISK", "FORE-INVENTORY_WEEKLY")

    # 模拟重启：同一文件重新加载，模型版本、预测结果、复评记录完全不变
    reloaded = LedgerStore(tmp_path / "forecast_ledger.json")
    engine2 = ForecastExecutionEngine(ledger_store=reloaded, module_store=store)
    artifact = engine2.list_artifacts()[0]
    assert artifact["model_version"] == "FORE-INVENTORY_WEEKLY#v1"
    assert artifact["confidence_level"] == 0.9
    assert len(artifact["evaluations"]) == 2
    again = engine2.submit({**INVENTORY_REQ, "run_id": "RUN-INV-3", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    assert again["exec_mode"] == "cache_reuse"
    assert again["artifact"]["prediction"] == artifact["prediction"]


def test_forecast_entry_synced_with_artifact(engine):
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-A"})
    row = next(r for r in store.rows("forecast") if r.get("方案编号") == "FORE-INVENTORY_WEEKLY")
    assert row["模型版本"] == "FORE-INVENTORY_WEEKLY#v1"
    assert row["置信水平"] == "0.8"
    engine.submit({**INVENTORY_REQ, "run_id": "RUN-B", "parameters": {"置信水平": 0.9, "算法名称": "lightgbm-quantile"}})
    assert row["模型版本"] == "FORE-INVENTORY_WEEKLY#v1"  # 复评不改模型版本
    assert row["置信水平"] == "0.9"
