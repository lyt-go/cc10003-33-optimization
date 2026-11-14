"""预测口径执行单元：artifact identity、复用/训练/复评/续跑策略与 run_ledger 台账。

执行单元由 target_key、feature_snapshot、train_span、forecast_span 组成 artifact identity，
sha256 即 artifact_sha256。围绕 identity 的复用与版本策略：

- identity 未变且参数区未变 -> cache_reuse：直接读已有预测结果，不训练；
- 特征快照或训练/预测区间变化 -> train_new：产出新 artifact 与新模型版本；
- 仅参数区变化 -> parameter_replay：只追加 evaluation_record，模型版本不变；
- 训练切片失败 -> 同一 run_id 续跑（resumed_slice）：从 checkpoint_seq 继续，
  成功切片不重训、不进入新 artifact，整个 run 只落一份产物。

并行任务各自带 input_snapshot 落台账，但共享同一份 artifact（按 sha 定位）。
预测中值由 identity 确定性生成，保证刷新/重进后结果一致。
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.ledger_store import LedgerStore, ledger as default_ledger

EXEC_MODES = ("cache_reuse", "train_new", "parameter_replay", "resumed_slice")

SPAN_SPLIT = re.compile(r"\s*(?:~|～|至|—|–)\s*")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_span(span: dict[str, Any]) -> dict[str, str]:
    try:
        start = date.fromisoformat(str(span.get("start", "")).strip())
        end = date.fromisoformat(str(span.get("end", "")).strip())
    except ValueError as exc:
        raise ValueError(f"区间日期必须是 YYYY-MM-DD：{span}") from exc
    if end < start:
        raise ValueError(f"区间结束日期不能早于开始日期：{span}")
    return {"start": start.isoformat(), "end": end.isoformat()}


def normalize_features(snapshot: dict[str, Any]) -> dict[str, Any]:
    fields = sorted({str(f).strip() for f in (snapshot or {}).get("fields") or [] if str(f).strip()})
    snapshot_id = str((snapshot or {}).get("snapshot_id") or "").strip()
    if not fields:
        raise ValueError("feature_snapshot.fields 不能为空")
    if not snapshot_id:
        raise ValueError("feature_snapshot.snapshot_id 不能为空")
    return {"fields": fields, "snapshot_id": snapshot_id}


def build_identity(
    target_key: Any,
    feature_snapshot: dict[str, Any],
    train_span: dict[str, Any],
    forecast_span: dict[str, Any],
) -> dict[str, Any]:
    """组装 artifact identity：target_key + feature_snapshot + train_span + forecast_span。"""
    key = str(target_key or "").strip()
    if not key:
        raise ValueError("target_key 不能为空")
    return {
        "target_key": key,
        "feature_snapshot": normalize_features(feature_snapshot),
        "train_span": normalize_span(train_span),
        "forecast_span": normalize_span(forecast_span),
    }


def artifact_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def span_days(span: dict[str, str]) -> list[str]:
    """训练区间按日切片（含首尾），seq 即相对起始日的偏移。"""
    start = date.fromisoformat(span["start"])
    end = date.fromisoformat(span["end"])
    days: list[str] = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def parse_span_text(text: Any) -> dict[str, str] | None:
    """把方案行里的「2026-09-01~2026-10-01」解析成 span 结构。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in SPAN_SPLIT.split(raw) if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return normalize_span({"start": parts[0], "end": parts[1]})
    except ValueError:
        return None


def _median_series(sha: str, target_key: str, forecast_span: dict[str, str]) -> list[dict[str, Any]]:
    """预测中值按 identity 确定性生成：同一 artifact 任何时候读都一致。"""
    series = []
    for day in span_days(forecast_span):
        digest = hashlib.sha256(f"{sha}|{target_key}|{day}|median".encode("utf-8")).hexdigest()
        series.append({"date": day, "median": 50 + int(digest[:8], 16) % 400})
    return series


def _eval_metrics(sha: str, parameters: dict[str, Any]) -> dict[str, float]:
    """参数复评指标按 (artifact, 参数区) 确定性生成，复评可重现。"""
    digest = hashlib.sha256(f"{sha}|{_canonical(parameters)}|eval".encode("utf-8")).hexdigest()
    return {
        "mape": round(3 + (int(digest[8:16], 16) % 700) / 100, 2),
        "coverage": round(0.7 + (int(digest[16:24], 16) % 290) / 1000, 3),
    }


def _version_no(model_version: str) -> int:
    try:
        return int(str(model_version).rsplit("#v", 1)[1])
    except (IndexError, ValueError):
        return 0


class ForecastExecutionEngine:
    """执行单元引擎：提交执行、续跑、复评，并维护台账与产物的一致性。"""

    def __init__(self, ledger_store: LedgerStore | None = None, module_store: Any = None) -> None:
        self.ledger = ledger_store or default_ledger
        self.modules = module_store

    # ---------- 查询 ----------
    def list_runs(self, target_key: str | None = None) -> list[dict[str, Any]]:
        runs = self.ledger.all("runs")
        if target_key:
            runs = [r for r in runs if r.get("target_key") == target_key]
        return sorted(runs, key=lambda r: (r.get("updated_at"), r.get("run_id")), reverse=True)

    def list_artifacts(self, target_key: str | None = None) -> list[dict[str, Any]]:
        artifacts = self.ledger.all("artifacts")
        if target_key:
            artifacts = [a for a in artifacts if a.get("target_key") == target_key]
        return sorted(artifacts, key=lambda a: (_version_no(a.get("model_version", "")), a.get("created_at")))

    def get_artifact(self, sha: str) -> dict[str, Any] | None:
        artifact = self.ledger.get("artifacts", sha)
        if artifact is None:
            return None
        artifact["evaluation_records"] = sorted(
            [e for e in self.ledger.all("evaluations") if e.get("artifact_sha256") == sha],
            key=lambda e: e.get("eval_version", ""),
        )
        artifact["analysis_readings"] = sorted(
            [r for r in self.ledger.all("readings") if r.get("artifact_sha256") == sha],
            key=lambda r: r.get("reading_id", ""),
        )
        return artifact

    def latest_artifact(self, target_key: str) -> dict[str, Any] | None:
        artifacts = self.list_artifacts(target_key)
        return artifacts[-1] if artifacts else None

    # ---------- 提交执行 ----------
    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        identity = build_identity(
            request.get("target_key"),
            request.get("feature_snapshot"),
            request.get("train_span"),
            request.get("forecast_span"),
        )
        sha = artifact_sha256(identity)
        run_id = str(request.get("run_id") or "").strip() or f"RUN-{uuid.uuid4().hex[:12]}"

        existing = self.ledger.get("runs", run_id)
        if existing is not None:
            if existing.get("identity") != identity:
                raise ValueError(f"run_id {run_id} 已绑定其他执行单元，不能复用该 run_id")
            if existing.get("status") == "succeeded":
                # 幂等：相同 run_id 重复提交直接返回原结果，不产生重复 artifact
                return self._result(existing, "相同 run_id 的执行已落账，直接返回原结果（幂等，不重复生成产物）")
            return self._resume(existing, request)

        artifact = self.ledger.get("artifacts", sha)
        params = request.get("parameters")
        if artifact is None:
            return self._train_new(self._new_run(run_id, identity, sha, request, "train_new"), request)
        if params is None or _canonical(params) == _canonical(self._current_parameters(artifact)):
            return self._cache_reuse(self._new_run(run_id, identity, sha, request, "cache_reuse"), artifact)
        return self._parameter_replay(self._new_run(run_id, identity, sha, request, "parameter_replay"), artifact)

    # ---------- 四种执行方式 ----------
    def _train_new(self, run: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        return self._run_slices(run, request, mode="train_new", attempt_no=1, start_seq=0)

    def _resume(self, run: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        if request.get("parameters") is not None:
            run["parameters"] = request["parameters"]
        if request.get("input_snapshot") is not None:
            run["input_snapshot"] = request["input_snapshot"]
        run["exec_mode"] = "resumed_slice"
        attempt_no = len(run.get("attempts") or []) + 1
        return self._run_slices(run, request, mode="resumed_slice", attempt_no=attempt_no, start_seq=int(run.get("checkpoint_seq") or 0))

    def _cache_reuse(self, run: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
        run.update({
            "status": "succeeded",
            "checkpoint_seq": run["total_slices"],
            "model_version": artifact["model_version"],
            "eval_version": artifact.get("latest_eval_version"),
            "recompute_scope": None,
            "updated_at": _now(),
        })
        run["attempts"].append({"mode": "cache_reuse", "attempt": 1, "status": "succeeded", "finished_at": _now()})
        self.ledger.put("runs", run["run_id"], run)
        return self._result(run, "artifact identity 未变化，直接复用已有预测结果（cache_reuse，未触发训练）", artifact)

    def _parameter_replay(self, run: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
        record = self._append_evaluation(artifact, run, note="参数复评：仅参数区变化，不重训")
        self.ledger.put("artifacts", artifact["artifact_sha256"], artifact)
        run.update({
            "status": "succeeded",
            "checkpoint_seq": run["total_slices"],
            "model_version": artifact["model_version"],
            "eval_version": record["eval_version"],
            "recompute_scope": None,
            "updated_at": _now(),
        })
        run["attempts"].append({"mode": "parameter_replay", "attempt": 1, "status": "succeeded", "finished_at": _now()})
        self.ledger.put("runs", run["run_id"], run)
        self._sync_forecast_entry(artifact)
        return self._result(
            run,
            f"仅参数区变化，已追加参数复评记录 {record['eval_version']}，模型版本 {artifact['model_version']} 不变",
            artifact,
        )

    # ---------- 切片训练与断点续跑 ----------
    def _run_slices(self, run: dict[str, Any], request: dict[str, Any], *, mode: str, attempt_no: int, start_seq: int) -> dict[str, Any]:
        days = span_days(run["identity"]["train_span"])
        fail_at = self._normalize_fail_slices(request.get("fail_at_slices"), days)
        run["attempts"].append({"mode": mode, "attempt": attempt_no, "started_at": _now()})

        for seq in range(start_seq, len(days)):
            if seq in fail_at:
                run["status"] = "failed"
                run["checkpoint_seq"] = seq
                run["attempts"][-1].update({"status": "failed", "finished_at": _now(), "failed_slice": days[seq]})
                run["updated_at"] = _now()
                self.ledger.put("runs", run["run_id"], run)
                return self._result(
                    run,
                    f"切片 {days[seq]} 训练失败，checkpoint_seq={seq} 已落账；修复后用同一 run_id 续跑即可从断点继续",
                )
            run["completed_slices"].append(seq)
            run["slice_attempts"][str(seq)] = attempt_no
            run["checkpoint_seq"] = seq + 1

        artifact = self._commit_artifact(run, days, start_seq=start_seq, mode=mode)
        run["status"] = "succeeded"
        run["model_version"] = artifact["model_version"]
        run["eval_version"] = artifact.get("latest_eval_version")
        run["recompute_scope"] = artifact.get("recompute_scope")
        run["attempts"][-1].update({"status": "succeeded", "finished_at": _now()})
        run["updated_at"] = _now()
        self.ledger.put("runs", run["run_id"], run)
        self._sync_forecast_entry(artifact)
        if mode == "resumed_slice":
            message = (
                f"从 checkpoint_seq={start_seq} 续跑成功（resumed_slice）："
                f"0~{start_seq - 1} 成功切片未重训、未进入新 artifact，模型版本 {artifact['model_version']}"
            )
        else:
            message = f"特征快照或训练/预测区间变化，已重新训练并生成新模型版本 {artifact['model_version']}（train_new）"
        return self._result(run, message, artifact)

    def _commit_artifact(self, run: dict[str, Any], days: list[str], *, start_seq: int, mode: str) -> dict[str, Any]:
        """训练完成才提交产物；同一 sha 已存在则只登记 run_id，绝不生成第二份。"""
        sha = run["artifact_sha256"]
        artifact = self.ledger.get("artifacts", sha)
        if artifact is None:
            version_no = 1 + max(
                (_version_no(a.get("model_version", "")) for a in self.ledger.all("artifacts") if a.get("target_key") == run["target_key"]),
                default=0,
            )
            artifact = {
                "artifact_sha256": sha,
                "target_key": run["target_key"],
                "predicted_target": run.get("predicted_target"),
                "identity": run["identity"],
                "model_version": f"{run['target_key']}#v{version_no}",
                "parameters": run.get("parameters") or {},
                "train_span": run["identity"]["train_span"],
                "forecast_span": run["identity"]["forecast_span"],
                "slices": [
                    {"seq": i, "date": days[i], "trained_in_attempt": run["slice_attempts"].get(str(i), 1)}
                    for i in range(len(days))
                ],
                "prediction": {"median": _median_series(sha, run["target_key"], run["identity"]["forecast_span"])},
                "evaluations": [],
                "latest_eval_version": None,
                "confidence_level": None,
                "recompute_scope": {"from": days[start_seq], "to": days[-1], "mode": mode},
                "run_ids": [],
                "created_at": _now(),
                "updated_at": _now(),
            }
            record = self._append_evaluation(artifact, run, note="训练产出评估")
            artifact["latest_eval_version"] = record["eval_version"]
        if run["run_id"] not in artifact["run_ids"]:
            artifact["run_ids"].append(run["run_id"])
        artifact["updated_at"] = _now()
        self.ledger.put("artifacts", sha, artifact)
        return artifact

    def _append_evaluation(self, artifact: dict[str, Any], run: dict[str, Any], *, note: str) -> dict[str, Any]:
        seq = len(artifact.get("evaluations") or []) + 1
        eval_version = f"EV-{artifact['artifact_sha256'][:8]}-{seq:02d}"
        parameters = run.get("parameters") or {}
        record = {
            "eval_version": eval_version,
            "artifact_sha256": artifact["artifact_sha256"],
            "model_version": artifact["model_version"],
            "run_id": run["run_id"],
            "parameters": parameters,
            "confidence_level": parameters.get("置信水平"),
            "metrics": _eval_metrics(artifact["artifact_sha256"], parameters),
            "note": note,
            "created_at": _now(),
        }
        self.ledger.put("evaluations", eval_version, record)
        artifact.setdefault("evaluations", []).append(eval_version)
        artifact["latest_eval_version"] = eval_version
        artifact["confidence_level"] = record["confidence_level"]
        artifact["updated_at"] = _now()
        return record

    # ---------- 内部工具 ----------
    def _new_run(self, run_id: str, identity: dict[str, Any], sha: str, request: dict[str, Any], mode: str) -> dict[str, Any]:
        days = span_days(identity["train_span"])
        return {
            "run_id": run_id,
            "task_id": str(request.get("task_id") or run_id),
            "target_key": identity["target_key"],
            "predicted_target": request.get("predicted_target"),
            "identity": identity,
            "artifact_sha256": sha,
            "exec_mode": mode,
            "status": "running",
            "checkpoint_seq": 0,
            "total_slices": len(days),
            "completed_slices": [],
            "slice_attempts": {},
            "input_snapshot": request.get("input_snapshot") or {},
            "parameters": request.get("parameters") or {},
            "model_version": None,
            "eval_version": None,
            "recompute_scope": None,
            "attempts": [],
            "created_at": _now(),
            "updated_at": _now(),
        }

    @staticmethod
    def _normalize_fail_slices(raw: Any, days: list[str]) -> set[int]:
        """fail_at_slices 支持日期（2026-09-08）或序号（7），用于演练/测试故障注入。"""
        seqs: set[int] = set()
        for item in raw or []:
            text = str(item).strip()
            if text in days:
                seqs.add(days.index(text))
                continue
            try:
                seqs.add(int(text))
            except ValueError:
                raise ValueError(f"fail_at_slices 无法识别：{text}（应为训练区间内的日期或切片序号）")
        return seqs

    def _current_parameters(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """当前生效参数区：以最新一条 evaluation_record 为准（复评不改产物本体）。"""
        latest = self.ledger.get("evaluations", artifact.get("latest_eval_version") or "")
        if latest and latest.get("parameters") is not None:
            return latest["parameters"]
        return artifact.get("parameters") or {}

    def _result(self, run: dict[str, Any], message: str, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
        if artifact is None and run.get("artifact_sha256"):
            artifact = self.ledger.get("artifacts", run["artifact_sha256"])
        return {
            "ok": run.get("status") == "succeeded",
            "exec_mode": run.get("exec_mode"),
            "run": run,
            "artifact": artifact,
            "message": message,
        }

    def _sync_forecast_entry(self, artifact: dict[str, Any]) -> None:
        """把最新模型版本/置信水平回写到预测方案行，列表页与产物保持一致。"""
        if self.modules is None:
            return
        for row in self.modules.rows("forecast"):
            if row.get("方案编号") == artifact.get("target_key"):
                row["模型版本"] = artifact.get("model_version")
                if artifact.get("confidence_level") is not None:
                    row["置信水平"] = str(artifact["confidence_level"])


def reconcile_forecast_entries(module_store: Any, ledger_store: LedgerStore) -> None:
    """启动时按台账回写方案行的模型版本/置信水平：重启后刷新、重进都一致。"""
    engine = ForecastExecutionEngine(ledger_store=ledger_store, module_store=module_store)
    for artifact in engine.list_artifacts():
        engine._sync_forecast_entry(artifact)
