"""
Server-side thread-safe telemetry store and PromptServer HTTP catch-up routes for OmniLoRA.

Prevents the frontend `OmniLoRALossTracker` widget from losing step metrics when the user's
PC enters idle mode, display sleep, browser tab throttling/freezing, or WebSocket disconnect
while Python/GPU training continues.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MAX_STORED_RUNS = 16
MAX_STEPS_PER_RUN = 50000
_DISK_FLUSH_INTERVAL_SEC = 10.0


def _default_state_file() -> Path:
    return Path(tempfile.gettempdir()) / "omni_lora_tracker_runs.json"


class OmniLoRATelemetryStore:
    """
    Thread-safe in-memory (with throttled atomic temp-file backup) store of OmniLoRA
    training runs and per-step metrics.
    """

    def __init__(self, state_file: Optional[Path] = None, max_runs: int = MAX_STORED_RUNS) -> None:
        self._lock = threading.RLock()
        self._max_runs = max(1, int(max_runs))
        self._explicit_state_file = state_file is not None
        self._state_file = state_file if state_file is not None else _default_state_file()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._run_order: List[str] = []
        self._deleted_run_ids: set[str] = set()
        self._last_flush_time: float = 0.0
        if self._explicit_state_file:
            self._load_from_disk()
        else:
            # Start each ComfyUI server process with a clean run store so stale runs
            # from previous sessions never stick around across restarts.
            self._flush_to_disk(force=True)

    def _load_from_disk(self) -> None:
        try:
            if not self._state_file.exists():
                return
            raw = self._state_file.read_text(encoding="utf-8")
            if not raw.strip():
                return
            parsed = json.loads(raw)
            runs_list = parsed.get("runs") if isinstance(parsed, dict) else None
            if not isinstance(runs_list, list):
                return
            for item in runs_list[-self._max_runs :]:
                if not isinstance(item, dict):
                    continue
                run_id = str(item.get("run_id") or "").strip()
                if not run_id or run_id in self._deleted_run_ids:
                    continue
                steps_by_num: Dict[int, Dict[str, Any]] = {}
                for pt in item.get("steps") or []:
                    if isinstance(pt, dict) and pt.get("step") is not None:
                        try:
                            step_num = int(pt["step"])
                        except (TypeError, ValueError):
                            continue
                        steps_by_num[step_num] = pt
                # Any run loaded from a previous server process that was still marked "active"
                # cannot still be executing in this fresh process.
                status = str(item.get("status") or "completed")
                if status == "active":
                    status = "interrupted"
                self._runs[run_id] = {
                    "run_id": run_id,
                    "run_title": str(item.get("run_title") or run_id),
                    "node_id": str(item.get("node_id") or ""),
                    "max_steps": int(item.get("max_steps") or 1000),
                    "lr": float(item.get("lr") or 0.0),
                    "optimizer": str(item.get("optimizer") or ""),
                    "model_id": str(item.get("model_id") or ""),
                    "rank": int(item.get("rank") or 0),
                    "alpha": float(item.get("alpha") or 0.0),
                    "prior_steps": int(item.get("prior_steps") or 0),
                    "status": status,
                    "started_at": float(item.get("started_at") or 0.0),
                    "updated_at": float(item.get("updated_at") or 0.0),
                    "final_step": int(item.get("final_step") or (max(steps_by_num.keys()) if steps_by_num else 0)),
                    "final_loss": item.get("final_loss"),
                    "final_ema_loss": item.get("final_ema_loss"),
                    "total_time_sec": item.get("total_time_sec"),
                    "reason": item.get("reason"),
                    "_steps": steps_by_num,
                }
                self._run_order.append(run_id)
        except Exception:
            # Never block startup if temp cache is unreadable or corrupt
            pass

    def _flush_to_disk(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_flush_time) < _DISK_FLUSH_INTERVAL_SEC:
            return
        self._last_flush_time = now
        try:
            serialized_runs: List[Dict[str, Any]] = []
            for rid in self._run_order[-self._max_runs :]:
                run = self._runs.get(rid)
                if run is None:
                    continue
                serialized_runs.append(self._serialize_run(run, after_step=0))
            payload = {"saved_at": now, "runs": serialized_runs}
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._state_file.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_path, self._state_file)
        except Exception:
            pass

    def _evict_if_needed(self) -> None:
        while len(self._run_order) > self._max_runs:
            oldest = self._run_order.pop(0)
            self._runs.pop(oldest, None)

    def _ensure_run_locked(self, run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        run = self._runs.get(run_id)
        now = time.time()
        if run is None:
            run = {
                "run_id": run_id,
                "run_title": str(payload.get("run_title") or run_id),
                "node_id": str(payload.get("node_id") or ""),
                "max_steps": int(payload.get("max_steps") or 1000),
                "lr": float(payload.get("lr") or 0.0),
                "optimizer": str(payload.get("optimizer") or ""),
                "model_id": str(payload.get("model_id") or ""),
                "rank": int(payload.get("rank") or 0),
                "alpha": float(payload.get("alpha") or 0.0),
                "prior_steps": int(payload.get("prior_steps") or 0),
                "status": "active",
                "started_at": now,
                "updated_at": now,
                "final_step": 0,
                "final_loss": None,
                "final_ema_loss": None,
                "total_time_sec": None,
                "reason": None,
                "_steps": {},
            }
            self._runs[run_id] = run
            self._run_order.append(run_id)
            self._evict_if_needed()
        else:
            if payload.get("run_title"):
                run["run_title"] = str(payload["run_title"])
            if payload.get("node_id"):
                run["node_id"] = str(payload["node_id"])
            if payload.get("max_steps") is not None:
                run["max_steps"] = int(payload["max_steps"])
            run["updated_at"] = now
        return run

    def start_run(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        with self._lock:
            self._deleted_run_ids.discard(run_id)
            run = self._ensure_run_locked(run_id, payload)
            run["status"] = "active"
            run["final_step"] = 0
            run["final_loss"] = None
            run["final_ema_loss"] = None
            run["total_time_sec"] = None
            run["reason"] = None
            run["_steps"] = {}
            if payload.get("lr") is not None:
                run["lr"] = float(payload["lr"])
            if payload.get("optimizer"):
                run["optimizer"] = str(payload["optimizer"])
            if payload.get("model_id"):
                run["model_id"] = str(payload["model_id"])
            if payload.get("rank") is not None:
                run["rank"] = int(payload["rank"])
            if payload.get("alpha") is not None:
                run["alpha"] = float(payload["alpha"])
            if payload.get("prior_steps") is not None:
                run["prior_steps"] = int(payload["prior_steps"])
            self._flush_to_disk(force=True)

    def record_step(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        try:
            step_num = int(payload.get("step") or 0)
        except (TypeError, ValueError):
            return
        if step_num <= 0:
            return
        now = time.time()
        with self._lock:
            if run_id in self._deleted_run_ids:
                return
            run = self._ensure_run_locked(run_id, payload)
            run["status"] = "active"
            started_at = float(run.get("started_at") or now)
            step_record = dict(payload)
            if "wall_time" not in step_record:
                step_record["wall_time"] = round(now, 3)
            if "rel_time_sec" not in step_record:
                step_record["rel_time_sec"] = round(max(0.0, now - started_at), 3)
            steps_map: Dict[int, Dict[str, Any]] = run["_steps"]
            steps_map[step_num] = step_record
            if len(steps_map) > MAX_STEPS_PER_RUN:
                oldest_step = min(steps_map.keys())
                steps_map.pop(oldest_step, None)
            if step_num > int(run.get("final_step") or 0):
                run["final_step"] = step_num
            self._flush_to_disk(force=False)

    def complete_run(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        with self._lock:
            if run_id in self._deleted_run_ids:
                return
            run = self._ensure_run_locked(run_id, payload)
            run["status"] = "completed"
            if payload.get("final_step") is not None:
                run["final_step"] = int(payload["final_step"])
            if payload.get("final_loss") is not None:
                run["final_loss"] = float(payload["final_loss"])
            if payload.get("final_ema_loss") is not None:
                run["final_ema_loss"] = float(payload["final_ema_loss"])
            if payload.get("total_time_sec") is not None:
                run["total_time_sec"] = float(payload["total_time_sec"])
            self._flush_to_disk(force=True)

    def interrupt_run(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            return
        with self._lock:
            if run_id in self._deleted_run_ids:
                return
            run = self._ensure_run_locked(run_id, payload)
            run["status"] = "interrupted"
            if payload.get("step") is not None:
                run["final_step"] = max(int(run.get("final_step") or 0), int(payload["step"]))
            if payload.get("reason") is not None:
                run["reason"] = str(payload["reason"])
            self._flush_to_disk(force=True)

    @staticmethod
    def _serialize_run(run: Dict[str, Any], after_step: int = 0) -> Dict[str, Any]:
        steps_map: Dict[int, Dict[str, Any]] = run.get("_steps") or {}
        filtered_steps = [
            steps_map[s]
            for s in sorted(steps_map.keys())
            if s > after_step
        ]
        return {
            "run_id": run["run_id"],
            "run_title": run["run_title"],
            "node_id": run.get("node_id", ""),
            "max_steps": run["max_steps"],
            "lr": run.get("lr", 0.0),
            "optimizer": run.get("optimizer", ""),
            "model_id": run.get("model_id", ""),
            "rank": run.get("rank", 0),
            "alpha": run.get("alpha", 0.0),
            "prior_steps": run.get("prior_steps", 0),
            "status": run.get("status", "active"),
            "started_at": run.get("started_at", 0.0),
            "updated_at": run.get("updated_at", 0.0),
            "final_step": run.get("final_step", 0),
            "final_loss": run.get("final_loss"),
            "final_ema_loss": run.get("final_ema_loss"),
            "total_time_sec": run.get("total_time_sec"),
            "reason": run.get("reason"),
            "steps": filtered_steps,
        }

    def get_snapshot(
        self,
        run_id: Optional[str] = None,
        after_step: int = 0,
        cursors: Optional[Dict[str, int]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        allowed_nodes = {str(n).strip() for n in (node_ids or []) if str(n).strip()}
        with self._lock:
            out_runs: List[Dict[str, Any]] = []
            if run_id:
                run = self._runs.get(run_id)
                if run is not None:
                    if not allowed_nodes or str(run.get("node_id") or "") in allowed_nodes:
                        cursor = (
                            int(cursors[run_id])
                            if cursors and run_id in cursors
                            else int(after_step)
                        )
                        out_runs.append(self._serialize_run(run, after_step=cursor))
            else:
                for rid in self._run_order:
                    run = self._runs.get(rid)
                    if run is None:
                        continue
                    if allowed_nodes and str(run.get("node_id") or "") not in allowed_nodes:
                        continue
                    cursor = (
                        int(cursors[rid])
                        if cursors and rid in cursors
                        else int(after_step)
                    )
                    out_runs.append(self._serialize_run(run, after_step=cursor))

            return {
                "server_time": time.time(),
                "runs": out_runs,
            }

    def delete_run(self, run_id: str) -> bool:
        target = str(run_id or "").strip()
        if not target:
            return False
        with self._lock:
            self._deleted_run_ids.add(target)
            existed = self._runs.pop(target, None) is not None
            if target in self._run_order:
                self._run_order = [rid for rid in self._run_order if rid != target]
            self._flush_to_disk(force=True)
            return existed

    def clear(self, keep_active: bool = False) -> int:
        with self._lock:
            if keep_active:
                to_remove = [
                    rid
                    for rid, r in self._runs.items()
                    if str(r.get("status") or "") != "active"
                ]
            else:
                to_remove = list(self._runs.keys())
            for rid in to_remove:
                self._deleted_run_ids.add(rid)
                self._runs.pop(rid, None)
            if keep_active:
                removed_set = set(to_remove)
                self._run_order = [rid for rid in self._run_order if rid not in removed_set]
            else:
                self._run_order.clear()
            self._flush_to_disk(force=True)
            return len(to_remove)

    def request_checkpoint(self, node_id: Optional[str] = None) -> None:
        with self._lock:
            self._manual_checkpoint_requested = True
            self._manual_checkpoint_node_id = str(node_id or "").strip()

    def consume_checkpoint_request(self, node_id: Optional[str] = None) -> bool:
        with self._lock:
            if not getattr(self, "_manual_checkpoint_requested", False):
                return False
            req_node = getattr(self, "_manual_checkpoint_node_id", "")
            if req_node and node_id and str(node_id).strip() != req_node:
                return False
            self._manual_checkpoint_requested = False
            self._manual_checkpoint_node_id = ""
            return True


TRACKER_STORE = OmniLoRATelemetryStore()


def _parse_cursors_param(raw: str) -> Dict[str, int]:
    cursors: Dict[str, int] = {}
    if not raw:
        return cursors
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        cursors[str(k)] = max(0, int(v))
                    except (TypeError, ValueError):
                        pass
                return cursors
        except Exception:
            pass
    for pair in stripped.split(","):
        if ":" not in pair:
            continue
        rid, val = pair.rsplit(":", 1)
        rid = rid.strip()
        if not rid:
            continue
        try:
            cursors[rid] = max(0, int(val.strip()))
        except (TypeError, ValueError):
            pass
    return cursors


def register_omni_lora_routes(prompt_server: Optional[Any] = None) -> bool:
    """
    Register `GET /omni_lora/metrics`, `POST /omni_lora/checkpoint`, and `POST /omni_lora/clear`
    on ComfyUI's `PromptServer` so the frontend tracker widget can recover missed steps, delete
    or clear runs on the server, and trigger an on-demand checkpoint during training.
    """
    if prompt_server is None:
        try:
            import server  # type: ignore

            if hasattr(server, "PromptServer"):
                prompt_server = getattr(server.PromptServer, "instance", None)
        except Exception:
            return False

    if prompt_server is None:
        return False

    if getattr(prompt_server, "_omni_lora_routes_registered", False):
        return True

    routes = getattr(prompt_server, "routes", None)
    if routes is None:
        return False

    try:
        from aiohttp import web  # type: ignore
    except Exception:
        return False

    async def handle_get_metrics(request: Any) -> Any:
        q = getattr(request, "rel_url", None)
        query = q.query if q is not None else getattr(request, "query", {})
        raw_delete = str(query.get("delete_run", "") or "").strip()
        if raw_delete:
            for rid in raw_delete.split(","):
                if rid.strip():
                    TRACKER_STORE.delete_run(rid.strip())
        raw_clear = str(query.get("clear_runs", "") or "").strip().lower()
        if raw_clear in ("1", "true", "yes", "all"):
            TRACKER_STORE.clear(keep_active=False)
        elif raw_clear == "inactive":
            TRACKER_STORE.clear(keep_active=True)

        run_id = query.get("run_id") or None
        raw_after = query.get("after_step", "0")
        try:
            after_step = max(0, int(raw_after))
        except (TypeError, ValueError):
            after_step = 0
        cursors = _parse_cursors_param(query.get("cursors", ""))
        raw_nodes = query.get("node_id", "")
        node_ids = [n.strip() for n in str(raw_nodes).split(",") if n.strip()] if raw_nodes else None
        if str(query.get("save_checkpoint", "")).strip() in ("1", "true", "yes"):
            TRACKER_STORE.request_checkpoint(node_ids[0] if node_ids else None)
        snapshot = TRACKER_STORE.get_snapshot(
            run_id=run_id,
            after_step=after_step,
            cursors=cursors,
            node_ids=node_ids,
        )
        return web.json_response(
            snapshot,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )

    async def handle_post_checkpoint(request: Any) -> Any:
        node_id = None
        try:
            data = await request.json()
            if isinstance(data, dict):
                node_id = data.get("node_id")
        except Exception:
            pass
        TRACKER_STORE.request_checkpoint(node_id)
        return web.json_response(
            {"status": "queued", "message": "Checkpoint will be saved at the end of the current step."},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    async def handle_post_clear(request: Any) -> Any:
        run_id = None
        keep_active = False
        try:
            data = await request.json()
            if isinstance(data, dict):
                run_id = data.get("run_id")
                keep_active = bool(data.get("keep_active", False))
        except Exception:
            pass
        if run_id:
            removed = 1 if TRACKER_STORE.delete_run(str(run_id)) else 0
        else:
            removed = TRACKER_STORE.clear(keep_active=keep_active)
        return web.json_response(
            {"status": "cleared", "removed": removed},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    try:
        routes.get("/omni_lora/metrics")(handle_get_metrics)
        routes.post("/omni_lora/checkpoint")(handle_post_checkpoint)
        routes.post("/omni_lora/clear")(handle_post_clear)

        # If PromptServer.add_routes() has already been executed before this module was imported
        # (e.g. hot-reload), also attach directly to an unfrozen aiohttp app router if needed.
        app = getattr(prompt_server, "app", None)
        router = getattr(app, "router", None) if app is not None else None
        if router is not None and not getattr(router, "frozen", False):
            existing_paths = {
                getattr(getattr(r, "resource", None), "canonical", None)
                for r in router.routes()
            }
            if existing_paths and "/omni_lora/metrics" not in existing_paths:
                router.add_get("/omni_lora/metrics", handle_get_metrics)
            if existing_paths and "/omni_lora/checkpoint" not in existing_paths:
                router.add_post("/omni_lora/checkpoint", handle_post_checkpoint)
            if existing_paths and "/omni_lora/clear" not in existing_paths:
                router.add_post("/omni_lora/clear", handle_post_clear)

        setattr(prompt_server, "_omni_lora_routes_registered", True)
        return True
    except Exception as exc:
        logging.debug(f"[OmniLoRA] Could not register PromptServer metrics route: {exc}")
        return False

