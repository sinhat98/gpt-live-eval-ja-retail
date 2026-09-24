"""Synchronous ToolExecutor over a per-trial, isolated JSONL worker."""

from __future__ import annotations

import atexit
import copy
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RetailExecutor:
    def __init__(self, initial_state=None, facts=None, *, task_id=None, task=None, log_dir=None):
        initial_state = initial_state or {}
        self.executions = []
        self._ids = {}
        self._sequence = 0
        self._lock = threading.Lock()
        self._responses = queue.Queue()
        python = ROOT / "vendor/j_tau/.venv/bin/python"
        if not python.exists():
            raise RuntimeError("Run uv sync --project vendor/j_tau --frozen --python 3.12")
        env = os.environ.copy()
        if initial_state.get("phone_profile"):
            env["RETAIL_PHONE_PROFILE"] = initial_state["phone_profile"]
        env.pop("RETAIL_NAME_MATCH", None)
        if initial_state.get("name_match"):
            env["RETAIL_NAME_MATCH"] = initial_state["name_match"]
        env["TAU2_DATA_DIR"] = str(ROOT / "vendor/j_tau/data")
        log_dir = log_dir or os.getenv("RETAIL_CAPTURE_DIR")
        self._stderr = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._stderr = (Path(log_dir) / "worker.stderr.log").open("w")
        self.process = subprocess.Popen(
            [str(python), str(ROOT / "voice_eval/retail/worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr or subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        atexit.register(self.close)
        try:
            self.info = self.rpc("initialize", {"task_id": task_id or initial_state.get("task_id", "21"), "task": task})
            self.schemas = self.rpc("tool_schema")
            self._completion_db = self.rpc("snapshot")
            self.checkpoint_count = len(initial_state.get("checkpoint_calls", []))
            for call in initial_state.get("checkpoint_calls", []):
                self.execute(**call)
            self.initial_snapshot = self.snapshot()
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            for line in self.process.stdout:
                self._responses.put(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
            self._responses.put({"error": {"kind": "infra_error", "message": str(exc)}})
        finally:
            self._responses.put(None)

    def rpc(self, method, params=None):
        with self._lock:
            self._sequence += 1
            request = {"request_id": self._sequence, "method": method, "params": params or {}}
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._responses.get(timeout=60)
            except (BrokenPipeError, queue.Empty) as exc:
                raise RuntimeError("Retail worker unavailable or timed out") from exc
            if response is None or response.get("request_id") != self._sequence:
                raise RuntimeError("Retail worker exited or RPC request_id mismatch")
            if "error" in response:
                error = response["error"]
                cls = ValueError if error["kind"] == "business_error" else RuntimeError
                raise cls(error["message"])
            return response["result"]

    def execute(self, name, arguments, *, call_id):
        start = time.monotonic()
        output = self.rpc("execute", {"name": name, "arguments": arguments, "call_id": call_id})
        if call_id not in self._ids:
            row = {
                "name": name,
                "arguments": arguments,
                "call_id": call_id,
                "output": output,
                "status": "completed" if output["ok"] else "failed",
                "duration_ms": (time.monotonic() - start) * 1000,
            }
            self.executions.append(row)
            self._ids[call_id] = row
            if name.startswith(("modify_", "cancel_", "exchange_", "return_")):
                self._completion_db = self.rpc("snapshot")
        return output

    def completion_evidence_state(self):
        """Small observer-only projection; full snapshots remain authoritative."""
        users, orders = set(), set()
        for call in self.executions:
            args = call["arguments"]
            if "user_id" in args:
                users.add(args["user_id"])
            if "order_id" in args:
                orders.add(args["order_id"])

        def project(db):
            return {
                "users": {k: db["users"][k] for k in users if k in db["users"]},
                "orders": {k: db["orders"][k] for k in orders if k in db["orders"]},
            }

        return copy.deepcopy(project(self.initial_snapshot)), copy.deepcopy(project(self._completion_db))

    def snapshot(self):
        return self.rpc("snapshot")

    def evaluate(self):
        return self.rpc("evaluate")

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            capture = os.getenv("RETAIL_CAPTURE_DIR")
            if capture and process.poll() is None and hasattr(self, "initial_snapshot"):
                from voice_eval.retail.data import save

                target = Path(capture)
                save(target / "executions.json", self.executions)
                save(target / "initial_state.json", self.initial_snapshot)
                try:
                    save(target / "final_state.json", self.snapshot())
                except RuntimeError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass
            atexit.unregister(self.close)
            if self._stderr:
                self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
