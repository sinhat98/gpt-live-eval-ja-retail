"""Regrade saved evidence without repeating an audio/text conversation."""

import argparse
import hashlib
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from voice_eval.retail.caller_audit import audit_caller
from voice_eval.retail.cli import ARTIFACTS
from voice_eval.retail.data import ROOT, save, task
from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.scoring import finalize, score


def rescore(path, *, caller_only=False):
    path = path.resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Select a trial inside artifacts/retail")
    result_path = path / "result.json"
    previous = json.loads(result_path.read_text())
    if previous.get("variant") == "digit-kana":
        os.environ["RETAIL_NUMBER_READING"] = "katakana-v1"
    else:
        os.environ.pop("RETAIL_NUMBER_READING", None)
    tid = previous["task_id"]
    mode = previous["mode"]
    t = task(tid)
    if previous.get("variant") in ("phone-sequence", "name-kana"):
        from voice_eval.retail.phone_sequence import project

        t = project(t, tid)
    if mode == "text":
        transcript = json.loads((path / "transcript.json").read_text())
        capture = path
        state = {"task_id": tid}
    else:
        transcript = next(path.glob("harness/**/conversation.transcript.txt")).read_text()
        scenario = json.loads((path / "inputs/scenarios.json").read_text())["scenarios"][0]
        state = scenario["application"]["initial_state"]
        if scenario["input"].get("context"):
            transcript = json.dumps(scenario["input"]["context"]["history"], ensure_ascii=False) + "\n" + transcript
        capture = path / "capture"
    client = OpenAI(max_retries=0, timeout=60)
    if caller_only:
        result = dict(previous)
        result["caller_audit"] = audit_caller(t, transcript, client)
        result["semantic"]["caller_validity"] = result["caller_audit"]["status"]
        result = finalize(result)
    else:
        with RetailExecutor(initial_state=state, task=t if tid == "control" else None) as e:
            for call in json.loads((capture / "executions.json").read_text()):
                e.execute(call["name"], call["arguments"], call_id=call["call_id"])
            if e.snapshot() != json.loads((capture / "final_state.json").read_text()):
                raise ValueError("Replay differs from recorded final state")
            result = dict(previous)
            result.update(score(e, t, transcript, client, partial=mode != "run" and mode != "text"))
    termination = previous.get("termination_reason")
    if termination is None and mode == "run":
        detail = next(path.glob("harness/**/conversation.result.json"), None)
        if detail:
            termination = json.loads(detail.read_text()).get("termination_reason")
    if termination == "duration_limit" or previous["status"] == "task_timeout":
        result.update(
            termination_reason=termination,
            task_reward=0.0,
            passed=False,
            status="caller_invalid" if result.get("caller_audit", {}).get("status") == "invalid" else "task_timeout",
        )
    sequence = len(list(path.glob("result.before_rescore-*.json"))) + 1
    save(path / f"result.before_rescore-{sequence}.json", previous)
    result["grading_revision"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "voice_eval/retail/scoring.py", ROOT / "voice_eval/retail/caller_audit.py")
    }
    save(result_path, result)
    return result


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--caller-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    print(rescore(args.trial, caller_only=args.caller_only)["status"])


if __name__ == "__main__":
    main()
