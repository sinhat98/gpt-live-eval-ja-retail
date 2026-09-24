"""Retail CLI, resumable sequential experiments, and artifact accounting."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from voice_eval.retail.data import DATA, PILOT, ROOT, build, save, task
from voice_eval.retail.executor import RetailExecutor

ARTIFACTS = ROOT / "artifacts/retail/name-phone-v1"
# The RUN harness accepts at most 600 seconds (max_duration_s le=600).
RUN_MAX_DURATION_SECONDS = 600


def validate():
    from run_harness.scenarios import validate_run_scenario
    from shared.scenarios import load_scenario_dataset

    for mode in ("crawl", "walk", "run"):
        ds = load_scenario_dataset(DATA / f"{mode}.json")
        for s in ds.scenarios:
            assert "ja" in s.tags
            if mode == "run":
                validate_run_scenario(s)
        print(f"{mode}: {len(ds.scenarios)} Japanese scenarios")
    with RetailExecutor(task_id="21") as e:
        assert len(e.schemas) == 15
    return True


def smoke():
    from voice_eval.retail.checkpoints import build_checkpoints

    build()
    build_checkpoints()
    validate()
    records = []
    for tid in PILOT:
        t = task(tid)
        with RetailExecutor(task_id=tid, task=t if tid == "control" else None) as e:
            for a in t["evaluation_criteria"]["actions"]:
                assert e.execute(a["name"], a["arguments"], call_id=a["action_id"])["ok"]
            r = e.evaluate()
            assert r["environment"]["db_check"]["db_match"] and r["replay_matches_snapshot"]
            records.append({"task_id": tid, "gold_replay": True})
    save(ARTIFACTS / "offline/smoke.json", {"execution_mode": "offline", "results": records})
    print("6 gold replays and 2 checkpoints passed; no model evaluation claimed")


def prepare(scenario=None):
    from crawl_harness.audio_cache import CallerAudioCache
    from shared.audio.effects import AudioRealismProcessor
    from shared.audio.gemini import INSTRUCTION, synthesize
    from shared.audio.pcm import write_mono_wav

    cache = CallerAudioCache(ARTIFACTS / "audio_cache")
    os.environ["GEMINI_USAGE_FILE"] = str(ARTIFACTS / "gemini_usage.jsonl")
    scenarios = json.loads((DATA / "crawl.json").read_text())["scenarios"]
    selected = [
        s for s in scenarios if not scenario or scenario in (s["id"], s["application"]["initial_state"]["task_id"])
    ]
    if not selected:
        raise ValueError("Unknown checkpoint scenario")
    for s in selected:
        params = {
            "text": s["input"]["text"],
            "model": os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
            "voice": os.getenv("GEMINI_TTS_VOICE", "Kore"),
            "sample_rate_hz": 24000,
        }
        audio = cache.load_or_create(
            s["id"],
            **params,
            synthesize=lambda p=params: synthesize(p["text"], model=p["model"], voice=p["voice"]),
        )
        for condition in ("clean", "noisy"):
            pcm = AudioRealismProcessor(condition, seed=41).process(audio.pcm)
            p = ARTIFACTS / "audio" / f"{s['id']}_{condition}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            write_mono_wav(p, pcm, 24000)
        save(
            ARTIFACTS / "audio" / f"{s['id']}.json",
            {
                **params,
                "instruction": INSTRUCTION,
                "source": "synthetic",
                "sha256": hashlib.sha256(audio.path.read_bytes()).hexdigest(),
                "duration_seconds": len(audio.pcm) / 48000,
                "listening_review": "pending",
                "path": str(audio.path),
            },
        )
        print(f"{s['id']}: {'reused' if audio.reused else 'generated'}")


def run_voice(mode, tid, output, variant, condition="clean"):
    from openai import OpenAI

    from voice_eval.retail.scoring import score

    scenarios = json.loads((DATA / f"{mode}.json").read_text())["scenarios"]
    if mode == "run":
        scenarios = json.loads((DATA / "all_run.json").read_text())["scenarios"]
    selected = next((s for s in scenarios if s["application"]["initial_state"]["task_id"] == tid), None)
    if selected is None:
        raise ValueError(f"No {mode} scenario for task {tid}")
    s = copy.deepcopy(selected)
    if variant in ("phone-sequence", "name-kana"):
        from voice_eval.retail.phone_sequence import TASK_USERS, project

        if tid not in TASK_USERS:
            raise ValueError(f"{variant} supports the six pilot scenarios")
        s = project(s, tid)
        s["application"]["initial_state"]["phone_profile"] = f"sequence-{tid}-v1"
    if variant == "name-kana":
        from voice_eval.retail.identity import READING_ONLY_MATCH

        s["application"]["initial_state"]["name_match"] = READING_ONLY_MATCH
    number_readings = None
    if variant == "digit-kana":
        from voice_eval.retail.number_reading import caller_projection

        s, number_readings = caller_projection(s)
    if mode == "walk":
        s["input"]["recordings"] = [
            {**r, "path": str((DATA / r["path"]).resolve())}
            for r in s["input"]["recordings"]
            if r["condition"] == condition
        ]
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if number_readings is not None:
        save(inputs / "number_readings.json", number_readings)
    save(inputs / "scenarios.json", {"schema_version": "1.0", "scenarios": [s]})
    for name in ("frontend.txt", "backend.txt"):
        (inputs / name).write_text((ROOT / f"prompts/retail/{variant}" / name).read_text())
    for name in ("tools.json", "facts.json", "source.json"):
        (inputs / name).write_bytes((DATA / name).read_bytes())
    if variant in ("digit-kana", "name-kana"):
        state = s["application"]["initial_state"]
        with RetailExecutor(initial_state=state, task=task(tid) if tid == "control" else None) as executor:
            save(
                inputs / "tools.json",
                [{"type": "function", **schema["function"], "strict": False} for schema in executor.schemas],
            )
    env = os.environ.copy()
    env.update(
        EVAL_DOMAIN="retail",
        EVAL_FRONTEND_PROMPT=str(inputs / "frontend.txt"),
        EVAL_BACKEND_PROMPT=str(inputs / "backend.txt"),
        EVAL_TOOLS_FILE=str(inputs / "tools.json"),
        EVAL_RESTAURANT_FACTS=str(inputs / "facts.json"),
        RETAIL_CAPTURE_DIR=str(output / "capture"),
    )
    cmd = [
        sys.executable,
        "-m",
        f"{mode}_harness.evaluate",
        "--data",
        str(inputs / "scenarios.json"),
        "--scenario",
        s["id"],
        "--assistant",
        "responses",
        "--concurrency",
        "1",
        "--results-dir",
        str(output / "harness"),
        "--run-name",
        f"{mode}-{tid}-{condition}",
    ]
    if mode == "crawl":
        cmd += [
            "--tts-model",
            os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
            "--audio-cache-dir",
            str(ARTIFACTS / "audio_cache"),
        ]
    if mode == "run":
        caller_model = os.getenv("OPENAI_CALLER_BACKEND_MODEL", "gpt-5.6-luna")
        (inputs / "run.toml").write_text("[simulation]\nsimulator_backend_model = " + json.dumps(caller_model) + "\n")
        cmd += [
            "--visualize",
            "--no-judge",
            "--max-duration-seconds",
            str(RUN_MAX_DURATION_SECONDS),
            "--config",
            str(inputs / "run.toml"),
        ]
    else:
        cmd += ["--response-timeout-seconds", "180"]
    result = {
        "mode": mode,
        "task_id": tid,
        "variant": variant,
        "condition": condition,
        "passed": False,
        "cost_usd": None,
        "confirmed_completion_ms": None,
    }
    if mode != "run":
        from crawl_harness.audio_cache import CallerAudioCache

        from voice_eval.retail.audio_review import validate_input_wav

        audio_path = (
            Path(s["input"]["recordings"][0]["path"])
            if mode == "walk"
            else CallerAudioCache(ARTIFACTS / "audio_cache").path_for(
                s["id"],
                text=s["input"]["text"],
                model=os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
                voice=os.getenv("GEMINI_TTS_VOICE", "Kore"),
                sample_rate_hz=24000,
            )
        )
        try:
            validate_input_wav(audio_path)
        except ValueError as exc:
            result.update(status="input_audio_invalid", error=str(exc))
            save(output / "result.json", result)
            return result
        result["input_audio_sha256"] = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        result["input_listening_review"] = "pending"
    try:
        with (output / "harness.log").open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Setup and grading need up to 240 seconds beyond the conversation itself.
                timeout=RUN_MAX_DURATION_SECONDS + 240 if mode == "run" else 420,
                check=False,
            )
        reports = list(output.glob("harness/*/results.json"))
        if proc.returncode or not reports:
            result["status"] = "infra_error"
        else:
            report = json.loads(reports[0].read_text())
            result["harness_diagnostic_only"] = True
            result["harness_metrics"] = [r.get("metrics", {}) for r in report.get("results", [])]
            if report.get("summary", {}).get("infrastructure_errors"):
                errors = [r.get("error", {}) for r in report.get("results", [])]
                result.update(
                    status="task_timeout"
                    if any(e.get("stage") == "response_timeout" for e in errors)
                    else "infra_error",
                    harness_errors=errors,
                )
                save(output / "result.json", result)
                return result
            transcripts = list(output.glob("harness/**/conversation.transcript.txt"))
            transcript = "\n".join(p.read_text() for p in transcripts)
            if s["input"].get("context"):
                transcript = json.dumps(s["input"]["context"]["history"], ensure_ascii=False) + "\n" + transcript
            calls = json.loads((output / "capture/executions.json").read_text())
            # Replay actual tool events in their recorded order, never captions.
            state = s["application"]["initial_state"]
            t = task(tid)
            if variant in ("phone-sequence", "name-kana"):
                from voice_eval.retail.phone_sequence import project

                t = project(t, tid)
            with RetailExecutor(initial_state=state, task=t if tid == "control" else None) as e:
                for c in calls:
                    e.execute(c["name"], c["arguments"], call_id=c["call_id"])
                try:
                    grade = score(e, t, transcript, OpenAI(max_retries=0, timeout=60), partial=mode != "run")
                    result.update(grade)
                except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
                    result.update(status="grader_error", error_type=type(exc).__name__)
                actual = json.loads((output / "capture/final_state.json").read_text())
                if e.snapshot() != actual:
                    result.update(status="grader_error", passed=False, replay_matches_snapshot=False)
            if report.get("summary", {}).get("infrastructure_errors"):
                result.update(status="infra_error", passed=False)
            # Preserve the harness timeout and simulator-validity classifications.
            for detail in output.glob("harness/**/conversation.result.json"):
                d = json.loads(detail.read_text())
                result["termination_reason"] = d.get("termination_reason")
                validity = (d.get("run_metadata") or {}).get("simulator_validity", {})
                if validity.get("status") == "invalid":
                    result.update(status="caller_invalid", passed=False)
                if d.get("termination_reason") in (
                    "max_duration",
                    "max_duration_exceeded",
                    "timeout",
                    "duration_limit",
                ):
                    result.update(
                        status="task_timeout" if result.get("status") != "caller_invalid" else "caller_invalid",
                        passed=False,
                        task_reward=0.0,
                    )
                if d.get("termination_reason") == "session_close_failed":
                    result.update(status="infra_error", passed=False)
    except subprocess.TimeoutExpired:
        result.update(status="infra_error", error_type="harness_timeout")
    except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
        result.update(status="infra_error", error_type=type(exc).__name__)
    save(output / "result.json", result)
    return result


def run(args):
    from voice_eval.retail.text import run_text

    if args.variant in ("digit-kana", "phone-sequence", "name-kana") and args.mode != "run":
        raise ValueError("This variant is a RUN-only controlled experiment")
    if args.variant == "digit-kana":
        os.environ["RETAIL_NUMBER_READING"] = "katakana-v1"
    else:
        os.environ.pop("RETAIL_NUMBER_READING", None)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing")
    if args.repeat != 1:
        raise ValueError("Retail repetitions require --new-attempt; --repeat is reserved for restaurant")
    if args.variant == "candidate" and not (ROOT / "prompts/retail/candidate/backend.txt").exists():
        raise ValueError("Candidate requires a documented baseline failure and a reviewed prompt change")
    tids = PILOT if args.suite or args.all else (args.scenario.removeprefix("retail-").removesuffix("-checkpoint"),)
    if args.mode in ("crawl", "walk") and (args.suite or args.all):
        tids = ("21", "22")
    for tid in tids:
        task(tid)
        for condition in ("clean", "noisy") if args.mode == "walk" else ("clean",):
            parent = ARTIFACTS / "live" / args.mode / args.variant / tid
            if args.mode == "walk":
                parent /= condition
            completed = list(parent.glob("attempt-*/result.json"))
            if completed and not args.new_attempt:
                print(
                    f"{args.mode}/{tid}/{condition}: preserved existing attempt (use --new-attempt for explicit repetition)"
                )
                continue
            if any(parent.glob("attempt-*")) and not completed and not args.new_attempt:
                raise ValueError("An unfinished attempt exists; inspect it before an explicit --new-attempt")
            index = max([int(p.name.split("-")[-1]) for p in parent.glob("attempt-*")], default=0) + 1
            output = parent / f"attempt-{index}"
            output.mkdir(parents=True)
            from voice_eval.retail.artifacts import freeze

            freeze(output)
            save(
                output / "manifest.json",
                {
                    "mode": args.mode,
                    "task_id": tid,
                    "variant": args.variant,
                    "condition": condition,
                    "source": json.loads((DATA / "source.json").read_text()),
                    "number_reading": os.getenv("RETAIL_NUMBER_READING"),
                    "max_duration_seconds": RUN_MAX_DURATION_SECONDS if args.mode == "run" else None,
                    "locks": {
                        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / "uv.lock", ROOT / "vendor/j_tau/uv.lock")
                    },
                    "settings": {
                        k: v
                        for k, v in os.environ.items()
                        if k.startswith(("OPENAI_", "GEMINI_")) and k.endswith(("MODEL", "VOICE", "EFFORT"))
                    },
                },
            )
            r = (
                run_text(tid, output, variant=args.variant)
                if args.mode == "text"
                else run_voice(args.mode, tid, output, args.variant, condition)
            )
            print(f"{args.mode}/{tid}/{condition}: {r['status']}", flush=True)
            if r["status"] in ("infra_error", "grader_error", "input_audio_invalid"):
                raise RuntimeError(f"Stopped after {r['status']}; inspect {output}")


def summarize():
    rows = []
    for p in sorted(ARTIFACTS.glob("live/**/result.json")):
        r = json.loads(p.read_text())
        rows.append(
            {k: r.get(k) for k in ("mode", "task_id", "variant", "condition", "status", "passed", "task_reward")}
            | {"path": str(p.relative_to(ROOT))}
        )
    valid = [r for r in rows if r["status"] in ("passed", "agent_failure", "task_timeout")]
    groups = []
    for mode, variant, condition in sorted({(r["mode"], r["variant"], r["condition"] or "clean") for r in rows}):
        selected = [
            r for r in rows if (r["mode"], r["variant"], r["condition"] or "clean") == (mode, variant, condition)
        ]
        assessed = [r for r in selected if r in valid]
        groups.append(
            {
                "mode": mode,
                "variant": variant,
                "condition": condition,
                "all_attempts": len(selected),
                "valid_attempts": len(assessed),
                "passed": sum(r["passed"] for r in assessed),
                "rate": sum(r["passed"] for r in assessed) / len(assessed) if assessed else None,
            }
        )
    save(
        ARTIFACTS / "summary.json",
        {
            "all_attempts": len(rows),
            "valid_attempts": len(valid),
            "passed": sum(r["passed"] for r in valid),
            "valid_pass_rate": None,
            "rate_note": "Use per-mode groups; partial and full-task assessments must not share a success rate.",
            "rows": rows,
            "groups": groups,
            "cost_usd": None,
        },
    )
    if rows:
        with (ARTIFACTS / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Retail experiment attempts",
        "",
        "All attempts are retained. Use per-mode groups; human review remains pending.",
        "",
        "| Mode | Task | Variant | Condition | Status | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        relative = str(Path(row["path"]).relative_to("artifacts/retail"))
        lines.append(
            f"| {row['mode']} | {row['task_id']} | {row['variant']} | {row['condition'] or 'clean'} | {row['status']} | [result]({relative}) |"
        )
    (ARTIFACTS / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} attempts, {len(valid)} valid, {sum(r['passed'] for r in valid)} passed")


def main(args):
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
                    "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
                    "language": "ja-JP",
                    "worker_python": (ROOT / "vendor/j_tau/.venv/bin/python").exists(),
                }
            )
        )
        validate()
    elif args.command == "prepare":
        prepare(args.scenario)
    elif args.command == "run":
        run(args)
    else:
        globals()[args.command]()
