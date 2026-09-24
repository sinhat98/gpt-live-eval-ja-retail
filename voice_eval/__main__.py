"""Run with `uv run python -m voice_eval --help`. No model calls by default."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from voice_eval.audit import audit_state
from voice_eval.build_data import ROOT, save

ARTIFACTS = ROOT / "artifacts"
VENDOR = ROOT / "vendor" / "gpt_live_evals"
MODEL = "gemini-3.1-flash-tts-preview"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(mode: str):
    from shared.scenarios import load_scenario_dataset

    return load_scenario_dataset(ROOT / "data" / f"{mode}.json")


def validate() -> None:
    from run_harness.scenarios import validate_run_scenario

    for mode in ("crawl", "walk", "run"):
        data = dataset(mode)
        if mode == "run":
            for scenario in data.scenarios:
                validate_run_scenario(scenario)
        print(f"{mode}: {len(data.scenarios)} valid scenarios")


def doctor() -> None:
    status = {
        "python": sys.version.split()[0],
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "tts_model": os.getenv("GEMINI_TTS_MODEL", MODEL),
        "model_access": "not_checked",
        "live_evaluations": "not_started_by_doctor",
    }
    save(ARTIFACTS / "doctor.json", status)
    print(json.dumps(status, indent=2))
    validate()


def cache():
    from crawl_harness.audio_cache import CallerAudioCache

    return CallerAudioCache(ARTIFACTS / "audio_cache")


def audio_parameters(scenario) -> dict:
    return {
        "text": scenario.input.text,
        "model": os.getenv("GEMINI_TTS_MODEL", MODEL),
        "voice": os.getenv("GEMINI_TTS_VOICE", "Kore"),
        "sample_rate_hz": 24000,
    }


def prepare(args) -> None:
    from shared.audio.effects import AudioRealismProcessor
    from shared.audio.gemini import INSTRUCTION, synthesize
    from shared.audio.pcm import write_mono_wav

    selected = [s for s in dataset("crawl").scenarios if not args.scenario or s.id == args.scenario]
    if not selected:
        raise ValueError("Unknown CRAWL scenario")
    os.environ["GEMINI_USAGE_FILE"] = str(ARTIFACTS / "gemini_usage.jsonl")
    manifest_path = ARTIFACTS / "audio_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for scenario in selected:
        params = audio_parameters(scenario)
        expected_path = cache().path_for(scenario.id, **params)
        prior = manifest.get(scenario.id)
        if expected_path.exists() and prior and prior.get("instruction") != INSTRUCTION:
            raise ValueError("TTS instruction changed; use a new cache directory before comparing runs")
        audio = cache().load_or_create(
            scenario.id,
            **params,
            synthesize=lambda s=scenario, p=params: synthesize(s.input.text, model=p["model"], voice=p["voice"]),
        )
        entry = {
            **params,
            "instruction": INSTRUCTION,
            "source": "synthetic",
            "path": str(audio.path.relative_to(ROOT)),
            "sha256": digest(audio.path),
            "duration_seconds": len(audio.pcm) / 48000,
            "listened_and_approved": prior.get("listened_and_approved", False)
            if prior and prior.get("sha256") == digest(audio.path)
            else False,
        }
        if scenario.id.startswith("correction_"):
            for condition in ("clean", "noisy"):
                pcm = AudioRealismProcessor(condition, seed=41).process(audio.pcm)
                path = ARTIFACTS / "audio" / f"{scenario.id}_{condition}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_mono_wav(path, pcm, 24000)
                entry[condition] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "seed": 41}
        manifest[scenario.id] = entry
        save(manifest_path, manifest)
        print(f"{scenario.id}: {'reused' if audio.reused else 'generated'} ({entry['duration_seconds']:.2f}s)")


def run(args) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing; set it in .env or pass --env-file PATH")
    selected = [s for s in dataset(args.mode).scenarios if args.all or s.id == args.scenario]
    if not selected:
        raise ValueError("Select a valid scenario with --scenario, or use --all")
    if args.mode in {"crawl", "walk"}:
        for s in selected:
            paths = (
                [cache().path_for(s.id, **audio_parameters(s))]
                if args.mode == "crawl"
                else [(ROOT / "data" / r.path).resolve() for r in s.input.recordings]
            )
            if any(not path.is_file() for path in paths):
                raise ValueError("Caller audio is missing. Run the prepare command first (Gemini TTS).")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    batch = ARTIFACTS / "live" / f"{stamp}-{args.mode}-{args.variant}"
    inputs = batch / "inputs"
    inputs.mkdir(parents=True)
    for name in ("frontend.txt", "backend.txt"):
        shutil.copyfile(ROOT / "prompts" / args.variant / name, inputs / name)
    shutil.copyfile(ROOT / "data" / "restaurant_facts.json", inputs / "restaurant_facts.json")
    frozen_scenarios = [s.model_dump(mode="json") for s in selected]
    for scenario in frozen_scenarios:
        for recording in scenario["input"].get("recordings", []):
            recording["path"] = str((ROOT / "data" / recording["path"]).resolve())
    save(inputs / "scenarios.json", {"schema_version": "1.0", "scenarios": frozen_scenarios})
    if (ARTIFACTS / "audio_manifest.json").exists():
        shutil.copyfile(ARTIFACTS / "audio_manifest.json", inputs / "audio_manifest.json")
    env = os.environ.copy()
    env.update(
        {
            "EVAL_FRONTEND_PROMPT": str(inputs / "frontend.txt"),
            "EVAL_BACKEND_PROMPT": str(inputs / "backend.txt"),
            "EVAL_RESTAURANT_FACTS": str(inputs / "restaurant_facts.json"),
            "GEMINI_USAGE_FILE": str(ARTIFACTS / "gemini_usage.jsonl"),
        }
    )
    settings = {
        key: env[key]
        for key in (
            "OPENAI_LIVE_MODEL",
            "OPENAI_LIVE_VOICE",
            "OPENAI_LIVE_BACKEND_MODEL",
            "OPENAI_LIVE_BACKEND_REASONING_EFFORT",
            "OPENAI_EVAL_JUDGE_MODEL",
            "OPENAI_COMPLETION_MODEL",
            "GEMINI_TTS_MODEL",
            "GEMINI_TTS_VOICE",
        )
        if key in env
    }
    save(
        batch / "manifest.json",
        {
            "created_at": stamp,
            "mode": args.mode,
            "variant": args.variant,
            "settings": settings,
            "upstream": json.loads((ROOT / "UPSTREAM.json").read_text()),
            "lock_sha256": digest(ROOT / "uv.lock"),
            "input_hashes": {p.name: digest(p) for p in inputs.iterdir()},
            "repetitions": args.repeat,
            "scenario_ids": [s.id for s in selected],
            "execution_mode": "live",
        },
    )
    for repeat in range(args.repeat):
        for scenario in selected:
            cmd = [
                sys.executable,
                "-m",
                f"{args.mode}_harness.evaluate",
                "--data",
                str(inputs / "scenarios.json"),
                "--scenario",
                scenario.id,
                "--assistant",
                "responses",
                "--concurrency",
                "1",
                "--results-dir",
                str(batch),
                "--run-name",
                f"{args.variant}-{scenario.id}-r{repeat + 1}",
            ]
            if args.mode == "crawl":
                cmd += [
                    "--tts-model",
                    os.getenv("GEMINI_TTS_MODEL", MODEL),
                    "--audio-cache-dir",
                    str(ARTIFACTS / "audio_cache"),
                ]
            elif args.mode == "run":
                cmd += ["--visualize", "--max-duration-seconds", "60"]
            before = set(batch.glob("*/results.json"))
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            reports = set(batch.glob("*/results.json")) - before
            if len(reports) != 1:
                raise RuntimeError("Expected one new results.json; inspect artifacts before retrying")
            report_path = reports.pop()
            report = json.loads(report_path.read_text())
            audits = []
            for item in report["results"]:
                detail_ref = item["artifacts"].get("details")
                detail = json.loads((report_path.parent / detail_ref).read_text()) if detail_ref else {}
                audits.append({"scenario_id": scenario.id, **audit_state(scenario.model_dump(mode="json"), detail)})
            save(report_path.parent / "reservation_audit.json", audits)
            if report["summary"]["infrastructure_errors"]:
                raise RuntimeError("Infrastructure failure: batch stopped without retrying. Inspect saved evidence.")
    summarize()


def smoke() -> None:
    """Official deterministic English fixtures, never a Japanese model benchmark."""
    from shared.audio.pcm import tone_for_text, write_mono_wav

    out = ARTIFACTS / "offline"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((VENDOR / "crawl_harness/data/scenarios.json").read_text())
    scenario = next(s for s in data["scenarios"] if s["id"] == "restaurant_003")
    wav = out / "fixture.wav"
    write_mono_wav(wav, tone_for_text(scenario["input"]["text"], 24000), 24000)
    scenario["input"]["recordings"] = [
        {"id": "fixture", "path": str(wav), "condition": "clean", "metadata": {"source": "offline_tone"}}
    ]
    save(out / "walk.json", {"schema_version": "1.0", "scenarios": [scenario]})
    env = os.environ.copy()
    for key in ("EVAL_FRONTEND_PROMPT", "EVAL_BACKEND_PROMPT", "EVAL_RESTAURANT_FACTS"):
        env.pop(key, None)
    for mode in ("crawl", "walk", "run"):
        cmd = [
            sys.executable,
            "-m",
            f"{mode}_harness.evaluate",
            "--offline",
            "--concurrency",
            "1",
            "--results-dir",
            str(out / mode),
        ]
        if mode == "run":
            cmd += ["--scenario", "restaurant_booking_complete", "--visualize"]
        else:
            cmd += ["--scenario", "restaurant_003", "--no-real-time"]
            if mode == "walk":
                cmd += ["--data", str(out / "walk.json")]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        report = json.loads(max((out / mode).glob("*/results.json"), key=lambda p: p.stat().st_mtime).read_text())
        if report["summary"]["infrastructure_errors"] or report["summary"]["failed"]:
            raise RuntimeError(f"{mode} offline smoke failed; inspect artifacts/offline")


def summarize() -> None:
    rows = []
    for path in sorted((ARTIFACTS / "live").glob("*/*/results.json")):
        report = json.loads(path.read_text())
        manifest = json.loads((path.parent.parent / "manifest.json").read_text())
        audits_path = path.parent / "reservation_audit.json"
        audits = json.loads(audits_path.read_text()) if audits_path.exists() else []
        for item in report["results"]:
            audit = next((a for a in audits if a["scenario_id"] == item["scenario_id"]), {})
            rows.append(
                {
                    "mode": manifest["mode"],
                    "variant": manifest["variant"],
                    "scenario_id": item["scenario_id"],
                    "harness_status": item["status"],
                    "exact_state_passed": audit.get("passed"),
                    "task_passed": item["status"] == "passed" and audit.get("passed") is True,
                    "response_latency_ms": item["metrics"]["audio"].get("response_latency_ms"),
                    "confirmed_completion_ms": None,
                    "completion_timing_review": "Requires listening and timestamp annotation",
                    "report": str(path.relative_to(ROOT)),
                }
            )
    save(
        ARTIFACTS / "summary.json",
        {
            "live_trials": len(rows),
            "rows": rows,
            "cost_usd": None,
            "cost_note": "Use vendor billing and raw usage logs; missing usage is not zero.",
        },
    )
    if rows:
        with (ARTIFACTS / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved summary: {len(rows)} live trials (offline fixtures excluded)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "validate", "smoke", "summarize"):
        child = sub.add_parser(name)
        child.add_argument("--domain", choices=("restaurant", "retail"), default="restaurant")
    p = sub.add_parser("prepare", help="Generate/cache Gemini TTS fixtures (incurs API usage)")
    p.add_argument("--scenario")
    p.add_argument("--domain", choices=("restaurant", "retail"), default="restaurant")
    p = sub.add_parser("run", help="Run live evaluation (incurs API usage)")
    p.add_argument("mode", choices=("text", "crawl", "walk", "run"))
    p.add_argument("--domain", choices=("restaurant", "retail"), default="restaurant")
    p.add_argument("--new-attempt", action="store_true")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenario")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--suite", choices=("pilot",))
    p.add_argument(
        "--variant", choices=("baseline", "candidate", "digit-kana", "phone-sequence", "name-kana"), default="baseline"
    )
    p.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    # The harness subprocess reads precisely the selected environment file.
    if args.env_file.is_file():
        os.environ["GPT_LIVE_EVALS_ENV_FILE"] = str(args.env_file.resolve())
    elif args.env_file != ROOT / ".env":
        parser.exit(1, f"Environment file does not exist: {args.env_file}\n")
    try:
        if args.domain == "retail":
            from voice_eval.retail.cli import main as retail_main
            retail_main(args)
            return
        if getattr(args, "mode", None) == "text" or getattr(args, "suite", None):
            raise ValueError("text and --suite require --domain retail")
        if args.command in {"prepare", "run"}:
            globals()[args.command](args)
        else:
            globals()[args.command]()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
