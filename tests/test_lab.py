import base64
import copy
import json
from unittest.mock import patch

import pytest
from crawl_harness.audio_cache import CallerAudioCache
from run_harness.scenarios import load_run_dataset
from shared.audio.effects import AudioRealismProcessor
from shared.audio.gemini import decode_audio, request_body, synthesize
from shared.audio.pcm import tone_for_text
from shared.scenarios import load_scenario_dataset

from voice_eval.audit import audit_state
from voice_eval.build_data import ROOT, arguments, expected_booking, single


def test_datasets_are_valid_and_paired():
    for mode, count in (("crawl", 8), ("walk", 4), ("run", 4)):
        dataset = (load_run_dataset if mode == "run" else load_scenario_dataset)(ROOT / "data" / f"{mode}.json")
        assert len(dataset.scenarios) == count
        assert sum("ja" in s.tags for s in dataset.scenarios) == count // 2
        for scenario in dataset.scenarios:
            assert scenario.application.initial_state == {"reservations": []}
            if mode == "run":
                assert all(s.critical for s in scenario.expected.procedure.steps)


def audio_payload(data=b"\x01\x00" * 10, mime="audio/L16;codec=pcm;rate=24000"):
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}}]},
            }
        ]
    }


def test_gemini_payload_and_audio_validation():
    assert "マヤ" in request_body("マヤ", "Kore")["contents"][0]["parts"][0]["text"]
    assert decode_audio(audio_payload()) == b"\x01\x00" * 10
    for payload in (
        audio_payload(b""),
        audio_payload(b"x"),
        audio_payload(mime="audio/mpeg"),
        {},
        {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    ):
        with pytest.raises(ValueError):
            decode_audio(payload)


def test_no_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("shared.audio.gemini.urlopen") as network, pytest.raises(ValueError, match="GEMINI_API_KEY"):
        synthesize("hello", model="gemini-3.1-flash-tts-preview", voice="Kore")
    network.assert_not_called()


def test_gemini_cache_reuses_and_distinguishes_settings(tmp_path):
    cache = CallerAudioCache(tmp_path)
    params = {"text": "hello", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore", "sample_rate_hz": 24000}
    with patch("builtins.input", side_effect=AssertionError("No interactive prompt")):
        first = cache.load_or_create("test", **params, synthesize=lambda: b"\x01\x00" * 20)
        second = cache.load_or_create("test", **params, synthesize=lambda: pytest.fail("Unexpected paid generation"))
    assert second.reused and first.pcm == second.pcm
    assert cache.path_for("test", **params) != cache.path_for("test", **{**params, "voice": "Puck"})


def test_noise_is_repeatable_and_preserves_length():
    pcm = tone_for_text("test", 24000)
    assert AudioRealismProcessor("clean", seed=41).process(pcm) == pcm
    noisy = AudioRealismProcessor("noisy", seed=41).process(pcm)
    assert noisy != pcm and len(noisy) == len(pcm)
    assert noisy == AudioRealismProcessor("noisy", seed=41).process(pcm)


def test_exact_state_rejects_wrong_date_duplicates_and_unrequested_booking():
    scenario = single("ja", "correction")
    correct = copy.deepcopy(expected_booking(arguments("ja"))["state"])
    assert audit_state(scenario, {"final_state": correct})["passed"]
    wrong = copy.deepcopy(correct)
    wrong["reservations"][0]["date"] = "2026-10-07"
    assert not audit_state(scenario, {"final_state": wrong})["passed"]
    duplicate = copy.deepcopy(correct)
    duplicate["reservations"].append(copy.deepcopy(duplicate["reservations"][0]))
    assert not audit_state(scenario, {"final_state": duplicate})["passed"]
    assert not audit_state(single("en", "availability"), {"final_state": correct})["passed"]
    assert audit_state(single("en", "missing_name"), {"final_state": {"reservations": []}})["passed"]
    assert audit_state(scenario, {})["evidence_error"]


def test_prompt_variants_only_change_backend():
    assert (ROOT / "prompts/baseline/frontend.txt").read_bytes() == (
        ROOT / "prompts/candidate/frontend.txt"
    ).read_bytes()


def test_live_batch_stops_after_infrastructure_error(tmp_path, monkeypatch):
    import argparse
    import shutil

    import voice_eval.__main__ as cli

    for folder in ("prompts", "data"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    for name in ("UPSTREAM.json", "uv.lock"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    calls = []

    def failed_run(cmd, **kwargs):
        calls.append(cmd)
        data_path = cli.Path(cmd[cmd.index("--data") + 1])
        assert data_path.parent.name == "inputs"
        output = cli.Path(cmd[cmd.index("--results-dir") + 1]) / "failed-run"
        output.mkdir()
        (output / "results.json").write_text(
            json.dumps(
                {
                    "summary": {"infrastructure_errors": 1},
                    "results": [{"artifacts": {}, "status": "infrastructure_error"}],
                }
            )
        )

    monkeypatch.setattr(cli.subprocess, "run", failed_run)
    args = argparse.Namespace(mode="run", all=True, scenario=None, variant="baseline", repeat=1)
    with pytest.raises(RuntimeError, match="batch stopped"):
        cli.run(args)
    assert len(calls) == 1
    manifest_path = next((tmp_path / "artifacts/live").glob("*/manifest.json"))
    assert "test-not-a-real-key" not in manifest_path.read_text()


def test_resource_override_does_not_leak_expectations(monkeypatch):
    from assistants.resources import assistant_resources

    monkeypatch.setenv("EVAL_FRONTEND_PROMPT", str(ROOT / "prompts/baseline/frontend.txt"))
    resources = assistant_resources()
    assert resources.system_prompt_file == ROOT / "prompts/baseline/frontend.txt"
    text = resources.system_prompt_file.read_text()
    assert "2026-10-06" not in text and "correction_ja" not in text


def test_prepare_reuses_audio_for_walk_without_api(tmp_path, monkeypatch):
    import argparse

    import voice_eval.__main__ as cli

    monkeypatch.setattr(cli, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "dataset", lambda _: load_scenario_dataset(ROOT / "data/crawl.json"))
    with patch("shared.audio.gemini.synthesize", return_value=tone_for_text("test", 24000)) as tts:
        cli.prepare(argparse.Namespace(scenario="correction_ja"))
        cli.prepare(argparse.Namespace(scenario="correction_ja"))
        assert tts.call_count == 1
    manifest = json.loads((tmp_path / "audio_manifest.json").read_text())
    assert manifest["correction_ja"]["source"] == "synthetic"
    assert not manifest["correction_ja"]["listened_and_approved"]
    assert (tmp_path / "audio/correction_ja_noisy.wav").is_file()
