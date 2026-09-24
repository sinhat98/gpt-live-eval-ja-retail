"""Reproducible local artifacts without credentials or unbounded environment dumps."""

import hashlib
import zipfile

from voice_eval.retail.data import ROOT, save


def freeze(output):
    sources = list((ROOT / "voice_eval").rglob("*.py"))
    sources += list((ROOT / "prompts/retail").rglob("*.txt"))
    sources += [
        ROOT / "vendor/gpt_live_evals/assistants/resources.py",
        ROOT / "vendor/gpt_live_evals/crawl_harness/audio_cache.py",
        ROOT / "vendor/gpt_live_evals/shared/audio/gemini.py",
        ROOT / "vendor/gpt_live_evals/run_harness/simulation/gpt_live_runner.py",
        ROOT / "vendor/j_tau/SOURCE_FILES.json",
        ROOT / "uv.lock",
        ROOT / "vendor/j_tau/uv.lock",
        ROOT / "UPSTREAM.json",
        ROOT / "data/retail/source.json",
        ROOT / "data/retail/identity.json",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save(output / "code_hashes.json", hashes)
    with zipfile.ZipFile(output / "implementation.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, arcname=str(p.relative_to(ROOT)))


def state_diff(before, after):
    differences = []
    for table in sorted(before.keys() | after.keys()):
        left, right = before.get(table, {}), after.get(table, {})
        for key in sorted(left.keys() | right.keys()):
            if left.get(key) != right.get(key):
                differences.append({"table": table, "id": key, "before": left.get(key), "after": right.get(key)})
    return differences
