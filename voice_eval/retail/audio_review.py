"""Independent ASR evidence for synthesized input; does not replace listening."""

import hashlib
import json
import os
import re
import wave

from voice_eval.retail.data import save


def validate_input_wav(path):
    from crawl_harness.audio_cache import CallerAudioCache

    try:
        pcm = CallerAudioCache._read_pcm(path, 24000)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise ValueError("Input must be a prepared 24kHz mono PCM16 WAV") from exc
    if not any(pcm):
        raise ValueError("Input audio is entirely silent")


def review(path, reference, client):
    output = path.with_suffix(".review.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["sha256"] == digest:
            return previous
    with path.open("rb") as audio:
        response = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_INPUT_ASR_MODEL", "gpt-transcribe"), file=audio
        )
    text = response.text
    expected_ids = re.findall(r"\d{8,}", reference)
    observed = re.sub(r"[\s,、，-]", "", text)
    result = {
        "sha256": digest,
        "reference": reference,
        "asr_text": text,
        "numeric_ids_match": all(value in observed for value in expected_ids),
        "listening_review": "pending",
        "automatic_review_only": True,
        "raw_response": response.model_dump(),
        "model": os.getenv("OPENAI_INPUT_ASR_MODEL", "gpt-transcribe"),
    }
    save(output, result)
    return result
