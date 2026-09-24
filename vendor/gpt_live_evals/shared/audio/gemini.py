"""Gemini TTS adapter: fixed PCM contract, no hidden retries, separate usage log."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INSTRUCTION = "Read the following text verbatim in its original language. Speak naturally. Do not add any words.\n"


def request_body(text: str, voice: str) -> dict:
    return {
        "contents": [{"parts": [{"text": INSTRUCTION + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }


def decode_audio(payload: dict) -> bytes:
    candidates = payload.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") not in (None, "STOP"):
        raise ValueError("Gemini TTS did not return one complete candidate")
    chunks = []
    for part in candidates[0].get("content", {}).get("parts", []):
        if "inlineData" not in part:
            continue
        data = part["inlineData"]
        mime = data.get("mimeType", "").lower().replace(" ", "")
        if not mime.startswith("audio/l16;") or re.search(r"(?:^|;)rate=24000(?:;|$)", mime) is None:
            raise ValueError(f"Gemini TTS requires 24 kHz PCM16, received {mime!r}")
        pcm = base64.b64decode(data["data"], validate=True)
        if not pcm or len(pcm) % 2:
            raise ValueError("Gemini TTS returned empty or incomplete PCM16")
        chunks.append(pcm)
    if not chunks:
        raise ValueError("Gemini TTS returned no audio")
    return b"".join(chunks)


def synthesize(text: str, *, model: str, voice: str) -> bytes:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("Set GEMINI_API_KEY in the project .env before generating caller audio")
    if not re.fullmatch(r"gemini-[a-z0-9.-]+", model):
        raise ValueError("Invalid Gemini TTS model ID")
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(request_body(text, voice)).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # Never emit response headers, API keys, or request objects to artifacts.
        raise RuntimeError(f"Gemini TTS HTTP {exc.code}; no automatic retry was made") from None
    except URLError:
        raise RuntimeError("Gemini TTS connection failed; no automatic retry was made") from None
    usage_file = os.getenv("GEMINI_USAGE_FILE")
    if usage_file:
        path = Path(usage_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "model": model,
                        "voice": voice,
                        "input_sha256": hashlib.sha256((INSTRUCTION + text).encode()).hexdigest(),
                        "usage": payload.get("usageMetadata"),
                    }
                )
                + "\n"
            )
    return decode_audio(payload)
