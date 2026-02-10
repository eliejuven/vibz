from __future__ import annotations

import base64
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_CLIENT: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _CLIENT


def transcribe_audio_file(file_path: str) -> str:
    """
    Speech -> text using OpenAI Audio Transcriptions endpoint.
    """
    model = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    with open(file_path, "rb") as f:
        # OpenAI Audio API: transcriptions.create(...)
        # Supported models include gpt-4o-mini-transcribe and gpt-4o-transcribe.  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/guides/speech-to-text?utm_source=chatgpt.com)
        result = _client().audio.transcriptions.create(
            model=model,
            file=f,
        )

    # For OpenAI Python SDK, transcription text is typically in result.text
    return (getattr(result, "text", "") or "").strip()


def analyze_voice_emotion_from_wav_bytes(wav_bytes: bytes) -> str:
    """
    Audio -> compact musical intent description (emotion, energy, tension, arc).
    Uses an audio-capable chat model that accepts audio input.  [oai_citation:5‡OpenAI Platform](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview?utm_source=chatgpt.com)
    """
    model = os.environ.get("OPENAI_AUDIO_MODEL", "gpt-4o-mini-audio-preview")
    b64 = base64.b64encode(wav_bytes).decode("utf-8")

    instruction = (
        "Analyze this spoken voice note as an emotional signal for instrumental music generation.\n"
        "Infer from PROSODY (pace, intensity, pauses, pitch variation) the following:\n"
        "- mood/emotion keywords (2-4)\n"
        "- energy level: low / medium / high\n"
        "- tension level: low / medium / high\n"
        "- a 3-part narrative arc over ~30s: intro -> build -> release\n"
        "Return ONE compact paragraph (max ~80 words). No bullet lists."
    )

    # Audio input via Chat Completions using an audio-capable model.  [oai_citation:6‡OpenAI Platform](https://platform.openai.com/docs/guides/audio?utm_source=chatgpt.com)
    completion = _client().chat.completions.create(
        model=model,
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
    )

    msg = completion.choices[0].message
    content = msg.content

    # msg.content can be str or (sometimes) structured — handle both safely.
    if isinstance(content, str):
        return content.strip()

    # If it's a list of parts, join any text parts.
    try:
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts).strip()
    except Exception:
        return str(content).strip()