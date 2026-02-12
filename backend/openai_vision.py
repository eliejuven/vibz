from __future__ import annotations

import base64
import io
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


def _convert_to_jpeg(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    SUPPORTED = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if mime_type in SUPPORTED:
        return image_bytes, mime_type

    from PIL import Image
    try:
        if mime_type in ("image/heic", "image/heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception as exc:
        raise ValueError(f"Cannot convert {mime_type} image: {exc}") from exc


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_image_for_music(image_bytes: bytes, mime_type: str, user_prompt: str = "") -> str:
    image_bytes, mime_type = _convert_to_jpeg(image_bytes, mime_type)

    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
    data_url = image_bytes_to_data_url(image_bytes, mime_type)

    instruction = (
        "You are helping generate instrumental music from an image.\n"
        "Describe the image as musical intent with:\n"
        "- mood/emotion (valence words)\n"
        "- energy level (calm/medium/intense)\n"
        "- a 3-part narrative arc over 30 seconds (intro -> build -> release)\n"
        "- 2-4 instrument suggestions\n"
        "Keep it concise (max ~80 words). No lists longer than 4 items.\n"
    )

    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": f"User theme (optional): {user_prompt.strip() or 'N/A'}"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip()
