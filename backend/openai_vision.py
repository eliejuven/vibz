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


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_image_for_music(image_bytes: bytes, mime_type: str, user_prompt: str = "") -> str:
    """
    Returns a compact description optimized to become a MusicGen prompt ingredient.
    """
    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini")
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

    # Responses API supports text + image inputs in one request.
    # We ask for plain text output we can blend into MusicGen prompts.
    resp = _client().responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": f"User theme (optional): {user_prompt.strip() or 'N/A'}"},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return (resp.output_text or "").strip()