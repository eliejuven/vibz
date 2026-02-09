from __future__ import annotations


def compose_musicgen_prompt(user_prompt: str, image_music_desc: str) -> str:
    user_prompt = (user_prompt or "").strip()
    image_music_desc = (image_music_desc or "").strip()

    parts = [
        "Instrumental music.",
    ]
    if user_prompt:
        parts.append(f"Theme: {user_prompt}.")
    if image_music_desc:
        parts.append(f"Image-derived intent: {image_music_desc}")
    parts.append("Make it coherent and emotionally aligned. Avoid vocals. 30-second structure.")

    return " ".join(parts)