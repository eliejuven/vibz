from __future__ import annotations


def compose_musicgen_prompt(
    user_prompt: str,
    image_music_desc: str,
    voice_emotion_desc: str = "",
    voice_transcript: str = "",
) -> str:
    user_prompt = (user_prompt or "").strip()
    image_music_desc = (image_music_desc or "").strip()
    voice_emotion_desc = (voice_emotion_desc or "").strip()
    voice_transcript = (voice_transcript or "").strip()

    parts = [
        "Instrumental music. Avoid vocals and spoken word.",
    ]

    if user_prompt:
        parts.append(f"User theme: {user_prompt}.")
    if image_music_desc:
        parts.append(f"Image-derived intent: {image_music_desc}")
    if voice_emotion_desc:
        parts.append(f"Voice-derived emotion/energy: {voice_emotion_desc}")
    if voice_transcript:
        parts.append(f"Voice transcript (narrative hint): {voice_transcript}")

    parts.append("Make it coherent and emotionally aligned. 3-part arc: intro -> build -> release.")
    return " ".join(parts)