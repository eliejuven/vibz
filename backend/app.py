from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from musicgen_engine import MusicGenEngine

from fastapi import File, Form, UploadFile
from openai_vision import describe_image_for_music
from prompt_composer import compose_musicgen_prompt

import tempfile

import threading

from openai_audio import transcribe_audio_file, analyze_voice_emotion_from_wav_bytes

app = FastAPI(title="Vibz MusicGen API", version="0.1.0")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Load model once at startup (baseline)
ENGINE: Optional[MusicGenEngine] = None


class GenerateTextRequest(BaseModel):
    model_type: Literal["baseline", "finetuned"] = "baseline"
    prompt: str = Field(min_length=1, max_length=2000)
    duration_sec: int = Field(default=30, ge=20, le=45)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_k: int = Field(default=250, ge=1, le=1000)
    seed: Optional[int] = Field(default=None, ge=0, le=2_000_000_000)


class GenerateResponse(BaseModel):
    audio_id: str
    used_prompt: str
    sample_rate: int
    download_url: str
    meta_url: str

ENGINE: Optional[MusicGenEngine] = None
_ENGINE_LOCK = threading.Lock()

def _get_engine() -> MusicGenEngine:
    global ENGINE
    if ENGINE is None:
        with _ENGINE_LOCK:
            if ENGINE is None:
                outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
                ENGINE = MusicGenEngine(outputs_dir=outputs_dir)
    return ENGINE

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate/text", response_model=GenerateResponse)
def generate_from_text(req: GenerateTextRequest):
    if req.model_type == "finetuned":
        # We’ll implement this later when we have LoRA weights.
        raise HTTPException(status_code=501, detail="finetuned model not available yet")

    engine = _get_engine()

    result = engine.generate_wav(
        prompt=req.prompt,
        duration_sec=req.duration_sec,
        temperature=req.temperature,
        top_k=req.top_k,
        seed=req.seed,
    )

    return GenerateResponse(
        audio_id=result.audio_id,
        used_prompt=result.used_prompt,
        sample_rate=result.sample_rate,
        download_url=f"/audio/{result.audio_id}.wav",
        meta_url=f"/meta/{result.audio_id}.json",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    model_type: Literal["baseline", "finetuned"] = Form("baseline"),
    duration_sec: int = Form(30),
    temperature: float = Form(1.0),
    top_k: int = Form(250),
    seed: Optional[int] = Form(None),
    text_prompt: str = Form(""),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
):
    engine = _get_engine()

    if not (20 <= int(duration_sec) <= 45):
        raise HTTPException(status_code=400, detail="duration_sec must be between 20 and 45")

    if model_type == "finetuned":
        raise HTTPException(status_code=501, detail="finetuned model not available yet")

    image_desc = ""
    if image is not None:
        image_bytes = await image.read()
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")
        try:
            image_desc = describe_image_for_music(image_bytes, image.content_type, user_prompt=text_prompt)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}")

    voice_transcript = ""
    voice_emotion_desc = ""

    if voice is not None:
        if not voice.content_type or not voice.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Uploaded voice file is not audio")

        voice_bytes = await voice.read()

        allowed_audio = ("audio/wav", "audio/x-wav")
        if voice.content_type not in allowed_audio:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {voice.content_type}"
            )

        voice_emotion_desc = analyze_voice_emotion_from_wav_bytes(voice_bytes)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(voice_bytes)
            tmp.flush()
            voice_transcript = transcribe_audio_file(tmp.name)

    final_prompt = compose_musicgen_prompt(
        text_prompt,
        image_desc,
        voice_emotion_desc=voice_emotion_desc,
        voice_transcript=voice_transcript,
    )

    result = engine.generate_wav(
        prompt=final_prompt,
        duration_sec=duration_sec,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )

    return GenerateResponse(
        audio_id=result.audio_id,
        used_prompt=result.used_prompt,
        sample_rate=result.sample_rate,
        download_url=f"/audio/{result.audio_id}.wav",
        meta_url=f"/meta/{result.audio_id}.json",
    )


@app.get("/audio/{filename}")
def get_audio(filename: str):
    # Basic safety: only allow .wav inside outputs
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    file_path = os.path.join(outputs_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.get("/meta/{filename}")
def get_meta(filename: str):
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    file_path = os.path.join(outputs_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/json",
        filename=filename,
    )


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))