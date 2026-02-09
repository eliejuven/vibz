from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from musicgen_engine import MusicGenEngine

app = FastAPI(title="Vibz MusicGen API", version="0.1.0")

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

@app.on_event("startup")
def _startup() -> None:
    global ENGINE
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    ENGINE = MusicGenEngine(outputs_dir=outputs_dir)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate/text", response_model=GenerateResponse)
def generate_from_text(req: GenerateTextRequest):
    if req.model_type == "finetuned":
        # We’ll implement this later when we have LoRA weights.
        raise HTTPException(status_code=501, detail="finetuned model not available yet")

    if ENGINE is None:
        raise HTTPException(status_code=500, detail="engine not initialized")

    result = ENGINE.generate_wav(
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