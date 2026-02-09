from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration


@dataclass
class GenerationResult:
    audio_id: str
    wav_path: str
    json_path: str
    sample_rate: int
    used_prompt: str
    duration_sec: int


class MusicGenEngine:
    """
    Loads MusicGen once and can generate WAV files from a text prompt.
    Saves:
      - outputs/<id>.wav
      - outputs/<id>.json (metadata for debugging + later A/B)
    """

    def __init__(
        self,
        model_id: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        outputs_dir: str = "outputs",
    ) -> None:
        self.model_id = model_id
        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        # Audio sampling rate
        self.sample_rate = int(self.model.config.audio_encoder.sampling_rate)

    @torch.inference_mode()
    def generate_wav(
        self,
        prompt: str,
        duration_sec: int = 30,
        temperature: float = 1.0,
        top_k: int = 250,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        duration_sec = int(duration_sec)
        if not (1 <= duration_sec <= 60):
            raise ValueError("duration_sec must be between 1 and 60")

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # Important: set generation config per request
        gen_cfg = self.model.generation_config
        gen_cfg.temperature = float(temperature)
        gen_cfg.top_k = int(top_k)

        # For MusicGen, the generation length is controlled by max_new_tokens.
        # We use a conservative mapping that behaves reasonably for 20–45s.
        # You can re-calibrate on GPU later, but this is stable enough for MVP.
        gen_cfg.max_new_tokens = int(duration_sec * 50)

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        t0 = time.time()
        audio_values = self.model.generate(**inputs)
        dt = time.time() - t0

        audio = audio_values[0].detach().cpu().float().numpy()
        if audio.ndim == 2:
            audio = audio[0]

        # Normalize to int16 WAV
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)

        audio_id = str(uuid.uuid4())
        wav_path = os.path.join(self.outputs_dir, f"{audio_id}.wav")
        json_path = os.path.join(self.outputs_dir, f"{audio_id}.json")

        wavfile.write(wav_path, self.sample_rate, audio_int16)

        meta = {
            "audio_id": audio_id,
            "model_id": self.model_id,
            "device": self.device,
            "duration_sec_requested": duration_sec,
            "used_prompt": prompt,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "seed": seed,
            "sample_rate": self.sample_rate,
            "generation_seconds": dt,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return GenerationResult(
            audio_id=audio_id,
            wav_path=wav_path,
            json_path=json_path,
            sample_rate=self.sample_rate,
            used_prompt=prompt,
            duration_sec=duration_sec,
        )