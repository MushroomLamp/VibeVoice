import io
import os
import time
import re
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


app = FastAPI(title="VibeVoice OpenAI-compatible TTS")


class VoicePresetMapper:
    """Scan demo/voices for available .wav presets and map by simple names."""

    def __init__(self) -> None:
        self._index: Dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "voices")
        if not os.path.exists(voices_dir):
            self._index = {}
            return
        index: Dict[str, str] = {}
        for filename in os.listdir(voices_dir):
            if not filename.lower().endswith(".wav"):
                continue
            key = os.path.splitext(filename)[0]
            # provide a few fallback keys derived from name
            parts = key.replace("_", "-").split("-")
            # original
            index[key] = os.path.join(voices_dir, filename)
            # simple last token
            if parts:
                index[parts[-1]] = os.path.join(voices_dir, filename)
        # stable ordering is not needed; store
        self._index = index

    def get(self, name: Optional[str]) -> Optional[str]:
        if not self._index:
            return None
        if not name:
            # default to first available preset deterministically
            return sorted(self._index.items())[0][1]
        # case-insensitive exact or partial match
        lower = name.lower()
        if name in self._index:
            return self._index[name]
        if lower in (k.lower() for k in self._index.keys()):
            for k, v in self._index.items():
                if k.lower() == lower:
                    return v
        for k, v in self._index.items():
            if lower in k.lower():
                return v
        # fallback default
        return sorted(self._index.items())[0][1]


# Global state populated on startup
class State:
    processor: Optional[VibeVoiceProcessor] = None
    model: Optional[VibeVoiceForConditionalGenerationInference] = None
    device: str = "cpu"
    attn_impl: str = "sdpa"
    dtype = torch.float32
    voices = VoicePresetMapper()
    cfg_scale: float = 1.3


@app.on_event("startup")
def on_startup() -> None:
    model_path = os.environ.get("MODEL_PATH", "microsoft/VibeVoice-1.5b")
    # Optional CFG scale override from environment
    cfg_env = os.environ.get("CFG_SCALE")
    if cfg_env is not None:
        try:
            State.cfg_scale = float(cfg_env)
        except ValueError:
            pass

    # device selection
    if torch.cuda.is_available():
        State.device = "cuda"
        State.dtype = torch.bfloat16
        State.attn_impl = "flash_attention_2"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        State.device = "mps"
        State.dtype = torch.float32
        State.attn_impl = "sdpa"
    else:
        State.device = "cpu"
        State.dtype = torch.float32
        State.attn_impl = "sdpa"

    State.processor = VibeVoiceProcessor.from_pretrained(model_path)

    try:
        if State.device == "mps":
            State.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=State.dtype,
                attn_implementation=State.attn_impl,
                device_map=None,
            )
            State.model.to("mps")
        elif State.device == "cuda":
            State.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=State.dtype,
                device_map="cuda",
                attn_implementation=State.attn_impl,
            )
        else:
            State.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=State.dtype,
                device_map="cpu",
                attn_implementation=State.attn_impl,
            )
    except Exception as e:
        # Fallback to SDPA if flash-attn fails
        if State.attn_impl == "flash_attention_2":
            State.attn_impl = "sdpa"
            State.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=State.dtype,
                device_map=(State.device if State.device in ("cuda", "cpu") else None),
                attn_implementation=State.attn_impl,
            )
            if State.device == "mps":
                State.model.to("mps")
        else:
            raise e

    assert State.model is not None
    State.model.eval()
    # keep steps modest for latency
    State.model.set_ddpm_inference_steps(num_steps=10)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": State.device,
        "attn": State.attn_impl,
        "dtype": str(State.dtype),
        "cfg_scale": State.cfg_scale,
        "voices": sorted(list(State.voices._index.keys()))[:16],
    }


def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    # ensure mono 1-D float32 array
    if audio.ndim > 1:
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)
        elif audio.shape[-1] == 1:
            audio = audio.squeeze(-1)
        else:
            # mixdown
            audio = audio.mean(axis=0)
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


@app.post("/v1/audio/speech")
async def tts(request: Request):
    if State.model is None or State.processor is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # OpenAI fields
    text: str = body.get("input") or body.get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Field 'input' is required and must be a string")

    # model name is ignored but accepted for compatibility
    # voice: use demo/voices mapping by name
    voice_name: Optional[str] = body.get("voice")
    response_format: str = (body.get("response_format") or "wav").lower()
    if response_format != "wav":
        # We only support wav for now
        raise HTTPException(status_code=400, detail="Only response_format 'wav' is supported")

    # instructions are currently ignored but accepted for API compatibility
    # instructions: Optional[str] = body.get("instructions")

    # Map voice
    voice_path = State.voices.get(voice_name)
    voice_samples = [voice_path] if voice_path else []

    # Build the same input shape as the file-based script, single speaker
    sanitized_text = text.replace("’", "'").strip()
    # If the user did not provide explicit Speaker labels, default to Speaker 1
    if re.search(r"(?mi)^\s*Speaker\s+\d+\s*:", sanitized_text):
        full_script = sanitized_text
    else:
        lines = [ln.strip() for ln in sanitized_text.splitlines() if ln.strip()]
        full_script = "\n".join([f"Speaker 1: {ln}" for ln in lines]) if lines else "Speaker 1:"

    # Prepare inputs
    try:
        inputs = State.processor(
            text=[full_script],
            voice_samples=[voice_samples] if voice_samples else [[]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    except ValueError as e:
        # Provide a clear client error when script parsing fails
        raise HTTPException(status_code=400, detail=f"Input parsing error: {str(e)}. Provide plain text or format as 'Speaker X: text'.")

    # Move to device
    target_device = State.device if State.device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    start = time.time()
    outputs = State.model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=State.cfg_scale,
        tokenizer=State.processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
    )
    duration = time.time() - start

    # Convert to wav bytes
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio_tensor = outputs.speech_outputs[0]
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.float().detach().cpu().numpy()
    else:
        audio_np = np.array(audio_tensor, dtype=np.float32)

    sample_rate = State.processor.audio_processor.sampling_rate if State.processor and State.processor.audio_processor else 24000
    wav_bytes = encode_wav_bytes(audio_np, sample_rate)

    headers = {
        "X-Generation-Time": f"{duration:.2f}",
        "Content-Disposition": "inline; filename=audio.wav",
    }
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers=headers)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "VibeVoice TTS API. POST /v1/audio/speech"})


