### VibeVoice TTS API Reference

Base URL examples:
- Local (uvicorn): `http://localhost:8000`
- Docker compose service: use the mapped host/port you configured

This API is OpenAI TTS compatible for the speech endpoint shape.

---

## Endpoints

### GET /health
Returns service status and basic runtime info.

Response JSON:
```json
{
  "status": "ok",
  "device": "cpu|cuda|mps",
  "attn": "sdpa|flash_attention_2",
  "dtype": "torch.float32|torch.bfloat16",
  "voices": ["en-Alice_woman", "en-Carter_man", "zh-Xinran_woman", "..."]
}
```

Example:
```bash
curl -s http://localhost:8000/health | jq .
```

Notes:
- `voices` reflects .wav presets found under `demo/voices/` at startup.

---

### POST /v1/audio/speech
Generate speech audio from text. Returns an audio/wav stream.

Request (JSON):
- `input` (string, required): The text to synthesize. Alias: `text`.
- `voice` (string, optional): Name/alias of a preset voice in `demo/voices`.
- `response_format` (string, optional): Must be `"wav"` (only supported option).
- `model` (string, optional): Accepted for compatibility; ignored.
- `instructions` (string, optional): Accepted for compatibility; currently ignored.

Response:
- Content-Type: `audio/wav` (streaming)
- Headers:
  - `X-Generation-Time`: seconds to generate (e.g., `"2.37"`)
  - `Content-Disposition`: `inline; filename=audio.wav`

Default sample rate is determined by the processor (`24_000 Hz` if not otherwise configured).

#### curl examples

Save output to a file (using default voice):
```bash
curl -s -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
        "input": "See you again!",
        "response_format": "wav"
      }' \
  --output out.wav
```

Pick a specific voice by name (exact filename stem or alias):
```bash
curl -s -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
        "input": "This is VibeVoice speaking.",
        "voice": "en-Alice_woman",
        "response_format": "wav"
      }' \
  --output alice.wav
```

Partial/case-insensitive matches also work (e.g., `"Alice"`):
```bash
curl -s -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Hello there!",
        "voice": "alice",
        "response_format": "wav"
      }' \
  --output hello.wav
```

#### Python example
```python
import requests

url = "http://localhost:8000/v1/audio/speech"
payload = {
    "input": "Welcome to VibeVoice.",
    "voice": "en-Carter_man",
    "response_format": "wav",
}

r = requests.post(url, json=payload)
r.raise_for_status()
with open("voice.wav", "wb") as f:
    f.write(r.content)
print("Saved voice.wav", r.headers.get("X-Generation-Time"))
```

#### Node.js (fetch) example
```javascript
const res = await fetch("http://localhost:8000/v1/audio/speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "VibeVoice ready.",
    voice: "en-Frank_man",
    response_format: "wav",
  }),
});
if (!res.ok) throw new Error(await res.text());
const buf = Buffer.from(await res.arrayBuffer());
require("fs").writeFileSync("out.wav", buf);
console.log("Saved out.wav", res.headers.get("X-Generation-Time"));
```

---

## Voice presets and mapping

At startup the server scans `demo/voices/` for `.wav` files and builds a map. You can reference a voice by:
- Exact filename stem, e.g., `"en-Alice_woman"`
- Case-insensitive last token of the stem, e.g., `"Alice"`
- Case-insensitive partial substring, e.g., `"carter"`

If `voice` is omitted or not found, the server picks a deterministic default from the available presets. To discover available names, call `GET /health` and inspect the `voices` array.

---

## Environment and runtime

- `MODEL_PATH` (env): Hugging Face model id or local path. Default: `microsoft/VibeVoice-1.5b`.
- Device selection is automatic: `cuda` (preferred), `mps`, then `cpu`.
- Attention impl may use `flash_attention_2` on CUDA, falling back to `sdpa` if unavailable.
- Internal generation config: `cfg_scale=1.3`, DDPM inference steps set to `10`, deterministic (`do_sample=False`).

---

## Errors

- 400: Invalid JSON body
- 400: `input`/`text` missing or not a string
- 400: Unsupported `response_format` (only `"wav"`)
- 503: Model not ready
- 500: No audio generated

Response bodies contain a JSON error message when applicable.

---

## Root

### GET /
Returns a small JSON message:
```json
{ "message": "VibeVoice TTS API. POST /v1/audio/speech" }
```


