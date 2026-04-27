# SenseVoice WebSocket ASR Server — API Reference

Version 1.1 | Protocol: WebSocket | Audio: PCM 16kHz 16-bit Mono

---

## 1. Connection

```
ws://<host>:8765/?language=zh
```

The `language` parameter is optional. Default is `zh`.

On connect, the server responds with an **info** message:

```json
{
  "type": "info",
  "message": "Connected. Send PCM 16kHz 16-bit mono audio.",
  "config": {
    "sample_rate": 16000,
    "model": "SenseVoiceSmall",
    "vad": "silero-vad",
    "language": "zh"
  }
}
```

---

## 2. Language

Specify language via URL query string when connecting:

| Code | Language | Example |
|------|----------|---------|
| `zh` | Chinese (Mandarin) | `ws://host:8765/?language=zh` |
| `en` | English | `ws://host:8765/?language=en` |
| `yue` | Cantonese | `ws://host:8765/?language=yue` |
| `ja` | Japanese | `ws://host:8765/?language=ja` |
| `ko` | Korean | `ws://host:8765/?language=ko` |
| `auto` | Automatic detection | `ws://host:8765/?language=auto` |

- Chinese mode (`zh`) also handles Chinese-English mixed speech naturally.
- If an unsupported code is provided, the server falls back to `zh` and logs a warning.

---

## 3. Client → Server Messages

### 3.1 Audio Data (Binary Frame)

Send raw audio as **WebSocket binary frames**.

| Field | Value |
|-------|-------|
| Format | Linear PCM |
| Sample Rate | 16000 Hz |
| Bit Depth | 16-bit signed integer |
| Byte Order | Little-endian |
| Channels | Mono |
| Chunk Size | Any size (32ms–200ms recommended) |

**Python example:**

```python
import asyncio
import websockets
import wave

async def send_audio(file_path):
    async with websockets.connect(
        "ws://server:8765/?language=zh"
    ) as ws:
        # Read welcome
        welcome = await ws.recv()

        with wave.open(file_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())

        # Send in 100ms chunks
        chunk_size = 3200  # 100ms @ 16kHz 16-bit
        for i in range(0, len(frames), chunk_size):
            await ws.send(frames[i:i + chunk_size])
            await asyncio.sleep(0.01)

        # Wait for results
        async for msg in ws:
            print(msg)
```

### 3.2 Control Commands (Text Frame)

| Command | Description |
|---------|-------------|
| `{"action": "reset"}` | Reset VAD state and audio buffer |

---

## 4. Server → Client Messages

All messages are JSON-encoded text frames.

### 4.1 `info`

Sent on connection or after a reset.

```json
{
  "type": "info",
  "message": "Session reset"
}
```

### 4.2 `speech_start`

Sent when VAD detects the beginning of a speech segment.

```json
{
  "type": "speech_start"
}
```

The client can use this signal to update UI (e.g., show a listening indicator).

### 4.3 `transcription`

Sent when VAD detects the end of a speech segment and ASR completes.

**⚠️ Important — Raw Output Format**: The `text` field is returned **exactly** as output by the SenseVoiceSmall model, without post-processing. It may contain special tags that must be stripped by the client.

**Example — actual server response:**

```json
{
  "type": "transcription",
  "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>在吗在吗看现在这个识别对不对",
  "duration_sec": 2.34,
  "inference_ms": 45.6
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Raw ASR output from SenseVoiceSmall (includes special tags, see below) |
| `duration_sec` | number | Audio duration of this utterance in seconds |
| `inference_ms` | number | Model inference time in milliseconds |

**Special Tags in `text`:**

The `text` field is prefixed with SenseVoiceSmall's task/language/emotion tags. The client must strip these before displaying the final text.

| Tag | Meaning | Notes |
|-----|---------|-------|
| `<\|zh\|>` | Language: Chinese | Also: `<\|en\|>`, `<\|yue\|>`, `<\|ja\|>`, `<\|ko\|>` |
| `<\|NEUTRAL\|>` | Emotion: neutral | Also: `<\|HAPPY\|>`, `<\|SAD\|>`, `<\|ANGRY\|>` |
| `<\|Speech\|>` | Modality: speech | Fixed tag, always present for voice input |
| `<\|woitn\|>` | Without ITN | Indicates raw output without inverse text normalization |
| `<\|withitn\|>` | With ITN | Present when `use_itn=True` (server default is `False`) |

**Client-side stripping (recommended):**

```javascript
// JavaScript — strip SenseVoice tags from text
function cleanSenseVoiceText(raw) {
  return raw.replace(/<\|[a-zA-Z_]+\|>/g, '').trim();
}

// Usage
const text = cleanSenseVoiceText(msg.text);
// "在吗在吗看现在这个识别对不对"
```

```python
# Python — strip SenseVoice tags from text
import re

def clean_sensevoice_text(raw: str) -> str:
    return re.sub(r'<\|[a-zA-Z_]+\|>', '', raw).strip()
```

```swift
// Swift — strip SenseVoice tags from text
func cleanSenseVoiceText(_ raw: String) -> String {
    return raw.replacingOccurrences(
        of: #"<\|[a-zA-Z_]+\|>"#,
        with: "",
        options: .regularExpression
    ).trimmingCharacters(in: .whitespaces)
}
```

**Notes:**
- The output text is raw ASR output without inverse text normalization (ITN). Numbers and punctuation are passed as-is, intended for downstream LLM processing.
- If the utterance contains no detectable speech, no `transcription` message is sent.
- The server does **not** strip these tags — it is the client's responsibility to clean the `text` field before display.

### 4.4 `error`

```json
{
  "type": "error",
  "message": "description of the error"
}
```

---

## 5. Integration Examples

### JavaScript / Browser

```javascript
const ws = new WebSocket('ws://server:8765/?language=zh');

// Strip SenseVoice tags helper
function cleanText(raw) {
  return raw.replace(/<\|[a-zA-Z_]+\|>/g, '').trim();
}

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) return;
  const msg = JSON.parse(event.data);

  if (msg.type === 'speech_start') {
    console.log('Listening...');
  } else if (msg.type === 'transcription') {
    // Clean the raw text before displaying
    const text = cleanText(msg.text);
    console.log('Recognized:', text);
  }
};

// Send audio from getUserMedia
navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
  const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  // ... convert to PCM 16kHz 16-bit and send via ws.send(audioChunk)
});
```

### Python

```python
import asyncio
import json
import re
import websockets
import pyaudio

def clean_sensevoice_text(raw: str) -> str:
    return re.sub(r'<\|[a-zA-Z_]+\|>', '', raw).strip()

async def microphone_client():
    async with websockets.connect(
        "ws://server:8765/?language=zh"
    ) as ws:
        welcome = await ws.recv()
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1600
        )

        async def send_audio():
            while True:
                data = stream.read(1600)
                await ws.send(data)
                await asyncio.sleep(0)

        async def recv_results():
            async for msg in ws:
                data = json.loads(msg)
                if data['type'] == 'speech_start':
                    print("[listening...]", end=" ", flush=True)
                elif data['type'] == 'transcription':
                    text = clean_sensevoice_text(data['text'])
                    print(f"\n[{text}]")

        await asyncio.gather(send_audio(), recv_results())
```

### C (using libwebsockets)

```c
// Connect to ws://server:8765/?language=zh
// Send raw PCM 16-bit 16kHz as binary frames via LWS_WRITE_BINARY
// Receive text frames and parse JSON
```

### curl (Not supported)

WebSocket protocol is required. HTTP-based endpoints are not available.

---

## 6. Audio Preprocessing

Before sending audio to the server, ensure the format matches:

### From WAV file

```python
import wave

with wave.open('input.wav', 'rb') as wf:
    assert wf.getnchannels() == 1,     "Must be mono"
    assert wf.getsampwidth() == 2,     "Must be 16-bit"
    assert wf.getframerate() == 16000, "Must be 16kHz"
    frames = wf.readframes(wf.getnframes())
```

### From other sample rates / formats

```bash
# Convert using ffmpeg
ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 output.wav
```

---

## 7. Session Lifecycle

```
Client connects  ws://host:8765/?language=zh
  ↓
Server sends 'info'  (config + language confirmation)
  ↓
┌────────────────────────────────────────────────┐
│  [Client sends PCM chunks continuously]         │
│    ↓                                            │
│  VAD detects speech ──→ 'speech_start'          │
│    ↓                                            │
│  VAD detects silence ──→ ASR inference          │
│    ↓                                            │
│  Server sends 'transcription'  (per utterance)  │
│    ↓                                            │
│  [Client continues sending audio]               │
│    ...                                          │
│  VAD detects next utterance ──→ next result     │
└────────────────────────────────────────────────┘
  ↓
Client disconnects
  ↓
Server flushes remaining audio, sends final results
```

The client keeps a persistent WebSocket connection for the entire session. No reconnection is needed between utterances.

---

## 8. VAD (Voice Activity Detection)

The server uses **silero-vad** to detect utterance boundaries.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | silero-vad v5 | Lightweight (1.7MB), high accuracy |
| Threshold | 0.5 | Speech probability threshold |
| Grace period | 600ms | Silence duration before utterance end |
| Frame size | 512 samples | 32ms at 16kHz |

How it works:
- Audio is processed in 32ms frames
- When speech probability exceeds 0.5, utterance starts
- When speech stays below 0.5 for 600ms, utterance ends and ASR triggers
- The server sends `speech_start` and `transcription` events accordingly

---

## 9. Error Handling

| Scenario | Behavior |
|----------|----------|
| No speech detected | No `transcription` sent for that segment |
| Audio too short (< 10ms) | Ignored, no transcription |
| Invalid audio format | Server may produce poor results; no error returned |
| Connection lost mid-utterance | Remaining audio is processed on close |
| Unsupported language code | Server falls back to `zh`, logs warning |
| Server overloaded | Connections are queued; all share one GPU model |

---

## 10. Performance

| Metric | Typical Value |
|--------|---------------|
| End-to-end latency (per utterance) | 300–800 ms |
| Audio chunk interval | 32–200 ms (per binary frame) |
| Max concurrent connections | ~10–20 per GPU |
| GPU memory usage | ~1–2 GB |
| CPU memory usage | ~2–4 GB |

---

## 11. Repository

- **GitHub**: https://github.com/lumicore-dev/sensevoice-ws
- **License**: MIT (SenseVoiceSmall) / MIT (silero-vad)
- **Contact**: lumicore@dpai.com
