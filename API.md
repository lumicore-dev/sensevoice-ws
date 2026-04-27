# SenseVoice WebSocket ASR Server — API Reference

Version 1.0 | Protocol: WebSocket | Audio: PCM 16kHz 16-bit Mono

---

## 1. Connection

```
ws://<host>:8765/
```

On connect, the server sends an **info** message:

```json
{
  "type": "info",
  "message": "Connected. Send PCM 16kHz 16-bit mono audio frames.",
  "config": {
    "sample_rate": 16000,
    "model": "SenseVoiceSmall",
    "vad": "silero-vad"
  }
}
```

---

## 2. Client → Server Messages

### 2.1 Audio Data (Binary Frame)

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
    async with websockets.connect("ws://server:8765") as ws:
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

### 2.2 Control Commands (Text Frame)

| Command | Description |
|---------|-------------|
| `{"action": "reset"}` | Reset VAD state and audio buffer |

---

## 3. Server → Client Messages

All messages are JSON-encoded text frames.

### 3.1 `info`

Sent on connection or after a reset.

```json
{
  "type": "info",
  "message": "Session reset"
}
```

### 3.2 `speech_start`

Sent when VAD detects the beginning of a speech segment.

```json
{
  "type": "speech_start"
}
```

### 3.3 `transcription`

Sent when VAD detects the end of a speech segment and ASR completes.

```json
{
  "type": "transcription",
  "text": "请问信用卡的授信额度通常是多少",
  "duration_sec": 2.34,
  "inference_ms": 45.6
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text (with ITN applied) |
| `duration_sec` | number | Audio duration in seconds |
| `inference_ms` | number | Model inference time in milliseconds |

### 3.4 `error`

```json
{
  "type": "error",
  "message": "description of the error"
}
```

---

## 4. Integration Examples

### JavaScript / Browser

```javascript
const ws = new WebSocket('ws://server:8765');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) return; // ignore binary echo
  const msg = JSON.parse(event.data);
  if (msg.type === 'transcription') {
    console.log('Recognized:', msg.text);
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
import websockets
import pyaudio

async def microphone_client():
    async with websockets.connect("ws://server:8765") as ws:
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
                if data['type'] == 'transcription':
                    print(f"[{data['text']}]")

        await asyncio.gather(send_audio(), recv_results())
```

### C (using libwebsockets)

```c
// See examples/c_client.c for a complete C implementation
// Key points:
//   - Connect to ws://server:8765
//   - Send raw PCM 16-bit 16kHz as binary frames via LWS_WRITE_BINARY
//   - Receive text frames and parse JSON
```

### curl (Not supported)

WebSocket protocol is required. HTTP-based endpoints are not available.

---

## 5. Audio Preprocessing

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

## 6. Session Lifecycle

```
Client connects
  ↓
Server sends 'info'
  ↓
┌─────────────────────────────────────────────┐
│  [Client sends PCM chunks]                   │
│    ↓                                         │
│  VAD detects speech → 'speech_start'         │
│    ↓                                         │
│  VAD detects silence → ASR inference         │
│    ↓                                         │
│  Server sends 'transcription'                │
│    ↓                                         │
│  [Client continues sending]                  │
└─────────────────────────────────────────────┘
  ↓
Client disconnects
  ↓
Server flushes remaining audio, sends final results
```

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| No speech detected | No `transcription` sent for that segment |
| Audio too short (< 10ms) | Ignored, no transcription |
| Invalid audio format | Server may produce poor results; no error returned |
| Connection lost mid-utterance | Remaining audio is processed on `close` |
| Server overloaded | Connections are queued; all share one GPU model |

---

## 8. Performance

| Metric | Typical Value |
|--------|---------------|
| End-to-end latency (utterance) | 300–800 ms |
| Audio chunk interval | 32–200 ms (per binary frame) |
| Max concurrent connections | ~10–20 per GPU |
| GPU memory usage | ~1–2 GB |
| CPU memory usage | ~2–4 GB |

---

## 9. Repository

- **GitHub**: https://github.com/lumicore-dev/sensevoice-ws
- **License**: MIT (SenseVoiceSmall) / MIT (silero-vad)
- **Contact**: lumicore@dpai.com
