# SenseVoice WebSocket ASR Server — API Reference

Version 2.0 | Protocol: WebSocket | Audio: PCM 16-bit Mono

---

## 1. Connection

```
ws://<host>:8765/?param1=value1&param2=value2
```

**All parameters are optional.** Default values apply if not specified.

On connect, the server responds with an **info** message containing the full resolved configuration.

---

## 2. Session Parameters

All parameters are passed via **URL query string** at connection time.
They can also be updated mid-session via the `{"action": "config"}` command (see §4.2.2).

### 2.1 SenseVoice ASR Parameters

These are passed directly to `model.generate()`.

| Parameter | Type | Default | Choices | Description |
|-----------|------|---------|---------|-------------|
| `language` | string | `zh` | `zh`, `en`, `yue`, `ja`, `ko`, `auto`, `nospeech` | Language code for ASR. `auto` = automatic detection, `nospeech` = ignore speech |
| `use_itn` | boolean | `false` | `true`, `false` | Inverse Text Normalization. When `true`, output includes **punctuation** (commas, periods, question marks) and **number formatting** (e.g., "一二三" → "123") |
| `ban_emo_unk` | boolean | `false` | `true`, `false` | When `true`, disables unknown emotion tags so every utterance gets an emotion label |
| `batch_size_s` | number | `60` | Any positive number | Dynamic batch size in seconds of audio (only used when `merge_vad=true`) |
| `merge_vad` | boolean | `false` | `true`, `false` | Merge VAD-split segments before transcription (improves context for long audio) |
| `merge_length_s` | integer | `15` | Any positive integer | Merged segment length in seconds (only used when `merge_vad=true`) |

### 2.2 Output Post-Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rich_postprocess` | boolean | `false` | When `true`, applies `rich_transcription_postprocess` to the raw model output. Strips all `<\|tag\|>` markers, converts emotion tags to emoji, and produces clean display text. See §6 for details |

### 2.3 VAD (Voice Activity Detection) Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `vad_threshold` | number | `0.5` | `0.0`–`1.0` | Speech probability threshold. Lower = more sensitive, higher = stricter |
| `vad_grace_period_ms` | integer | `600` | `100`–`5000` | Silence duration (ms) before VAD declares utterance end |
| `ptt_mode` | boolean | `false` | `true`, `false` | Push-to-talk mode. When `true`, VAD never auto-triggers transcription. Only `{"action":"eof"}` triggers it. See §3 for PTT flow |

### 2.4 Audio Parameters

| Parameter | Type | Default | Choices | Description |
|-----------|------|---------|---------|-------------|
| `sample_rate` | integer | `16000` | `8000`, `16000` | Audio sample rate. Must match client's audio format |

---

### 2.5 Connection Example

```bash
# Chinese with punctuation, relaxed VAD
ws://server:8765/?language=zh&use_itn=true&vad_threshold=0.3

# English push-to-talk, no auto segmentation
ws://server:8765/?language=en&ptt_mode=true

# Japanese with rich postprocessing, strict VAD
ws://server:8765/?language=ja&rich_postprocess=true&vad_threshold=0.7

# Auto language detection, merge VAD segments
ws://server:8765/?language=auto&merge_vad=true&merge_length_s=30
```

---

## 3. Protocol Modes

### 3.1 VAD Mode (Default)

VAD automatically detects utterance boundaries.

```
Client connects → sends PCM chunks
  ↓
VAD detects speech start → Server sends {"type": "speech_start"}
  ↓
VAD detects silence (after grace period) → Server transcribes
  ↓
Server sends {"type": "transcription", ...}
  ↓
Continues listening for next utterance
```

### 3.2 Push-to-Talk (PTT) Mode

Set `ptt_mode=true` to disable automatic VAD segmentation.
The client **must** send `{"action": "eof"}` to trigger transcription.

```
Client connects → sends PCM chunks while button held
  ↓
Client releases button → sends {"action": "eof"}
  ↓
Server force-transcribes all buffered audio
  ↓
Server sends {"type": "transcription", ...}
  ↓
Server sends {"type": "done"}
  ↓
Session resets, ready for next utterance
```

---

## 4. Client → Server Messages

### 4.1 Audio Data (Binary Frame)

Send raw audio as **WebSocket binary frames**.

| Field | Value |
|-------|-------|
| Format | Linear PCM |
| Sample Rate | 16000 Hz (default) |
| Bit Depth | 16-bit signed integer |
| Byte Order | Little-endian |
| Channels | Mono |
| Chunk Size | Any size (32–200 ms recommended) |

### 4.2 Control Commands (Text Frame)

Send as JSON-encoded text frames.

#### 4.2.1 `eof` — Force Transcribe

```json
{"action": "eof"}
```

Force-transcribes all buffered audio immediately, bypassing VAD silence detection.
Returns `transcription` (if speech detected) then `done`.

#### 4.2.2 `config` — Update Parameters Mid-Session

```json
{
  "action": "config",
  "params": {
    "use_itn": true,
    "language": "en",
    "vad_threshold": 0.3
  }
}
```

Dynamically updates session parameters **without reconnecting**.
Only the following keys can be changed mid-session:

- `language`
- `use_itn`
- `ban_emo_unk`
- `batch_size_s`
- `merge_vad`
- `merge_length_s`
- `rich_postprocess`
- `ptt_mode`

VAD parameters (`vad_threshold`, `vad_grace_period_ms`, `sample_rate`) and `batch_size_s` are **fixed at connection time** — changing them requires reconnecting.

#### 4.2.3 `reset` — Reset Session

```json
{"action": "reset"}
```

Resets VAD state and audio buffer. Session parameters are preserved.

---

## 5. Server → Client Messages

All messages are JSON-encoded text frames.

### 5.1 `info`

Sent on connection, after reset, or after config update.

```json
{
  "type": "info",
  "message": "Connected. Send PCM 16-bit mono audio.",
  "config": {
    "sample_rate": 16000,
    "model": "SenseVoiceSmall",
    "vad": "silero-vad",
    "language": "zh",
    "use_itn": true,
    "ban_emo_unk": false,
    "batch_size_s": 60,
    "merge_vad": false,
    "merge_length_s": 15,
    "rich_postprocess": false,
    "vad_threshold": 0.5,
    "vad_grace_period_ms": 600,
    "ptt_mode": false
  }
}
```

### 5.2 `speech_start`

Sent when VAD detects the beginning of a speech segment.

```json
{
  "type": "speech_start"
}
```

### 5.3 `transcription`

Sent when ASR completes.

```json
{
  "type": "transcription",
  "text": "...",
  "duration_sec": 2.34,
  "inference_ms": 45.6
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text. Format depends on config parameters (see §6 below) |
| `duration_sec` | number | Audio duration of this utterance in seconds |
| `inference_ms` | number | Model inference time in milliseconds |

### 5.4 `done`

Sent after an EOF processing cycle completes.

```json
{
  "type": "done"
}
```

### 5.5 `error`

```json
{
  "type": "error",
  "message": "Invalid JSON command"
}
```

---

## 6. Output Text Formats

The format of `text` in transcription messages depends on the `use_itn` and `rich_postprocess` parameters.

### 6.1 Raw Format (`use_itn=false`, `rich_postprocess=false`)

The model's raw output with special tags.

```
<|zh|><|NEUTRAL|><|Speech|><|woitn|>测试测试一二三
```

| Tag | Meaning |
|-----|---------|
| `<\|zh\|>` | Detected language |
| `<\|NEUTRAL\|>` | Detected emotion |
| `<\|Speech\|>` | Audio event type |
| `<\|woitn\|>` | Without ITN (no punctuation) |

### 6.2 With ITN (`use_itn=true`, `rich_postprocess=false`)

Inverse text normalization applied. Includes punctuation and number formatting.

```
<|zh|><|NEUTRAL|><|Speech|><|withitn|>测试测试，一二三。
```

### 6.3 Rich Postprocessed (`rich_postprocess=true`)

All special tags stripped. Emotion converted to emoji. Clean display text.

```
测试测试，一二三。
```

When `rich_postprocess=true`, the `<\|NEUTRAL\|>` tag is stripped (no emoji for neutral). Other emotions like `<\|HAPPY\|>` become 😊, `<\|SAD\|>` becomes 😔, etc.

### 6.4 Client-Side Stripping (if not using rich_postprocess)

```javascript
// JavaScript — strip all SenseVoice tags
function cleanSenseVoiceText(raw) {
  return raw.replace(/<\|[a-zA-Z_]+\|>/g, '').trim();
}
```

```python
# Python — strip all SenseVoice tags
import re
def clean_sensevoice_text(raw: str) -> str:
    return re.sub(r'<\|[a-zA-Z_]+\|>', '', raw).strip()
```

---

## 7. Full Parameter Matrix

| `use_itn` | `rich_postprocess` | Output Example |
|-----------|-------------------|----------------|
| `false` | `false` | `<\|zh\|><\|NEUTRAL\|><\|Speech\|><\|woitn\|>在吗在吗看现在这个识别对不对` |
| `true` | `false` | `<\|zh\|><\|NEUTRAL\|><\|Speech\|><\|withitn\|>在吗，在吗，看现在这个识别对不对？` |
| `false` | `true` | `在吗在吗看现在这个识别对不对` |
| `true` | `true` | `在吗，在吗，看现在这个识别对不对？` |

---

## 8. Integration Examples

### Python — Full Configuration

```python
import asyncio
import json
import websockets

async def client():
    # Full parameter set via URL
    params = "?language=zh&use_itn=true&rich_postprocess=true&vad_threshold=0.3"
    uri = f"ws://server:8765/{params}"

    async with websockets.connect(uri) as ws:
        welcome = json.loads(await ws.recv())
        print("Connected with config:", json.dumps(welcome['config'], indent=2))

        # Send audio...
        with open('audio.wav', 'rb') as f:
            data = f.read()
            chunk_size = 3200  # 100ms
            for i in range(0, len(data), chunk_size):
                await ws.send(data[i:i+chunk_size])
                await asyncio.sleep(0.01)

        # EOF
        await ws.send(json.dumps({"action": "eof"}))

        async for msg in ws:
            result = json.loads(msg)
            if result['type'] == 'transcription':
                print(f"Text: {result['text']}")
            elif result['type'] == 'done':
                break
```

### Python — Dynamic Config Change Mid-Session

```python
import asyncio
import json
import websockets

async def dynamic_config():
    async with websockets.connect("ws://server:8765/?language=zh") as ws:
        welcome = await ws.recv()

        # Send some Chinese audio
        # ...
        await ws.send(json.dumps({"action": "eof"}))
        result = json.loads(await ws.recv())
        print("Chinese:", result.get('text', ''))

        # Switch to English mid-session
        await ws.send(json.dumps({
            "action": "config",
            "params": {"language": "en", "use_itn": True}
        }))
        config_ack = await ws.recv()
        print(config_ack)

        # Send some English audio
        # ...
        await ws.send(json.dumps({"action": "eof"}))
        result = json.loads(await ws.recv())
        print("English:", result.get('text', ''))

asyncio.run(dynamic_config())
```

### JavaScript — Browser Push-to-Talk

```javascript
const ws = new WebSocket('ws://server:8765/?language=zh&use_itn=true&ptt_mode=true');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
  if (typeof event.data !== 'string') return;
  const msg = JSON.parse(event.data);
  if (msg.type === 'transcription') {
    console.log('Recognized:', msg.text);
  } else if (msg.type === 'done') {
    console.log('Ready for next utterance');
  }
};

// When user releases button:
function onReleaseButton() {
  ws.send(JSON.stringify({ action: 'eof' }));
}
```

### Swift — iOS/macOS

```swift
// Connect with full parameter set
let url = URL(string: "ws://server:8765/?language=en&use_itn=true&ptt_mode=true")!
let ws = WebSocket(url: url)
ws.connect()

// Send audio chunks as Data
ws.send(audioChunk)

// Force transcribe
ws.send(try! JSONEncoder().encode(["action": "eof"]))
```

---

## 9. Audio Preprocessing

**Required format:** PCM 16-bit signed integer, 16 kHz, mono.

```bash
# Convert from common formats using ffmpeg
ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 output.wav
ffmpeg -i input.m4a -ac 1 -ar 16000 -sample_fmt s16 output.wav
```

```python
# Verify WAV file format
import wave
with wave.open('audio.wav', 'rb') as wf:
    assert wf.getnchannels() == 1,     "Must be mono"
    assert wf.getsampwidth() == 2,     "Must be 16-bit"
    assert wf.getframerate() == 16000, "Must be 16kHz"
```

---

## 10. Session Lifecycle

### VAD Mode Flow

```
Connect → info (with full config)
  ↓
[Binary PCM frames] ──────────────→ speech_start → transcription → ...
  ↓
[{"action": "eof"}] ─────────────→ transcription → done → reset
  ↓
[Disconnect] ────────────────────→ flush remaining audio
```

### Dynamic Config Flow

```
Connect with ?language=zh
  ↓
Send audio → Chinese transcription
  ↓
{"action": "config", "params": {"language": "en", "use_itn": true}}
  ↓
info (config updated)
  ↓
Send audio → English transcription with punctuation
```

---

## 11. Error Handling

| Scenario | Behavior |
|----------|----------|
| No speech detected | No `transcription` sent for that segment |
| Audio too short (< 10 ms) | Ignored, no transcription |
| EOF with no buffered speech | No `transcription` sent; `done` is still sent |
| Invalid query parameter | Server falls back to default, logs warning |
| Connection lost mid-utterance | Remaining audio processed and sent on close |
| Unsupported language code | Server falls back to `zh`, logs warning |

---

## 12. Performance

| Metric | Typical Value |
|--------|---------------|
| End-to-end latency (VAD mode) | 300–800 ms |
| End-to-end latency (PTT mode) | 100–300 ms (no grace period) |
| Audio chunk interval | 32–200 ms (per binary frame) |
| Max concurrent connections | ~10–20 per GPU |
| GPU memory usage | ~1–2 GB |
| CPU memory usage | ~2–4 GB |

---

## 13. Server Launch Options

```bash
# Start with default config
python server.py

# Specify host, port, device
python server.py --host 0.0.0.0 --port 8765 --device cuda:0

# Custom model path
python server.py --model-dir /path/to/SenseVoiceSmall

# Debug logging
python server.py --debug
```

All ASR and VAD parameters are runtime-configurable via URL query string — no server restart needed for parameter changes.
