# SenseVoice WebSocket ASR Server

A real-time, streaming-capable Automatic Speech Recognition (ASR) server that wraps Alibaba's **SenseVoiceSmall** model with **silero-vad** (Voice Activity Detection) and exposes a **WebSocket** interface for low-latency client-server communication.

```
┌──────────────┐     WebSocket (PCM chunks)     ┌──────────────────────┐
│  External    │ ──────────────────────────────> │  SenseVoice WS Server │
│  Device      │                                  │                      │
│  (Client)    │ <────────────────────────────── │  ┌────┐  ┌────────┐  │
└──────────────┘      JSON (transcriptions)      │  │VAD │→ │ASR Model│  │
                                                 │  └────┘  └────────┘  │
                                                 └──────────────────────┘
```

- **Input**: Streaming 16kHz 16-bit mono PCM audio via WebSocket binary frames
- **Output**: JSON text frames with transcribed speech per utterance
- **VAD**: Silero-VAD automatically detects utterance boundaries (no client-side split needed)
- **Language**: Supports Chinese, English, Cantonese, Japanese, Korean, and auto-detection
- **Model**: SenseVoiceSmall (~80M params, GPU or CPU)

## Features

- **Persistent connection**: One WebSocket connection for the entire session. Client sends audio continuously, server returns results per utterance.
- **Automatic utterance segmentation**: VAD detects speech start/end. No manual endpointing required.
- **Multi-language**: Specify language via URL query string: `ws://host:8765/?language=zh`
- **Raw ASR output**: No formatting or ITN applied — ideal for downstream LLM consumption.
- **Per-utterance timing**: Each result includes audio duration and inference latency.

## Quick Start

### Install

```bash
pip install funasr torch websockets silero-vad numpy pyyaml
```

For microphone testing: `pip install pyaudio`

### Start Server

```bash
python server.py --device cuda:0 --port 8765
```

### Test

```bash
python tests/client.py mic --language zh
python tests/client.py send /path/to/audio.wav --language en
```

## Language Support

| Code | Language | Example URL |
|------|----------|-------------|
| `zh` | Chinese (Mandarin) | `ws://host:8765/?language=zh` |
| `en` | English | `ws://host:8765/?language=en` |
| `yue` | Cantonese | `ws://host:8765/?language=yue` |
| `ja` | Japanese | `ws://host:8765/?language=ja` |
| `ko` | Korean | `ws://host:8765/?language=ko` |
| `auto` | Automatic detection | `ws://host:8765/?language=auto` |

## Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | WebSocket port |
| `--device` | `cuda:0` | Inference device (`cuda:0`, `cpu`) |
| `--model-dir` | (auto) | Path to SenseVoiceSmall model |
| `--debug` | (off) | Debug logging |

## API Documentation

See **[API.md](API.md)** for the complete developer reference, including:

- Protocol specification
- Integration examples (Python, JavaScript, C)
- Audio preprocessing guide
- Session lifecycle
- Performance metrics

## Architecture

### VAD State Machine

```
SILENT ──(speech detected)──> SPEAKING
                                 │
                   (silence > 600ms)
                                 │
                                 v
                              SILENT  ──emit speech_end──> ASR inference
```

- Model: silero-vad v5 (1.7MB, 95%+ accuracy)
- Frame: 512 samples (32ms at 16kHz)
- Grace period: 600ms configurable silence timeout

### Performance

- End-to-end latency: 300–800ms per utterance
- GPU memory: ~1–2 GB
- Concurrent connections: ~10–20 per GPU

## Deployment

```ini
# systemd service example
[Service]
ExecStart=/usr/bin/python3 /opt/sensevoice-ws/server.py --host 0.0.0.0 --port 8765
WorkingDirectory=/opt/sensevoice-ws
Restart=always
```

## Repository

- GitHub: https://github.com/lumicore-dev/sensevoice-ws
- License: MIT
