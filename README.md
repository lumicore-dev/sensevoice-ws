# SenseVoice WebSocket ASR Server

A real-time, streaming-capable Automatic Speech Recognition (ASR) server that wraps Alibaba's **SenseVoiceSmall** model with **silero-vad** (Voice Activity Detection) and exposes a **WebSocket** interface for low-latency client-server communication.

## Overview

```
┌──────────────┐     WebSocket (PCM chunks)     ┌──────────────────────┐
│   External   │ ──────────────────────────────> │  SenseVoice WS Server │
│   Device     │                                  │                      │
│  (Client)    │ <────────────────────────────── │  ┌────┐  ┌────────┐  │
└──────────────┘      JSON (transcriptions)      │  │VAD │→ │ASR Model│  │
                                                 │  └────┘  └────────┘  │
                                                 └──────────────────────┘
```

- **Input**: Streaming 16kHz 16-bit mono PCM audio via WebSocket binary frames
- **Output**: JSON text frames with transcribed speech, inference timing, and session metadata
- **VAD**: Silero-VAD detects speech/silence boundaries; transcription is triggered when a complete utterance is detected
- **Model**: SenseVoiceSmall (FunASR) — ~80M parameters, GPU or CPU inference

## Features

- Real-time WebSocket server with asynchronous I/O
- Voice Activity Detection (silero-vad) for automatic utterance segmentation
- SenseVoiceSmall ASR with Chinese language support and Inverse Text Normalization (ITN)
- Per-utterance inference timing for performance monitoring
- Graceful connection handling — processes remaining audio on client disconnect
- Configurable model path, device (CUDA/CPU), and ITN settings
- Includes a test client with file send, batch directory, and live microphone modes

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (CPU fallback available)
- FunASR + SenseVoiceSmall model (auto-downloaded or cached)

### Dependencies

```
funasr
torch>=2.0.0
websockets>=10.0
silero-vad>=0.2.0
numpy
pyyaml
```

For microphone mode in the test client:
```
pyaudio
```

## Quick Start

### 1. Install

```bash
pip install funasr torch websockets silero-vad numpy pyyaml
```

For microphone testing:
```bash
pip install pyaudio
```

### 2. Start the Server

```bash
python server.py --host 0.0.0.0 --port 8765 --device cuda:0
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | WebSocket port |
| `--device` | `cuda:0` | Inference device (`cuda:0`, `cuda:1`, `cpu`) |
| `--model-dir` | (auto) | Path to SenseVoiceSmall model directory |
| `--no-itn` | (ITN on) | Disable Inverse Text Normalization |
| `--debug` | (off) | Enable debug logging |

### 3. Test with a Client

**Send a single WAV file:**
```bash
python tests/client.py send /path/to/audio.wav --host localhost --port 8765
```

**Batch process a directory:**
```bash
python tests/client.py dir /path/to/wavs/ --host localhost --port 8765
```

**Live microphone input:**
```bash
python tests/client.py mic --host localhost --port 8765
```

## WebSocket Protocol

### Client → Server

Send raw PCM audio frames as **binary messages**:
- Format: 16-bit signed integer (little-endian), mono
- Sample rate: 16000 Hz
- Chunk size: any size (internally processed in 512-sample frames for VAD)

Optionally, send **text messages** for control:
```json
{"action": "reset"}
```

### Server → Client

All messages are **JSON-encoded text frames**:

**Connection info (sent on connect):**
```json
{
  "type": "info",
  "message": "Connected...",
  "config": {
    "sample_rate": 16000,
    "model": "SenseVoiceSmall",
    "vad": "silero-vad"
  }
}
```

**Speech detected:**
```json
{"type": "speech_start"}
```

**Transcription result:**
```json
{
  "type": "transcription",
  "text": "请问信用卡的授信额度通常是多少",
  "duration_sec": 2.34,
  "inference_ms": 45.6
}
```

**Error:**
```json
{
  "type": "error",
  "message": "description of the error"
}
```

### Audio Format Requirements

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| Bit Depth | 16-bit signed |
| Channels | Mono (1) |
| Byte Order | Little-endian |
| Encoding | Linear PCM |

## Architecture

### Server Components

1. **`server.py`** — Main entry point. Sets up WebSocket server and shared ASR engine.
2. **`vad/vad.py`** — Voice Activity Detection using silero-vad with a 3-state machine: `SILENT → SPEAKING → GRACE → SILENT`.
3. **`tests/client.py`** — Test client with file, batch, and microphone modes.

### VAD State Machine

```
SILENT ──(speech detected)──> SPEAKING
                                 │
                   (silence > 600ms)        (speech resumes)
                                 │                  │
                                 v                  │
                              GRACE ──(timeout)──> SILENT
                                 │
                           emit speech_end
```

- **Grace period**: 600ms of silence before cutting an utterance (configurable)
- **Threshold**: Speech probability > 0.5

### Performance

Measured on testYL dataset (2589 utterances, 16kHz clean audio):
- **Average inference time**: ~100-400ms per utterance (GPU)
- **End-to-end latency**: ~300-800ms (including VAD grace period)
- **Model**: SenseVoiceSmall, ~80M parameters
- **WER**: 14.76% on conversational Chinese test set

## Deployment

### As a System Service (systemd)

```ini
[Unit]
Description=SenseVoice WebSocket ASR Server
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/sensevoice-ws/server.py --host 0.0.0.0 --port 8765
WorkingDirectory=/opt/sensevoice-ws
Restart=always
User=asr
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN pip install funasr websockets silero-vad numpy pyyaml

COPY . /app
WORKDIR /app

EXPOSE 8765

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765"]
```

### Resource Requirements

- **GPU Memory**: ~1-2GB (CUDA inference)
- **CPU Memory**: ~2-4GB (CPU inference)
- **Disk**: ~1GB for model cache
- **Python**: 3.8+

## Limitations

- SenseVoiceSmall is a **non-streaming** (full-utterance) model. True character-by-character streaming is not supported. The VAD + chunk approach provides utterance-level streaming with ~300-800ms latency.
- Input is limited to 16kHz mono PCM. Other formats must be pre-converted on the client side.
- The server processes one utterance at a time per connection. Concurrent connections share the same GPU model instance.

## License

This project is provided as-is. SenseVoiceSmall is licensed under the MIT License by Alibaba DAMO Academy. Silero-VAD is licensed under the MIT License.

## Author

LumiCore (CEO Assistant)
