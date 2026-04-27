#!/usr/bin/env python3
"""
SenseVoice WebSocket ASR Server

A real-time speech recognition server that accepts streaming audio
via WebSocket and returns transcribed text using SenseVoiceSmall.

Architecture:
  Client → WebSocket (binary PCM chunks) → VAD (silero-vad)
    → Speech Segment Detected → SenseVoiceSmall → Text Result → WebSocket → Client

Usage:
  python server.py --host 0.0.0.0 --port 8765

Clients can specify language via URL query string:
  ws://host:8765/?language=zh
  ws://host:8765/?language=en
  ws://host:8765/?language=yue
  ws://host:8765/?language=ja
  ws://host:8765/?language=ko
  ws://host:8765/?language=auto

EOF Protocol:
  Client sends {"action": "eof"} to immediately transcribe buffered audio
  without waiting for VAD silence detection. Useful for push-to-talk apps
  where the user releases the button to indicate end-of-speech.

Requirements:
  - funasr
  - torch
  - websockets
  - silero-vad
  - pyyaml
"""

import asyncio
import json
import logging
import os
import argparse
import time
from urllib.parse import urlparse, parse_qs

import websockets
import numpy as np

from vad.vad import VoiceActivityDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensevoice-ws')

# Supported language codes
SUPPORTED_LANGUAGES = {'zh', 'en', 'yue', 'ja', 'ko', 'auto'}

# ---------------------------------------------------------------------------
# ASR Engine
# ---------------------------------------------------------------------------

class SenseVoiceEngine:
    """
    Wraps the SenseVoiceSmall model from FunASR.
    Loaded once and shared across all connections.
    """

    def __init__(self, model_dir: str = None, device: str = 'cuda:0'):
        self.device = device
        self.model = None
        self.model_dir = model_dir or os.environ.get(
            'SENSEVOICE_MODEL_DIR',
            '/home/zhyi/.cache/modelscope/hub/iic/SenseVoiceSmall'
        )
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading SenseVoiceSmall from {self.model_dir} on {self.device} ...")
        from funasr import AutoModel

        self.model = AutoModel(
            model=self.model_dir,
            trust_remote_code=True,
            device=self.device,
            disable_update=True,
        )
        logger.info("SenseVoiceSmall loaded successfully")

    def transcribe(self, audio_bytes: bytes, language: str = 'zh', sample_rate: int = 16000) -> dict:
        """
        Run ASR on raw PCM audio bytes.

        Args:
            audio_bytes: Raw PCM 16-bit mono audio data
            language: Language code (zh, en, yue, ja, ko, auto)
            sample_rate: Audio sample rate (must match model expected rate)

        Returns:
            dict with 'text', 'duration_sec', 'inference_ms'
        """
        if not audio_bytes or len(audio_bytes) < 320:  # < 10ms audio
            return {'text': '', 'duration_sec': 0, 'inference_ms': 0}

        duration_sec = len(audio_bytes) / 2 / sample_rate
        t0 = time.time()

        # Write to temp file (FunASR generate() requires file path)
        # Using tmpfs to avoid disk I/O
        tmp_path = f'/dev/shm/_sensevoice_tmp_{id(audio_bytes)}.wav'
        try:
            self._write_wav(tmp_path, audio_bytes, sample_rate)
            res = self.model.generate(
                input=tmp_path,
                language=language,
                use_itn=False,
            )
            text = res[0]['text'].strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        inference_ms = (time.time() - t0) * 1000

        return {
            'text': text,
            'duration_sec': round(duration_sec, 2),
            'inference_ms': round(inference_ms, 1),
        }

    @staticmethod
    def _write_wav(path: str, audio_bytes: bytes, sample_rate: int):
        """Write PCM data to a WAV file."""
        import wave
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)


# ---------------------------------------------------------------------------
# WebSocket Handler
# ---------------------------------------------------------------------------

class AudioSession:
    """
    Manages one client connection's audio session.
    Accumulates chunks and runs VAD + ASR.
    """

    def __init__(self, engine: SenseVoiceEngine, language: str = 'zh', sample_rate: int = 16000):
        self.engine = engine
        self.language = language
        self.sample_rate = sample_rate
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.buffer = bytearray()
        self.samples_accumulated = 0
        self.total_audio_ms = 0

    def reset(self):
        """Reset VAD and audio buffer for a new utterance."""
        self.vad.reset()
        self.buffer.clear()
        self.samples_accumulated = 0
        self.total_audio_ms = 0

    async def feed_audio(self, chunk: bytes):
        """
        Feed incoming audio chunk. Returns list of result events.
        """
        self.buffer.extend(chunk)
        self.samples_accumulated += len(chunk) // 2

        results = []

        # Process in 512-sample frames (silero-vad requirement)
        frame_size = 1024  # 512 samples * 2 bytes
        while len(self.buffer) >= frame_size:
            frame = bytes(self.buffer[:frame_size])
            self.buffer = self.buffer[frame_size:]

            event = self.vad.process_chunk(frame)
            self.total_audio_ms += 32

            if event['event'] == 'speech_end':
                audio = event['buffer']
                transcription = self.engine.transcribe(
                    audio,
                    language=self.language,
                    sample_rate=self.sample_rate
                )
                if transcription['text']:
                    results.append({
                        'type': 'transcription',
                        'text': transcription['text'],
                        'duration_sec': transcription['duration_sec'],
                        'inference_ms': transcription['inference_ms'],
                    })
            elif event['event'] == 'speech_start':
                results.append({'type': 'speech_start'})

        return results

    async def force_transcribe(self) -> dict:
        """
        Immediately transcribe all buffered audio (triggered by EOF signal).
        Bypasses VAD grace period — returns result right away.

        Returns:
            dict with transcription result, or None if audio is too short.
        """
        # Collect remaining partial VAD frame + any speech buffer
        partial_frame = bytes(self.buffer)
        self.buffer.clear()

        speech_audio = self.vad.force_flush()

        full_audio = partial_frame + speech_audio

        if not full_audio or len(full_audio) < 320:
            return None

        transcription = self.engine.transcribe(
            full_audio,
            language=self.language,
            sample_rate=self.sample_rate
        )
        if transcription['text']:
            return {
                'type': 'transcription',
                'text': transcription['text'],
                'duration_sec': transcription['duration_sec'],
                'inference_ms': transcription['inference_ms'],
            }
        return None

    def flush(self):
        """Process any remaining audio (on connection close)."""
        if len(self.buffer) >= 320:
            audio = bytes(self.buffer)
            transcription = self.engine.transcribe(
                audio,
                language=self.language,
                sample_rate=self.sample_rate
            )
            self.buffer.clear()
            if transcription['text']:
                return [{
                    'type': 'transcription',
                    'text': transcription['text'],
                    'duration_sec': transcription['duration_sec'],
                    'inference_ms': transcription['inference_ms'],
                }]
        return []


def parse_language_from_path(path: str) -> str:
    """Extract language from WebSocket URL query string."""
    parsed = urlparse(f"http://localhost{path}")
    params = parse_qs(parsed.query)
    lang = params.get('language', ['zh'])[0].lower()
    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language '{lang}', falling back to 'zh'")
        return 'zh'
    return lang


async def handle_client(websocket: websockets.WebSocketServerProtocol, engine: SenseVoiceEngine):
    """
    Handle one WebSocket client connection.

    Protocol:
      - Client connects via ws://host:port/?language=zh
      - Client sends raw PCM 16kHz 16-bit mono audio as binary frames
      - Client may send {"action": "eof"} to force immediate transcription
      - Server sends JSON text frames:
        {"type": "speech_start"}
        {"type": "transcription", "text": "...", "duration_sec": 1.23, "inference_ms": 45.6}
        {"type": "done"}  (after EOF transcription is complete)
        {"type": "info", "message": "..."}
        {"type": "error", "message": "..."}
    """
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

    # Parse language from URL query string
    language = parse_language_from_path(websocket.path)
    logger.info(f"Client connected: {client_id}, language={language}, path={websocket.path}")

    session = AudioSession(engine, language=language)

    try:
        await websocket.send(json.dumps({
            'type': 'info',
            'message': 'Connected. Send PCM 16kHz 16-bit mono audio.',
            'config': {
                'sample_rate': session.sample_rate,
                'model': 'SenseVoiceSmall',
                'vad': 'silero-vad',
                'language': language,
            }
        }))

        async for message in websocket:
            if isinstance(message, bytes):
                results = await session.feed_audio(message)
                for result in results:
                    await websocket.send(json.dumps(result, ensure_ascii=False))
            else:
                # text message — control commands
                try:
                    cmd = json.loads(message)
                    action = cmd.get('action')

                    if action == 'reset':
                        session = AudioSession(engine, language=language)
                        await websocket.send(json.dumps({'type': 'info', 'message': 'Session reset'}))

                    elif action == 'eof':
                        # Force transcribe buffered audio immediately
                        result = await session.force_transcribe()
                        if result:
                            await websocket.send(json.dumps(result, ensure_ascii=False))
                        # Signal that EOF processing is complete
                        await websocket.send(json.dumps({'type': 'done'}))
                        # Reset for next utterance
                        session.reset()

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON command'
                    }))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    finally:
        # Flush remaining audio
        remaining = session.flush()
        for result in remaining:
            try:
                await websocket.send(json.dumps(result, ensure_ascii=False))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Server Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='SenseVoice WebSocket ASR Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Bind host')
    parser.add_argument('--port', type=int, default=8765, help='Bind port')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='SenseVoiceSmall model directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Inference device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 50)
    logger.info("SenseVoice WebSocket ASR Server")
    logger.info("=" * 50)
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.model_dir or 'default'}")
    logger.info(f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}")

    # Initialize ASR engine (shared across all connections)
    engine = SenseVoiceEngine(
        model_dir=args.model_dir,
        device=args.device,
    )

    # Start WebSocket server
    start_server = websockets.serve(
        lambda ws: handle_client(ws, engine),
        args.host,
        args.port,
        ping_interval=30,
        ping_timeout=10,
        max_size=2**20,
    )

    logger.info(f"Server listening on ws://{args.host}:{args.port}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    main()
