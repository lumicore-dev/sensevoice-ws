#!/usr/bin/env python3
"""
SenseVoice WebSocket ASR Server

A real-time speech recognition server that accepts streaming audio
via WebSocket and returns transcribed text using SenseVoiceSmall.

Architecture:
  Client -> WebSocket (binary PCM chunks) -> VAD (silero-vad)
    -> Speech Segment Detected -> SenseVoiceSmall -> Text Result -> WebSocket -> Client

Usage:
  python server.py --host 0.0.0.0 --port 8765

Clients can specify parameters via URL query string:
  ws://host:8765/?language=zh&use_itn=true

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
SUPPORTED_LANGUAGES = {'zh', 'en', 'yue', 'ja', 'ko', 'auto', 'nospeech'}

# Default values for all session parameters (matches API.md v2.0)
DEFAULT_PARAMS = {
    # SenseVoice ASR
    'language': 'zh',
    'use_itn': False,
    'ban_emo_unk': False,
    'batch_size_s': 60,
    'merge_vad': False,
    'merge_length_s': 15,
    # Output post-processing
    'rich_postprocess': False,
    # VAD
    'vad_threshold': 0.5,
    'vad_grace_period_ms': 600,
    'ptt_mode': False,
    # Audio
    'sample_rate': 16000,
}

# Types for each parameter (for conversion)
PARAM_TYPES = {
    'language': str,
    'use_itn': bool,
    'ban_emo_unk': bool,
    'batch_size_s': int,
    'merge_vad': bool,
    'merge_length_s': int,
    'rich_postprocess': bool,
    'vad_threshold': float,
    'vad_grace_period_ms': int,
    'ptt_mode': bool,
    'sample_rate': int,
}


# ---------------------------------------------------------------------------
# Parameter Parsing
# ---------------------------------------------------------------------------

def parse_params_from_path(path: str) -> dict:
    """
    Extract all session parameters from WebSocket URL query string.
    Returns a dict with all keys from DEFAULT_PARAMS, overridden by any
    values present in the query string.
    """
    parsed = urlparse(f"http://localhost{path}")
    query_params = parse_qs(parsed.query)

    params = dict(DEFAULT_PARAMS)

    for key, default_val in DEFAULT_PARAMS.items():
        if key in query_params:
            raw = query_params[key][0]
            param_type = PARAM_TYPES[key]
            try:
                if param_type == bool:
                    # Accept 'true'/'false', '1'/'0', 'yes'/'no'
                    params[key] = raw.lower() in ('true', '1', 'yes')
                else:
                    params[key] = param_type(raw)
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for '{key}': '{raw}', using default '{default_val}'")
                params[key] = default_val

    # Validate language
    if params['language'] not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language '{params['language']}', falling back to 'zh'")
        params['language'] = 'zh'

    # Validate sample_rate
    if params['sample_rate'] not in (8000, 16000):
        logger.warning(f"Unsupported sample_rate '{params['sample_rate']}', falling back to 16000")
        params['sample_rate'] = 16000

    return params


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
        self.postprocess_fn = None
        self.model_dir = model_dir or os.environ.get(
            'SENSEVOICE_MODEL_DIR',
            '/home/zhyi/.cache/modelscope/hub/iic/SenseVoiceSmall'
        )
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading SenseVoiceSmall from {self.model_dir} on {self.device} ...")
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.model = AutoModel(
            model=self.model_dir,
            trust_remote_code=True,
            device=self.device,
            disable_update=True,
        )
        self.postprocess_fn = rich_transcription_postprocess
        logger.info("SenseVoiceSmall loaded successfully")

    def transcribe(
        self,
        audio_bytes: bytes,
        params: dict = None,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Run ASR on raw PCM audio bytes.

        Args:
            audio_bytes: Raw PCM 16-bit mono audio data
            params: Dict of session parameters (use_itn, ban_emo_unk, etc.)
            sample_rate: Audio sample rate (must match model expected rate)

        Returns:
            dict with 'text', 'duration_sec', 'inference_ms'
        """
        if not audio_bytes or len(audio_bytes) < 320:  # < 10ms audio
            return {'text': '', 'duration_sec': 0, 'inference_ms': 0}

        if params is None:
            params = {}

        duration_sec = len(audio_bytes) / 2 / sample_rate
        t0 = time.time()

        # Write to temp file (FunASR generate() requires file path)
        tmp_path = f'/dev/shm/_sensevoice_tmp_{id(audio_bytes)}.wav'
        try:
            self._write_wav(tmp_path, audio_bytes, sample_rate)

            # Build kwargs for model.generate() from params
            generate_kwargs = {
                'input': tmp_path,
                'language': params.get('language', 'zh'),
                'use_itn': params.get('use_itn', False),
            }

            # Pass optional SenseVoice params only if explicitly provided
            if 'ban_emo_unk' in params:
                generate_kwargs['ban_emo_unk'] = params['ban_emo_unk']
            if 'batch_size_s' in params:
                generate_kwargs['batch_size_s'] = params['batch_size_s']
            if 'merge_vad' in params:
                generate_kwargs['merge_vad'] = params['merge_vad']
            if 'merge_length_s' in params:
                generate_kwargs['merge_length_s'] = params['merge_length_s']

            res = self.model.generate(**generate_kwargs)
            text = res[0]['text'].strip()

            # Apply rich post-processing if enabled
            if params.get('rich_postprocess', False) and self.postprocess_fn:
                try:
                    text = self.postprocess_fn(text)
                except Exception as e:
                    logger.warning(f"rich_postprocess failed: {e}")

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

    def __init__(self, engine: SenseVoiceEngine, params: dict = None):
        if params is None:
            params = dict(DEFAULT_PARAMS)
        self.engine = engine
        self.params = params
        self.sample_rate = params.get('sample_rate', 16000)

        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            grace_period_ms=params.get('vad_grace_period_ms', 600),
            threshold=params.get('vad_threshold', 0.5),
            ptt_mode=params.get('ptt_mode', False),
        )
        self.buffer = bytearray()
        self.samples_accumulated = 0
        self.total_audio_ms = 0
        # Stores the last VAD-triggered transcription result for EOF fallback.
        # Cleared after use to prevent duplicate delivery.
        self.last_vad_result = None

    def reset(self):
        """Reset VAD and audio buffer for a new utterance."""
        self.vad.reset()
        self.buffer.clear()
        self.samples_accumulated = 0
        self.total_audio_ms = 0
        # NOTE: last_vad_result persists intentionally — it's cleared by
        # force_transcribe() / flush() after fallback use, not by reset().

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
                logger.info(f"VAD speech_end: transcribing {len(audio)} bytes ({len(audio)/2/self.sample_rate:.2f}s)")
                transcription = self.engine.transcribe(
                    audio,
                    params=self.params,
                    sample_rate=self.sample_rate,
                )
                if transcription['text']:
                    result = {
                        'type': 'transcription',
                        'text': transcription['text'],
                        'duration_sec': transcription['duration_sec'],
                        'inference_ms': transcription['inference_ms'],
                    }
                    # Save for EOF fallback
                    self.last_vad_result = result
                    results.append(result)
                else:
                    logger.info("VAD speech_end: no text in transcription")
            elif event['event'] == 'speech_start':
                results.append({'type': 'speech_start'})

        return results

    async def force_transcribe(self) -> dict:
        """
        Immediately transcribe all buffered audio (triggered by EOF signal).
        Bypasses VAD grace period — returns result right away.

        - If substantial buffered audio (> 50ms / 1600 bytes), transcribe fresh.
        - If only residual audio (<= 50ms) remains (VAD already triggered
          speech_end), falls back to last_vad_result.
        - Fallback result is cleared after use to prevent duplicate delivery.

        Returns:
            dict with transcription result, or None if nothing available.
        """
        # Collect remaining partial VAD frame + any speech buffer
        partial_frame = bytes(self.buffer)
        self.buffer.clear()
        speech_audio = self.vad.force_flush()
        full_audio = partial_frame + speech_audio

        MIN_EOF_AUDIO_BYTES = 1600  # 50ms of 16kHz PCM

        if full_audio and len(full_audio) >= MIN_EOF_AUDIO_BYTES:
            logger.info(f"force_transcribe: transcribing {len(full_audio)} bytes ({len(full_audio)/2/self.sample_rate:.2f}s)")
            transcription = self.engine.transcribe(
                full_audio,
                params=self.params,
                sample_rate=self.sample_rate,
            )
            if transcription['text']:
                result = {
                    'type': 'transcription',
                    'text': transcription['text'],
                    'duration_sec': transcription['duration_sec'],
                    'inference_ms': transcription['inference_ms'],
                }
                # New transcription supersedes any previous VAD result
                self.last_vad_result = None
                return result

        # Audio too short or no text — fall back to VAD result (once only)
        if self.last_vad_result:
            logger.info(f"force_transcribe: residual audio ({len(full_audio) if full_audio else 0} bytes), "
                        f"re-sending last VAD result: '{self.last_vad_result['text']}'")
            result = self.last_vad_result
            self.last_vad_result = None  # ← Clear to prevent duplicate delivery
            return result

        logger.info("force_transcribe: no audio and no VAD result available")
        return None

    def flush(self):
        """Process any remaining audio (on connection close)."""
        # Collect all remaining audio (same logic as force_transcribe)
        partial_frame = bytes(self.buffer)
        self.buffer.clear()
        speech_audio = self.vad.force_flush()
        full_audio = partial_frame + speech_audio

        MIN_EOF_AUDIO_BYTES = 1600

        if full_audio and len(full_audio) >= MIN_EOF_AUDIO_BYTES:
            logger.info(f"flush: transcribing {len(full_audio)} bytes on disconnect")
            transcription = self.engine.transcribe(
                full_audio,
                params=self.params,
                sample_rate=self.sample_rate,
            )
            if transcription['text']:
                self.last_vad_result = None
                return [{
                    'type': 'transcription',
                    'text': transcription['text'],
                    'duration_sec': transcription['duration_sec'],
                    'inference_ms': transcription['inference_ms'],
                }]

        if self.last_vad_result:
            logger.info(f"flush: re-sending last VAD result: '{self.last_vad_result['text']}'")
            result = self.last_vad_result
            self.last_vad_result = None  # ← Clear to prevent duplicate delivery
            return [result]

        return []


async def handle_client(websocket: websockets.WebSocketServerProtocol, engine: SenseVoiceEngine):
    """
    Handle one WebSocket client connection.

    Protocol:
      - Client connects via ws://host:port/?language=zh&use_itn=true&...
      - Client sends raw PCM 16kHz 16-bit mono audio as binary frames
      - Client may send {"action": "eof"} to force immediate transcription
      - Client may send {"action": "config", "language": "en", "use_itn": true} to update params mid-session
      - Server sends JSON text frames:
        {"type": "speech_start"}
        {"type": "transcription", "text": "...", "duration_sec": 1.23, "inference_ms": 45.6}
        {"type": "done"}  (after EOF transcription is complete)
        {"type": "info", "message": "...", "config": {...}}
        {"type": "error", "message": "..."}
    """
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

    # Parse all parameters from URL query string
    params = parse_params_from_path(websocket.path)
    logger.info(f"Client connected: {client_id}, path={websocket.path}")

    # Log resolved params
    param_log = {k: v for k, v in params.items() if k != 'language'}
    logger.info(f"Client config: {client_id}, language={params['language']}, params={param_log}")

    session = AudioSession(engine, params=params)

    try:
        await websocket.send(json.dumps({
            'type': 'info',
            'message': 'Connected. Send PCM 16kHz 16-bit mono audio.',
            'config': {
                'model': 'SenseVoiceSmall',
                'vad': 'silero-vad',
                **params,
            }
        }))

        async for message in websocket:
            if isinstance(message, bytes):
                results = await session.feed_audio(message)
                for result in results:
                    await websocket.send(json.dumps(result, ensure_ascii=False))
            else:
                # text message — control commands
                logger.info(f"Received text from {client_id}: {message}")
                try:
                    cmd = json.loads(message)
                    action = cmd.get('action')

                    if action == 'reset':
                        session = AudioSession(engine, params=params)
                        await websocket.send(json.dumps({'type': 'info', 'message': 'Session reset'}))

                    elif action == 'config':
                        # Update session parameters mid-session
                        # Parameters can be at top level OR nested under "params" key
                        config_source = cmd.get('params', cmd)
                        for key in DEFAULT_PARAMS:
                            if key in config_source:
                                session.params[key] = config_source[key]
                        # Recreate VAD if VAD params changed
                        session.vad = VoiceActivityDetector(
                            sample_rate=session.params.get('sample_rate', 16000),
                            grace_period_ms=session.params.get('vad_grace_period_ms', 600),
                            threshold=session.params.get('vad_threshold', 0.5),
                            ptt_mode=session.params.get('ptt_mode', False),
                        )
                        await websocket.send(json.dumps({
                            'type': 'info',
                            'message': 'Config updated',
                            'config': dict(session.params),
                        }))

                    elif action == 'eof':
                        t_eof = time.perf_counter()
                        # Force transcribe buffered audio immediately
                        result = await session.force_transcribe()
                        elapsed_ms = (time.perf_counter() - t_eof) * 1000
                        if result:
                            logger.info(f"EOF result: '{result['text']}' "
                                        f"(server_time={elapsed_ms:.1f}ms, "
                                        f"inference={result['inference_ms']}ms)")
                            await websocket.send(json.dumps(result, ensure_ascii=False))
                        else:
                            logger.info(f"EOF result: no transcription available "
                                        f"(server_time={elapsed_ms:.1f}ms)")
                        # Signal that EOF processing is complete
                        await websocket.send(json.dumps({'type': 'done'}))
                        logger.info(f"Sent 'done' to {client_id} (total={elapsed_ms:.1f}ms)")
                        # Reset for next utterance (last_vad_result already cleared by force_transcribe)
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
        import traceback
        logger.error(traceback.format_exc())
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
