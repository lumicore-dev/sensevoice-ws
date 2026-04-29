#!/usr/bin/env python3
"""
VAD module using silero-vad for voice activity detection.
Detects speech segments from raw PCM audio chunks.

Model is loaded once at module level and cached for all instances.
Uses local cached JIT model — no GitHub dependency.
"""

import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Global model cache — loaded once, reused by all VAD instances
# ---------------------------------------------------------------------------
_VAD_MODEL = None

# Path to locally cached silero-vad JIT model
_VAD_JIT_PATH = os.path.expanduser(
    '~/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit'
)


def _load_vad_model():
    """Load silero-vad JIT model from local disk cache (once)."""
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL

    if not os.path.exists(_VAD_JIT_PATH):
        raise RuntimeError(
            f"silero-vad JIT model not found at {_VAD_JIT_PATH}. "
            "Please run: python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')\" "
            "to download the model first."
        )

    print(f"[VAD] Loading JIT model from: {_VAD_JIT_PATH}")
    _VAD_MODEL = torch.jit.load(_VAD_JIT_PATH)
    _VAD_MODEL.eval()
    print("[VAD] Model loaded and cached globally")
    return _VAD_MODEL


class VADState:
    SILENT = 0
    SPEAKING = 1
    GRACE = 2  # silence timeout before cutting


class VoiceActivityDetector:
    """
    Real-time VAD using silero-vad model.
    
    State machine:
      SILENT → (speech detected) → SPEAKING
      SPEAKING → (silence > grace_period) → GRACE → emit event → SILENT
    
    In PTT (push-to-talk) mode, the VAD will NOT auto-trigger speech_end.
    Speech end is only triggered by explicit EOF command from client.

    The underlying ML model is loaded once globally and shared across all
    instances — creating a VoiceActivityDetector is cheap.
    """

    def __init__(self, sample_rate: int = 16000, grace_period_ms: int = 600, threshold: float = 0.5, ptt_mode: bool = False):
        self.sample_rate = sample_rate
        self.grace_frames = int(sample_rate / 512 * grace_period_ms / 1000)  # silero uses 512-frame windows
        self.threshold = threshold
        self.ptt_mode = ptt_mode

        # Use globally cached model — no redundant loading per connection
        self.model = _load_vad_model()

        self.state = VADState.SILENT
        self.silent_count = 0
        self.speech_buffer = []

    def reset(self):
        self.state = VADState.SILENT
        self.silent_count = 0
        self.speech_buffer = []

    def force_flush(self) -> bytes:
        """
        Immediately return all accumulated speech audio regardless of VAD state.
        Resets the VAD state afterwards. Used for explicit EOF from client.

        Returns:
            bytes: Accumulated speech audio (empty if no speech was detected)
        """
        audio = b''.join(self.speech_buffer)
        self.reset()
        return audio

    def process_chunk(self, audio_chunk: bytes) -> dict:
        """
        Process a PCM chunk (must be 512 samples = 32ms at 16kHz).
        Returns event dict:
          - {'event': 'silence'}
          - {'event': 'speech_start'}
          - {'event': 'speaking', 'buffer': accumulated_audio}
          - {'event': 'speech_end', 'buffer': accumulated_audio}  (only in non-PTT mode)
          - {'event': 'error', 'message': ...}
        """
        # Convert bytes to numpy
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Get speech probability
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0)
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob > self.threshold

        if self.state == VADState.SILENT:
            if is_speech:
                self.state = VADState.SPEAKING
                self.speech_buffer = [audio_chunk]
                self.silent_count = 0
                return {'event': 'speech_start'}
            return {'event': 'silence'}

        elif self.state == VADState.SPEAKING:
            self.speech_buffer.append(audio_chunk)
            if is_speech:
                self.silent_count = 0
                return {'event': 'speaking', 'buffer': b''.join(self.speech_buffer)}
            else:
                self.silent_count += 1
                # In PTT mode, never auto-trigger speech_end
                if not self.ptt_mode and self.silent_count >= self.grace_frames:
                    full_audio = b''.join(self.speech_buffer)
                    self.reset()
                    return {'event': 'speech_end', 'buffer': full_audio}
                return {'event': 'speaking', 'buffer': b''.join(self.speech_buffer)}

        return {'event': 'error', 'message': 'unknown state'}
