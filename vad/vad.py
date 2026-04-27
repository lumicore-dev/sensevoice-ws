#!/usr/bin/env python3
"""
VAD module using silero-vad for voice activity detection.
Detects speech segments from raw PCM audio chunks.
"""

import numpy as np
import torch


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
    """

    def __init__(self, sample_rate: int = 16000, grace_period_ms: int = 600, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.grace_frames = int(sample_rate / 512 * grace_period_ms / 1000)  # silero uses 512-frame windows
        self.threshold = threshold

        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        self.model.eval()

        self.state = VADState.SILENT
        self.silent_count = 0
        self.speech_buffer = []

    def reset(self):
        self.state = VADState.SILENT
        self.silent_count = 0
        self.speech_buffer = []

    def process_chunk(self, audio_chunk: bytes) -> dict:
        """
        Process a PCM chunk (must be 512 samples = 32ms at 16kHz).
        Returns event dict:
          - {'event': 'silence'}
          - {'event': 'speech_start'}
          - {'event': 'speaking', 'buffer': accumulated_audio}
          - {'event': 'speech_end', 'buffer': accumulated_audio}
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
                if self.silent_count >= self.grace_frames:
                    full_audio = b''.join(self.speech_buffer)
                    self.reset()
                    return {'event': 'speech_end', 'buffer': full_audio}
                return {'event': 'speaking', 'buffer': b''.join(self.speech_buffer)}

        return {'event': 'error', 'message': 'unknown state'}
