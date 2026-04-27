#!/usr/bin/env python3
"""
WebSocket ASR Client - sends audio files and prints recognition results.

Usage:
  # Send a single WAV file
  python tests/client.py send /path/to/file.wav

  # Send all WAV files from a directory
  python tests/client.py dir /path/to/wavs/

  # Interactive microphone mode (requires pyaudio)
  python tests/client.py mic

Options:
  --host HOST       Server host (default: localhost)
  --port PORT       Server port (default: 8765)
  --language LANG   Language code: zh, en, yue, ja, ko, auto (default: zh)
  --rate RATE       Sample rate (default: 16000)
"""

import asyncio
import json
import sys
import os
import argparse

import websockets


def build_ws_url(host: str, port: int, language: str) -> str:
    return f"ws://{host}:{port}/?language={language}"


async def send_audio(websocket, audio_path: str):
    """Send a WAV file and receive transcription."""
    import wave

    with wave.open(audio_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "Mono audio required"
        assert wf.getsampwidth() == 2, "16-bit audio required"
        assert wf.getframerate() == 16000, "16kHz audio required"

        frames = wf.readframes(wf.getnframes())

    # Read welcome message
    welcome = await websocket.recv()
    info = json.loads(welcome)
    print(f"  Server: {info['message']}  (language: {info['config']['language']})")

    # Send audio in chunks (simulating streaming)
    chunk_size = 3200  # 100ms of audio
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        await websocket.send(chunk)
        await asyncio.sleep(0.01)

    # Wait for results
    await asyncio.sleep(0.5)
    try:
        while True:
            msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
            data = json.loads(msg)
            if data['type'] == 'transcription':
                print(f"  Result [{os.path.basename(audio_path)}]: {data['text']}")
                print(f"    Audio: {data['duration_sec']}s, Inference: {data['inference_ms']}ms")
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
        pass


async def send_directory(host: str, port: int, language: str, directory: str, ext: str = '.wav'):
    """Send all WAV files from a directory."""
    files = sorted([f for f in os.listdir(directory) if f.endswith(ext)])
    if not files:
        print(f"No {ext} files found in {directory}")
        return

    print(f"Sending {len(files)} files from {directory} ...")

    for fname in files:
        fpath = os.path.join(directory, fname)
        try:
            async with websockets.connect(build_ws_url(host, port, language)) as ws:
                await send_audio(ws, fpath)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")


async def mic_mode(host: str, port: int, language: str):
    """Interactive microphone mode."""
    try:
        import pyaudio
    except ImportError:
        print("pyaudio required for microphone mode. Install: pip install pyaudio")
        sys.exit(1)

    CHUNK = 1600  # 100ms at 16kHz
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    async with websockets.connect(build_ws_url(host, port, language)) as ws:
        welcome = await ws.recv()
        info = json.loads(welcome)
        print(f"Connected: {info['message']}  (language: {info['config']['language']})")
        print("Speak into the microphone. Press Ctrl+C to stop.\n")

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                await ws.send(data)
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(msg)
                    if data['type'] == 'speech_start':
                        print("[listening...]", end=' ', flush=True)
                    elif data['type'] == 'transcription':
                        print(f"\n[{data['text']}]")
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    parser = argparse.ArgumentParser(description='WebSocket ASR Client')
    parser.add_argument('mode', choices=['send', 'dir', 'mic'],
                        help='send: single file, dir: directory, mic: microphone')
    parser.add_argument('path', nargs='?', default=None,
                        help='File path or directory path')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--language', default='zh',
                        help='Language code: zh, en, yue, ja, ko, auto (default: zh)')
    parser.add_argument('--rate', type=int, default=16000, help='Sample rate')

    args = parser.parse_args()

    if args.mode == 'mic':
        asyncio.run(mic_mode(args.host, args.port, args.language))
    elif args.mode == 'send':
        if not args.path:
            print("Error: path required for 'send' mode")
            sys.exit(1)
        async def _send():
            async with websockets.connect(build_ws_url(args.host, args.port, args.language)) as ws:
                await send_audio(ws, args.path)
        asyncio.run(_send())
    elif args.mode == 'dir':
        if not args.path:
            print("Error: path required for 'dir' mode")
            sys.exit(1)
        asyncio.run(send_directory(args.host, args.port, args.language, args.path))


if __name__ == '__main__':
    main()
