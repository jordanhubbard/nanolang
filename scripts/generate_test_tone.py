#!/usr/bin/env python3
"""Generate the bundled WAV test asset without external packages.

The repository needs a small, unambiguously license-clean WAV that
``Mix_LoadWAV`` can play, so ``examples/audio/sdl_audio_wav.nano`` has
something to default to. Generating it here rather than importing a
third-party file keeps provenance self-evident: the asset is whatever this
script emits, and nothing has to be downloaded at build time.

Output is deterministic -- the same bytes on every run, on every platform --
so the committed asset can be regenerated and diffed as a verification step.

Usage:
    python3 scripts/generate_test_tone.py [output.wav]

Defaults to examples/audio/nanolang-test-tone.wav relative to the repository
root. Verify a checkout with:

    python3 scripts/generate_test_tone.py && git diff --exit-code examples/audio
"""

from __future__ import annotations

import math
import pathlib
import struct
import sys
import wave

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1
DURATION_S = 1.5
AMPLITUDE = 0.35  # headroom below full scale; avoids clipping on mixdown

# An A-major arpeggio: A4, C#5, E5, A5. Equal-tempered, rounded to 3 decimals
# so the frequencies are exact literals rather than platform-dependent
# computations.
NOTES_HZ = (440.000, 554.365, 659.255, 880.000)

# Per-note and whole-file fades. Without these the note boundaries and the
# buffer edges produce audible clicks, which is exactly what a playback smoke
# test should not be teaching people to ignore.
NOTE_FADE_S = 0.008
FILE_FADE_S = 0.050


def _envelope(index: int, total: int, note_index: int, note_len: int) -> float:
    """Amplitude scale at ``index``, combining the note and file envelopes."""
    note_pos = index - (note_index * note_len)
    note_fade = NOTE_FADE_S * SAMPLE_RATE
    file_fade = FILE_FADE_S * SAMPLE_RATE

    note_env = min(1.0, note_pos / note_fade, (note_len - note_pos) / note_fade)
    file_env = min(1.0, index / file_fade, (total - index) / file_fade)
    return max(0.0, min(note_env, file_env))


def render() -> bytes:
    """Render the arpeggio as little-endian signed 16-bit PCM frames."""
    total = int(SAMPLE_RATE * DURATION_S)
    note_len = total // len(NOTES_HZ)
    frames = bytearray()

    for index in range(total):
        note_index = min(index // note_len, len(NOTES_HZ) - 1)
        freq = NOTES_HZ[note_index]
        phase = 2.0 * math.pi * freq * (index / SAMPLE_RATE)
        value = AMPLITUDE * _envelope(index, total, note_index, note_len) * math.sin(phase)
        frames += struct.pack("<h", int(value * 32767.0))

    return bytes(frames)


def write(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(render())


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        target = pathlib.Path(argv[1])
    else:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        target = repo_root / "examples" / "audio" / "nanolang-test-tone.wav"

    write(target)
    print("wrote %s (%d bytes)" % (target, target.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
