# Audio example assets

## `nanolang-test-tone.wav`

A 1.5 second A-major arpeggio (A4, C#5, E5, A5) used as the bundled default
for `sdl_audio_wav.nano`, so the WAV playback example is runnable without
supplying a file.

| Property | Value |
| --- | --- |
| Format | RIFF/WAVE, uncompressed Microsoft PCM |
| Bit depth | 16-bit signed |
| Sample rate | 44100 Hz |
| Channels | 1 (mono) |
| Duration | 1.500 s |
| Size | 132,344 bytes |

**Provenance and licensing.** This file is original project content. It is not
sampled, derived from, or adapted from any third-party recording. It is
produced entirely by `scripts/generate_test_tone.py`, which synthesizes it
from sine waves using only the Python standard library (`wave`, `struct`,
`math`). It is released under the same terms as the rest of the repository,
and is additionally dedicated to the public domain under CC0 1.0.

Generating the asset rather than importing one keeps its provenance
self-evident: there is no external source to audit, and nothing is downloaded
at build time.

**Regenerating.** The generator is deterministic — identical bytes on every
run and every platform — so the committed asset can be verified by
regenerating it and checking that nothing changed:

```sh
python3 scripts/generate_test_tone.py
git diff --exit-code examples/audio/nanolang-test-tone.wav
```

To validate the container independently:

```sh
file examples/audio/nanolang-test-tone.wav
python3 -c "import wave; w=wave.open('examples/audio/nanolang-test-tone.wav'); \
print(w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())"
```

## `gabba-studies-12.mod`

A tracker module, used as the default for the `.mod`-playing examples
(`sdl_mod_visualizer.nano`, `sdl_audio_player.nano`, `sdl_nanoamp.nano`).
`Mix_LoadWAV` cannot play it, which is why the WAV example needs its own
asset rather than pointing at this one.
