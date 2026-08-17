# Audio Examples `# Default Args:` Sibling Audit

## Scope

Parent task `task_50f2668605ce4d79b54fabf91b4b8f18` fixed the missing/broken
`# Default Args:` header on `examples/audio/sdl_nanoamp.nano` (the launcher was
silently running the player with no music directory). This audit checks the
**sibling** audio examples under `examples/audio/` for the same defect and
reports findings.

The `# Default Args:` header is parsed by the launcher's real header parser
(`examples/lib/example_discovery.nano :: parse_example_header`) and consumed by
`examples/lib/process_manager.nano :: pm_launch`, which appends the value to the
launch command. When the header is absent, `default_args` is the empty string,
so the launcher runs the example with **no arguments**. For an example whose
`main()` requires an argument, that no-argument launch fails (usage message,
non-zero exit), which is the defect under audit.

## Files audited

| File | `# Default Args:` | Value | Verdict |
|------|-------------------|-------|---------|
| `examples/audio/sdl_nanoamp.nano` | present | `examples/audio` (dir) | OK — parent-fixed; scan is non-empty |
| `examples/audio/sdl_tracker_shell.nano` | present | `examples/audio` (dir) | OK — scan is non-empty |
| `examples/audio/sdl_mod_visualizer.nano` | present | `examples/audio/gabba-studies-12.mod` (file) | OK — bundled, playable MOD |
| `examples/audio/sdl_audio_player.nano` | present | `examples/audio/gabba-studies-12.mod` (file) | OK — bundled, playable MOD |
| `examples/audio/sdl_audio_wav.nano` | **absent** | — | **Same defect class**; see Findings |
| `examples/audio/nanoamp_playlist_harness.nano` | n/a | — | Not launcher-facing (headless harness); excluded |

## Findings

1. **`sdl_audio_wav.nano` has the same missing-`# Default Args:` defect.**
   Its `main()` returns `1` with a usage message when `argc < 2`, and it loads
   the file via `Mix_LoadWAV`, which only accepts WAV data. The repository
   bundles exactly one audio asset — `examples/audio/gabba-studies-12.mod` —
   and **no** `.wav`. There is therefore no valid default this example could
   point at:
   - Pointing it at the bundled `.mod` would be a false fix: `Mix_LoadWAV`
     cannot decode a tracker module, so a no-argument run would still fail (now
     at load time instead of at the usage check).
   - Adding a header naming a non-existent `.wav` would fail the same header
     validity check the parent task introduced.

   **Recommendation:** keep `sdl_audio_wav.nano` argument-required (no
   `# Default Args:` header) until a small, license-clean `.wav` asset is added
   to `examples/audio/`. Once such an asset exists, add
   `# Default Args: examples/audio/<file>.wav` and extend the guard below.
   The example already prints a helpful tip pointing users at
   `sdl_mod_visualizer` for a zero-argument audio smoke test, so no runtime
   change is required now.

2. **The four MOD/directory players are correct.** All resolve to an existing
   path that surfaces at least one playable audio file, so the launcher can run
   them with no arguments.

3. **`nanoamp_playlist_harness.nano` is correctly excluded.** It is a headless
   verification harness (no `# Example:` / `# Build:` metadata) that is not
   surfaced in the launcher UI, so it needs no `# Default Args:` header. Its own
   prose already documents the `examples/audio` default it mirrors.

## Regression guard

`tests/nl_functions_audio_examples_default_args.nano` generalises the parent's
single-file regression (`tests/nl_functions_nanoamp_default_args.nano`) to the
whole family. It runs under `make test-quick` (auto-discovered by the
`tests/nl_functions_*.nano` glob in `tests/run_all_tests.sh`) and asserts:

- each launcher-facing audio example has a non-empty `# Default Args:` header
  that resolves to a playable default (a directory whose audio scan is
  non-empty, or a file with a `.mod/.mp3/.ogg/.wav` extension), and
- `sdl_audio_wav.nano` intentionally has **no** `# Default Args:` header, so the
  documented exclusion cannot silently regress into a broken `.mod` default.

The guard reuses the repository's real header parser and mirrors the players'
audio-extension predicate rather than re-implementing them.
