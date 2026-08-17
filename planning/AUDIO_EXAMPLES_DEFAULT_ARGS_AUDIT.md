# Examples `# Default Args:` Sibling Audit

## Scope

Parent work fixed the missing `# Default Args:` header on
`examples/audio/sdl_nanoamp.nano` (the launcher was silently running the player
with no music directory). This audit enumerates **every** example under
`examples/` that reads `get_argv` / `argv` (or otherwise takes a path argument),
cross-references the launcher-consumed `# Default Args:` header, and reports
which hits need a header-only follow-up versus a header-plus-asset follow-up.

No example sources were changed in this task — findings are for sibling filing.

The `# Default Args:` header is parsed by the launcher's real header parser
(`examples/lib/example_discovery.nano :: parse_example_header`) and consumed by
`examples/lib/process_manager.nano :: pm_launch`, which appends the value to the
launch command. When the header is absent, `default_args` is the empty string,
so the launcher runs the example with **no arguments**.

Bundled audio inventory (whole tree): exactly one playable asset —
`examples/audio/gabba-studies-12.mod`. No `.wav` / `.mp3` / `.ogg` / `.xm` /
`.s3m` / `.it` files are present.

## Inventory (argv / path consumers)

| File | `# Default Args:` | Path/value | Diagnosis |
|------|-------------------|------------|-----------|
| `examples/audio/sdl_nanoamp.nano` | present | `examples/audio` (dir, non-empty scan) | OK — parent-fixed |
| `examples/audio/sdl_tracker_shell.nano` | present | `examples/audio` (dir) | OK |
| `examples/audio/sdl_mod_visualizer.nano` | present | `examples/audio/gabba-studies-12.mod` | OK — bundled MOD |
| `examples/audio/sdl_audio_player.nano` | present | `examples/audio/gabba-studies-12.mod` | OK — bundled MOD |
| `examples/audio/sdl_audio_wav.nano` | **absent** | — | **HIT — header + asset** (see Findings) |
| `examples/graphics/sdl_nanoviz.nano` | **absent** | self-defaults to `examples/audio/gabba-studies-12.mod` | **HIT — header-only** |
| `examples/run_examples.nano` | present | `examples/language` | OK |
| `examples/language/nl_pi_calculator.nano` | absent | optional **integer** (digits), not a path | Cleared — full demo runs with no args |
| `examples/opl/opl_cli.nano` | absent | developer CLI; usage exit by design | Cleared as launcher Default-Args defect; see ancillary note |
| `examples/api_lessons/env_api.nano` | absent | teaches `argc`/`argv` API only | Excluded — not a path consumer |
| `examples/audio/nanoamp_playlist_harness.nano` | n/a | headless harness (no `# Example:` / `# Build:`) | Excluded — not launcher-facing |
| `examples/lib/example_discovery.nano` | n/a | parses the header; not an example | Excluded — launcher machinery |

## Findings (file for sibling tasks)

1. **`examples/audio/sdl_audio_wav.nano` — header + asset (expensive class).**
   - No `# Default Args:` header, so the launcher runs it with no argument and
     `main()` exits 1 with a usage message (`argc < 2`).
   - It loads via `Mix_LoadWAV`, which only accepts WAV data. The repository
     ships no `.wav`, so a header alone cannot make a no-argument run succeed.
     Pointing a header at the bundled `.mod` would be a false fix (`Mix_LoadWAV`
     cannot decode a tracker module).
   - Stale tip path in the usage message: prints
     `examples/gabba-studies-12.mod` but the file lives at
     `examples/audio/gabba-studies-12.mod`.
   - **Signal:** same class that made the nanoamp follow-up more expensive than
     a pure header patch — needs a license-clean bundled `.wav` (or a redesign)
     before a `# Default Args:` line is useful.
   - Owned for remediation by the dedicated sibling WAV task; this audit does
     not change the example.

2. **`examples/graphics/sdl_nanoviz.nano` — header-only (cheap class).**
   - Reads `get_argv(1)` with no `# Default Args:` header.
   - When `argc < 2` it already self-defaults to the existing
     `examples/audio/gabba-studies-12.mod`, so a no-argument manual run works.
   - Adding `# Default Args: examples/audio/gabba-studies-12.mod` restores
     launcher consistency only; no new asset required.

## Default Args path hygiene

- No declared `# Default Args:` value is machine-specific.
- All present headers name repo-relative paths that exist:
  `examples/audio`, `examples/audio/gabba-studies-12.mod`, `examples/language`.
- Stale path found outside the header (usage tip only): see Finding 1.

## Ancillary (not a Default-Args defect)

- `examples/opl/opl_cli.nano` embeds a machine-local `nanoc` absolute path inside
  `cmd_build` (developer compile helper). That is outside the Default-Args
  contract but should be filed separately as a portability cleanup using a
  fleet-generic placeholder (for example `bin/nanoc` or `$NANO_ROOT/bin/nanoc`),
  not a personal checkout path.

## Regression guard

`tests/nl_functions_audio_examples_default_args.nano` generalises the parent's
single-file regression (`tests/nl_functions_nanoamp_default_args.nano`) to the
launcher-facing `examples/audio/` family. It asserts:

- each required audio example has a non-empty `# Default Args:` header that
  resolves to a playable default (non-empty directory scan, or a
  `.mod`/`.mp3`/`.ogg`/`.wav` file), and
- `sdl_audio_wav.nano` intentionally has **no** `# Default Args:` header until a
  bundled `.wav` exists, so the exclusion cannot silently regress into a broken
  `.mod` default.

The guard reuses the repository's real header parser and mirrors the players'
audio-extension predicate.
