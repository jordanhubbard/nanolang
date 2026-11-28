# My current Linux native component gate — 2026-11-27

At source `6e09c07a9f1651294e4669282cdc17ee14126775` I pass the original
bootstrap17, native36 and emitter12 sequence on Sparky Linux ARM64. My selected
source/tool/head guards match before/after. My actual generated index matches
the bootstrap receipt, as do all six host-library hashes after native execution.

| Original gate | Result | Make wall time |
| --- | --- | ---: |
| Bootstrap17, raw equality, installation/no-C-seed | Pass | 1160.081790s |
| Native36 | Pass | 34.866411s |
| Emitter12 | Pass | 257.231898s |

The emitter runner reports256.998s for all12 methods. Each setup retains the
original300s command limit. My bootstrap retains1800s stages,30s compiler shadows
and the ordinary10s default. Stage1/Stage2 generation takes456.165s/364.571s;
I make no cross-host or paired speedup claim from those times.

My raw Stage1 and Stage2 modules are byte-identical1,064,904-byte files, SHA-256
`8920ee0546671f9fa13ea98f7aaf5030b235d6d7349b277cdb875b4a5aa46833`.
The installed image matches Stage2 and the original no-C-seed probe succeeds.
I retain actual products, logs, module cache/host closure, generated index,
selected tools, observer and the terminal receipt.

My manifest SHA-256 is
`4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f`.
My [independent reader](read_current_native.py) verifies1,889 records/1,039 chunks,
4,676 Git source files, all17 step logs and products, source/tool/head equality,
the actual index/host closure, raw module equality and both original method counts.
I retain symlinks as metadata and verify `nanoc -> nanoc_stage2` without restoring
links on the replay host.

```sh
python3 docs/evidence/c-repairs-linux-20261126/native-current-linux-20261127/read_current_native.py \
  --archive docs/evidence/c-repairs-linux-20261126/native-current-linux-20261127/archive \
  --repo . --workspace "$HOME/.local/share/fleet-migration/puck-volume/nanolang/tmp"
```

I temporarily reconstruct about331MB on chosen storage and remove only my own
temporary directory. I establish no current liveness. My native tests use their
original cleanup; passing per-case temporary products are not all retained.
The retained compiler images and frozen tests support replay.

This accepts the Linux native component gate. Current Darwin qualification and
required exact-candidate CI remain. My separate broad structured-C target still
has2456 passes/two frozen CAST_STRING refusal failures; that contract decision
remains pending. I preserve every original assertion/deadline and all69 release
requirements. I accept no paired platform result or complete release here.
