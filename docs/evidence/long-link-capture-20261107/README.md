# My long Darwin link capture repair

I reproduce #1077 with 40 shared C sources under a long path containing an
apostrophe and a dollar sign. The actual shared-object paths exceed 8,192
bytes. Before this change, the library builds but `source_hashes.json` is
absent: Darwin's 8 KiB recording wrapper cannot hold the admitted link recipe.

I allocate recording storage from the actual command and the worst-case
quoted dependency-record path, within the existing 65,536-byte recipe and
2,047-byte record-path bounds. I free that storage on every completed capture
or ordinary-link path. I change no linker arguments, snapshot policy, final-link
failure handling, cache-admission checks or bootstrap closure equality.

My focused native regression now saves the dependency evidence and reuses
the exact generation on an unchanged build. A changed private source creates
a new generation returning the new value; the old library's bytes and value
remain intact. My retained before/after logs record the same test's refusal
and success.

My local Darwin arm64 checks pass:

- The focused shared-closure regression in 7.287 seconds.
- `make test-module-builder-self-capture`: six methods in 16.832 seconds;
  three GCC-only methods are explicitly skipped on this Apple Clang run.
- Actual failed-final-link publication preservation, framework flag pairing,
  tail flags and repeated-library controls: three methods in 3.377 seconds.
- `make check-fast`: five real-lexer fixture controls, 3,120 inspected static
  fragments with zero unacknowledged reserved names, five prerequisite controls,
  223 strict Clang and 223 strict GCC product compilations and the bootstrap-tool
  control. I still report 133 lexer-refused static fragments separately.

I retain the earlier complete long-path refusal under
`docs/evidence/long-path-host-closure-20261107/`. I require a new fresh full
Stage 1 bootstrap at a path of at least 300 characters after this correction.
These targeted checks do not establish that complete acceptance or a release.
