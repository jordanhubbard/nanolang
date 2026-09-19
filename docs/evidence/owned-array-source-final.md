# I qualify exact owner-array source lowering on Linux and Darwin

I qualify the corrected six-method source fixture at frozen
`1c4e29067aa21ee843e00997d42851877f7082fe` on both platforms. This includes
canonical parser827 and File-priority public activation842. My paired producer
files remain byte-identical to reviewed99fdb4804; the corrected fixture remains
byte-identical to32f2f1398. I retain the earlier cumulative Linux qualification
and both fixture-phase failures in [their separate seal](owned-array-source-linux.md).

| Phase | Linux GCC | Darwin Apple producer / Homebrew native |
| --- | ---: | ---: |
| Fresh bootstrap | PASS272.544s | PASS342.563s |
| C tools/probes | PASS27.076s | PASS81.772s |
| Corrected six source methods | PASS408.316s | PASS438.494s |
| Source files unchanged |2268|2268|
| Post-setup inputs unchanged |799|646|
| Retained producers/drivers |13|13|
| Host tools |7|7|
| Retained artifact files |401|401|

Each full source-method duration includes its separately inventoried emitter/
shadow-driver setup. I compare the unchanged original Bundle/PREFIX through
Cseed, Stage1, Stage2 and NanoVirt, preserving every original shadow. Canonical
program/shadow dumps, retained and stripped metadata, raw emitters, verified VM
and generated-native execution pass. Integer minimum/wrap/zero-divisor/range/
Boolean controls, nested return/prepared-root/alias/empty-array controls, optional
FLOAT operand runtime checks and cleanup, refusal/output sentinels and original
Samples/STRING/scalar/borrow neighbors all pass.

Native generated C uses strict C11/O2 with address/undefined sanitizers, `-lm`,
and `detect_leaks=1:halt_on_error=1`. Linux explicitly selects GCC. Darwin builds
producers with `/usr/bin/clang` and the recorded Xcode macOS27 SDK, then selects
`/opt/homebrew/opt/llvm/bin/clang` after driver setup for native checks. The
resolved Homebrew binary hash is captured separately from producer selection.
Darwin uses verified GNU Make3.81 at `/usr/bin/make`; gmake was absent during
preflight. Its real Git checkout retains the same original commit, not a
synthetic source snapshot. No shared host toolchain is changed.

The [combined report manifest](owned-array-source-final/report-sha256.json)
covers75 report files, including each platform's36-report seal and manifest.
Each sealing script checks current source/tool/producer/native-compiler hashes
against retained before/after maps and verifies post-setup input equality before
creating the archive. I retain setup-created outputs separately from inputs.

- Linux archive: `/tmp/nanolang-owner-array-source-final-linux-artifacts.tar.gz`,
  SHA256 `31d2a4cc4e44f5dc233be444845862cca6fd1bd33a7ddda718abe3bdb3f24457`.
- Darwin archive on CXWWHGGJX0:
  `/private/tmp/nanolang-owner-array-source-final-darwin-artifacts.tar.gz`,
  SHA256 `16632830aab248cb4c51bfae03565656a0efa2e8b402a82fef476761e15e94f4`.

My [ready integration](owned-array-source-final/ready-integration.json) adds
canonical19b0 private File CODE query providers/target in a new worktree after
qualification. The two producer files, corrected fixture, parser and public
verifier/VM/native selection paths remain exact. I retain both original1c4e
trees/tools and claim no new bootstrap or gate on this additive private-query
integration. ROADMAP conflicts preserve both histories and canonical completion
rows; no production conflict was resolved by editing source.

The bounded source18731 and fixture repaird27a acceptance is qualified and awaits
actual merge-ledger reconciliation. Mutation set/push/length taskbba622, owner-array
parent430220, mixed4be, product and release remain open. This source slice does
not claim ordinary Samples plus owner-array composition, borrowed managed fields,
bare ARRAY signatures, mutable owner fields or new File execution authority.
