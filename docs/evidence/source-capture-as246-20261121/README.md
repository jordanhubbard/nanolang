# My refused GNU assembler 2.46 candidate

I retain complete failure evidence from private candidate run
[44930824853](https://github.com/jordanhubbard/nanolang/actions/runs/44930824853)
at `4f74234b6` and focused native-debug run
[44935014281](https://github.com/jordanhubbard/nanolang/actions/runs/44935014281)
at `0885f2d50`. Both helpers restore their checkout changes. My product table
still admits GNU assembler 2.40 and 2.42; I have not promoted 2.46.

My actual hosted tools are GCC 15.2.0-16ubuntu1, GNU assembler 2.46 and Ubuntu
Clang 21.1.8 (6ubuntu1). My isolated candidate admits the exact assembler token
and adds malformed-banner controls before running the unchanged original
source-snapshot suite. All 128 methods run in 514.793 seconds: 39 subtest failures,
four errors and 24 platform skips. The candidate does not reach its subsequent
cold compiler-support gate.

I separate the failures:

- Under #1158, native selected-GCC response-file backslashes differ from my
  parser in two quoted forms and three flag-transport origins. My older grammar
  always consumes the next byte after a backslash; the actual driver preserves
  selected backslashes inside quotes. I require selected-driver semantics and
  old-driver controls before changing parsing; I do not infer grammar from a
  GCC major version.
- Under #1159, actual Ubuntu Clang 21.1.8 reports reach an unqualified banner
  boundary. I preserve single-command parsing, report validation and actual
  backend identities. I do not execute reported shell text or guess a backend.
- Under #1157, my original debug-provenance guard errors require actual object
  evidence. My separate four-method diagnostic now retains it.

The four-method diagnostic runs in 1.539 seconds, with four errors and no
assertion failures or skips. Two GCC cases retain `payload.S`, original `-g`,
native and physical object bytes, fixture trees and original normalized dumps.
Both objects contain `.debug_line` and `.debug_info`, with a compilation unit,
but their `DW_AT_name` and line table name `/tmp/cc….s` rather than `payload.S`.
This is an ASCII source-name failure, not a Unicode rendering mismatch.
My local independent readelf audit verifies all four retained object hashes and
reads the same temporary names from the actual bytes. I preserve every original
flag and assertion. The two Clang methods stop earlier during production capture
at the still-unqualified report boundary, before this native-provenance guard.

My diagnostic mode selects only these four methods and cannot claim complete
candidate or cold-capture acceptance. My default qualification still requires
all 128 original methods and actual cold compiler-support success. I retain the
first local workflow-test import refusal under the Homebrew Python without
PyYAML; the installed system Python runs all 11 original workflow controls
successfully. This setup correction does not alter hosted qualification.

`manifest.json` retains 111 files and 3,879,590 raw bytes as lossless gzip members,
with raw and compressed SHA-256 identities. It includes actual probes/helpers,
compiler/assembler command observations, source deltas, full test terminals,
failed native objects, source fixtures and independent object dumps. These are
failed toolchain qualifications. Original source/snippet/platform requirements
and the release remain open.
