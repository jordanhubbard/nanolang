# My native preprocessed-assembly provenance controls

I investigate the unchanged native `.S` debug assertion under #1157. My
[original failure receipt](../source-capture-as246-20261121) retains the complete
private qualification failures, actual native objects and flags. I change
neither product admission nor those assertions in this diagnostic.

[44945288089](https://github.com/jordanhubbard/nanolang/actions/runs/44945288089)
first shows that a simple `.S` source keeps `payload.S` identity under actual
hosted GCC 15/assembler 2.46. The original include/macro failure therefore
cannot be generalized to every preprocessed assembly input.

[44947026342](https://github.com/jordanhubbard/nanolang/actions/runs/44947026342)
compares seven controlled variants with original native `-c -fPIC -g` flags.
GCC preserves identity for an inline macro but reports `/tmp/cc*.s` for plain
assembler includes, included macros and the original include/incbin form.
Clang 21 preserves identity for all variants. Raw `.s` marker witnesses keep
`payload.S` even without an added marker after the include. The actual local
GCC 13/assembler 2.42 and Clang controls preserve source identity throughout.

[44950231043](https://github.com/jordanhubbard/nanolang/actions/runs/44950231043)
then replays actual saved preprocessing bytes through the selected compiler's
assembler path. Hosted GCC/assembler still records the replay `.s` name for
the include case. Removing either the working-directory marker or the
`stdc-predef.h` marker restores the original `payload.S` identity; replacing
all preprocessing markers with a single source marker also restores it.
The local older-tool replay keeps `payload.S` in every variant. I retain the
actual source bytes, literal argv, native/replay objects, debug reports,
selected-tool hashes and preprocessing output for these comparisons.

These controls localize a selected-tool interaction after preprocessing.
They do not establish an exact upstream source-code cause or make the
modified-marker counterfactual a passing original native test. I preserve
the original compiler flags and the failed general assembler qualification.
GNU assembler 2.46 admission and complete source/platform acceptance remain
open. My manifest retains 681 logical inputs, 2,108,670 raw bytes and 134,672
unique compressed bytes, with checked raw and compressed SHA-256 identities.
