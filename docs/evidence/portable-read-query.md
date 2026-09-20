# I describe read-text imports without host authority

I qualified private query production `9005b9f50` at fixture pin `e9e6144cb`.
My later fixture `76ec98223` adds only a linked-mode zero-failure-counter
assertion; its fresh Homebrew linked result is attributed separately. The
query/report does not select a profile, prove operand types or managed lifetime,
open a file, call an adapter, emit code or execute a module. Taskb7ef and parent2d2
remain open for real adapters and the remaining host/module linkage contract.

## My measured configurations

| Host and compiler | Query acceptance |
| --- | --- |
| Linux AArch64, GCC13.3 ordinary | e9e: 5994 observed +266 linked checks |
| Linux AArch64, GCC13.3 ASan/UBSan/LSan | e9e: 5994 observed +266 linked checks |
| Linux AArch64, Clang23git11af80 ASan/UBSan/LSan | e9e: 5994 observed +266 linked checks |
| puck, Apple17, SDK26.2 ordinary | e9e: 5994 observed +266 linked checks |
| puck, Homebrew23.1.1 ASan/UBSan/LSan | e9e observed5994; fresh76ec linked267 |

I built fresh ordinary providers with GCC on Linux and Apple Clang on puck.
Each query configuration built its four query/decoder/verifier/type translation
units twice, with and without allocation hooks. Sanitizer flags instrument those
four units and the fixture; other linked providers remain ordinary. All sanitizer
runs use `detect_leaks=1`, `halt_on_error=1`, and empty `LSAN_OPTIONS`. I inventory
selected actual compiler/runtime files, not every transitive library or system
tool. Linux Clang explicitly selects the installed GCC13 toolchain; no warning
is suppressed. No source bootstrap or adapter acceptance is claimed.

Every observed run reports29 attributed structural INVALID failures,7 report
MEMORY failures and10 advisory-type-only successful transient queries. The
fixture visits all23 baseline allocation indices in both persistent and transient
modes, checks exact input bytes/pointers and zero tracked allocation balance,
and recovers successfully after each attempt. These results preserve the shared
verifier's advisory type limitation rather than disguising it as a type proof.
Copied report rows remain valid after input destruction. All three existing
closed verifier profiles still refuse the exact import-bearing module.

Fresh unchanged neighbors pass on both hosts:96 generic verifier tests, the
verifier allocation-cleanup fixture, and2968 NanoISA tests. Their binaries remain
retained. I copied their Make recipes with external binary destinations; no old
failed binary was replayed and no successful query configuration was repeated
merely to replace a later harness terminal.

## My retained terminals and bounded corrections

1. Initial Linux runtime inventory stopped before build: Clang returned an
   unresolved architecture-suffixed ASan basename. I preserve command status0
   and the external runner's status1 separately. Corrected inventory requires
   the actual target-directory shared/static runtime paths.
2. After all three Linux query configurations and two neighbors passed, the ISA
   neighbor compilation failed on missing `nanoisa.h`. The external include
   path was corrected to Make's `modules/nanoisa`. A fresh build/run passes.
3. Puck Homebrew observed checks passed, then linked compilation rejected the
   fixture's unread failure counter under strict warnings. The two-line76ec
   assertion correction changes no production or prior assertion. Fresh linked
   build/run and the three previously unreached neighbors pass. Earlier
   source/tool trees and all successful/failed products remain unchanged.

The roots are `/tmp/nanolang-portable-read-{e9-linux,corrected-linux,resume-linux}`
and puck's `/private/tmp/nanolang-portable-read-{corrected-puck,resume-puck}`.
The fixture correction is supplied as a separate retained C input to the new
Homebrew link; the qualified checkout still contains exact e9e source. Original
passes are not relabeled as execution of the corrected checkpoint.

## My seal and limits

[The report manifest](portable-read-query/report-sha256.json) seals committed
reports, commands, logs, input maps and exact drivers. Linux contributes285 raw
reports and puck195. I independently rehashed634 unique retained artifacts
(423363786 bytes), consolidated at `/tmp/nanolang-portable-read-artifacts`.
The summary records22 equal before/after input-map pairs, plus successful current
rehashes of every endpoint input. Source scope is participating C/header/include
files plus named fixtures/Make/contract; tools/providers are explicitly listed.
These endpoint comparisons do not establish intermediate immutability or a
complete system dependency closure. The report records the exact scope.

The complete puck seal archive is retained at
`/tmp/nanolang-portable-read-puck-seal.tar.gz`, SHA256
`7d731dc4378903a5d58d03f80a88fa30acc0714bbf699baafb2ffdc94e8a5a7f`.
Its downloaded content and each consolidated artifact were rehashed. No release,
public admission, linked-module semantics, filesystem adapter or full2d2 closure
is inferred from this declaration-only checkpoint.
