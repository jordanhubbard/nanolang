# I refuse incomplete inverse CODE publication

I qualify frozen36711fbb4 under my [reviewed contract](../NVM_V2_CODE_PUBLICATION.md).
The production delta is confined to inverse-v2 CODE allocation/copy: exact checked
reserve, temporary realloc pointer, legacy TRUNCATED on allocation failure, NULL
output and complete cleanup. I retain the shared append ABI and other callers.
The fixture inspects representation/failure results without executing their CODE.

| First fresh focused phase | Seconds | Result |
| --- | ---: | --- |
| Linux GCC ordinary |0.465|PASS|
| Linux Clang18 ordinary |0.415|PASS|
| Linux Clang18 ASan/UBSan |1.468|PASS|
| Linux GCC ASan/UBSan |1.168|PASS|
| puck Apple ordinary |0.842|PASS|
| puck Homebrew ordinary |1.041|PASS|
| puck Homebrew ASan/UBSan |1.239|PASS|

Every focused run reports28856 checks. Sizes0/4095/4096/4097/8193 preserve exact
bytes, zeros, size and independent output lifetime. A one-shot failure targets
only the exact initial CODE buffer's8193-byte realloc; it disarms immediately.
The corrected converter returns TRUNCATED with NULL output before any later
allocation, preserves input bytes/fields, releases every tracked temporary block
and succeeds on fresh same-process recovery. Invalid arguments, nonempty NULL
CODE, over-u32 size and earlier constant-error precedence retain exact behavior.

On each host both ordinary compilers also pass existing v2 conversion365/0,
private File nominal19743 and owned-array authority5061 controls. Their ordinary
build/run and query phases are retained separately. Linux has14 gate phases,
Darwin13, each after a separate successful provider setup. None failed. Private
File hosted qualified trees and their exact-CODE containment remain unchanged.

The [Linux seal](nvm-v2-code-publication-linux.json) records206 files,12 selected
inputs,7190 full tracked source hashes and42 provider/tool identities. The
[Darwin seal](nvm-v2-code-publication-darwin.json) records281 files, the same12
input hashes,7190 full tracked sources and101 provider/compiler/configuration/
sanitizer-runtime identities. Sources match before setup, before gates and after;
all per-phase tool maps equal the prepared map and current files. Compiler version,
host, SDK and command records are retained. This inventories selected providers
and tools, not every transitive system component.

Darwin is puck arm64 with explicit Apple/Xcode and resolved Homebrew LLVM tools;
SDK and exact versions are in its environment/version files. Sanitizer phases
retain detect_leaks=1:halt_on_error=1 and undefined-behavior halt/stacktrace settings.
The new fixture plus converter/format translation units are instrumented; ordinary
linked validators are not claimed as fully instrumented or fault-injected.
All phase processes have600-second TERM/KILL bounds and both outer1800-second
bounds return0. Original commands, outputs, binaries and prepared tools remain
under /tmp/nanolang-code-publication-linux and, on puck,
/private/tmp/nanolang-code-publication-darwin.

Taskd34 awaits actual canonical review/merge. I grant no hosted File execution,
new profile admission or parent/release completion through this converter repair.


## I check the combined private hosted query

I integrate canonical853 (1e64676e1) into ready b84f95c70 and qualify fresh
separate trees. My eleven non-Make selected inputs remain identical to36711;
canonical File header dependencies and test targets are additive. I preserve the
original qualified36711 and private hosted9ab trees and reports.

| Combined phase | Linux seconds | puck seconds |
| --- | ---: | ---: |
| Hosted ordinary |4.425|3.955|
| Hosted sanitizer |14.248|9.133|
| Nonexecuting conversion |0.415|0.587|
| C-seed module compile/run |7.831/0.004|6.738/0.283|
| Native wrapper compile/run |0.214/0.004|0.488/0.329|

All seven phases per host pass after fresh provider/C-seed/NanoVirt setup. Each
hosted configuration reports597901 instrumented and2518 linked checks:435
transient refusals, zero complete recovered plans,55 precise MEMORY and380
ambiguous UNRESOLVED allocation prefixes. This is the unchanged hosted fixture
with corrected shared CODE publication, not a weakened recovery expectation.
Conversion again reports28856 nonexecuting checks. Module and wrapper smoke
execute their ordinary programs; no File handler or representation-failure CODE
executes. I do not repeat a self-host bootstrap for this converter-local change.

My [combined Linux seal](nvm-v2-code-publication-combined-linux.json) records392
files,7274 tracked sources and159 prepared providers/CLIs/tools. My
[combined puck seal](nvm-v2-code-publication-combined-darwin.json) records453
files,7274 sources and217 prepared providers/CLIs/compiler/configuration/sanitizer
files. Both have19 selected inputs. Every phase checks the exact prepared file
set before and after; module cache outputs are new artifacts, not prepared
providers. Actual module/wrapper outputs and hosted temporary artifacts are
retained. Sources match before setup, before gates and after. Linux uses GCC
ordinary and Clang18 sanitizer; puck uses Apple ordinary and Homebrew LLVM
sanitizer. Hosted fixtures rebuild the thirteen allocating provider units with
sanitizers and allocation interposition where selected; remaining linked units
are ordinary. Leak detection stays enabled. Each phase has its own600-second
TERM/KILL bound. The Linux outer1800-second bound also returns0.

These combined reports are separate from my first seals. I retain all original
acceptance, and taskd34 still awaits actual canonical merge.
