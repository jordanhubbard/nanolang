# My source snapshot boundary

On 2026-09-12 I tested the production module builder at `34fab3e9` with
Apple clang 21.0.0 (`clang-2100.1.1.101`) on Darwin. My cache checks preprocessing
before and after compiling live source. Those observations can agree even when
the compiler read different bytes between them.

```sh
make obj/test_module_generation_probe
python3 -m tests.characterize_source_snapshot
python3 -m tests.characterize_source_snapshot --require-consistent
```

The first experiment command exits zero when measurement succeeds. It is not
an acceptance gate. The second exits nonzero when warm and fresh results differ.

I use a controlled compiler wrapper to replace `42` with `43` only during the
actual C compilation, then restore the original bytes and modification time
before returning. This models a concurrent edit-and-restore without timing
guesses. It does not claim to sandbox an adversarial compiler. I separately
exercise a direct source edit and an included-header edit, in local and shared
caches. Each library is loaded in a fresh process.

| Input changed and restored | Cache | Cold | Warm | Fresh | C compilations, cold → warm |
| --- | --- | ---: | ---: | ---: | --- |
| Source | Local | 43 | 43 | 42 | 1 → 1 |
| Source | Shared | 43 | 43 | 42 | 1 → 1 |
| Header | Local | 43 | 43 | 42 | 1 → 1 |
| Header | Shared | 43 | 43 | 42 | 1 → 1 |

All cases restore bytes, size and modification time, create a reuse record,
and retain the same generation on the warm build. The consistency gate fails.
I have reproduced incorrect reuse, not just a theoretical race.

## Required repair

I must bind reuse evidence to inputs actually consumed by compilation. Another
before/after content hash alone cannot close this edit-and-restore gap. A
supported snapshot mode must compile retained input bytes, not merely inspect
them in a separate probe. Its acceptance needs ordinary and shared-only C
sources, includes and search precedence, original diagnostic paths, and
failed-capture recovery without publishing partial artifacts.

Preprocessed C is a candidate for ordinary translation units, not a universal
replacement for compiler modes. PCH, compiler modules, assembly inclusion,
indirect flags and compiler-specific inputs need explicit treatment. I must
test their behavior rather than silently changing or declaring support for
them. Link input snapshots and runtime dynamic-library retention remain
separate requirements. This experiment changes no production cache behavior.

## Retained ordinary Clang C

I now capture preprocessed `.c` translation units into private `.i` files and
compile those exact retained files for ordinary Clang builds. Eligibility
requires a successful Clang version query, no custom compiler flags on any
platform, and no pkg-config entries. Declared include directories still work.
Other modes retain their original compilation and cache checks; the full
snapshot requirement remains open.

I compute the preprocessing fingerprint while writing the retained bytes,
then require a fresh matching fingerprint before recording reuse. This closes
the experiment's observation gap: a restored edit during actual C compilation
cannot change the retained input. If an edit occurs during capture instead,
I compile the captured bytes but withhold reuse when the later observation
differs. Failed or empty capture falls back to the original compile without
reuse evidence. Failed compilation preserves the previous generation.

`python3 -m unittest tests.test_source_snapshots` checks the four restored-edit
cases, multiple and shared-only sources, edits during capture, original-source
diagnostics, failed replacement preservation, recovery, and unchanged reuse.
On ordinary Clang, the original consistency experiment now returns 42 for
cold, warm and fresh builds and retains its generation with one C compilation.
These tests explicitly skip other compilers; they are not GCC acceptance.

The retained files are part of the immutable generation. My v15 build context
invalidates older records. Compiler/version wrappers remain trusted; this is
not compiler authentication, filesystem isolation, or an atomic snapshot of
all inputs. Implicit PCH selection on GCC, custom compiler modes, assembler
inputs and transitive tool identity need separate acceptance before widening
this boundary.

All four methods pass on Apple clang 21 normally and with ASan/UBSan at `-O1`
and recovery disabled. The sanitizer build instruments the production builder
through its probe and the linked cJSON, UTF-8, module-build-directory and FFI
loader support sources. Fixture compilers/libraries and system libraries are
not instrumented. A separate isolated, network-disabled Linux GCC 12 `-O3
-Werror` probe passes all four Linux linker tests and explicitly skips the
Clang-only snapshot suite. That verifies the tested Linux fallback behavior,
not a Linux snapshot repair.

## GCC ordinary inputs and implicit PCH

I extended ordinary retained translation units to GCC after reproducing an
implicit-PCH cache defect against `397d7d17` on Linux arm64, GCC 12.2.0
(`Debian 12.2.0-14+deb12u1`). With `answer.h` defining 43, I built a reusable
generation returning 43. I then created a usable `answer.h.gch` defining 42
and restored the header's bytes and modification time. Fresh compilation
returned 42, but the old builder retained its 43 generation.

Plain `-E` returned text containing 43. With `-fpch-preprocess`, GCC instead
emitted `#pragma GCC pch_preprocess` naming the selected PCH. I now use that
option for GCC capture and warm validation. A PCH marker makes capture
ineligible: I compile the original source and withhold reuse evidence. I do
not compile a purported snapshot that still loads live PCH bytes. Without
the marker, I compile and validate the retained translation unit just as for
ordinary Clang C. My v16 build context invalidates older records.

All six snapshot methods pass on GCC, including the four restored source/header
cases, multiple/shared-only sources, original-source diagnostics, failed
replacement recovery, capture-time edits, PCH appearance/removal and every
internal marker split across the 4096-byte read boundary. The PCH split test
checks the actual compiler argument path, not only the absence of a cache
record: compilation must use the original source. Removing the PCH restores
ordinary capture and unchanged generation reuse.

All six also pass with GCC `-O1` ASan/UBSan and recovery disabled. The production
builder and linked cJSON, UTF-8, module-build-directory and FFI-loader support
sources are instrumented; fixture and system libraries are not. Testing runs
inside a disposable, network-disconnected container based on the same image
recorded in [my Linux linker evidence](LINKER_INPUT_EVIDENCE.md).

Retained PCH contents, custom compiler flags, pkg-config compiler flags,
assembler inputs and transitive tool snapshots remain unfinished. Identifying
Clang or GCC from version output does not authenticate a compiler or wrapper.

The full Linux compiler/VM gates also pass: 28 shadows, 45 cache methods
(8 platform-specific skips), four Linux linker methods, six snapshot methods,
five wrapper link tests, seven wrapper boundaries, 63 codegen tests, 19 FFI
tests and dependency gates. A clean run exposed a missing C-seed prerequisite
in `test-bytecode-shadows`; after adding it, I confirmed that the seed was
absent and that this target built it before passing. Darwin's corresponding
gates pass too, with four ordinary snapshot methods and two GCC-specific
skips. These results do not establish complete source snapshots or release
readiness.

## Configured scalar flags

Against `648eb106`, a restored source edit still produced 43 instead of 42
when ordinary flags selected the original compilation path. I reproduced it
with common flags, active-platform flags, and common flags alongside an
inactive-platform option. I now recognize these individual token spellings:

| Kind | Spellings | Capture | Retained compilation |
| --- | --- | --- | --- |
| Optimization | `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, `-Oz`, `-Og` | Keep | Keep |
| Debug | `-g`, `-g0`, `-g1`, `-g2`, `-g3` | Keep | Keep |
| PIC | `-fPIC`, `-fpic` | Keep | Keep |
| C dialect | `-std=` with `c89`, `c90`, `c99`, `c11`, `c17`, `c18`, or their `gnu` counterparts | Keep | Keep |
| Warnings | `-Wall`, `-Wextra`, `-Werror`, `-Wpedantic`, `-Wno-unused-parameter`, `-Wno-unused-variable`, `-Wno-unused-function` | Keep | Keep |
| Macro/include tokens | Simple `-DNAME`, `-DNAME=value`, `-UNAME`, `-Idirectory` | Keep | Omit |
| Declared include directories | Manifest `include_dirs` | Keep, with existing path quoting | Omit |

Macro/include flag tokens use only ASCII letters, digits, `_`, `.`, `/`, `=`,
`+` and `-`; macro names must be C identifiers. Bare `-D`, `-U` and `-I`,
the special `-I-`, quoted tokens, whitespace fragments and response files are
not recognized for this mode. They still reach the original compiler path
unchanged. I am not interpreting arbitrary shell syntax or changing the
shared-link recipe. Pkg-config entries still select the original path.

I preserve flag order and consider only common plus active-platform flags.
Inactive-platform flags cannot disable a supported mode or leak into its
command. I omit preprocessor-only options when compiling the retained `.i`
file, including my own platform macro for shared-only inputs. This avoids
unused-command-line diagnostics under `-Werror` without disabling warnings.
My v17 context invalidates older records.

The regression checks cold result 42, unchanged warm reuse, exactly one C
compilation, actual phase argument vectors, and a declared include directory
containing spaces. Every one of the 33 scalar spellings above executes a real
build. Six unrecognized forms verify original compilation rather than a silent
conversion to retained input. These tests establish the exercised compilation
and result behavior, not complete debugger metadata equivalence or support
for all configured compiler modes.

All ten snapshot methods pass on GCC 12; Darwin Clang 21 passes eight and skips
the two GCC-specific PCH methods. Negative cases compare against direct compiler
rejection for unused variables, macro redefinition and C89 comments under
`-Werror`, then verify that the old generation remains byte-for-byte intact.
GCC ASan/UBSan passes the nine-method suite plus the added warning-error method,
with production-builder and support-source instrumentation as described above.
I separately reran the strengthened reuse-record assertion for all 33 spellings
normally on both compilers.

The full Darwin/Linux compiler and VM gates pass: 28 shadows, cache acceptance
(53 on Darwin; 45 plus eight skips on Linux), Linux's four linker methods,
wrappers, 63 codegen tests, 19 FFI tests and dependency gates. The gate runs
preceded the added negative method and stricter record assertion; both additions
were verified afterward. No production code changed between those checks.

## Literal fragments and captured package flags

I extend the configured boundary with a bounded literal-word decoder. I test
its output against `/bin/sh` argument vectors for quote concatenation, empty
words, single and double quotes, escaped spaces, literal expansion characters,
backslash-newline and the 4095-byte word limit. I reject oversized words,
unclosed quotes, dangling escapes and unquoted expansion/operator/glob syntax
for snapshot eligibility. I do not execute a fragment to discover its words.

The same decoder handles manifest fragments and the already captured
pkg-config cflags. I classify decoded arguments using the scalar table above;
I also accept paired `-D`, `-U` and `-I` arguments within one fragment. Macro
names remain C identifiers, but literal replacements and include paths can
contain spaces and quotes. Retained compilation re-quotes scalar arguments
individually and omits preprocessing-only arguments. Original preprocessing
and fallback compilation retain the supplied shell text unchanged.

My v18 context invalidates older records. Retained-input fingerprints include
captured package names, cflags and libs so changing either query still prevents
reuse. Native macOS frameworks bypass package queries and their absent flag
strings. Existing package consistency tests check changed responses, failed
post-build queries, recovery and successful empty flags.

The restored-source regression now also exercises a quoted multiword manifest
fragment and a pkg-config fragment, with paired options, a string-valued macro,
and include paths containing spaces and quotes. Cold/warm results remain 42;
the generation is reused with exactly one C compilation. Captured argument
vectors verify flag placement. Expansion fragments, response files, unknown
options and paired options split across manifest entries retain fallback.

This is not a shell interpreter or sandbox. Unsupported trusted fragments
still execute through the existing shell command path. It does not complete
PCH, assembler, transitive tool or arbitrary compiler-mode snapshots.

All 11 snapshot methods pass on Linux GCC 12, normally and with `-O1`
ASan/UBSan, recovery disabled, instrumenting the production builder and support
sources described above. The full Linux compiler/VM gates pass as well. The
stronger quoted-path/string-macro case was added after that full gate and then
rerun in the complete snapshot suite normally and under sanitizers. My first
broader Linux cache invocation lacked the CLI executables; I do not count
those missing-prerequisite errors as a pass. The successful gate built its
prerequisites before running.

Darwin Clang 21 passes nine snapshot methods with two GCC-specific skips, plus
the full shadow/cache/wrapper/codegen/FFI/dependency gates. After those gates,
I added a framework-backed CoreFoundation package subcase and strengthened the
phase assertions to check every paired preprocessing argument. The configured
regression passes again on Darwin and on Linux normally and under ASan/UBSan. Production code is
unchanged between these checks.

## Assembler inputs: reproduced gap

Against `5259507d`, I can still reuse code compiled from restored external
input bytes. Inline assembly reads `answer.bin` through `.incbin`; that read
happens after C preprocessing. The retained `.i` contains the directive, not
the binary payload. My experiment changes the payload from ASCII `42` to
`43` only during the first C compilation and restores its bytes, size and
nanosecond mtime before compilation returns. It loads each resulting library
in a fresh process and independently compiles the restored source.

```sh
make obj/test_module_generation_probe
python3 -m tests.characterize_source_snapshot --assembler --require-consistent
```

Both Apple Clang 21 and Debian GCC 12.2.0 report these results:

| Input | Cache | Cold | Warm | Independent fresh | Reuse record contains input |
| --- | --- | --- | --- | --- | --- |
| C source | Local and shared | 42 | 42 | 42 | Yes |
| C header | Local and shared | 42 | 42 | 42 | Yes |
| Assembler binary | Local and shared | 43 | 43 | 42 | No |

All cases retain a translation unit, reuse the same generation and execute
exactly one C compilation across cold and warm builds. Every controlled edit
is restored. The command exits nonzero because the assembler cases disagree;
it is deliberately not part of the passing snapshot acceptance suite yet.
Without `--assembler`, the original four-case acceptance remains unchanged.

I ran GCC in the network-disconnected, disposable Linux environment described
above, using a production-builder probe built at `-O3 -Werror`. My first GCC
fixture left assembly in the data section and failed to execute the library;
after explicitly returning to `.text`, both platforms reproduced the same
cache defect. That failed fixture run is not cache evidence.

Capturing C preprocessing alone is insufficient even with no custom flags,
PCH or package metadata. I need to capture assembler-consumed bytes and make
the assembler use them. Merely adding a post-build hash, or excluding a few
directive spellings, does not establish that invariant. The capture and
compiler-variant acceptance work remains open in my roadmap.

After adding the experiment, the existing 11-method snapshot suite passes on
GCC and passes on Darwin with its two GCC-specific skips. No production
compiler code changed in this characterization step.

## Clang assembly-output candidate

Apple Clang 21's normal `-S` pipeline expands inline `.incbin` to literal data
and expands nested assembler `.include` files and macros. I test the resulting
assembly as a candidate retained input in
`test_clang_retained_assembly_expands_external_inputs`:

1. Direct compilation executes 42.
2. I capture assembly with `-S` while the binary input still contains 42.
3. I change the binary; independent direct compilation executes 43.
4. I remove the binary and both nested include files. New assembly capture fails.
5. I remove the C source too, assemble/link only the retained output, and execute 42.

The four subcases combine direct versus nested reads with default flags versus
`-O2 -g -std=c11 -Wall -Wextra -Werror`. They exercise assembler macros, include
paths containing spaces, a binary path containing spaces and apostrophes, and
the offset/count operands of `.incbin`. No production cache implementation
changed in this trial. It establishes the exercised data/result behavior,
not complete debug metadata equivalence or all assembler dialects.

GCC 12's `-S` output retains the external `.incbin` directive. I confirmed this
in a disposable, network-disabled Debian container. Using that output as if it
were self-contained would preserve the reproduced defect. Production Clang
integration must capture and validate assembly bytes, preserve C diagnostics
and dependencies, and apply C code-generation flags during capture rather than
passing them blindly to assembly. GCC needs a separately tested capture path.

The expanded 12-method snapshot suite passes on Darwin with two GCC-specific
skips. The new trial skips compilers whose version output does not identify
Clang; a skip does not establish an equivalent GCC path.

## Production Clang assembly capture

My v19 context replaces Clang's retained preprocessed C with retained assembly.
I run the supported compiler with `-S` and all configured C flags, preserve
the source dependency and include-trace records, and hash the exact assembly
bytes written to private `__snapshot_<group>_<index>.s` files. Final object
assembly reads those files with `-x assembler`, without C-only flags. GCC keeps
its existing preprocessed-C path. Unknown compiler modes retain their existing
fallback; this does not finish the broader source-snapshot requirement.

The original restored `.incbin` experiment now produces 42 for cold, warm and
independent fresh builds on Clang, in both local and shared caches. Nested
assembler-input tests through the production builder also check permanent
binary changes (42 to 43), include changes (43 to 44), missing-input failure
with the old generation still executable, and recovery. They exercise ordinary
and shared-only sources with optimization, debug and strict-warning flags.

I validate reuse with fresh assembly capture. This runs C code generation on
warm builds, not just preprocessing. The experiment now reports object-build
and assembly-capture counts separately, and asserts that warm validation adds
an assembly capture without adding an object build. This is a correctness
tradeoff, not a performance improvement. C source/header dependencies remain
recorded; external assembler bytes are covered by the assembly fingerprint,
not by a newly invented dependency-path inventory.

Failures during capture retain the original compilation fallback without a
reuse record. Errors compiling retained input fail the build without replacing
the prior generation. The existing warning-error, capture-edit, scalar-flag,
package-response, source-diagnostic and multiple-source regressions remain
part of acceptance. GCC assembler snapshots and complete debug-metadata/tool
equivalence remain open.

The 14-method snapshot suite passes on Darwin Clang 21, with two GCC-specific
skips, normally and with Clang `-O1` ASan/UBSan and recovery disabled. The
instrumented build includes production `module_builder.c` and cJSON, UTF-8,
module-build-directory and FFI-loader support; fixture libraries and system
libraries are not instrumented. Full Darwin and Linux GCC compiler/VM gates
pass: 28 shadows, cache acceptance, platform linker tests, snapshots, wrapper
tests, 63 codegen cases, 19 FFI cases and dependency checks. GCC skips the three
Clang-only snapshot methods. After the Linux GCC full gate, the clarified
capture counters were rerun through its snapshot suite.

I also installed Debian Clang 14 only inside the disposable Linux container,
disconnected its network, and passed the snapshot, cache and Linux linker
suites using that compiler. This exercises the new assembly path on Linux;
the GCC pass alone would not. The temporary compiler is not a new product
dependency.

## GCC object-output candidate and cost

GCC 12 emits a reproducible object for the four exercised combinations of
direct/nested assembler reads and default/configured flags. The configured
set is `-O2 -g -std=c11 -Wall -Wextra -Werror`; objects also use `-fPIC`.
`test_gcc_retained_object_reproducibility_and_external_inputs` checks:

- Different private output directories and filenames produce identical objects.
- Changing the binary payload from 42 to 43 changes object bytes despite
  restoring the original mtime.
- Restoring the payload and mtime restores the original object bytes.
- Missing assembler inputs fail compilation.
- After deleting C, assembler include and binary inputs, the retained original
  and changed objects still link and execute 42 and 43 respectively.

This is a compiler-output boundary. It does not retain an auditable snapshot
of every original source file, make source-tree reads atomic, or establish
reproducibility for arbitrary compiler options. It is a candidate for binding
linking and reuse to actual compiler output without parsing assembler syntax.
Production integration remains open.

Warm output validation would repeat the complete C compilation and assembly.
I measured the cost on repository `src/cJSON.c` inside the same isolated Linux
arm64 GCC 12 environment, with these commands, three sequential runs each:

```sh
time cc -O2 -fPIC -Isrc -E src/cJSON.c -o /work/cjson.i
time cc -O2 -fPIC -Isrc -c src/cJSON.c -o /work/cjson.o
```

Preprocessing took 12, 12 and 11 ms; object generation took 214, 214 and 211 ms
of wall time. The median ratio is about 18 times for this workload, not a
general performance estimate. This path could establish stronger output
consistency, but calling it an incremental compilation speedup would be false.
The same distinction applies to Clang's already integrated assembly capture:
its warm validation also repeats C code generation.

The expanded 15-method snapshot suite passes on both Darwin Clang and Linux
GCC, with three compiler-specific skips each. This trial changes tests and
evidence, not production compiler behavior.

## Production GCC object validation

My v20 context adds actual GCC object bytes to the retained-C fingerprint.
Cold compilation still reads the retained `.i` files and follows the existing
error path. Before linking, I hash the ordinary and shared objects that will
be consumed. Fresh validation captures C into a private temporary directory,
compiles it with the same retained-C recipe, and hashes those objects in the
same order. A mismatch or failed validation withholds reuse evidence.

This does not retain every original file or make source reads atomic. In the
restored `.incbin` experiment, the first GCC result is still 43, but it has no
reuse record; the next build returns 42, matching independent fresh compilation.
Clang's assembly-capture boundary continues to return 42 on the first build.
Ordinary C source/header cases remain 42 throughout. They perform two object
compilations for a GCC cold build (build plus validation) and a third for warm
validation. Tests assert those counts rather than claiming no recompilation.

The production tests cover nested assembler macros/includes, permanent binary
and include changes, ordinary and shared-only inputs, missing-input recovery,
and preservation of the old generation after an injected cold compilation
failure. A compiler that fails once during cold compilation is not silently
retried. Validation uses private `nano-gcc-check-*` directories under `TMPDIR`
or `/tmp`, with the existing non-recursive, symlink-safe private cleanup helper.
Normal cleanup is tested with a space-containing temporary root. Termination
can leave an orphan; temporary-directory collection remains separate work.

All 17 snapshot methods pass on GCC 12 with two Clang-only skips, normally and
with GCC `-O1` ASan/UBSan, recovery disabled, instrumenting the production
builder and support sources described above. The full Linux compiler/VM gate
sequence passes, including 53 cache methods (eight platform skips), four linker
methods, shadows, wrappers, 63 codegen cases, 19 FFI cases and dependency checks.
Darwin's corresponding gates pass with five GCC-specific snapshot skips. Its
four changed cache-count methods were rerun after the full gate.

Initial Linux gate failures identified old no-recompilation count assertions;
the corrected tests assert exact validation counts while retaining result,
generation-preservation and failed-link assertions. An early sanitizer run
failed the new injection fixture because its own directory name matched the
validation-only trigger. After fixing that fixture prefix, the complete
instrumented suite passed. No sanitizer finding was reported in that failure.
The strengthened test verifies cleanup after validation failure and subsequent
successful recovery, not just normal cleanup.

## GCC literal assembler-file capture

I now insert `-S -x cpp-output` between retained C and GCC object assembly.
For literal, line-leading `.include` and `.incbin` directives, I copy the named
files into private staging and rewrite only their path operands. Includes are
processed recursively; binary payloads are copied whole, leaving offset/count
expressions to the assembler. Relative input names resolve against the build
working directory. I hash the original assembly and the bytes actually copied,
not a later reread of the source files. I retain object-output validation.

The copier accepts at most 16 MiB per file, 64 MiB cumulatively, 256 file visits
and 16 include levels. It rejects nonregular inputs, NULs or backslashes in
assembly text, alternate macro/MRI modes, nonliteral operands and file directives
outside the supported line form. Rejection removes partial capture files and
keeps the previous retained-C/object-validation path. Even an inactive or
commented unsupported spelling can cause fallback. This is a bounded copier,
not a complete assembler parser or an atomic snapshot of the source tree.
Copied inputs survive generation publication, but assembly paths name the
original private staging directory: these are evidence, not a portable replay
bundle. Compiler/assembler variants remain separate acceptance work.

The restored-edit fixture now checks direct binary reads, nested relative
includes with literal macros and offset/count expressions, and changed include
text, each with local and shared caches. GCC returns cold/warm/fresh 42 and
reuses the generation. It performs three assembly captures and three object
assemblies across the cold and warm builds. A macro-argument filename exercises
fallback instead: cold 43 without reuse evidence, then warm/fresh 42, with no
partial assembler capture left in the published generation.

The direct copier tests include binary NUL/255 bytes, empty input, same-line
multiple directives, labels, macro expansion, alternate macro/MRI modes, missing files,
FIFOs, recursive includes, embedded textual NUL and oversized files. Existing
tests retain permanent-edit invalidation, shared-only C sources, configured
flag separation, failed-build preservation and validation cleanup.

Both bootstraps and the 111-method bytecode/cache/link/snapshot regression set
pass on Darwin and Linux, with ten platform skips each. The 19-method GCC
snapshot suite also passes with production-builder/support ASan/UBSan and leak
detection. After that run I reduced buffers to observed file size plus one
growth-detection byte and rejected MRI mode explicitly. The final 19-method
suite passes again normally on both hosts and with GCC ASan/UBSan (leak detection
off); the final copier-boundary method passes separately with leak detection
on. Dependency-rebuild and FFI gates pass on both platforms. A file that grows
beyond its observed size during a copy now triggers fallback (2026-09-13).

## GNU assembler read-boundary trial

The literal copier cannot expand a macro-argument filename. I tested a different
boundary against GNU as 2.40 on Linux arm64: interpose its observed `fopen` and
`fopen64` input reads, copy each regular input, and return a stream for the copy.
Replay resolves reads only against that capture, never against live originals.
The isolated helper lives in `tests/fixtures/assembler_snapshot_preload.c`.
It is not linked into or selected by my production compiler.

```sh
make test-assembler-snapshot-trial
NANO_AS_TRIAL_UBSAN=1 make test-assembler-snapshot-trial
```

Four subcases pass with macro-argument binary names containing ordinary text,
spaces/apostrophes/dollar/hash, a backslash, or a newline. The assembler decodes
the operands; I do not parse them. Native-endian, length-delimited records
preserve the names it actually opened. Captured assembly, a relative include,
and binary payload produce byte-identical replay objects after the originals
are replaced, and again after they are deleted. The linked replay returns 42.
An existing but unrecorded input fails replay.

A pipe barrier pauses capture after the first read of a binary. I replace its
42 with 43 before its second read. Two distinct copies are retained under the
same pathname. Ordered replay matches the original object and exposes 4243
after deletion of the originals. A pathname-to-single-copy map would lose this
information. Removing one retained copy fails replay even when a live original
is present. Both methods pass normally and with UBSan on the helper; GNU as
itself is not instrumented. Darwin reports two platform skips. The unchanged
production snapshot suite passes on both hosts (19 methods, six Darwin/two
GCC skips).

[GNU as dependency reporting](https://sourceware.org/binutils/docs/as/MD.html)
provides dependency names, not a byte-retention mechanism. The read-boundary
trial demonstrates a route beyond source parsing, but it adds a shared helper,
dynamic-loader configuration, copy storage and an assembly capture/replay pass.
The fixture first emits assembly with the C compiler, captures with the
assembler, then replays; it does not establish end-to-end driver integration
or a performance improvement.

I will investigate this boundary for production rather than add macro expansion
to the literal copier. Before integration I need executable/tool selection,
helper build identity, child-only loader settings, complete read-hook coverage
for supported assemblers, failed-read semantics and robust transactional
records. The trial assumes a single-threaded assembler and a trusted private
directory; it supports only the tested stdio modes, 256 opens and 32 MiB per
input. It does not cover static assemblers, arbitrary direct syscalls, host
sandboxing, concurrent writers to the capture, or other compiler/assembler
variants. General assembler snapshot acceptance remains open (2026-09-13).

## Sealed assembler capture records

`src/runtime/assembler_capture.c` now owns a runtime helper and
`src/runtime/assembler_capture.h` its shared record validator. I build it
explicitly on Linux with `make bin/nano_as_capture.so`; at this stage the production
module builder did not select it yet. The earlier trial remains historical evidence,
not the implementation used by this helper.

The `NASCAP01` format uses little-endian fixed-width headers and length-delimited
path bytes. Each ordered open records either the original error number or a
copy's size and FNV-1a content digest. The seal binds the complete record stream.
I normalize only the first, invocation-private pathname in that digest and
require the helper's expected-input setting to match its actual first read.
The validator checks every successful copy, record bounds, the seal and EOF.
These checks detect corruption; they are not cryptographic authentication.

Capture writes an exclusive partial record and exclusive copy files, returns
streams for the copies, and publishes the sealed record without replacing an
existing manifest. A killed capture leaves only partial evidence. Replay checks
the entire capture, preserves failed opens even if those paths now exist,
and consumes repeated reads in order. Unknown reads, missing/changed copies,
symlink substitutions and an unread tail fail. Each replay first clears its
completion marker; only a completed replay writes `NACDONE1`. The driver must
check both child success and completion evidence: a sealed open stream alone
does not establish successful assembly.

The helper requires `NANO_AS_CAPTURE_PREFIX`, `NANO_AS_CAPTURE_PHASE` and
`NANO_AS_CAPTURE_INPUT`. Those settings and `LD_PRELOAD` must eventually be
confined to the selected assembler child. I reject unsupported stdio modes,
other threads/processes, nonregular source inputs and excess capture sizes.
Original symlinks are followed during capture; retained-copy symlinks are not.
Nonblocking source opens prevent a FIFO replacement from blocking capture.
The limits are 256 opens, 4095 pathname bytes, 32 MiB per file and 64 MiB total.

```sh
make test-assembler-capture-records
make bin/nano_as_capture.so
NANO_AS_CAPTURE_TEST_HELPER="$PWD/bin/nano_as_capture.so" make test-assembler-capture-records
NANO_AS_TRIAL_UBSAN=1 make test-assembler-capture-records
```

Eight methods pass against the packaged Linux helper and against its UBSan
build. Real GNU as replays a macro include and path bytes containing double
quotes, apostrophes, dollar/hash, backslash and newline after source deletion.
A deterministic stdio driver verifies repeated reads, preserved `ENOENT`, and
rejection of an unread tail even when the driver itself exits zero. Tests also
cover killed capture, header/seal truncation, extra trailing bytes, bad sizes,
changed/missing copy evidence, source/copy symlink distinctions, FIFO replacement
and existing-capture preservation. Darwin skips these Linux-only methods.

The private capture directory remains trusted and must not be modified during
replay; this is not hostile-writer isolation or power-loss durability. Only the
tested stdio read boundary is covered, not arbitrary assembler syscalls. Tool
identification, complete supported read coverage, child-only configuration,
build/install integration and actual production selection remain in the
integration gate. At this stage I had not enabled the helper as a default compiler path.

## Production GNU assembler read replay

When GCC's literal assembler copier cannot represent a source, I now try the
sealed read-capture helper on Linux. I resolve GCC's `-print-prog-name=as`
result, require a dynamically linked ELF64 little-endian executable reporting
GNU as 2.40 or 2.42, and fingerprint its path and bytes. I copy the selected helper
into private staging and fingerprint those actual bytes. GNU/Linux builds and
installs include `nano_as_capture.so` beside the native/bytecode drivers;
`NANO_AS_CAPTURE_HELPER` is an explicit path override and cache-context input.

GCC still chooses the assembler arguments. Its private `-B` entry selects a
wrapper that opens the retained helper on descriptor 3 and sets `LD_PRELOAD`
only for the assembler child. Neither the compiler driver nor the linker gets
that loader setting. Descriptor loading avoids the loader's whitespace-delimited
path interpretation. Existing ambient `LD_PRELOAD`/`LD_AUDIT` configurations
decline this path instead of being silently replaced.

I emit assembly from retained C, run a capture assembly, validate its completed
record and copies, then use ordered replay for the objects I will link. The
wrapper clears completion before replay and requires both a successful child
and `NACDONE1`. A replay failure fails the build and preserves the old generation.
Capture or tool-selection failure removes partial read-capture files and keeps
the previous retained-C/object-validation fallback. That fallback is still not
general assembler snapshot support.

The restored macro-input fixture now changes the live payload during final
object generation, after capture, and verifies cold/warm/fresh 42 with reuse
for local and shared caches. Capture invocations are distinguished explicitly
from final assembly: a change during capture itself can be retained and then
withhold reuse when fresh validation differs. Across cold and warm builds this
path runs six object assemblies, including captures and validation. It is not
a no-recompilation cache hit; the existing literal path remains cheaper.

Production tests also cover shared-only assembly, inactive conditional inputs,
permanent payload changes, missing-input preservation, quoted helper paths,
default adjacent-helper discovery, failed assembler lookup, replay-helper
loss, recovery and temporary cleanup. A compiler-driver assertion verifies
that it never inherits my `LD_PRELOAD`. The private copy directory remains
trusted during replay. GNU-as variants, arbitrary syscall reads, helper/assembler
authentication and transitive shared-library/toolchain snapshots are not proved
by this integration and remain separate acceptance work.

Both full bootstraps and the 121-method bytecode/cache/link/snapshot/helper
regression set pass (ten Linux skips, twenty Darwin skips). The expanded 21
snapshot methods pass again with the GCC production builder and support sources
under ASan/UBSan, leak detection disabled. The final negative checks distinguish
missing helpers from ignored loads with successful native assembly, and decline
an assembler shell wrapper even when its version output looks supported.
Dependency-rebuild and FFI gates also pass on both hosts.

The Linux install gate first exposed a pre-existing `-Werror` socket-path copy
in `vmd_server_run`. I added an explicit length rejection before unlink/bind;
three injected exact-fit/oversized cases preserve old files and pass on both
hosts. Linux `make install` then succeeded, including the adjacent helper.
These gates do not establish complete relocatable installation of the whole
language or release readiness (2026-09-13).

## GNU assembler 2.42 acceptance

I exercised the same production path on Ubuntu 24.04, Linux arm64, with GCC
13.3.0 and GNU as 2.42. My 22 snapshot methods pass with two Clang-only skips,
including restored macro inputs, shared-only assembly, replay failure,
preservation of old generations, cleanup and recovery. The eight capture-record
and two replay-trial methods pass normally and with the helper under UBSan.
GCC 12.2.0 / GNU as 2.40 on Debian bookworm, Linux arm64, passes the same
snapshot, record and trial gates, including helper UBSan.

GCC 13 first rejected a discarded diagnostic `write` result under `-Werror`.
I now handle partial writes and EINTR, and stop on other failures before the
existing exit 125. I did not suppress the warning. Darwin's production build
also required keeping the version classifier inside its Linux-only boundary.

I now accept exact 2.40 and 2.42 version tokens on the first GNU assembler
banner line. Tests reject neighboring versions, suffixes, a missing newline,
another tool's banner, and a supported version appearing only on later lines.
The separate dynamic-ELF check remains. Neither check authenticates an
executable. Other assembler versions, arbitrary syscall reads and transitive
toolchain snapshots remain outside this tested boundary (2026-09-13).

The Ubuntu bootstrap passes. After explicitly building `nano_virt`, `nano_vm`
and `nano_cop`, my 122-method bytecode-shadow/cache/link/snapshot/helper suite
passes with ten expected skips. Initial suite attempts lacked those test
executables; they were prerequisite failures, not passing runs.
The Darwin bootstrap also passes, followed by all 22 snapshot methods with
nine Linux/GCC-only skips on the final source.

## GNU assembler input and auxiliary-output modes

My real-assembler matrix now runs ordinary, `-g -alh` listing/debug, and
`--alternate` macro modes, each with two include search directories and `--MD`
dependency output. The fixture uses nested includes, repeated macro expansion,
an inactive include, and `.incbin` offset/count expressions. I extract `.data`
and require the repeated payload `4242`; I also require a nonempty debug
listing and a dependency record naming the root source.

After capture I delete the root, both includes and the binary payload. I add
error-producing files to the earlier include search directory, then replay.
The object, listing, diagnostics and dependency file must match capture
byte-for-byte, and replay must produce its completion receipt. These checks
exercise remembered search failures as well as successful reads; a new earlier
candidate must not replace the captured include.

All nine capture-record methods pass normally and with helper UBSan on Linux
arm64 with GCC 12.2 / GNU as 2.40 (Debian bookworm) and GCC 13.3 / GNU as 2.42
(Ubuntu 24.04). Both toolchains also pass the two historical trial methods
under helper UBSan. The 22 production snapshot methods pass on GNU as 2.40
with two expected skips. Darwin imports the new suite and skips all nine
Linux-only methods; that is not runtime-helper evidence. No production code
changed in this acceptance step. Arbitrary syscall reads and configured
compiler-wrapper inputs remain separate work (2026-09-13).

## Retained GCC precompiled headers

My relocation trial establishes that GCC can compile retained preprocessed C
against a copied `.gch`, including a private directory containing spaces. The
result stays 42 after the original PCH is rebuilt for 43 and after the original
source, header and PCH are removed. Removing the private PCH makes that retained
compilation fail.

I now rewrite canonical `#pragma GCC pch_preprocess` references to private
binary copies during production capture. I combine the original pragma text,
original PCH path and actual copied bytes in the fingerprint. Destination paths
are invocation-private and do not enter identity. The existing copier enforces
regular files, 16 MiB per file, 64 MiB per translation unit and 256 visits.
Malformed pragmas, escaped paths and failed copies still decline snapshot reuse.
A rejected rewrite preserves the original retained input and does not remove a
pre-existing rewrite temporary file. Other partial private copies belong to
the normal staging cleanup.

GCC's `-H` trace emits `! ` for a selected PCH and a space-prefixed root source
line. My previous dot-only parser rejected that evidence. I now hash both paths
as dependencies, so a valid retained-PCH build can acquire a reuse record.
PCH appearance, replacement and removal invalidate prior selections.

The production restoration test changes the live PCH during final assembly and
restores its bytes and timestamp. Cold output remains 42, warm builds reuse the
generation, and a permanent replacement yields 43. I run this for ordinary and
shared-only C sources under local and shared caches. Deleting the private PCH
before retained C emission fails replacement, preserves the old generation,
cleans staging, and permits a later retry. Every split position in the pragma
marker still selects retained assembly and permits reuse.

This retains GCC PCH inputs; it does not retain arbitrary compiler modules,
plugins, response files or wrapper-owned inputs. Those configured-mode gates
remain open.

The 25 snapshot methods pass on Linux arm64 with GCC 12.2 / GNU as 2.40
(Debian bookworm) and GCC 13.3 / GNU as 2.42 (Ubuntu 24.04), two expected skips
each. The final matrix also passes with GCC 12's production builder and linked
support sources under ASan/UBSan, leak detection disabled. The Linux bootstrap
and 126-method bytecode/cache/link/snapshot/helper regression set pass with ten
expected skips (2026-09-13).
Darwin also passes its bootstrap and final 126-method regression set with 24
expected skips; its production probe exercises the rewrite rejection checks.

## Restored compiler response-file arguments

At `16d932e1`, I measured `@flags.rsp` containing `-DANSWER=42`. The compiler
wrapper changes it to `-DANSWER=43` only while the final object compilation
runs, then restores its bytes, size and nanosecond modification time. The source
returns `ANSWER`. Fresh compilation uses the same restored response file.

On GCC 12.2 / GNU as 2.40, Linux arm64, both local and shared caches publish
43 and reuse that generation, while fresh compilation returns 42. On Apple
Clang 21, Darwin arm64, both caches publish 43 but withhold the reuse record;
the next build and fresh compilation return 42. Neither compiler retains the
response-file arguments in this path. Withholding Clang reuse contains later
reuse, but does not make the cold output a snapshot.

My previous `--require-consistent` condition compared only warm and fresh
answers, so it incorrectly passed the Clang cold mismatch. I now require cold,
warm and fresh answers to agree. Four predicate cases distinguish a consistent
run, a cold-only mismatch, a warm-only mismatch and both mismatches. With the
corrected gate, this command fails on both tested compilers:

```sh
python3 -m tests.characterize_source_snapshot --response --require-consistent
```

This is a reproduced production defect and a repaired acceptance gate, not
response-file retention. The repair must capture arguments once for the build
and use them consistently across compilation phases, preserving compiler
response-file quoting, nested-file lookup and error semantics. Reusing my
literal-shell-word parser for response contents would change their meaning.
The roadmap response-file item remains open (2026-09-13).

All 26 snapshot regression methods pass on GCC 12 Linux arm64 and Apple Clang
21 Darwin arm64, with two and eleven expected skips respectively. The default
source/header consistency gate remains green; the response-file gate above is
intentionally red until argument retention is implemented.

## Literal response-file argument retention

I now capture bounded literal Clang/GCC response arguments before entering the
build. An invocation-local metadata copy owns expanded common and active-platform
flag strings; package compiler flags use the same expansion during their existing
capture. The original metadata remains caller-owned and unchanged. The public
rebuild query captures fresh arguments too. I include selected metadata argument
strings in build-context version 21; package arguments remain part of the
preprocessing fingerprint. Compilation and linking use the captured strings.

My response tokenizer is separate from my literal-shell-word parser. Quotes
group response bytes; backslashes quote the following byte, including within
quotes. Dollar signs and shell operators remain literal arguments. I quote each
expanded argument before passing it to my shell command runner. Differential
tests compare that actual shell transport with direct compiler `@file` handling
for spaces, dollar signs, backslashes, macro strings and separated flags.
Nested response names use the invocation's working directory, not the parent
response file's directory; competing files in both places verify the choice.

The restored-edit reproducer now reports cold/warm/fresh 42 with reuse for both
local and shared caches. Root, active-platform and package flag tests cover
same-timestamp permanent replacement, public rebuild decisions, missing files,
cycles, FIFO rejection, old-generation preservation and recovery. The missing
and cycle cases fail during argument capture instead of entering an incomplete
replacement. FIFO inputs are opened nonblocking and rejected as nonregular.

I do not silently label every response form retained. The current limits are
16 nesting levels, 64 KiB cumulative response bytes, 4095-byte words and a
2048-byte serialized fragment. Over-budget and noncanonical forms keep the old
path, as do shell-expanded fragments and unrecognized drivers. Boundary tests
verify that those fragments are preserved rather than partially expanded.
The full roadmap response-file item remains open for large-argument transport,
shell-expanded/noncanonical forms and other dialects. The default
`--response --require-consistent` reproducer now passes for the implemented
literal forms; that does not establish the remaining boundaries.

I also exclude named `clang-cl` drivers and explicit `--driver-mode` overrides.
Capture is transactional across metadata and package argument groups: a pending
response or shell fragment keeps the original argument set. A source-free
build-info regression checks metadata overrides, package overrides, nested
package response overrides and a named `clang-cl` wrapper without asking the
host compiler to execute another driver's dialect. Escaped spellings of those
overrides are checked after literal argument decoding too.

All 30 snapshot methods pass on GCC 12.2 and GCC 13.3, Linux arm64, with two
expected skips each. The final GCC 12 production builder and support sources
also pass those methods under ASan/UBSan with leak detection disabled. The
Linux bootstrap and 130-method bytecode/cache/link/snapshot/helper suite pass
with ten expected skips; the final boundary test was then added and included
in the 30-method snapshot and sanitizer runs (2026-09-13).

After the driver-mode rollback review, all 31 snapshot methods pass on GCC 12.2
Linux arm64, both normally and under ASan/UBSan on the production builder and
support sources (two expected skips; leak detection disabled). The broader
Darwin bytecode/cache/link/snapshot/helper run passes 132 methods with 24 expected
skips before the final escaped-override decoding guard.
With that final guard, all 31 Darwin snapshot methods pass with eleven expected
skips (Apple Clang 21, arm64, 2026-09-13).

## Large response-file transport remains unretained

I now measure a 10,212-byte response file: 600 harmless identical macro flags,
followed by `-DANSWER=42`. The selected driver changes 42 to 43 only during
object compilation, then restores the original bytes and timestamp. This
exceeds my inline expansion limit while remaining well below the response
reader's 64 KiB input budget.

```sh
python3 -m tests.characterize_source_snapshot --response-large --require-consistent
```

Both local and shared caches give these results on arm64 (2026-09-13):

| Compiler | Cold | Warm | Fresh | Generation reused |
|---|---:|---:|---:|---|
| Apple Clang 21.0.0 | 43 | 42 | 42 | no |
| GCC 12.2.0 | 43 | 43 | 42 | yes |

Neither driver retains a translation unit or assembly snapshot for this case.
GCC publishes a reuse record that does not name the response input. Clang
withholds reuse, but that does not repair its incorrect cold output. The command
above exits one on both hosts. My unit test checks the measurement's restoration,
size, fresh output and consistency predicate; it deliberately does not claim
that this production boundary passes.

The next implementation must retain argument ownership across every consumer:
`module_compile_prefix` filters arguments by compilation phase;
`module_shared_link_command` also consumes metadata compiler flags; and
`ModuleBuildInfo.compile_flags` outlives the invocation-local metadata copy and
is returned to native callers. An invocation-private response file removed at
the end of `module_build` would leave those returned flags dangling. A random
temporary pathname must not become cache identity either. I need retained
transport with defined lifetime and content identity, not a larger inline
buffer or a flag saying this unsupported path is safe.

All 32 snapshot test methods pass on Apple Clang 21 Darwin arm64 and GCC 12
Linux arm64, with eleven and two expected skips respectively. The separate
large-response acceptance command remains red on both hosts.

## Retained response transport

I now transport long literal compiler fragments through content-addressed GNU
response sidecars in the module cache. I keep the decoded strings for build
identity and phase filtering, and use sidecar references only for compiler
commands and returned native flags. Build-context version 22 distinguishes this
transport from the earlier inline-only path.

Each file contains separately double-quoted GNU response words, with backslashes
and double quotes escaped. I write and sync a private file, make it read-only,
then publish it with a no-replacement hard link. Concurrent creators verify the
same complete bytes. A content-derived name is not sufficient evidence: I check
the existing file's type, size and bytes, and reject mismatches rather than
overwrite them. These are trusted local cache files, not a sandbox against an
owner who modifies files during compiler reads.

Sidecars outlive invocation metadata, build-info objects and compiler processes.
They remain under the module cache until that cache is removed. The lifetime
test gets native flags from a completed probe process, changes the original
response file, then runs the host compiler with those returned flags. Its output
still matches the original arguments, including literal dollar signs.

Compilation, GCC post-preprocessing flags and shared linking use this transport.
Darwin's linker observation recognizes only exact transports of captured
compiler flags; unrelated indirect response arguments remain excluded. The
10,212-byte reproducer now returns cold/warm/fresh 42 with generation reuse in
both local and shared caches on Apple Clang 21 and GCC 12.

Common, active-platform and package argument tests cover changed sidecars,
symlinks, FIFOs, directories, removal and recovery. Concurrent source-free builds
exercise publication without relying on the C-generation lock. Long repeated
`-O2` arguments also exercise phase filtering and permanent argument changes.

I retain the 16-level nesting limit, 64 KiB response-input and serialized-fragment
budgets, and 4095-byte words. Transport begins above 1024 serialized bytes per
fragment. This does not remove aggregate command limits for many short
fragments, capture shell-expanded or noncanonical forms, or implement other
driver dialects and indirect linker response retention. The parent roadmap
item remains open.

The Linux GCC 12 bootstrap and 134-method bytecode/cache/link/snapshot/helper
regression set pass with ten expected skips. All 33 snapshot methods also pass
with the production builder and support sources compiled under ASan/UBSan
(two expected skips, leak detection disabled). The sanitizer run includes the
final quoting and concurrent-publication cases (2026-09-13).
The final sidecar-corruption check preserves the prior published generation
and reuses it after transport repair, for common, active-platform and package
flags. It passes in the full Linux sanitizer run and a focused Darwin run.
The Darwin bootstrap and 134-method broad regression set also pass, with
24 expected skips. The focused Darwin run includes the final quoting,
concurrency and prior-generation preservation checks.

## Returned compiler-flag capacity

Aggregate argument work exposed a prerequisite memory-safety defect. Both
returned compile-flag collectors allocated 1024 pointers, then appended common
flags, active-platform flags and include directories without checking capacity.
The 1300-entry production fixtures fail under GCC 12 UBSan with an insufficient
object-space store at each append loop. This is an out-of-bounds write, not
just a compiler command-length limit.

I now use one collector for source-free and compiled modules. It checks count
addition and allocation-size multiplication, allocates enough pointer storage,
preserves package/include/common/platform order, and publishes the result only
after every string copy succeeds. Include flags use their complete string
length instead of a 256-byte allocation. This preserves their existing shell
spelling; it does not establish a new path-quoting contract.

Tests cover 1300 returned entries for each metadata group, an include path over
256 bytes, and a compiled module with 1300 empty fragments. The latter reaches
the result collector without exceeding the still-open command-length limit.
Five injected allocation failures and count overflow leave no partial result;
each is followed by a successful retry with exact order checks. All 35 snapshot
methods pass on Darwin and under GCC 12 ASan/UBSan (eleven and two expected
skips respectively, leak detection disabled for the full sanitizer suite).

Returned link flags and shared-link assembly still have fixed pointer budgets.
They have a separate roadmap item before aggregate transport; this collector
repair does not establish that arbitrary-sized compiler or linker commands
are supported.

The Linux bootstrap and 136-method regression set pass with ten expected skips.
The focused allocation-failure test also passes with leak detection enabled,
in addition to the full 35-method ASan/UBSan snapshot run (2026-09-13).
Darwin's rebuilt native VM tools and 136-method regression suite pass with
24 expected skips. Aggregate command transport and the separate link-flag
collector work remain open.

## Returned link flags and shared-link assembly

The link collectors had the same fixed 1024-pointer allocation. Linux sanitizer
fixtures reproduce out-of-bounds writes for common linker flags, system libraries
and the compiled-module result path. Active-platform linker flags instead stop
at 1024 entries without reporting failure. On Darwin, shared-link deduplication
also turns `-framework Foundation -framework Security` into a command with a
bare `Foundation`; the real compiler rejects it as a missing input file.

I now size one returned-link collector from checked metadata counts. Both
source-free and compiled results use it, preserving object/package/common/
platform/framework/system-library order. System-library strings use their full
length. I publish the owned array only after every allocation succeeds.

Shared linking no longer builds temporary 1024-entry arrays. It appends package,
system-library, common, platform and framework fragments directly through the
checked command writer. Explicit library repetitions and framework pairs stay
in order. A command-capacity failure prevents execution; it does not silently
drop tail arguments. Build-context version 23 invalidates earlier link recipes.

Tests cover 1300 returned entries, the compiled result path, a real Foundation
and Security link, and a rejected option after 1300 space-only fragments. That
tail option reaches the linker; rejection leaves the previous generation intact
and a repaired build succeeds. Failure injection covers all six Linux or eight
Darwin collector allocations, count-addition and pointer-allocation overflow,
then successful retry with exact order checks. All 39 snapshot methods pass on
Darwin and under GCC 12 ASan/UBSan (eleven and three expected skips). The focused
Linux allocation test also passes with leak detection enabled (2026-09-13).

Aggregate command-length limits and cross-module flag-merging policy are not
changed by this per-module collection repair.

The Linux bootstrap and 140-method regression set pass with eleven expected
skips. The final large-list fixture also checks a system-library name over
256 bytes; returned spelling must remain complete.
The final Linux ASan/UBSan run passes all 39 snapshot methods with three expected
skips. Darwin's rebuilt native tools, C reference compiler and 140-method broad
suite pass with 24 expected skips; its final long-library-name fixture also
passes. The link-flag capacity roadmap item is complete; aggregate command
transport remains open.

## Coalesced compiler fragments

I now combine eligible common, active-platform and package compiler groups when
their combined size exceeds 1024 bytes, up to the existing 64 KiB budget. The
first nonempty slot holds the ordered sequence; other owned strings become
empty, and native-framework NULL slots remain NULL. This keeps package indexing
stable and lets the existing response sidecars carry many short fragments.
Compilation and shared linking skip those empty slots. Caller-owned metadata
and `module.json` are unchanged. Build-context version 24 identifies this recipe.

The regression covers 1300 short flags, forty references to a small response
file, and forty package compiler fragments. The final `-DANSWER=42`, `-UANSWER`,
`-DANSWER=43` sequence returns 43 and reuses its generation. Returned compiler
flags from the large-list fixture decode to the complete original argument
sequence, including order.

Four injected allocation failures retain every original pointer and string;
each is followed by a successful retry. The fixture includes an empty slot and
a NULL slot. The full 41-method snapshot suite passes on Darwin and under Linux
ASan/UBSan (eleven and three expected skips respectively); the focused Linux
coalescing test also passes with leak detection enabled (2026-09-13).

This is not unbounded aggregate transport. Include-directory lists and linker
fragments are not coalesced, and over-budget, shell-expanded, noncanonical and
other-driver forms remain on their previous path. The parent response-file
roadmap item stays open.
Against the previous revision (`a897aa3e`), all four aggregate fixture variants
fail to compile on GCC 12. The same variants pass with coalescing enabled.
The Linux bootstrap and 142-method regression set pass with eleven expected
skips. The complete 41-method snapshot sanitizer run and leak-enabled allocation
checks use the same production code.
Darwin's rebuilt native tools and C reference compiler also pass the 142-method
regression set, with 24 expected skips (2026-09-13).

## Include-directory transport

I quote include paths consistently for object compilation and returned native
flags, then coalesce their ordered argument list through retained response
transport. Original include paths remain in metadata and cache validation;
build-context version 25 identifies the new recipe. Partial allocation failure
frees every initialized slot before returning failure.

The regression uses 200 missing include directories followed by competing
headers in paths containing spaces, quotes and a dollar sign. Reordering the
headers changes 42 to 43 and selects a new generation; warm builds reuse it.
Both source-free and compiled modules return flags that a later compiler
process can use from another directory. The earlier 1300-directory fixture
still checks complete argument order. Six collector allocation failures and
the capacity overflow guard each permit a clean retry.

Against revision `344950ff` on GCC 12, the compiled fixture fails to build and
the source-free fixture does not produce retained transport. The new fixture
passes on Darwin. Linker fragments, over-budget groups and unsupported driver
forms remain outside this boundary; the parent roadmap item stays open.

On GCC 12, all 42 snapshot methods pass normally and with ASan/UBSan on the
production builder probe (three expected skips). The two focused allocation
failure methods also pass with leak detection enabled (2026-09-13).
Darwin's rebuilt native tools and C reference compiler pass the 143-method
compiler, shadow, cache and assembler regression suite (24 expected skips).

## Aggregate linker arguments: reproduced, not repaired

I measure four linker groups with:

```sh
python3 -m tests.characterize_link_argument_transport --require-consistent
```

On Apple Clang 21 and Debian GCC 12.2 (2026-09-13), all four module builds
fail while direct driver links succeed and return 42. Common, active-platform
and package groups contain 1200 search flags followed by `-lm -lc -lm`
(19,211 bytes). The system-library group contains 1200 `m,c` pairs followed
by `m` (9,603 bytes). The shared-link command still has a fixed-size buffer.

Source-free returned flags retain every argument in order, including repeated
libraries. A later compiler process uses them successfully and returns 42.
Failed module builds and explicit invalid-tail replacements leave the old
published generation intact. None of this establishes working cold builds
or warm reuse for these long groups: the acceptance command exits nonzero.

Two unit methods check the acceptance gate, including each status, each cold,
warm, direct and later answer, each invariant, and an empty measurement. They
pass on both hosts and run with the bytecode-shadow make gate. The next change
must repair shared-link transport while preserving these observations and
Darwin linker-input validation. Raw linker response files, other dialects and
larger budgets remain separate unfinished requirements.
