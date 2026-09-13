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

## Shared-link group transport

I now serialize the shared-link argument group in its original order: package
libraries, system libraries, common linker flags, active-platform linker flags,
then Darwin framework pairs. Literal groups use my retained driver response
transport within a 64 KiB input budget. This does not expand raw linker response
files. Any user `@` keeps the group on its previous path, so I do not hide an
indirect input from Darwin observation. Build-context version 26 identifies
this recipe. Returned native linker flags keep their previous spelling and
ownership; no temporary link sidecar escapes into them.

Darwin observation admits the exact retained shared-link fragment in addition
to the existing captured compiler fragments. It still rejects other indirect
arguments. The four-case characterization now passes on Apple Clang 21 and
GCC 12: direct, cold, warm and later answers are 42; warm generations are
reused; returned argument sequences are complete; failed replacements preserve
the prior generation. The formerly failing acceptance command now exits zero.

Tests also corrupt a retained linker file before a source replacement, check
that publication fails without changing the old generation, remove the damaged
file and verify recovery. A fragment allocation failure permits a clean retry.
The existing shared-link tail test decodes my response transport before checking
repeated library order and the final invalid option. Framework-pair tests remain
unchanged. Unsupported dialects, indirect linker input capture and larger
argument budgets remain open.

The 47-method snapshot/link suite passes normally on Darwin and GCC 12
(eleven and three expected skips). With the added corruption-recovery test,
all 48 methods pass under Linux ASan/UBSan on the production builder probe;
the focused fragment-allocation test also passes with leak detection enabled
(2026-09-13).
Darwin's rebuilt native tools and C reference compiler pass the complete
149-method compiler, shadow, cache, assembler and linker regression set
(24 expected skips).

## Driver responses supplied through linker flags

My restored-input experiment now covers common linker metadata, active-platform
linker metadata and package-library output, in local and shared caches:

```sh
python3 -m tests.characterize_source_snapshot --link-response --require-consistent
```

The fixture switches only the selected archive name during the shared link,
then restores the response bytes, size and timestamp. Its directory deliberately
contains `42`; only `selected42.a` changes to `selected43.a`. Fresh reference
links place archives after the referring source. The original generic byte
replacement and archive ordering were fixture defects, corrected before the
final Linux baseline. At revision `8e9f17fb`, GCC 12 reproduces cold/warm/fresh
43/42/42 in all six cases. Apple Clang 21 reproduces the same mismatch for
common linker metadata in both caches.

I now capture driver response arguments in all four common/platform compiler
and linker metadata groups. Package compiler and library candidates are captured
and published atomically. Unsupported response or driver-mode forms roll back
the candidate set. Decoded linker arguments remain in build identity and owned
returned flags. Build-context version 27 invalidates older recipes.

The six restored-response cases now produce 42/42/42 with generation reuse on
both drivers. Source-free consumers retain complete ordered arguments after the
original response is removed and can pass them to a later compiler process.
Missing, cyclic and FIFO responses fail; repaired inputs work again. Thirty-two
allocation budgets exercise fallback, partial-copy cleanup and
successful retry without changing caller-owned arrays.

This captures compiler-driver response syntax, not response files forwarded
as `-Wl,@file`, linker scripts or the contents of selected archives. Those
external-input boundaries and larger argument budgets remain open.

All 51 snapshot/link methods pass on GCC 12 normally and with ASan/UBSan on
the production builder probe (three expected skips). Metadata-allocation and
returned-lifetime checks also pass with leak detection enabled. The final
lifetime check includes literal and escaped driver-mode overrides and verifies
that both compiler and linker response references remain unexpanded; it passes
on Darwin and under Linux sanitizers (2026-09-13).
The existing Darwin `-Xlinker @file` test now requires a reuse record and an
unchanged warm generation: these are captured driver arguments, not an
uncaptured linker response. The forwarded-response visibility test remains.

Darwin's rebuilt tools completed a 152-method run with 24 expected skips and
one ten-second timeout in the existing response-recovery method. That method
passed its isolated recheck in 9.246 seconds. An earlier 40-package stress
build also timed out and passed its isolated recheck; only those stress builds
now receive a thirty-second test bound. I do not claim a clean full Darwin
gate. MAC `task_44c2d5851e1948e6a474036e263a0a73` and the roadmap retain that
verification requirement.

### Completed Darwin gate

At `d9bfd86c`, I rebuilt `nano_virt`, `nano_vm`, `nano_cop`, `bin/nanoc_c` and
the production generation probe, then ran the same complete regression set
with verbose reporting. All 152 methods finished in 341.417 seconds: 24
expected skips, no failures and no timeouts (2026-09-13). This closes the
missing full-run requirement; it does not establish that a loaded host can
never exceed a test deadline. I did not change ordinary deadlines for this run.

```sh
python3 -m unittest tests.test_bytecode_shadows tests.test_module_cache_publication \
  tests.test_linux_link_cache tests.test_source_snapshots \
  tests.test_assembler_capture_records tests.test_link_argument_transport -v
```

Forwarded linker response inputs and larger argument budgets remain separate
open work. This is not a release-readiness claim.

## Forwarded linker responses: reproduced, not repaired

At `d352c643`, I measure response files passed as `-Wl,@file`, separately from
the driver `@file` cases:

```sh
python3 -m tests.characterize_source_snapshot --forwarded-response --require-consistent
```

The forwarded cases cover common linker metadata, active-platform linker
metadata and package-library output, each in local and shared caches. The
controlled shared link changes `selected42.a` to `selected43.a` in the response
file and restores its bytes, size and timestamp before returning.

Apple Clang 21 and GCC 12 both produce cold/warm/fresh 43/42/42 in all six
forwarded cases. Warm builds do not reuse the cold generation. The explicit
acceptance command exits nonzero. The earlier driver-response capture remains
separate: retaining a driver's arguments does not retain arguments that the
linker reads from another file.

This fixture establishes the archive-selection mismatch, not equivalence of
driver and linker response grammars. The repair must preserve linker quoting,
nested response resolution, selected-linker identity and argument order before
forwarded inputs gain snapshot eligibility. No production capture code changes
in this measurement increment.

The existing 51-method snapshot/link suite still passes on Darwin and GCC 12
(eleven and three expected skips respectively, 2026-09-13). This regression
result does not turn the new forwarded-response acceptance gate green.

### Linker response grammar boundary

I compare the installed linker's native `-Wl,@file` interpretation with my
driver decoder's words, each passed through its own `-Xlinker` pair:

```sh
python3 -m tests.characterize_linker_response_grammar --require-equivalent
```

My twelve-case corpus covers plain arguments, single and double quotes,
embedded quotes, escaped spaces, quoted backslashes, commas, vertical tabs,
form feeds, CWD-relative nesting, repeated response files and an unterminated
quote. A successful shared library must load in a fresh process and return 42.
The gate recomputes the observed comparison; it does not trust a stored
`equivalent` flag. A declined decoding makes no substitution claim.

On Apple Clang 21, the native linker rejects the vertical-tab, form-feed and
repeated-response cases, while my driver decoder turns all three into
successful links returning 42. The other eight admitted cases agree. The
unterminated quote is declined. The explicit gate exits nonzero. On GCC 12
in Debian Bookworm, all eleven admitted cases agree and the gate passes
(2026-09-13). These are observations over this corpus, not a grammar proof.

[GNU's response-file documentation](https://sourceware.org/binutils/docs/ld/Options.html)
describes recursive expansion, quoting and escaping. Apple's published
[cctools argument expansion](https://github.com/apple-oss-distributions/cctools/blob/main/libstuff/args.c)
uses a narrower whitespace set and tracks repeated response paths. That
published implementation helps explain the experiment; it is not evidence
that every installed Apple linker uses the same implementation.

I cannot reuse my driver decoder unchanged for forwarded capture. The repair
must identify the selected linker or retain its interpretation without
changing accepted inputs. Alternate linker selection, nested dependencies,
failure behavior and retained-input lifetime remain acceptance requirements.
No production capture code changes in this increment. My nine linker-transport
unit/integration methods pass on both hosts; the forwarded snapshot repair
remains open.

### Retained response identity experiment

I extend the grammar corpus to sixteen cases with distinct equal-content
responses, alternate path spelling, a symlink alias and a hardlink alias:

```sh
python3 -m tests.characterize_linker_response_grammar --require-retained-equivalent
```

My prototype copies response bytes and rewrites only the fixture's explicitly
known nested references. I remove the original root and nested response files
before linking the retained graph. I do not infer nested references with a
production parser in this experiment.

Apple Clang 21 rejects repeated paths, `inner.rsp` plus `./inner.rsp`, and a
symlink alias. It accepts two separate equal-content files and two distinct
hardlink paths. GCC 12 accepts all five cases. An initial spelling-keyed
prototype changed Apple's alternate-spelling rejection into success. A
content-only prototype instead changes Apple's distinct-file and hardlink
successes into rejection. Neither representation preserves the boundary.

Retention keyed by resolved path preserves all sixteen observed native
statuses and loaded answers on both hosts, including malformed quoting and
control whitespace (2026-09-13). The hardlink case verifies equal inode numbers
before the native link. This supports resolved-path identity for these tested
linkers, not an identity contract for every selectable linker. Absolute nested
references in this fixture contain no spaces; general nested token rewriting
still needs quoting and selected-linker coverage.

My retained acceptance gate recomputes native/retained outcomes and rejects an
empty measurement, changed success/failure, missing loaded answers and wrong
answers. Ten linker-transport methods pass on Darwin and GCC 12. This selects a
capture representation to implement; it does not repair production forwarded
capture or close the six restored-selection failures.

### Bounded C graph-capture mechanism

I now implement the retained graph in `module_capture_link_response`, with an
internal contract in `src/module_link_response.h`. The caller selects a GNU or
Apple grammar explicitly; I do not infer the selected linker from the host OS.
The probe selects the known grammar for the installed toolchain experiment:

```sh
python3 -m tests.characterize_linker_response_grammar --require-captured-equivalent
python3 -m unittest tests.test_link_response_graph
```

I retain every non-reference byte. When I find a decoded token beginning with
`@`, I capture its CWD-resolved input and replace that token span with a quoted
retained path. Resolved-path memoization preserves repeated references and
distinguishes equal-content and hardlink paths. The sixteen native-linker cases
match on Apple Clang 21 and GCC 12 after original root and nested responses
are removed. Unlike the earlier fixture prototype, this path uses my C token
scanner and graph traversal.

I bound capture to sixteen active levels, sixty-four distinct resolved paths,
64 KiB of unique input bytes and 64 KiB per rewritten response. Missing inputs,
cycles, non-regular files, embedded NULs, unsupported nested-token quoting and
over-budget graphs return failure. Non-reference leaf bytes retain their
original quoting, including malformed quoting for the linker to interpret.
I use the existing private-file publication and exact-byte verification for
both driver transports and retained linker files; linker identity additionally
includes the resolved source path. Retained files live with the module cache.
This preserves the existing trusted-local-cache boundary, not hostile-owner
protection or a cryptographic identity claim.

Seven focused tests cover quoted/escaped paths, both whitespace rules,
resolved-path identity, original-file removal, depth/node/byte bounds,
malformed inputs, FIFO and directory rejection, changed/symlinked retained
children, recovery and thirty-two allocation budgets with same-process retry.
They pass with ASan/UBSan and leak detection on Linux. The sixteen-case C
capture comparison also passes with the probe instrumented. All fifty-three
snapshot/link methods pass under ASan/UBSan, with three expected skips and
leak detection disabled for that broader set. An initial run instrumented the
assembler preload helper too and failed three assembler-capture assertions;
rebuilding that helper normally restored the passing broader result.

Darwin's rebuilt tools pass the 159-method regression set with 24 expected
skips and no failures (285.254 seconds). After the final argument guard and
two additional boundary tests, I rebuild again: all seventeen graph/link
methods and the sixteen-case native capture comparison pass.

Invocation-wide ownership, selected-linker admission, returned flag rewriting
and Darwin cache eligibility are not wired to this helper yet. The six
forwarded restored-selection failures remain open. A captured graph is not,
by itself, permission to reuse a cached library.
Retained cache paths containing commas also need deliberate driver transport:
inserting such a path into `-Wl,` splits it, and this Apple driver rejects the
joined spelling `-Xlinker=@path`. I retain that integration requirement on the
roadmap rather than claiming the graph helper solves argument transport.

### Selected-linker query boundary

I implement `module_query_link_response_grammar` as an internal query over a
complete literal compiler command. I do not execute a shell to discover its
arguments. I supervise a private process group, bound stdout to 8 KiB and
share a five-second monotonic deadline across the GNU `--version` and Apple
`-version_details` requests. I terminate remaining members of the query's
process group on completion as well as failure; this is not a security sandbox.
Diagnostics on stderr are discarded, not mixed into the
machine-readable version report.

This identifies a supported tool contract, not toolchain authenticity. I
recognize GNU ld's version banner and the tested Apple ld 1267 JSON report
with its architecture list and Apple TAPI vendor field. Unrecognized linkers,
failed queries, malformed/binary/oversized reports and exceeded deadlines do
not select a grammar. A fixture that prints a recognized banner can emulate
that contract; this is not executable attestation.

On installed Clang, `-print-prog-name=ld -fuse-ld=lld` still names Apple ld,
while the actual dry-run recipe selects ld64.lld. My native tests instead
route the complete command through a controlled `-B` linker wrapper. They
verify default recognition and rejection of an unsupported selected linker,
including a `-fuse-ld` selector supplied in a driver response file. Both Apple
Clang 21 and GCC 12 pass those routing checks.

The caller must supply disposable probe outputs. Apple's version-details
request can complete a link when inputs are present; my native test verifies
that it creates the disposable library. A version flag is not a read-only
guarantee. Query integration must not target a published generation, and must
retain the full invocation's selection flags while constructing private
probe artifacts.

Seven query tests cover native routing, overrides, literal arguments, rejected
shell syntax, argument/output bounds, report validation, allocation retry,
deadline cleanup and successful reporters that leave background descendants.
All fourteen query/graph methods pass with ASan/UBSan and leak detection on
Linux; all ten linker-transport methods pass with ASan/UBSan and leak detection
disabled (2026-09-13). Invocation-wide ownership and cache admission remain
unwired; this query does not close the six forwarded snapshot failures.
Darwin's rebuilt compiler/VM tools and generation probe pass the complete
168-method regression set in 302.000 seconds, with 24 expected skips and no
failures or timeouts.

### Disposable primary query outputs

My internal query API now accepts a parent directory and creates a private
`.nano-link-query-*` directory beneath it. The unchecked complete-command
runner is private to the builder implementation and its test probe. I append
both a driver `-o` and a separately forwarded linker `-o` targeting the private
file, without dropping or reordering the existing selection flags. Separate
`-Xlinker` arguments preserve commas in that output path.

The production shared-link recipe supplies the test commands. On Apple
Clang 21 and GCC 12, twenty-four combinations cover common/platform compiler
and linker metadata, package compiler/library flags, direct driver/linker
output overrides and overrides inside driver/forwarded response files. Both
the recipe's original output and a metadata-selected output retain their
bytes and nanosecond timestamps. The query directory is removed after each
successful case, including with a comma and space in its parent path.

I reject visible end-of-options controls and nonliteral shell commands before
creating a query directory. Thirty-two allocation budgets exercise cleanup
and same-process retry. If cleanup cannot remove an owned directory, I report
the retained files and decline grammar admission; I do not claim cleanup
succeeded. A nested-directory fixture checks that failure path.

The rebuilt Darwin compiler/VM tools pass all twenty-seven query/graph/link
methods, and the sixteen-case native grammar comparison remains green. On
Linux, all seventeen query/graph methods pass with ASan/UBSan and leak
detection; the ten linker-transport methods pass with sanitizers and leak
detection disabled (2026-09-13). The preceding 168-method Darwin gate remains
separate evidence; I did not rerun that entire set for this API wrapper.

This pins primary outputs, not every possible compiler side effect. Auxiliary
outputs and controls hidden in indirect inputs still require admission and
retention before a query runs. A validated original response must not be
reread from a mutable path during execution. Invocation-wide installation of
captured flags, returned-flag lifetime and cache admission remain open; the
six forwarded restored-selection failures are not repaired by this wrapper.

### Transactional response-root sets

I implement `module_capture_link_responses` for an ordered set of one to
sixty-four roots. The single-root API delegates to it. All roots share one
resolved-path map and a 64 KiB unique-input budget. I also freeze each source
spelling at its first observed binding, so later references do not resolve a
removed or retargeted path again. I bound this table to 128 spellings of at
most 4095 bytes each; the existing sixty-four-node and sixteen-level bounds
remain.

I return the complete owned path array or NULL. A later missing input,
exceeded shared budget or allocation failure cannot expose an earlier partial
path set. Verified content-addressed files can already exist in the cache
when a later root fails; the transaction governs the returned set, not an
atomic filesystem publication or a filesystem-wide point-in-time snapshot.

Six controlled cases rewrite, remove or retarget a shared response between
root captures, under both explicit grammars. Later roots retain the first
observed input; a new transaction observes its replacement or fails if it is
gone. Sixty-four allocation budgets check same-process retry. Root-count,
spelling-count, spelling-length and shared-byte limits are exercised.

Native multi-root links on Apple Clang 21 and GCC 12 preserve search order:
one root order returns 42, and its reverse returns 43. Distinct equal-content
roots remain distinct; repeated roots preserve Apple's rejection and GNU's
success. Retained links still produce those outcomes after the original roots
are removed. The existing sixteen-case single-root comparison stays green.

The final rebuilt Darwin tools pass all thirty-two graph/query/link methods.
On Linux, all twelve graph methods pass with ASan/UBSan and leak detection
(82.885 seconds); the ten linker-transport methods also pass with sanitizers
and leak detection disabled (2026-09-13). This does not replace the earlier
full regression gate or complete invocation-wide flag installation. Indirect
option admission, returned flag transport and cache eligibility remain open.

### End-of-input tokens and explicit linker arguments

I compare a second transport hypothesis: decode the retained graph with my
C linker token scanner, then pass each resulting word as a separate
`-Xlinker` argument. The experiment reads retained files after removing the
original graph. Its cache paths contain commas and spaces; quoted CRLF in a
filename remains two literal bytes, not text-mode newline normalization.
This is a fixture prototype, not invocation admission or production transport.

The experiment exposes a scanner mismatch. Apple Clang 21 and GCC 12 both
successfully link `'selected.a` without a closing quote, and `selected.a\`
with a trailing backslash. My scanner previously declined both spellings.
I now accept an open quote at end of input and discard the final escape,
matching those observed tools. I leave my separate compiler-driver response
decoder unchanged. Nested references with either ending are captured and
rewritten as complete token spans; the retained graph outlives the originals.

I keep transport refusal separate from executed linker failure. My prototype
refuses repeated resolved identities under the Apple profile. That is not a
successful native-equivalence measurement, even when the native linker also
rejects the original. The materialization gate rejects missing outcomes and
does not trust a stored `materialized_equivalent` flag. Production handling of
these rejections, selected-tool admission and installation remain open.

All twenty-two native comparisons preserve retained-graph outcomes on Apple
Clang 21 and GCC 12.2. Materialized arguments preserve all twenty-two GNU
outcomes and the nineteen executed Apple outcomes. The other three Apple
cases remain explicit refusals, leaving its full materialization gate red.
I do not replace those missing executions with fabricated linker statuses.

The rebuilt Darwin tools pass thirty-six targeted graph/query/link methods
in 28.886 seconds. Linux passes sixteen graph/scanner/prototype methods with
ASan/UBSan and leak detection in 98.277 seconds. The final twenty-two-case
Linux comparison also passes with the sanitizer-instrumented probe, followed
by the CRLF prototype regression. These are targeted gates, not a new full
release gate (2026-09-13). The six forwarded restored-selection failures
remain open; this parser correction does not install invocation-wide capture.

### Owned C linker-argument transactions

I add `module_capture_link_arguments` as a second output form of my existing
graph transaction. I parse the bytes already read for each node and return
one owned literal shell fragment per root, containing ordered `-Xlinker`
and value pairs. I do not publish a response sidecar or reopen one to decode
it. The retained-path API keeps its existing behavior and shares the same
source-identity map, frozen spellings, bounded reads and cleanup.

I preserve the sixty-four-root/node, 128-spelling, sixteen-depth and 64 KiB
unique-input limits. I also cap quoted output at 64 KiB across all roots,
including repeated expansions. GNU repeats reuse the first captured value
in order. Apple repeats of a resolved identity fail with `ELOOP`, even if
the original spelling has since been removed or retargeted. Distinct paths
to hardlinked files remain distinct. Any failure frees the whole returned
set; an empty response returns an owned empty fragment, not failure.

The tests cover exact empty/quoted/control-whitespace words, CRLF filenames,
shell punctuation, no sidecar creation, aggregate expansion and input limits,
nonregular inputs, eighty allocation budgets across both profiles and
same-process retry. Six read-boundary mutation cases rewrite, remove or
retarget a shared input after its first read. Native multi-root links preserve
the 42/43 search-order distinction and remain independent of removed source
responses.

The native grammar comparison now checks the C transaction separately from
the Python transport prototype. Its gate distinguishes capture failure,
explicit identity rejection and executed linker results; allocation failure
or process death cannot stand in for native rejection. All twenty-two GNU
cases execute with matching results. Apple has nineteen matching executions
and three explicit identity rejections agreeing with native failure. These
are API-level rejection checks, not a claim that ordinary compilation already
uses the new argument form. Invocation-wide installation, indirect-control
admission and cache eligibility remain open.

My rebuilt Darwin tools pass forty targeted methods and the complete
`make test-bytecode-shadows` target: 175 methods, fifteen platform/configuration
skips, and 278.132 seconds of reported test time. Linux passes twenty-six
ordinary graph/query methods, then seventeen graph/acceptance methods with
ASan/UBSan and leak detection in 146.355 seconds. Its final twenty-two-case
native comparison also passes with the instrumented probe, including the
stricter capture-status oracle (2026-09-13). The six restored-selection
failures remain open; I have not enabled cache admission for this API.

### Public query option admission

I put a literal option check in front of `module_query_link_response_grammar`.
Unresolved driver/linker responses, indirect controls, plugins, auxiliary
output switches and unknown options return zero before I create a query
directory or start the configured tool. I check both `-Wl,` fields and
`-Xlinker` pairs, retain operand state between forwarded words, and refuse
dangling operands or ambiguous interleaving with driver arguments.

This is an explicit initial option profile, not a general compiler option
parser. I admit the existing scalar snapshot flags, selected driver controls,
canonical separate library/search/output operands and listed linker switches.
I preserve the original command and selection flags after admission; the
existing driver/linker primary-output pins still apply. I do not accept an
arbitrary `-l` prefix as a library: Apple's published parser handles
`-lto_library` as a libLTO override before its ordinary library handling.
[Apple ld64 option parser](https://github.com/apple-oss-distributions/ld64/blob/main/src/ld/Options.cpp).

The unclassified primary-output mechanism is now private and is exposed only
through an explicitly unchecked test-probe mode. Its twenty-four raw-response
fixtures still measure output pinning, not public admission. The public tests
check refusal before tool execution, exact admitted argument order, operand
boundaries and limits, native `-B`/`-fuse-ld` overrides, and queries using C
captured arguments after the original response file is removed.

I classify option controls, not the contents of positional native inputs.
Those inputs may carry indirect linker behavior, and configured wrappers have
their own authority. Their admission/retention remains the caller's boundary;
this API is not a filesystem sandbox. I must resolve that boundary and install
one captured set across the real invocation before claiming the six restored-
selection failures repaired.

The rebuilt Darwin tools pass all forty-three targeted query/graph/transport
methods in 30.860 seconds. Linux passes twenty-nine normal graph/query methods
in 8.889 seconds and all thirteen query methods with ASan/UBSan and leak
detection in 111.323 seconds (2026-09-13). The latter includes allocation
recovery, deadlines, cleanup, native tool overrides and the new admission
cases. I did not rerun the preceding full 175-method gate for this option
check. No ordinary compilation or cache-reuse path invokes this query yet.

### Installing forwarded arguments in ordinary builds

I now use the captured argument graph in ordinary module builds and public
rebuild checks. I collect `-Wl,@path` roots across common/platform compiler
and linker flags and package compiler/linker flags in shared-link contribution
order. I prepare complete GNU and Apple candidates before querying either.
Each candidate owns its metadata and package fragments; I install only the
candidate whose selected linker confirms that candidate's grammar.

The query uses the shared-link flag recipe with a controlled empty C input,
`-x c /dev/null -x none`, and private primary output pins. It does not compile
module sources or require module objects to exist. Explicit native inputs,
configured compiler wrappers and toolchains remain trusted. I classify options;
I do not sandbox native input contents. The two grammar captures are separate
transactions, not an atomic observation of the whole filesystem.

My v28 build context includes the captured common/platform linker fragments
as well as compiler fragments. Package fragments remain in the preprocessing
fingerprint. Post-build validation uses the selected owned package arguments;
the next invocation captures fresh arguments. Editing a response without
changing its size or modification time therefore changes selection identity.
Returned fragments own their strings and do not depend on the response paths.

I remove confirmed linker-only `-Xlinker` pairs from source phases, not from
the shared link. The new Darwin compiler-group cases initially produced the
right results but no reuse record: unused-linker-argument diagnostics polluted
the dependency evidence. The source-phase filter repairs that measured gap.
It applies only after successful invocation-wide grammar admission.

Unadmitted options, unsupported drivers and incomplete candidates leave the
complete preceding flag path intact. This fallback does not acquire a new
snapshot guarantee. A confirmed Apple linker with repeated resolved response
identities is explicitly rejected; repairing the response permits a later build.
Allocation checks exercise 180 budgets, require unchanged caller metadata and
all-original or all-captured argument groups, and retry in the same process.

My restored-selection fixture changes the response during actual shared
linking, not during version queries. Apple's first GNU-version attempt is
expected to fail; counting that probe as the publication link would invalidate
the experiment. The six cases require cold, warm and fresh answers of 42,
restored bytes/size/mtime, a reuse record and actual generation reuse.
Additional cases edit responses across all six argument groups in a directory
containing a comma and space, test source-less returned-flag lifetime, reject
Apple repeated roots without publishing, and decline cyclic or unclassified
controls without starting a linker query. These checks do not close arbitrary
compiler modes, native-library snapshots or the entire cache transaction item.

The rebuilt tools pass the full `make test-bytecode-shadows` target on both
hosts (2026-09-13): 184 methods each, fifteen skips and 421.594 seconds of
reported test time on Darwin; twelve skips and 109.158 seconds on Linux.
Platform/configuration skips are not evidence for the skipped behavior.
Linux also passes 32 graph/query/invocation methods with ASan/UBSan and leak
detection in 269.241 seconds. That run includes allocation rollback,
source-less returned-flag lifetime and unadmitted-candidate fallback. Native
fixture tools and the assembler preload helper are not sanitizer-instrumented.

## Clang macro-argument capture acceptance

On Apple Clang 21.0.0, ARM64 Darwin, I expanded the retained-assembly trial to
literal inputs, nested includes and macro-supplied binary paths. The macro case
also contains an inactive reference to a missing file. Plain flags and
`-O2 -g -std=c11 -Wall -Wextra -Werror` pass all six cases. Direct libraries
return 42, changing the input produces 43, and retained assembly returns 42
after the source, nested includes and binary payload have all been removed.
This checks actual replay, not the appearance of captured assembly text.

The production restored-edit fixture now includes macro arguments under both
local and shared caches. Cold, warm and fresh results are 42, bytes/size/mtime
are restored, and the cache generation is actually reused. The full snapshot
suite passes 43 methods with eleven platform/configuration skips in 88.555
seconds; the expanded production case passes separately in 4.045 seconds.
This adds acceptance evidence without changing production code. It does not
establish arbitrary assembler file-read coverage, external assembler modes,
other Clang versions, or completion of the general snapshot requirement.

## External assembler: reproduced stale reuse

Apple Clang 21.0.0 with `-fno-integrated-as` does not expand the tested
`.incbin` during `-S`. Its emitted assembly still reads the original payload:
changing 42 to 43 changes the linked result, and deleting the input makes
assembly fail. The existing `capture-assembly` helper retains this literal
input; replay with the external assembler then returns 42 after the source and
payload are deleted. That trial is not production integration.

Before the v29 build-context repair, my production builder excluded this flag from retained-input mode
but still publishes and reuses a cache record. The restored-edit fixture gives
cold/warm/fresh **43/43/42** under both local and shared caches, with restored
bytes, size and mtime, no retained assembly, and actual generation reuse.
The exclusion was not containment.

`python3 -m tests.characterize_source_snapshot --external-assembler --require-consistent`
now accepts the literal case: cold/warm/fresh 42/42/42 under both caches,
one retained binary input, and actual generation reuse. I preprocess C into
private storage, emit assembly, copy literal assembler inputs with my existing
bounded copier, and assemble the rewritten input. Warm validation repeats
private capture and hashes the resulting object. The invocation fixture checks
that all three object compilations preserve `-fno-integrated-as`.

I also exercise nested includes, quoted binary paths, offset/count reads,
ordinary and shared source groups, permanent replacement, missing inputs,
preservation of the previous library on failure, and recovery. These cases use
`-O2 -std=c11 -Wall -Wextra -Werror` with the external selector.

At v29 this was a bounded repair, not completed external-assembler support. On this
Apple toolchain, macro-supplied filenames and `-O2 -g` still exceed my literal
copier's grammar: debug assembly contains octal escapes in `.ascii` strings,
and I conservatively reject backslashes. Both restored-edit variants give
43/42/42 without a reuse record under local and shared caches. My consistency
gate rejects them. Declining reuse does not repair their cold compilation.
Both variants and general assembler file-read coverage remained open at v29.

The rebuilt tools pass `make test-bytecode-shadows` on ARM64 Darwin:
188 methods, fifteen platform/configuration skips, and 375.944 seconds of
reported test time. This includes all 47 snapshot methods, publication,
argument transport and response graph/query checks. At that checkpoint I had
not rerun the repair's full gate on Linux; Linux results above belong to earlier
commits.

## Fixed-width octal debug data

My v30 capture parser accepts three-digit octal byte escapes (`\000` through
`\377`) in a single `.ascii`, `.asciz` or `.string` string. I retain the line
verbatim; I do not decode bytes into assembler source. I still decline short
or non-octal escapes, escaped file paths, named macro substitutions, malformed
strings, alternate/MRI macro modes and ambiguous trailing statements. Apple
semicolon comments and GNU statement separators receive different treatment.

Direct and retained assembly reproduce all 256 byte values through each data
directive after I delete the original assembly. The `.ascii` case also passes
inside a named-parameter macro, without interpreting numeric escapes as that
parameter. These checks pass with Apple Clang 21's external assembler on
ARM64 Darwin and GCC 12.2/GNU assembler 2.40 on ARM64 Linux.

On the Apple toolchain, the production `-fno-integrated-as -O2 -g` restored-edit
fixture now returns cold/warm/fresh 42/42/42 under local and shared caches,
retains the binary input, preserves the external selector and actually reuses
the generation. Optimized/debug nested includes also pass permanent replacement,
missing-input recovery and last-good-library checks for ordinary and shared
source groups. At v30, macro-expanded filenames still gave 43/42/42 without
reuse and their cold-build capture remained open. Numeric debug data is not general
assembler macro support.

The Linux full `make test-bytecode-shadows` gate passes 190 methods with sixteen
platform/configuration skips in 108.938 seconds of reported test time. A
separate ASan/UBSan build of the production capture probe passes three boundary
and replay methods with leak detection in 20.693 seconds. Rejected inputs must
produce no diagnostics, so a sanitizer failure cannot masquerade as an expected
exit code. Supporting object files and external assemblers are not sanitizer
instrumented. I removed the disposable Linux build container after these checks.

The rebuilt Darwin tools also pass the full gate: 190 methods, fifteen skips
and 426.654 seconds of reported test time, including all 49 snapshot methods.
These are tested toolchain boundaries, not proof of arbitrary assembler syntax
or identical semantics across every supported platform.

## Selected Apple external-assembler expansion

My v31 path follows the selected external assembler's dry-run report, then
that assembler's backend report. On the tested Xcode toolchain, `as` is a
wrapper around its Clang assembler. I do not infer a backend path from the
driver's installation directory or silently switch to another assembler.
I admit a single literal command from the Apple Clang 21.0.0 report family,
decode bounded arguments without a shell, and hash the selected assembler,
backend bytes and backend arguments into the capture fingerprint.

I request assembly-text output from that backend and retain the expanded
bytes privately. Text mode initially produced repeated empty labels in debug
assembly. Naming temporary labels during text capture fixes that failure;
final object assembly retains the selected assembler's original symbol policy.
I do not apply `-msave-temp-labels` to final object assembly. My existing
supervised-process helper gives the queries and normalization a shared
five-second subprocess deadline, bounded output and process-group cleanup.
Each report must describe one command; malformed, multiple, oversized,
truncated, failed and timed-out reports decline this capture path.

Six direct/replay cases cover literal, nested and macro-supplied paths with
plain and optimized/debug C flags. They include all 256 payload byte values,
an inactive missing-file reference, offset/count operands and a temporary-label
relocation. Direct output returns 42, replacement returns 43, and replay after
deleting source and input files returns 42. Direct and replayed object bytes
are identical, including the debug cases.

Production restored-edit tests now return 42/42/42 for plain and debug macro
inputs under local and shared caches, with actual generation reuse and the
external selector on all three object compilations. The mutation fixture
explicitly excludes `-###`: it changes input during real object compilation,
not a dry run. Production-retained assembly also replays after deleting
originals for ordinary and shared source groups. Permanent edits, missing
inputs, preservation of the previous library, query failure cleanup and later
recovery are checked. Warm validation repeats capture in private storage.

At v31, unknown report families or failed capture still used the uncaptured
fallback without a reuse record. That fallback does not establish cold-build
snapshot consistency. This repair covers the tested Apple selection; other
assembler implementations, unadmitted flags and source modes still need their
own acceptance. I trust the configured tools and their reports; this is not
executable authentication or a filesystem-wide snapshot.

The Linux full gate passes 194 methods with twenty platform/configuration
skips in 108.702 seconds of reported test time. Those skips include the
Apple-only capture path. On Darwin, a separate ASan/UBSan production probe
passes three report-boundary, retained-macro/recovery and query-failure methods
in 53.122 seconds. Supporting objects and external compilers are not sanitizer
instrumented, and this Darwin run disables leak detection. I removed the
disposable Linux container and moved the sanitizer scratch directory to Trash.

An additional restored-edit control forces the admitted driver's dry-run query
to fail. That v31 fallback gives cold/warm/fresh 43/42/42 under both
caches, without a reuse record. The consistency gate rejects it. This is why
the external-assembler umbrella remains open even though successful selected
backend capture now handles macro reads: a failed admitted capture must not
silently publish an uncaptured cold result. Unsupported-mode compatibility and
failure of an admitted capture need separate outcomes.

The rebuilt Darwin full gate passes 194 methods with fifteen skips in 516.378
seconds of reported test time. Its 53 snapshot methods pass. The failed-query
restored-edit control was added after that suite loaded and passes separately
in 4.230 seconds; it deliberately verifies that the consistency gate rejects
the remaining defect, not that the cold result is correct.

## Failed admitted external capture

My v32 build context keeps external-capture admission separate from the actual
retained format returned by capture. If admitted Clang external capture fails,
I stop before object compilation, report the module, and remove private staging.
I no longer silently compile live source. Source/flag modes classified outside
the retained path keep their existing compatibility behavior; this change does
not establish snapshot consistency for those modes.

The forced-query restored-edit regression now returns a build failure before
the mutation hook or any object compilation runs. Neither local nor shared
cache publishes a generation or current pointer, and no staging directory leaks.
The characterization gate reports a failed build explicitly rather than calling
absence of an answer consistent.

Twelve query-failure cases cover empty, multiple, truncated, oversized, failed
and timed-out reports in both cache roots. Cold failure leaves no artifact.
Warm failure leaves the previous library and reuse record byte-for-byte intact,
does not add object compilations, and cleans validation and build staging.
Repairing the query permits changed input to produce 43 and then reuse that
new generation. Compiler errors and missing assembler-input diagnostics remain
visible; an unadmitted-flag control still uses the original build path.

The Linux full gate passes 197 methods with twenty-three platform/configuration
skips in 111.619 seconds of reported test time. A separate Darwin ASan/UBSan
probe passes three failure/recovery, diagnostic and unadmitted-mode methods in
118.647 seconds. Supporting objects and external compilers are not sanitizer
instrumented, and leak detection is disabled in this Darwin run. I removed the
disposable Linux container and moved the sanitizer scratch directory to Trash.

My user-guide build and validation pass for thirteen pages in six editions.
The new capture-failure explanation is English authority text; this check does
not establish that the localized drafts have been translated.

On Darwin, the rebuilt gate's shadow, publication, Linux-link and snapshot
suites pass 148 methods with fifteen platform skips. Their reported times are
52.774, 141.823, 0.000 and 256.652 seconds. The final gate output was lost after
the snapshot suite; I do not claim an observed exit status for that invocation.
I reran the remaining transport, response-graph and response-query suites:
all 49 methods pass in 111.956 seconds. Together these verified runs cover all
197 methods in the gate with fifteen skips.

## Complete admitted capture or failure

My v33 context extends capture-failure refusal to ordinary Clang and GCC C
builds admitted by the existing source/flag checks. GCC no longer accepts a
preprocessed translation unit alone when both literal assembly capture and
read replay fail: later assembler file reads would still consume live inputs.
Failure before final compilation leaves publication to a later successful
build. Unsupported source and flag modes remain outside this guarantee.

Before the repair, the GCC missing-helper characterization passed its defect
expectation: cold/warm/fresh 43/42/42 under both cache roots (0.713 seconds on
GCC 12.2 Linux ARM64). It now requires a failed build with no mutation hook,
object compilation, generation, current pointer or staging leak. Ordinary
capture failure has the same cold-cache checks. Separate preprocessing and
assembly-phase failures test local/shared cold refusal, warm byte preservation,
diagnostics, cleanup, recovery to changed output and actual subsequent reuse.
Missing helper and rejected assembler selection also preserve prior generations.

My capture subprocesses now use the existing source diagnostic filter; failed
capture previously left errors in a private include-trace file. GCC retained
C-to-assembly and read-capture commands no longer discard diagnostics. Initial
Linux regression failures exposed both the hidden errors and a fixture setup
problem: the probe lives in `obj/`, not beside the installed helper in `bin/`.
The scalar-flag fixture now selects that helper explicitly. Its GCC `-g3`
output contains quoted macro debug strings outside the literal copier's grammar
and uses read replay; I did not exclude `-g3` or restore live-input fallback.

The corrected Linux snapshot suite passes 58 methods with fourteen platform
skips in 29.419 seconds. Four Darwin phase-failure and diagnostic methods pass
in 6.924 seconds. I then added local/shared missing-helper recovery before the
full rebuilt compiler/shadow/cache gates reported below.

The final GCC/Linux full gate passes all 201 methods with twenty-four platform
or configuration skips in 109.456 seconds of reported test time. Its existing
CLI preprocessing-failure test previously required successful live compilation;
I replaced that expectation with failure, no output, no object compilation and
no published generation for both partial and empty capture output.

Four focused Linux ASan/UBSan methods pass first with leak detection disabled
(5.091 seconds), then with leak detection enabled (63.263 seconds). They cover
phase failure/recovery, missing-helper warm preservation/recovery, source
diagnostics and replay cleanup/tool selection. The production builder is
instrumented through the probe; supporting objects, external compilers and the
assembler helper are not sanitizer instrumented. My user-guide build and check
also pass thirteen pages in six editions; localized draft validation is not
translation acceptance.

The same GCC-built production probe also passes all 59 snapshot methods using
Debian Clang 14 on Linux ARM64 (nineteen skips, 26.184 seconds). I select that
compiler through a private `cc` symlink in the disposable container; this is
cross-driver input-capture evidence, not a Clang-built compiler bootstrap.

An additional permanent Linux Clang regression passes in 2.382 seconds. Its
four plain/debug macro cases yield cold/warm/fresh 42/42/42 under local and
shared caches, with retained read manifests, actual generation reuse and the
external assembler selector on all six capture/replay object invocations.
This checks the Linux read-replay path separately from Apple backend expansion.

The final Darwin full gate passes 200 methods with sixteen skips in 589.397
seconds of reported test time. Its snapshot process loaded before I added the
Linux Clang regression; that additional method separately confirms its Darwin
skip in 0.026 seconds. Together these runs cover the current 201-method gate.
I removed the disposable Linux container, including its sanitizer scratch and
temporary Clang installation. No host toolchain installation changed.

## Assembler filename spelling

I compare direct native compilation with my production builder for five payload
filenames: `space name.bin`, `single'quote.bin`, `double"quote.bin`,
`back\slash.bin`, and `naïve-λ.bin`. I spell quote and backslash escapes in the
assembler string, preserve literal UTF-8 there, then encode that assembly in a
C string. A direct library must return 42 before I test its captured build.

Each supported compiler/assembler mode runs under both local and shared caches.
I require a retained generation, actual warm reuse, replacement to 43, failure
after deletion without changing the previous generation, and recovery to a
reusable 44. Separate characterization temporarily replaces 42 with 43 or removes
the payload during real object compilation, restores its bytes and timestamps,
and requires cold/warm/fresh 42/42/42 with actual reuse. Capture invocations are
excluded from the mutation hook; changing an input during capture would test a
different boundary.

The initial four-spelling recovery test passes on Apple Clang 21 (64.973
seconds) and GCC 12.2 Linux (2.902 seconds). The initial restored-edit/deletion
method passes on Darwin in 51.303 seconds; both Linux methods pass in 8.465
seconds. I then add UTF-8 and run the final checks below. No production code
changes are needed: the selected Apple backend and GNU read-replay paths already
handle escaped filenames that the literal copier declines.

With all five spellings, the GCC/Linux snapshot suite passes 62 methods with
fifteen skips in 43.268 seconds. Debian Clang 14 on the same ARM64 host passes
62 methods with nineteen skips in 58.121 seconds. Both use the GCC-built
production probe. Apple Clang 21 passes seven new and existing characterization
methods in 161.665 seconds, including source/header restoration, literal and
macro assembler reads, both assembler modes, and capture-failure refusal.

I then strengthen the restored-input method to require the external selector on
every real object invocation in external mode. That final method passes on
Darwin in 59.286 seconds; both filename methods pass on Linux Clang in 28.746
seconds. Each Clang host covers forty restored-edit/deletion cases and twenty
native-baseline/recovery sequences; GCC covers twenty and ten respectively.
These checks establish the five tested spellings with plain C flags, not every
assembler string escape or toolchain mode. I changed only tests and evidence;
the previous production gate results remain recorded above.
I removed the disposable Linux container and its compiler installation and
scratch files after verification; no host toolchain installation changed.

## Assembler include-search phases

My v34 build context admits literal `-Wa,-I,dir`, `-Wa,-Idir`,
`-Xassembler -I -Xassembler dir`, and `-Xassembler -Idir` arguments. Before
this repair, the production classifier declined them: restored payload edits
produced cold/warm/fresh 43/43/42 with actual stale reuse. I measured both
cache roots on Apple Clang 21, with ordinary and external assembly, and on
GCC 12.2 / GNU as 2.40 Linux ARM64.

I now classify preprocessing, C generation, assembler, and forwarded linker
arguments separately. Search paths stay ordered within package, common and
platform groups. Separate preprocessing and retained C generation do not get
assembler arguments; actual assembly does. Link-only jobs and their grammar
queries omit assembler search arguments while preserving linker operands.
Unrelated and unadmitted compatibility fragments keep their existing spelling.
Darwin's owned-response check derives its expected transport from the same
filtered link fragment.

Clang needs a separate capture detail. Its `-S` driver job drops assembler
include paths, whereas its integrated `-c` job forwards them to the frontend
before ordinary C include paths. When these search flags are present I use
`-c -Xclang -S`: the real driver's search order with assembly output. The
regression compares directly compiled code against captured builds with
different headers in the assembler and C include directories. The native
answer is 142 with integrated Clang and 42 with external Clang or GCC; I require
the captured answer to match, not merely to compile.

Strict `-Werror` testing exposed two additional phase errors. Assembler search
flags reached link-only Clang jobs and failed as unused arguments. After
filtering those, the linker-grammar query's trailing `-x none` still failed
under `-Werror`. I now create a private empty `probe.c`, query with that input,
and remove its owned directory afterward. I do not suppress user warnings.
The combined search-path / forwarded linker-response regression requires
actual reuse, not just successful output without a reuse record.

The three new methods cover flag admission and malformed/output-option
rejection; restored edits and deletion during final compilation; and search
order, native header precedence, strict warnings, phase argv, new earlier
candidates, missing-input preservation, cleanup and recovery. The latter
matrix has 36 local/shared cases on Clang and 18 on GCC. General assembler
options such as `--alternate` and auxiliary outputs are not newly admitted;
the broader assembler snapshot item remains open.

The final rebuilt GCC/Linux `make test-bytecode-shadows` gate passes all 206
methods with 24 platform/configuration skips in 146.035 seconds of reported
test time. This includes all 65 source snapshot methods, the strict-warning
search matrix, and linker response/transport regressions. The user guide
builds and validates thirteen pages in six editions; this is not acceptance
of the localized draft translations.

Two focused Linux ASan/UBSan methods pass with leak detection enabled in
197.766 seconds: flag-phase admission/rejection and all eighteen GCC ordered
search/failure/recovery sequences. Neither sanitizer produces a diagnostic
log. The production builder is instrumented through a separate probe;
supporting objects, native compilers and the assembler helper are not.

Debian Clang 14.0.6 on Linux ARM64 passes all 65 snapshot methods with
nineteen skips in 109.740 seconds, including ordinary/external search paths
and the strict-warning response combination. I select Clang through a private
`cc` symlink in the disposable container and use the GCC-built production
probe; this is cross-driver capture evidence, not a Clang bootstrap.

The final rebuilt Apple Clang 21 Darwin gate passes all 206 methods with
seventeen platform skips in 885.141 seconds of reported test time. Its 65
snapshot methods pass in 596.717 seconds, including the 36-case search matrix.
The later transport, response-graph and guarded-query suites also pass. These
results verify the admitted flag fragments; they do not establish arbitrary
assembler option support or compiler/VM semantic equivalence.

## Paired arguments across metadata fragments

My v35 context treats adjacent literal flag fragments as one argument
sequence for phase selection when they contain paired forms. The former
1024-byte coalescing threshold served response transport, not argument
semantics. A short `-Xassembler`, `-I`, `-Xassembler`, `dir` array must receive
the same capture treatment as its joined spelling.

The first Apple Clang 21 characterization failed before producing a library:
metadata loading interpreted bare `-I` as an empty C include path, rebased it
to an ancestor directory, and changed the forwarded assembler arguments.
Clang then rejected the directory operand as an unsupported assembler option.
I now track paired operands while applying legacy C include fallback and
leave bare `-I` and forwarded operands unchanged. Ordinary attached C include
fallback remains tested from a different working directory.
An unclassified shell fragment does not disable that fallback for later
literal flags. The ownership test includes this compatibility boundary.

Invocation capture copies metadata before coalescing, keeps array slots and
native-framework NULL entries stable, and leaves original strings untouched
on allocation failure. I reuse the existing literal parser and owned-copy
path. Shell expansions and malformed shell fragments remain outside admitted
normalization; I do not evaluate them to discover arguments. Coalescing does
not newly admit arbitrary compiler or assembler options.

The split restored-input characterization now yields 42/42/42 with retained
assembly and actual reuse on Apple Clang's ordinary/external modes under both
cache roots. Five new methods exercise paired C/assembler/linker words,
allocation failure and retry, include-operand ownership, restored edits and
deletion, plus six GCC or twelve Clang package/common/platform search-order
and failure/recovery cases. The common split case also carries a split
`-Xlinker -lm` pair and a captured linker response. The initial four-method
GCC/Linux run passes in 10.511 seconds.

The previous unknown-fragment test required split `-D`, `NAME=42` to avoid
capture. I replace that outdated expectation with a positive split/joined C
test: definitions and undefinition affect the compiled function, a selected
header change changes its answer, and unchanged input reuses its generation.
Genuinely unsupported scalar and shell-expanded flags retain their separate
compatibility tests.

Three focused Linux ASan/UBSan methods pass in 70.132 seconds, followed by the
positive C capture test in 13.231 seconds. Leak detection is enabled; neither
sanitizer produces a diagnostic log. The production builder is instrumented
through a separate probe, not its supporting objects or external tools.
After the compatibility guard, all four methods pass together in 84.635
seconds, again with leak detection and no sanitizer diagnostic logs.

The final rebuilt GCC/Linux `make test-bytecode-shadows` gate passes all 211
methods with 24 platform/configuration skips in 155.140 seconds of reported
test time.
This includes all 70 snapshot methods and the publication, transport, response
graph and guarded-query suites. The guide builds and validates thirteen pages
in six editions; localized draft validation is not translation acceptance.

Debian Clang 14 on Linux ARM64 passes the final 70-method snapshot suite with
nineteen skips in 129.825 seconds. I use the GCC-built production probe and
select Clang through a private `cc` symlink; this is cross-driver capture
evidence, not a Clang-built bootstrap.

The final rebuilt Darwin `make test-bytecode-shadows` gate passes all 211
methods with seventeen platform skips in 1081.933 seconds of reported test
time. All 70 snapshot methods pass in 782.050 seconds; transport, response
graph and guarded-query suites pass afterward.

An initial targeted Darwin run, concurrent with another gate, exceeded the
ten-second deadline in the fresh Python library reader for one external
package-flags case. It reported no wrong answer. The final full gate and an
isolated rerun of all twelve split-search cases pass with unchanged deadlines;
the isolated method takes 132.640 seconds. I have not established the cause
of that initial timeout.

### GNU alternate-macro production baseline

With production code at `d2082006`, I reproduce stale reuse on Debian arm64,
GCC 12.2 and GNU assembler 2.40. My new `assembler-alternate` characterization
uses a nested macro include whose `payload <path>` invocation requires GNU
alternate syntax. I require compilation without the alternate flag to fail;
the fresh native build with the flag must succeed. I do not insert `.altmacro`
into the source and thereby bypass the driver option under test.

Both `-Wa,--alternate` and separate metadata entries `-Xassembler`,
`--alternate` produce cold/warm/fresh answers **43/43/42**, under both local
and shared cache roots. Each warm build reuses the original generation and
performs no additional object compilation. Input bytes, length and mtime are
restored. No retained translation unit, assembly or assembler read manifest
is present, and the payload is absent from the reuse record. The source and
header controls remain **42/42/42** with actual generation reuse.

I reproduce each spelling with:

```sh
python3 -m tests.characterize_source_snapshot /usr/bin/cc --alternate-assembler --require-consistent
python3 -m tests.characterize_source_snapshot /usr/bin/cc --alternate-assembler --split-search --require-consistent
```

Both commands correctly exit one for the observed inconsistency. This is a
failing production baseline, not acceptance. Apple Clang 21 rejects
`-Wa,--alternate` in its native driver query; I do not claim this GNU mode is
supported by that backend. Production admission, phase routing, failure
recovery and the remaining compiler/metadata matrix are still open under
MAC `task_8c4127e1aeea4325acda9bca51eacf76`.

### GNU alternate-macro production repair

I admit the literal `--alternate` selector in `-Wa` groups and paired
`-Xassembler` arguments. My phase filter preserves it for assembler capture
and replay, removes it from separate C and link-only phases, and does not
mistake it for an include-search path. I advance the build context to v36 so
old generations cannot satisfy the changed capture contract.

The regression fixture requires angle-bracket macro arguments: without the
selector, native compilation must fail. With the selector, restored edits and
temporary deletion produce **42/42/42** with actual generation reuse and
retained GNU read manifests. Both flag spellings and both cache roots pass on
GCC 12.2/GNU as 2.40, Clang 14 with external GNU as 2.40, and GCC 13.3/GNU as
2.42, all Linux arm64.

Permanent payload edits produce a new generation. Missing payloads fail while
preserving the previous generation byte-for-byte; restoring the payload
recovers and permits reuse. A twelve-case matrix per compiler combines
alternate syntax with include-search precedence, common/platform/package
flags, mixed comma groups, split forwarded operands, and a captured linker
response. It checks native results and phase arguments, introduces an earlier
include, removes all matching includes, then restores one and verifies reuse.

The final GCC 12 gate passes 214 methods with 24 platform skips: 35 bytecode,
53 publication, four Linux link-cache, 73 source-snapshot, 20 transport,
16 response-graph, and 13 guarded-query methods. The snapshot suite takes
98.232 seconds. The full Clang 14 snapshot suite passes 73 methods with
nineteen skips in 169.890 seconds. GCC 13/GNU as 2.42 passes the four focused
phase, restored-input, payload-recovery and search-matrix methods in 19.414
seconds.

Three focused methods pass with leak-enabled ASan/UBSan in 155.278 seconds,
with no sanitizer report files. This instruments the production builder in
the probe, not its supporting objects or external compiler/helper processes.
The guide builds and validates thirteen pages in each of six editions;
validation is not translation acceptance. Apple Clang does not gain GNU
alternate-mode support. Arbitrary assembler options and compiler variants
remain outside this acceptance.

On Darwin, the phase-filter method and both existing include-search matrices
pass in 409.515 seconds: three methods covering 48 production build/recovery
cases plus the phase assertions. This checks the shared parser and preserves
integrated/external Clang include behavior; it is not a GNU alternate-mode
execution claim on Darwin.

### Mixed C and assembler translation-unit baseline

At `8724288c`, snapshot admission requires every ordinary and shared source
to have a `.c` suffix. A standalone assembler input disables capture for the
whole module. I add `--assembler-units` to the characterization CLI to measure
one C function reading a payload defined by a separate `.s` or `.S` file.
The `.S` fixture requires preprocessing to expand its payload filename macro.
The wrapper changes the payload only during compilation of that assembler
unit, then restores bytes, length and mtime. Changing it during the C sibling
compile would not exercise this defect.

Native mixed-source builds return 42 on Apple Clang 21 and Debian GCC 12.2,
both arm64. Production results under both local and shared cache roots are:

| Driver | Unit | Cold / warm / fresh | Publication |
| --- | --- | --- | --- |
| Apple Clang 21 | `.s` | 43 / 43 / 42 | stale generation reused |
| Apple Clang 21 | `.S` | 43 / 43 / 42 | stale generation reused |
| GCC 12.2 | `.s` | unavailable | rejected incomplete artifacts |
| GCC 12.2 | `.S` | 43 / 43 / 42 | stale generation reused |

Each successful cold build compiles two objects; the warm build compiles
neither. No retained assembly or translation unit exists for the C sibling,
and the payload is absent from its reuse record. Raw `.s` failures on Linux
leave no published generation, current pointer or staging directory. Their
native control succeeds, so rejection is not snapshot acceptance.

A separate GCC invocation with the builder's `-c -H -MD -MT
nano_module_dependencies -MF payload.d` options emits `payload.o` successfully
but no `payload.d` for raw `.s`. My publication path requires the per-source
depfile. Source-kind-aware capture must supply genuine dependency evidence;
an empty placeholder or a relaxed publication check cannot establish it.

```sh
python3 -m tests.characterize_source_snapshot --assembler-units --require-consistent
```

The command exits one on both hosts. Existing source/header controls still
return 42/42/42 with reuse. This is a reproduced failure, not a production
repair. Source-specific capture, shared-only assembler sources, include and
flag semantics, permanent edits and recovery remain work under MAC
`task_3f96ba3373db49a6b1c2a1987c1c0349`.
