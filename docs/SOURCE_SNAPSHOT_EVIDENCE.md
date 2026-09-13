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
