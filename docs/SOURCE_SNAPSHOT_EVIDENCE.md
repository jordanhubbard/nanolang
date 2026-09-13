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
