# My Foreign Compiler Input Evidence

I have not finished lossless foreign-library cache validation. This experiment
keeps the remaining failure reproducible and tests candidate boundaries before
I change the compilation pipeline.

```sh
make nano_virt nano_vm
python3 -m tests.characterize_compiler_inputs
```

The optional argument selects a Clang-compatible executable. I use temporary
fixtures, run only their small integer-returning programs, and print JSON.
Exit zero means the experiment ran, **not** that cache acceptance passed.
I do not add this characterization to a green release acceptance gate.

## What I measured

On Darwin with Apple clang 21.0.0 (`clang-2100.1.1.101`), after `b084764c`:

| Boundary | Observation |
| --- | --- |
| Make dependency path | A literal `hidden\answer.h` becomes `hidden/answer.h`. Both files exist, so I hash the wrong one and create a reusable record. |
| Actual header edit | A same-timestamp change from 42 to 43 leaves stale code. The updated root shadow rejects publication; the retained program still returns 42. |
| Ordinary preprocessed input | Saved input and standalone `-E` output match; unchanged replay matches; the actual header edit changes the output. |
| Include search | Adding a header in an earlier include directory changes preprocessed output, although the old dependency list could not name that previously absent file. |
| Precompiled header | With a same-size, same-timestamp header edit and the old PCH retained, direct compilation returns 42. Adding `-save-temps=obj` returns 43. Standalone preprocessing follows the changed header. |
| Clang dependency graph | DOT output preserves the literal backslash, but system-header labels include logical `/usr/include/...` paths rather than the selected SDK's physical paths. It is not a ready-to-hash file inventory. |

The JSON reports these observations independently. A future compiler or cache
repair can change the results; the script does not assert that a known defect
must remain present.

## Include-trace repair

I now supplement Make dependencies with `-H` output from the original
compilation. In the tested Apple clang, the trace preserves escaped backslash
bytes and physical SDK paths. I accept literal paths and
[LLVM-style escaped](https://llvm.org/doxygen/classllvm_1_1raw__ostream.html)
paths only when their interpretation is unambiguous; if both spellings exist
and differ, I withhold reuse even when they currently share an inode. I hash
the resolved trace spelling as well as Make dependencies, so a readable slash
alias cannot hide a changed literal-backslash header.

I preserve warnings and errors when capturing the trace. Unexpected trace
content, including mixed diagnostics, withholds reuse instead of being silently
discarded. I recognize GCC's English guard-advice section only when it repeats
already recorded paths; my parser fixture tests this format, not a complete
GCC build on Linux. Empty traces are valid for compilations without textual includes;
Make records and the existing source checks are still required. I do not
claim that an empty trace establishes complete PCH, module or plugin inputs.

My acceptance regression exercises single-source, multi-source and shared-only
alias edits. This repair does not change the saved-input/PCH observation or
finish source snapshot and include-search validation.
The updated experiment observes the header edit through the cache and returns
42 in both direct and traced PCH builds, versus 43 in saved-input mode.

## Fresh search observations

I now use fresh preprocessing and its include trace as a supplemental veto on
cache reuse. I do not use that output as the original compiler's input. The
existing source, manifest, driver, environment, header and trace checks still
apply. This distinction preserves the configured compilation mode while
detecting the newly earlier header in the experiment.

I compare probes before and after a cold compilation and repeat the probe
before warm reuse. Failed or empty observations cannot authorize reuse.
Differing before/after observations withhold the new hash record, even if the
normal compilation produced valid code. I still need captured source inputs
to exclude changes that happen and revert between those observations, and I
still need compiler-mode-specific evidence for external inputs such as PCH.

## What this rules out

I cannot repair a lossy Make record by adding more escape decoding: the
compiler already replaced bytes. Nor can I assume a readable decoded name is
the file it read.

I cannot silently enable saved-input compilation for every manifest. The PCH
case shows a behavior change. A separate `-E` digest is not universally the
input of the original compilation either. The compiler documentation describes
[saved intermediate results](https://clang.llvm.org/docs/ClangCommandLineReference.html)
and [GCC's PCH preprocessing mode](https://gcc.gnu.org/onlinedocs/gcc/Preprocessor-Options.html);
neither establishes equivalence for every existing compiler configuration.

## Next implementation boundary

I need explicit, tested compiler-mode handling, not a second generic path
parser. For ordinary source compilation, retained preprocessed inputs are a
candidate for both lossless evidence and source snapshots. PCH, modules,
assembler inputs, plugins and other external compiler inputs require their
own captured dependencies or an explicit unsupported-cache decision. That
decision must preserve compilation behavior; it cannot silently change modes.

Before replacing current cache validation I require:

- The alias edit and newly earlier include to invalidate actual cached code.
- Unchanged input to retain warm reuse.
- Ordinary, multi-source and shared-only compilations to use the same rules.
- Compilation and reuse evidence to refer to the same captured inputs, even
  when source files change during a build.
- PCH and other external input changes to invalidate reuse without silently
  changing the configured compiler mode.
- Failed capture, incomplete evidence and failed publication to retain the
  previous generation and permit a clean retry.

These requirements remain in my roadmap and MAC cache task. This experiment
does not establish full toolchain identity or make 5.0 release-ready.

My separate [linker-input experiment](LINKER_INPUT_EVIDENCE.md) reproduces
same-timestamp archive changes and newly earlier library selections that my
cache still misses. It measures linker records, unusual paths and omitted
response-file inputs; it does not establish cache acceptance.
