# I retain exact VOID calls and ordered BOOL assertions

I implement the approved [bounded contract](../../NANOISA_OWNED_VOID_ASSERT.md) on `fix/5.1-owned-void-assert` from `9071eadb9`. Independent production review found no blocker. Original verifier and affine APIs are unchanged. I rebuilt only the facts tool with the existing consistent Homebrew ASan/UBSan object set (`facts-build.log`); I ran no frontend/full bootstrap and did not edit the active aggregate.

## My original sources and exact providers

`receipt.json`/`hashes.json` retain the previous read-only accepted-source gap mapping, including the unmodified `source_owned_value_graph.nano`: both retained producers produce, verify and execute it, but the old facts profile stopped at its VOID consumer. `value-graph-partial-facts.json` and its error retain that original refusal. The mapping includes other still-open producer families and is not successful reconstruction evidence for them.

`qualification-inputs.json` pins the retained C producer, retained changed-source C-hosted Nano producer, VM, facts/reconstruction/translation tools, AOT runtime object and production source inputs. These are not fresh aggregate stages. I qualify the unchanged graph fixture plus source-driven VOID calls inside branches/loops and BOOL-returning ownership calls feeding assertions. Exact nominal layouts, parameter/result tags and result counts survive roundtrip; VOID calls remain zero-result statements with unchanged call counts.

## My actual trap behavior

The source negative fixture has a successful helper shadow using value42 and an intentionally runtime-only main using value43. The first assertion fails before a later zero-divisor expression and before owner cleanup. The source still contains complete cleanup on its successful path. This tests order without claiming that a trap proves ownership cleanup.

Original and reconstructed-Nano VM runs exit1 with `Assertion failed` diagnostics. Reconstructed C uses explicit `fputs` plus `abort`, returns signal status -6 on this worker and prints `Assertion failed`; O0/O2 and NDEBUG variants retain that behavior. The existing owned nvm2c route sets internal assertion status2, runs its existing cleanup, then its public main maps any nonzero status to exit1 with empty stderr. I retain that exact public outcome and add an instrumented NDEBUG status probe through its existing `nvm_owned_entry` API, which returns2 and prints `owned status=2`. The probe supplements public execution; it does not replace it or modify product source. I do not equate abort/exit codes or claim equal cleanup behavior across engines.

Only child-local RLIMIT_CORE=0 prevents core files during intentional native aborts. No sanitizer options or checks are suppressed. Native builds use O0/O2, warnings as errors, ASan/UBSan and no recovery, with the pinned retained AOT object.

## My refusals and retained harness failures

The first run (`void-tests-first.log`, `first-command-receipt.json`, `first-artifact-hashes.json`) passed positive roundtrips and analyzer controls but exposed two harness assumptions. I initially used legacy function-entry offsets for a v2 signature-count mutation. The purported value-count-zero file was byte-identical to the original, so its verification success demonstrated a no-op test, not a verifier gap. I now read v2 FUNCTIONS signature indices and padded SIGNATURES records, assert the expected old count, require changed bytes and require original verifier refusal. Correct result-count, VOID layout/parameter and missing-authority mutations all refuse without replacing prior C/Nano output.

The first run also expected a generic nvm2c diagnostic from its owned route. I traced the actual owned emitter and retained its public exit1/empty stderr behavior plus the assertion status2 probe described above. No broad diagnostic normalization or product change was made.

Supplementary actual-facts controls reject a leftover VOID return operand, a fabricated value after a VOID call, non-BOOL assertions, a VOID entry and a false assertion followed by missing owner consumption. The latter proves the analyzer does not treat a possible trap as successful cleanup. These controls supplement original module admission.

`void-tests-rerun.log` passes all four new tests in 28.996 seconds. My required full neighbor qualification command is retained in `focused-qualified.log`:

```sh
NANOLANG_OWNED_RECONSTRUCTION_COMPILERS=bin/nanoc_u8_context \
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
python3 -m unittest tests.test_owned_void_reconstruction tests.test_owned_reconstruction tests.test_owned_call_reconstruction tests.test_owned_branch_reconstruction tests.test_owned_loop_reconstruction
```

Fresh aggregate qualification, other accepted borrowed/nested/union/array/global paths and full #999 remain open.

## My final qualified result

All 23 VOID/local/call/branch/loop tests pass in 170.248 seconds. `qualified-void-commands.json` retains 221 commands including all exact false-assert outcomes; `qualified-void-artifacts.json` pins source and module artifacts. The branch/loop command receipts retain their actual neighbor executions. All required focused suites run against the final production inputs pinned above. I preserve the original failed attempts instead of replacing them with the green run.
