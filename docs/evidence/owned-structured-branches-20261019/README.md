# I retain bounded structured ownership branch evidence

I work from aggregate `5bb06c298` in isolated `fix/5.1-owned-structured-branches`. Original verifier admission is unchanged. I qualify only the approved contract in [my structured branch document](../../NANOISA_OWNED_STRUCTURED_BRANCHES.md); full #999 and fresh current-stage qualification remain open.

## My producers and controls

`qualification-inputs.json` pins every tool and the two changed production inputs. `bin/nano_virt` is my retained C producer; `bin/nanoc_u8_context` is my retained changed-source C-hosted Nano producer. These are not fresh stages from aggregate `5bb06c298`. My facts tool is rebuilt serially with consistent Homebrew Clang ASan/UBSan objects; `facts-build.log` retains the invocation output. My native checks use `/opt/homebrew/opt/llvm/bin/clang`, O0/O2, `-fsanitize=address,undefined -fno-sanitize-recover=all`, warnings as errors, and the pinned retained AOT runtime object.

The eight initial probe families preserve sources, original production/verify/VM commands and dumps in `receipt.json` and `conditions-receipt.json`. The committed test fixtures strengthen their shadows and add a nested four-path case. All actual branch fixtures execute both outcomes; the nested fixture executes all four combinations. I check exact nominal layouts and signatures, owned-call counts, original VM execution, reconstructed C native execution, and reconstructed Nano through both producers followed by VM and native execution. I reject hidden VM/goto/dispatcher output.

CRC-valid original-module mutations test backedges, non-boundary targets, ephemeral-read interior targets, missing owner transfer and missing authority, preserving prior C/Nano output. Supplementary actual-facts analyzer controls test unequal live owners, branch-local escape, partial scalar initialization, staged operand owners, crossing regions, nesting budget and a truncated ephemeral sequence. Existing owner/call suites cover original corruption, exact transfers, scalar suffix isolation and the unchanged generated-function budget. Their formerly rejected supported branches now have positive tests; an actual source loop retains atomic refusal.

## My retained failed attempts

`branch-tests.log` records invoking the Python file directly without the repository import path. I corrected the invocation to `python3 -m unittest`.

`branch-tests-module.log` records a harness-only invalid executable suffix (`-O0.exe`); I corrected it to `.O0.exe`. `branch-tests-rerun.log` passes all initial three tests in 57.745 seconds.

`focused-final.log` passes the new branch matrix including nested source and all other existing tests except the old scalar-branch refusal expectation. Its traceback source line reflects an edit made while that already-imported test was running. I replaced the stale expectation with an actual positive roundtrip and a separate source loop refusal, preserving the failing log.

`signature-control-failure.log` records an overstrict test assumption that generated Nano adds no functions. The existing emitter adds checked scalar arithmetic helpers; I require every original exact signature and unchanged owned-call counts while allowing those helpers. Production reconstruction was unchanged. All other 13 controls passed in that run.

`focused-qualified.log` is my final qualification command:

```sh
NANOLANG_OWNED_RECONSTRUCTION_COMPILERS=bin/nanoc_u8_context \
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
python3 -m unittest tests.test_owned_branch_reconstruction tests.test_owned_reconstruction tests.test_owned_call_reconstruction
```

I do not claim loops, general owner phi, borrow escape, broader resource layouts or full #999. The aggregate owns fresh producer/bootstrap qualification.

## My qualified result

All 14 focused tests pass in 132.501 seconds. The two final supplemental staged-owner/crossing controls were added during that run and pass in the separately executed `join-state-final.log`; the remaining tests were unchanged. `branch-command-receipt.json` retains all 520 commands from the final nine-fixture branch run, including outcomes/stdout/stderr; `branch-artifact-hashes.json` pins its generated source and modules. `join-state-command-receipt.json` retains the final supplementary run. Independent production review found no remaining blocker after the region-end variable collision was corrected before qualification.

I remove only extra trailing blank lines from tracked dump logs for repository whitespace checks; raw probe outputs remain in `/private/tmp/nanolang-owned-branches-20261019`.
