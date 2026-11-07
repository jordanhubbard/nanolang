# I retain bounded pure-pretest ownership loops

I work on isolated `fix/5.1-owned-pretest-loops` from reviewed `b6ee168d7`. The integration owner approved [my contract](../../NANOISA_OWNED_PRETEST_LOOPS.md) and reviewed the production diff without a blocking finding. Original facts/verifier admission is unchanged. I built no compiler or facts tools and ran no full bootstrap.

## My source and authority evidence

`receipt.json` and `providers.json` retain the original eight-case source mapping through the retained C producer and changed-source C-hosted Nano producer. Both accept preserved outer owners, body borrows, fresh per-iteration owners, conditional early returns and borrowed-field conditions. Both reject consumption of an outer owner on continuing paths and the reassignment probe. These are retained producers, not fresh aggregate stages.

My six positive source fixtures cover zero/multiple iterations, fresh lifetimes, nested loops, ephemeral body reads, conditional early returns and an unconditional returning body (whose actual producer code has no backedge). I compare original verified VM behavior with structured C and Nano, recompile Nano with both providers, preserve exact nominal layouts/original signatures, and run O0/O2 ASan/UBSan native outputs. Native flags include `-Wall -Wextra -Werror -fsanitize=address,undefined -fno-sanitize-recover=all`; the retained AOT runtime object and tools are pinned in `qualification-inputs.json`.

I additionally retain a producer-derived CRC-valid inverted pretest (LT→GE and JMP_FALSE→JMP_TRUE) and a dump/assembly variant inserting one unreachable backedge after the returning body. The original verifier actually admits both; VM and both reconstruction routes are tested. `return-backedge-receipt.json` retains the initial admission probe, not a fabricated producer success.

Actual accepted borrowed-field and arithmetic pretests remain atomic reconstruction refusals. CRC-valid malformed-edge controls cover mid-header entry, wrong exit, conditional backedge and borrow-interior target. Supplementary state controls use actual producer facts to reject a consumed carried owner, an unconsumed fresh owner, an operand owner at loop entry and scalar initialization that exists only in the body. These supplement, never replace, original module admission.

## My retained attempts

`loop-tests-first.log` ran the branch neighbors as well because the harness initially imported their TestCase into its module namespace. All branch and positive loop roundtrips passed. Its sole failure was a negative control that tried scalar POP on a staged owner and correctly reached the existing scalar suffix guard before the intended carried-owner check. I corrected the control to OWN_UNPACK followed by two scalar POPs; `state-edge-tests.log` passes the corrected state and CRC-valid edge tests. I changed the harness import to module form to avoid unintentionally collecting the neighbor class.

`focused-qualified.log` runs:

```sh
NANOLANG_OWNED_RECONSTRUCTION_COMPILERS=bin/nanoc_u8_context \
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
python3 -m unittest tests.test_owned_loop_reconstruction tests.test_owned_reconstruction tests.test_owned_call_reconstruction
```

My original loop refusal neighbor now uses an arithmetic pretest, preserving an actual admitted source outside this deliberately narrow slice. All arithmetic, calls, ownership/borrow conditions and operand duplication/discard remain excluded from pretests. Full #999, general owner phi, borrow escape and fresh aggregate qualification remain open.

## My qualified result

All 15 final focused tests pass in 74.180 seconds. `qualified-commands.json` retains all 423 commands from the loop suite, including provider commands, original verification, VM outcomes, reconstruction and native builds/runs. `qualified-artifacts.json` pins original/derived modules and generated source. The four structured-branch neighbor methods also passed in the first run under the same production implementation. Fresh aggregate stages remain separate.

I trim only trailing blank lines from the tracked return-backedge assembly; its original remains in the temporary probe directory.
