# My helper-local-owner reconstruction qualification

I qualify the bounded exact v2 [2,1] two-function profile from unchanged
`source_borrow_helper_owners.nano`, including its else/zero-loop variant.
My ancestor is3446ec677, followed by design6225c7a46/3df0982f9 and
baseline5431252b3. Full #999 and platform release qualification remain open.

`full-first.log` passes18 methods in98.944s: eleven defensive witness methods
and seven actual source/native methods. `goldens.log` passes four old-profile
byte goldens in0.408s. Strict Clang/GCC16 facts syntax, Python syntax and
whitespace checks pass. The existing `test-owned-reconstruction` target now
includes both new modules with unchanged prerequisites.

I run the unchanged and alternate source through copied current NanoVirt and
both freshly installed3446 Stage1/Stage2 compilers. Original VM and O0/O2
ASan+UBSan native outputs agree with reconstructed C, regenerated Nano VM and
regenerated native products. The independently retained original-shadow driver
produces a separate accepted two-function executable; I do not recover original
shadows from main metadata. Generated consistency shadows exercise their own
literal owner fixtures and consumption, not universal branch coverage.

Original ordinary and original-shadow inputs have12/41 locals. Ordinary
regenerated entry/helper counts are49/120. All three producers measure the
consistency-shadow/helper at34/120; the conservative existing shadow estimate38
remains unchanged. Actual128-local helper publication passes;129 refuses
atomically, preserving prior output. The original179-helper estimate/refusal is
retained in `../owned-helper-local-first-gate-20261019`;179 was never claimed as
an actual regenerated count. The reviewed finite planner operates only on the
new Nano helper after immutable origin/work/witness proofs, covers151 statements
and retains all19 assertions. It changes neither caller nor C nor old profiles.

A later call with a different shared value fails the original and regenerated
VM assertion and native assertion boundary. Native controls require exit1 for
original/regenerated native or SIGABRT for reconstructed C and reject sanitizer
or runtime-error diagnostics. Descriptor overwrite is rejected by the original
verifier; malformed descriptor2.0/bool/operand mismatches are separately tested
as defensive facts refusals. Draft missed loop-invariant refusals remain in
`preinvalidating-before.log`; the corrected proof invalidates every recursively
written scalar and borrowed field before symbolic iteration. Eleven final
witness controls include generation/backedge, typed equality, literal ranges,
immutable snapshots and cumulative proof-cap boundaries.

The historical single-exclusive [2] former-refusal helper-owner variant remains
a separate open composition. This exact [2,1] component does not claim that
variant or arbitrary helper signatures, escaping owners or unbounded loops.

## My exact commands and providers

I run from `/Users/jkh/Src/nanolang-owned-nested-depth2` with:

```
NANOLANG_OWNED_RECONSTRUCTION_COMPILERS="$PWD/bin/helper_fresh_nanoc_stage1 $PWD/bin/helper_fresh_nanoc_stage2"
NANOLANG_OWNED_HELPER_SHADOW_EMITTER=/private/tmp/owned-helper-locals-baseline-20261019/shadow_driver
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang
python3 -m unittest tests.test_owned_helper_local_witness tests.test_owned_helper_local_reconstruction -v
```

The environment assignments apply to the command. `fresh-provider-pins.json`
records the copied current seed/stages/VM/assembler/translator/AOT object plus
immutable host-library/runtime closure. `provider-after.json` confirms all19
pins unchanged. The parent generated this fresh compiler pair with all17 steps
and raw fixed point SHA
`f1f25eb29c823bfcfdfcd1273288ecdc36035542ce3830bf10b8d963bff2e164`.
The original-shadow driver remains the separately pinned C-hosted driver from
`../owned-helper-local-baseline-20261019`, not three fresh shadow producers.

`commands.json` retains540 commands/statuses with bounded output excerpts and
full-output hashes; `artifacts.json` hashes796 retained temporary artifacts.
Representative sources accompany this receipt; binaries are not duplicated.
`source-pins.json` identifies the tested production/test inputs. The earlier
narrow facts build and first copied-seed count are retained separately; no
frontend bootstrap was run by this component.
