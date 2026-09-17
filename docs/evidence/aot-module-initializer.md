# Native module startup

I now invoke the first function named `__init__` before my native entry point,
matching `vm_execute` in `src/nanovm/vm.c`. I require zero arguments and discard
the initializer's result. Host arguments are installed before this call, and
entry-lifetime owned maps and snapshots remain alive until entry returns.

An initializer failure stops the process before entry. If `__init__` is also
the entry function, I invoke it once for initialization and once as entry,
matching the VM rather than silently removing one invocation.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,176
AOT checks and 990 shape checks. The sanitizer driver verifies fresh ASan/UBSan
translator and graph objects; opcode case-parity and driver tests pass.
`git diff --check` passes.
My new tests check initializer-before-entry output, both definition orders,
void/integer/string/map initializer results, failure before entry, initializer
entry identity, and rejection of an initializer that requires an argument.

## Remaining global-variable work

This fixes startup ordering; it does not implement `LOAD_GLOBAL` or
`STORE_GLOBAL`. Full compiler acceptance still stops at `LOAD_GLOBAL` in
function 323 at offset 21.

The VM sizes globals from the highest referenced slot, enforces its 4,096-slot
limit, and zero-initializes their tagged values to void. Loads retain their
values; stores replace the previous owned value. Native implementation must
preserve those semantics, including reads before the first store.

My compiler artifact uses fourteen globals. Its initializer stores booleans,
strings and empty arrays; later functions replace several empty arrays with
string arrays. Global inference must retain boolean tags and handle these
empty-array declarations without treating them as proved integer elements.
This work remains in `task_bcc4271b0de244c2810c09e004f6cd2e`.

MAC still refuses my claim with `agent_status_unavailable`; the global task
remains open.
