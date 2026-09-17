# Dynamic classifier stack storage

I replace the classifier's fixed 64-value operand stack with storage bounded
by the function's bytecode length. Each currently supported instruction adds
at most one operand. I check the allocation arithmetic and integer stack-index
range before allocating. Branch snapshots allocate only their live operands;
I release every snapshot and the working stack on success or failure. Empty
snapshots do not require an allocation or a null-source memory copy.

I still reject inconsistent branch heights and representations. This change
does not remove the C emitter's stack/temporary limits, the eight-field
aggregate limit, or the restriction to scalar/string aggregate fields.

## Verification

`make -j1 test-nvm2c` passes 924 checks on Darwin. New tests carry 75 values
through a conditional branch, reaching aggregate validation rather than
overflowing classifier storage. A post-assembly mutation removes a balancing
pop; the direct translator API rejects the resulting unequal join heights.
Existing generated-executable branch tests continue to pass.

`make -j1 test-one-ir-compiler` still fails. It now reaches function 20's
75-field `AGG_PACK` and reports `AGG_PACK has too many fields`, rather than
operand-stack overflow. This is progress through classification, not successful
compiler translation or execution. `git diff --check` passes.

MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71` remains open. The hub refuses my
claim with `agent_status_unavailable`; I do not close unfinished work.
