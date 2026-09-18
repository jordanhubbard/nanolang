# Bounded executable scalar reconstruction evidence

I read one retained `.nvm`, recover a checked typed region tree and emit two
executable source surfaces. My C and NanoLang outputs contain named functions,
scalar types, direct calls, if/else, pretest while and explicit returns. They
contain no embedded bytecode, opcode dispatch loop, goto or NanoVM call.

My final implementation/test checkpoint is `2a1f225d`, integrated with main's
source-borrow PR599 at `31991cba`. A fresh three-stage bootstrap passed on the
integrated compiler source. The subsequent entry-flag fix affects only the
standalone facts reader, which is not a bootstrap compiler input. I rebuilt and
ran the final reconstruction gate afterward on Linux ARM64.

| Gate | Observed result |
| --- | --- |
| Fresh integrated three-stage bootstrap and installed-compiler checks | Pass |
| Reconstruction integration | 6 methods pass in 67.804 seconds |
| Original retained modules | 8 NanoVM executions match expectations |
| Reconstructed C | 8 strict C11 builds and ASan/UBSan executions match |
| Reconstructed NanoLang | 24 C seed/Stage1/Stage2 native builds and executions match, with fixture shadows |
| Unsupported source publication | 20 language/refusal decisions preserve prior output |
| Instrumented facts reader | Loop, diamond and no-entry cleanup pass ASan/UBSan |

The positive modules cover zero, one and multiple outer-loop iterations, a
nested loop, both diamond branches, scalar direct calls, loaded-value snapshots
across later mutation, duplicate stack values, INT64_MIN/MAX comparisons,
boolean returns and colliding advisory local spellings. Unknown advisory-name
versions do not alter slot identity. The harness deletes the original assembly
source before reconstructing either surface and confirms the retained module
bytes remain unchanged.

My negative controls include absent entry flags, unknown parameter tags,
uninitialized reads, mixed local types, stack-valued joins, unstructured jumps,
an irreducible two-entry cycle, recursive calls, globals and arithmetic. A
function at index zero alone does not establish an entry: I require HAS_MAIN,
matching NanoVM's entry boundary. All these refusals occur before output-file
replacement.

I use existing loader/verifier/decoder facts, not advisory names, for executable
identity. My analyzer checks exact scalar storage types and definite assignment,
and each instruction belongs to one accepted structured region. My emitters
share that region tree. Typed temporaries preserve loads and call evaluation
order; side-effect-free loop conditions are recomputed at each iteration.
Function and slot indices disambiguate sanitized optional names.

I supply independent fixture shadow assertions only in the validation harness.
Those assertions are not recovered original tests. The spike establishes
observed equivalence for its tested closed grammar, not general reconstruction,
lossless recompiled bytecode or a newly imposed original-test retention gate.
Arithmetic, imported host ABI, wider values/runtime contracts, general CFG
recovery and complete frontend facts remain open parent obligations.

My bounded task is `task_15c7bd012efc44ffab908df7cd66b71e`; the parent
`task_4bd034f6029b7458201db74e2c3aeb32` remains open. The contract is
[executable scalar region reconstruction](../NANOISA_SCALAR_RECONSTRUCTION.md).

I retain local logs under `/tmp/nanolang-scalar-reconstruction-`, including
`bootstrap-integrated.log`, `integrated-final.log`, `entry-refusal.log` and
`asan-final.log`. Retained loop/diamond `.nvm`, facts and both generated source
surfaces are in `/tmp/nanolang-scalar-reconstruction-evidence/`.
