# My passive short-circuit node contract

I repair task_4c107eb5835743dd9e8c0683eadd78ea separately from the generic-call
repair. Fresh canonical af8809 and repaired8538 both reject the unchanged
passive-flow fixture; the baseline attribution remains in
`FULL_SUITE_PASSIVE_SHADOW_ATTRIBUTION.md`.

My version-2 scalar source expressions now short-circuit. I keep the same
metadata encoding and source semantics. I extend the node proof to finite
internal forward control flow; I do not merely add branch opcodes to a list.
Version-1 node admission stays unchanged.

For each version-2 node I decode at most 65,536 instructions, retain exact
instruction starts, and propagate stack heights in ascending instruction order.
My entry height is zero. Every instruction must be reachable in the conservative
CFG, every edge must reach an exact instruction start inside the same node,
and every branch must move strictly forward. Conditional branches propagate
both successors. Joined heights must agree; every instruction consumes only
available values and the checked height arithmetic cannot overflow. This
finite graph requires no iterative convergence or execution-time assumption.

Only the final instruction may store a local. It must store the declared result
and leave an empty stack. Every admitted path reaches this store: earlier
returns, halts, backwards edges, cross-node exits and branches past the store
remain refused. All loads still name a declared producer or guarded immutable
input. I collect the union of reads across reachable paths and require exact
agreement with the declared dependencies and inputs. Conditional omission at
runtime does not erase the static dependency. Closed scalar calls retain their
existing independent proof and bounds.

At the enclosing function boundary, a branch into the interior of a passive
block is allowed only when its source and target are in the same version-2
node and the node proof admits that edge. Outside entry into the complete block
remains supported. Cross-node entry, skipped producers, operand-interior
entry, mutable external inputs and effects remain refused.

I qualify selected and skipped AND/OR, nested boolean expressions, branch-local
reads, equal joins, and a shared final store. Refusal controls cover unequal
joins, underflow, cycles, wrong or omitted dependencies, unreachable padding,
internal and external operand-interior targets, cross-node targets, missing
result stores and side effects. I retain version-1 refusals and existing codec,
assembler, VM/native and passive frontend tests. Fresh Linux and Darwin
providers must compile and execute the unchanged passive-flow fixture. This
repair does not close the separate emitter-shadow timeout or full release gate.

## My first qualification observations

At35957 I pass fresh Linux26 passive methods,273 instrumented metadata checks,
both source-producer flow/par comparisons, the unchanged flow program in VM and
native C, and the scoped passive-TU ASan/UBSan allocation fixture. My first
Darwin driver setup lacked the external tracked-file manifest required by a
Git archive; no build or test launched. After supplying that manifest, fresh
build/discovery and273 C checks pass. The26-method suite reports25 passes and
one strict native C compile failure in the original arctan test: unused
`nparse_binary64`. I retain that terminal and track the adjacent helper omission
as task_1520319827d64ec9abee585fb1dfb08f. The four new CFG methods passed on both
hosts. I do not claim the Darwin suite or full release is green.
