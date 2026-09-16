# My self-hosted ownership-flow checkpoint

I replace the flat top-level ownership scan with recursive function flow
checking. Each lexical binding has one state slot; ordinary bindings mask
outer resource names without erasing their cleanup obligations. I explicitly
copy branch snapshots rather than assume array assignment isolates them.

I check fallthrough scope exits, early returns, branch joins, loop backedges,
break and continue. Returning branches do not participate in continuation
joins. I check overwrite-before-resolution and restore ownership after a
resolved destination receives a new value. Short-circuit right operands join
with the path that skips them. I check declared collection element types and
discarded resource results from declared function-valued parameters.

## Tested boundary

`make bootstrap3` completes its smoke checks. The combined frontend-parity and
contract-boundary suite passes 18 methods on my C seed, Stage 1 and Stage 2.
It includes the three original native rejection fixtures, parameter leaks,
observations, moves, more than 256 owners, branch disagreement, early returns,
loop-local exits, outer loop ownership, assignment, collection declarations,
indirect results, short-circuit moves and hidden outer obligations.

Source-only rejection probes require a static ownership diagnostic and preserve
prior output. Source-only positive probes do not call the declared foreign
consumer and do not establish runtime cleanup. Helper shadows check independent
snapshot updates and distinguish observation from a move.

The first broader quick run passes the 17 language cases but stops at the
GLUT build: source-aware emitter lookup leaves imported OpenGL constants as
undefined C identifiers. I track that regression in
`task_c4064660752d4717a8284da8388a43b2`. My repair marks global bindings
explicitly, keeps emitted locals' exact declaration bounds and complete type
metadata, and initializes header-created symbols. Ten lexical/global methods
and the headless GLUT guard pass afterward. The interactive window check is
not run. A full quick rerun remains required; the first failed run is not
release acceptance.

The final repair also keeps nominal record names on emitted local declarations;
one function's same-named local must not supply another function's record type.
A fresh `make bootstrap3 test-one-ir-compiler` passes bootstrap smoke and all
21 compiler-to-native methods after that repair. The contract-boundary module
alone passes 17 methods on the rebuilt compilers; the 18-method combined count
above includes the separate frontend-parity fixture suite.

## Still incomplete

My self-hosted classifier currently recognizes explicit resource declarations;
recursive records, union payloads, generic substitution and module-owned nominal
identity need further work. Resource match payloads, captures, borrows,
whole-owner destructuring, inferred higher-order signatures and ownership facts
in NanoISA are not established by this matrix. Unsupported expression forms
produce an ownership-lowering diagnostic in resource-bearing programs; that
rejection is not their implementation. Allocation-failure conformance for the
self-hosted state arrays is also not established by these tests.

The full task `task_c60a8d2e14b7494f8875e75b16e9b087` remains open. I have not
cleared full ownership acceptance or the release gate.
