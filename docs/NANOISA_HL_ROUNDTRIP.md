# NanoISA high-level reconstruction

I ask whether a reader of one `.nvm`, without its original source, can recover
named functions, types, structured control and a declared host ABI. I now have
a bounded executable C + NanoLang reconstruction spike. I do not claim general
reconstruction or a target for almost any high-level language.

Canonical disassembly is a different gate: exact bytecode text transport does
not itself recover high-level structure. A bytecode blob, opcode dispatch loop,
embedded NanoVM or daemon client in another language also fails this contract.

## Retained facts and remaining prerequisites

| Need | Current boundary |
| --- | --- |
| Function identity and signatures | Function indices, names, exact parameter/result tags are retained. My bounded spike requires known int/bool signatures. |
| Local names | `nano.local.v1` records function/slot/PC interval and exact original bytes. Ordinary scalar producers are tested; broader producers remain separate. Names are advisory. |
| Local types | No general per-slot typed lexical table. My spike infers exact scalar types, requires definite assignment and refuses mixed-type reuse. |
| Structured control | My spike recovers a bounded region grammar from decoded branch targets: diamonds and pretest loops. Arbitrary reducible graphs and irreducible graphs remain outside it. |
| Host ABI | Existing native translators have bounded direct host contracts. The reconstruction spike refuses imports and module links; it does not complete the general ABI criterion. |
| Nominal layouts and ownership | Retained layout and ownership tables support separate verified subsets. My scalar spike refuses these contracts rather than reconstructing their semantics. |
| Source maps | Accepted DEBUG entries and empty section presence roundtrip through canonical text. This preserves facts; it does not recover original source. |
| Frontend facts | Bounded passive/ownership metadata exists. General purity, affine-use, generics, effects and exhaustiveness reconstruction remains open. |
| Original mandatory tests | Production bytecode does not retain the original shadow harness. My generated-source tests supply independent validation assertions; they are not recovered original shadows. |

Unknown advisory keys remain transportable. Unknown wire section types remain
refused. Neither names nor DEBUG records establish type or ownership authority.

## Executable scalar regions

`make nvm2hl` builds a separate host tool. It uses my existing loader, verifier
and decoder to extract checked identities and boundaries, then builds one typed
region tree consumed by both source emitters:

```sh
bin/nvm2hl --language c program.nvm -o recovered.c
bin/nvm2hl --language nano program.nvm -o recovered.nano
```

The initial grammar admits explicit scalar returns, constants, local snapshots,
int comparisons, exact wrapping I64 addition/subtraction/negation/multiplication, boolean operators,
direct acyclic calls, forward if/else
regions and canonical pretest loops. It requires empty operand stacks at joins
and loop backedges. I use checked helper implementations for these four integer operations and
refuse remaining arithmetic until its exact cross-language behavior is implemented. Imports, globals, aggregates, ownership/effects,
unknown signatures and unstructured jumps also remain outside this grammar.
The precise caps and admission rules are in
[my scalar reconstruction contract](NANOISA_SCALAR_RECONSTRUCTION.md).

For a retained loop fixture, the NanoLang surface contains nested `while`
statements, typed mutable locals and `if`/`else` state transitions. The C surface
contains the same recovered regions and direct functions. Neither output
contains goto, an opcode dispatch loop, bytecode arrays or NanoVM calls.
Function/slot indices make names collision-free even when advisory spellings
are identical. Temporaries preserve evaluation order and loaded-value snapshots.

My test harness deletes the assembly source before reconstructing both surfaces
from the retained module. It runs NanoVM, compiled C and reconstructed NanoLang
compiled by C seed, Stage1 and Stage2. The harness appends meaningful fixture
shadows for generated NanoLang functions. This validates selected behaviors;
it does not reconstruct the source's original mandatory tests.

Production `nvm2c` remains a separate, broader translator. Its typed storage,
scalar/aggregate/host support is not limited to this spike, and some of its
control flow still uses labels and goto. Native computation without an embedded
VM is useful, but those jumps do not satisfy the structured-region criterion.
I do not replace its product path or remove a compiler backend from this spike.

## Finding

**Sufficient for the tested closed scalar region grammar:** one retained module
can become two executable high-level surfaces with functions, exact scalar types
and structured control. I retain the measured pins and cases in
[my evidence](evidence/nanoisa-scalar-reconstruction.md).

**Insufficient for full high-level reconstruction:** remaining arithmetic, general control
flow, wider values and runtime contracts, imported host calls, complete frontend
facts remain unmet obligations. Original mandatory tests are not reconstructed;
that limitation does not invent an additional release criterion. A passing
bounded spike does not close MAC `task_4bd034f6029b7458201db74e2c3aeb32` or the
full v5.1 roadmap acceptance.
