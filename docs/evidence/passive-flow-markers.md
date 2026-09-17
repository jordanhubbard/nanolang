# My structured flow producer checkpoint

I recorded the marker contract before implementation in `b8e9e611`, under
`task_d83213c008654ff4a561843889e778aa`.

`.flow_begin` declares a node count. Each `.flow_node` names its source ID,
result local, dependency IDs and external parameter reads. Instructions remain
in stable topological order. I collect their real ranges, then serialize the
complete unique node set in source-ID order. My existing version-2 verifier
checks graph, ordering, scalar provenance and actual instruction effects.

I changed no verifier rule, binary format, VM semantics, native lowering or AST
schema. Canonical disassembly still uses exact `.passive` hexadecimal bytes.
The marker interface does not implement frontend `flow` syntax or a scheduler.

## Acceptance

```
make nanoisa_dump nano_vm nvm2c
make test-passive-metadata test-disasm-roundtrip test-verifier
```

Twenty passive methods pass, including four new flow-marker methods. The new
positive cases cover a forward dependency with an exact expected v2 payload,
a diamond whose ready nodes have a source-order tie, a closed scalar helper,
guarded input, mixed `par`/`flow` blocks, and repeated flow blocks. Each accepted
module roundtrips to identical bytes and executes in both VM and native output.
Incomplete or mixed text markers refuse publication. The retained roundtrip
gate passes 210 checks; the verifier passes 96 checks.

I extended the existing allocation fixture for flow-index allocation failure
and ordered-record allocation failure followed by recovery. It passes with
ASan, UBSan and leak detection at `-O0`. Separately, all seven directive methods
pass through an assembler-instrumented CLI under those sanitizers. Other linked
objects were the ordinary build; I do not claim a fully instrumented toolchain.

I corrected a test setup omission before these final checks: the assembler's
internal function-processing fixture needs its first-pass function symbol.
The public assembler already creates that symbol; no production repair was
needed for that test assertion.

Frontend graph extraction is the next dependent slice. Broader immutable input
proof remains `task_bf571298c10d4cc5a387b9f233ff3c40`; exact foreign-purity identity
remains `task_20f6cb36fbf24bba987b4ea503529438`. No resource permission or foreign
summary is inferred from this producer interface.
