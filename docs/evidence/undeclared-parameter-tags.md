# My undeclared parameter tags

The selfhost compiler's raw assembly has no `.parameters` directives. Before
this repair, its saved bytecode nevertheless gave `purity_identifier` the
parameter tags `void struct bool string`, although its last two source
parameters are `name: string` and `count: int`. Native inference correctly
refused the resulting false integer/string conflict.

My v2 bridge interns equal signatures in one shared tag pool. On a duplicate,
it rewinds the unused candidate span. The next function or import without
parameter declarations previously reused those bytes without clearing them.
The pool's original zero initialization does not protect a reused span.

I now write `TAG_VOID` for each absent declaration span. This is my existing
unknown-proof placeholder, not a claim about the runtime argument. Explicit
parameter tags remain exact. I change no native signature check or instruction
semantics.

My regression interleaves duplicate explicitly typed signatures with larger
undeclared signatures, then does the same across function/import boundaries.
It checks interning, every parameter tag and a full wire round trip. Before the
repair, 21 assertions fail; afterward all 365 bridge checks pass. The
instrumented converter and test also pass those 365 checks with ASan, UBSan and
leak detection.

I reassemble the retained complete selfhost assembly with the corrected bridge
and the merged assembler fixes. The 348,980-byte output has unknown parameter
declarations, round-trips byte-for-byte through dump and assembly, and runs
`--help` in NanoVM. I retain it at
`/tmp/nanolang-parameter-proofs-compiler.nvm`, SHA256
`82d286c1120d6d3de4606d9256c1f47a5694cc52b3f51a998149e96ceefe4a4c`.
Native translation passes the original parameter guard and next refuses an
unresolved reconstructed field in `remap_diagnostics`. I track that separate
blocker as `task_250092bed54749ad988f06af5b88c228`; this artifact's native compiler
product remains unproved.

The seeded compiler/native bridge also passes its full explicit `--emit-nvm`
hello product test in 97.022 seconds. That is distinct from executing the
complete selfhost-emitted artifact natively. MAC
`task_04376d3e430c478d968af69e26543a0f` records this bounded signature repair.
I must rebuild immutable assembler facade artifacts before claiming subsequent
selfhost generations contain this repair.

I retain baseline, fixed, integrated and product logs under
`/tmp/nanolang-parameter-proofs-*.log`, with sanitizer output at
`/tmp/nanolang-parameter-proofs-asan/result.log`.
