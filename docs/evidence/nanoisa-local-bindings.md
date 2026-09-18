# Optional lexical local names: bounded evidence

I retain `nano.local.v1` advisory records for function, slot, lexical bytecode
interval and exact original name bytes. My ordinary C and selfhosted producers
name scalar parameters and scalar `let` bindings, including nested conditionals
and loops. I do not derive execution permission from these records.

I tested integrated source `919d86a878ecf54a3605b5609fd3bde1ea72c4c0` on Linux
ARM64. This includes main PR594 and my assembler startup teardown repair. My
fresh three-stage bootstrap passed at `cdf1b6659604ce56e0307c859b4ef982dfe0f600`;
subsequent changes add the C assembler allocation teardown, main's native integer
arithmetic fix, and documentation. My `.nano` compiler source is unchanged from
that successful bootstrap. The bootstrap builds the actual canonical compiler
through its host-module source manifest; this is not just a standalone tool link.

| Check | Observed result |
| --- | --- |
| Lexical codec, lookup, assembly and canonical text | 119 checks pass |
| Metadata append allocation failures | 5 boundaries pass |
| Assembler marker allocation failures | 20 boundaries pass |
| All three preceding suites under ASan/UBSan | Pass, including leak checks |
| Actual NanoVirt code generator | 89 checks pass |
| Paired C seed / Stage 1 / Stage 2 local-name producers | 2 methods, 6 source artifacts pass |
| VM and generated native C execution of those artifacts | Exact expected results 9 and 2 |
| Integrated native total integer arithmetic regression | 1 method passes |
| Earlier broader emitter gate | 86 checks and 88 integration methods pass |
| Earlier advisory transport / host closure gate | 188 metadata checks, 18 conversion allocation boundaries, 22 ownership methods and 2 Forth host-build methods pass |

My producer tests check parameter entry intervals, nested shadowed spelling,
branch interval separation, loop scopes and omitted nonscalar names. The C API
tests include a reused slot over disjoint intervals, empty intervals, unknown
versions, unusable advisory entries and unchanged executable eligibility. I
validate instruction boundaries before returning a name. An inconsistent
function-table pointer yields INVALID; my public lookup leaves its output
untouched when no usable matching name exists.

My builder publishes no partial metadata entry on allocation failure. Unused
string-pool constants may remain after a failed append; I do not claim complete
pool rollback. My assembler owns and frees marker allocations on every exit.
Nested C code generation shares the final name-list drain; names are disabled
for unsupported nested-function production rather than attributed to the outer
function.

My allocation sweep exposed an existing startup leak: the assembler created a
module after its initial code-buffer allocation failed, then returned without
freeing the module. I recorded MAC `task_d2326e6883ed4cbe93728cd1393024c2` before
repairing that branch. The original sanitizer report retained 10,128 bytes in
10 allocations. The unchanged sweep passes after the one-line module teardown.

I preserve exact advisory bytes, name intervals and executable code through
canonical text, then require byte stability on the second canonical cycle. My
existing disassembler omits DEBUG source maps, so I do not claim exact original
source-artifact byte roundtrip. That distinct obligation remains open as MAC
`task_1466d452d48c4a558c3be8a51765dd8f`.

My bounded producer task is `task_d62e26f741bf47b9810a7cfa43fca44a`. Pattern
binders, closures/upvalues, effects, specialized borrowed producers and other
unnamed families remain outside this evidence. Full frontend facts, structured
high-level reconstruction and release acceptance remain open.

I retained local gate logs under `/tmp/nanolang-local-bindings-`: the integrated
bootstrap, `final-core.log`, `final-paired.log`, `asan-final.log` (original failed
allocation check), `asan-repaired.log` and `asan-integrated.log`. Broader source
checks are in `producer-regressions.log`; earlier host/metadata checks are in
`/tmp/nanolang-local-binding-adjacent.log`.
