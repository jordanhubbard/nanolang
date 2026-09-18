# My original scalar callback acceptance audit

I audit task_d0997d4a11184689ac99a91b430340de against its retained live
[description](original-scalar-callback-acceptance/original-task-scope.json).
Its three concrete obligations are direct FLOAT reduce-result inference,
legacy `fn(float,float)->float` ABI selection, and reviewed canonical native
callable transport **or named producer specialization**. General function-value
support is not an additional condition on that original task.

PR740's source child8618 supplies the checked inference and legacy selection;
PR759 supplies named producer specialization and its checker prerequisites.
I retain the failed worker's history and artifacts. I authored a new small
[source](original-scalar-callback-acceptance/fresh_named_float_callbacks.nano)
after those reviewed repairs, with distinct named float map/reduce callbacks,
NaN input bits, inline `float_to_bits(reduce(...))`, a typed reduced local and
unchanged input assertions. I did not read or execute the failed artifact.

| Fresh route | Actual result |
|---|---|
| Interpreter | Pass |
| Cseed, Stage1, Stage2 legacy native | All compile and execute successfully |
| Cseed, Stage1, Stage2 canonical | All publish, verify normally and execute in VM |
| Each canonical module through generated C | GCC and Clang ASan/UBSan builds and executions pass |
| Each canonical dump | No FUNCREF or CALL_INDIRECT |

All 34 recorded commands pass. Eight tool hashes remain unchanged. These are
the integrated8a751c1e tools already qualified by PR759; Stage1/2 retain the
actual2c68c01c bootstrap identity. Canonical merge79b7177f contains the reviewed
production. I make no new bootstrap or full-platform claim from this audit.
My [sealed manifest](original-scalar-callback-acceptance/report-sha256.json)
retains the source, runner, command logs, result and original task scope.

I assess the original three d099 acceptance clauses as satisfied. Its failed
worker attempt remains history; any ledger reconciliation must append the
new canonical evidence rather than erase that history. Broader function values,
U8 source tags, tuple/function expression annotations and collection callbacks
retain their separately tracked scopes. They do not turn this named task into
a different requirement. Parent5009 still needs the route-matrix review below;
full reconstruction and release stay open.

# My scalar-policy route matrix for review

I compare the explicit inventory and acceptance in
[NANOISA_BINARY64_ARITHMETIC_POLICY](../NANOISA_BINARY64_ARITHMETIC_POLICY.md)
with merged implementation and bounded evidence. I do not replace the policy's
default rounding/gradual-underflow/unsupported-environment boundary with an
unmeasured universal promise.

| Required scalar route or property | Qualified evidence |
|---|---|
| Shared result policy, target guards, volatile per-operation rounding, O0/O2/O3/contraction/LTO | [Helper730](binary64-arithmetic-helper.md); C99/C11 storage-guard companion and repeated helper gates in [public C748](public-c-binary64.md) |
| VM typed and generic, including mixed numeric promotion and both dispatches | [Backend731](binary64-arithmetic-backends.md): exact-bit controls and 274541 VM assertions |
| Native typed, known generic and boxed generic scalar paths | Backend731: GCC/Clang sanitizer controls, 2422 native and 1365 shape checks |
| LLVM/O2 LLVM, linked native LLVM, Wasmtime and import-free Node; managed scalar consumers | Backend731 and PR759's fresh adjacent source arithmetic methods |
| Main interpreter and both optimized callback evaluators | [Source734](binary64-arithmetic-source.md): 120 interpreter controls; 24 optimized callback results plus eight original input patterns; this fresh callback audit |
| Cseed and both selfhost legacy scalar emission, ordered globals, ordinary helper-like user names | Source734 and PR759's fresh19-method source suite |
| Canonical producer typed opcode selection and exact observers | Source734's emitted modules; fresh PR759 paired source/LLVM/Wasm methods |
| Separate public C-seed `--target c` route | Public C748: strict C99/C11, O0/O2, sanitizer/API/refusal controls; parent070db is completed |
| Original named float callback source pattern across all recorded routes | Source child8618, PR759 and this fresh34-command audit |
| Input/transport/negation unchanged; qNaN/sNaN, invalid infinity, signed-zero divisor precedence, ties/subnormals/overflow and separate-operation rounding | Helper730/backend731/source734/public C748, with new bit-observed source controls in PR759 |
| Unsupported owned/profile arithmetic and nondefault environments | Explicit refusal/contract boundaries; no accidental admission from policy helpers |

The later shared-header change is only the C99/C11 constant storage assertion;
its arithmetic bodies are unchanged and its current hash matches public-C748's
qualified manifest. I retain historical reports as historical; their old
"callback/public C still open" sentences are superseded by the linked later
acceptance, not silently reinterpreted as original passes.

I identify no remaining implementation clause in the original scalar5009
contract after this matrix is accepted. I leave its ledger state open for the
independent matrix decision. Existing aggregate-route task3717 is separately
scoped; adding native/LLVM/Wasm aggregate admission is not an invented condition
on either original scalar5009 or original3717 acceptance.

My next dependency-ordered implementation is a separate contract for typed
F64_ADD/SUB/MUL/DIV reconstruction. `scripts/nanoisa_reconstruction.py` still
admits transport/comparison/negation but excludes these four operations. That
slice should require two exact FLOAT operands, once-evaluated snapshots, the
same shared C policy and freshly qualified post-policy Nano producers, plus
VM/C/Nano integer-bit equivalence for endpoint/NaN/rounding/branch/call controls.
Mixed-tag, generic and broader heap/host reconstruction stay under parent4bd.
I propose no production changes or reconstruction admission in this audit.
