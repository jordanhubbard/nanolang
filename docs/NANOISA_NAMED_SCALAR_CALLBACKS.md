# My named scalar callback specialization

I begin task_fd15bbaf8acd4e448347ce3cce4e7a1d on canonical main
`a5985842`, after PR752. This is a local, no-dispatch child of the remaining
callback obligation task_d0997d4a11184689ac99a91b430340de under scalar policy5009.
I preserve d099's failed-worker evidence. I do not replay its old source or
failed artifacts. This contract precedes production and requires review.

## My existing asymmetry

My C-seed canonical `compile_builtin_call` snapshots a map/reduce callback as a
value and emits FUNCREF/CALL_INDIRECT, even when it is a known declared function.
My ordinary VM accepts that representation; native translation retains a checked
refusal. My selfhost `nisa_functional_index` already validates declared scalar
callbacks and `nisa_compile_functional` emits ordinary CALL. This child supplies
that bounded C-seed producer path. It does not implement native function values.

My existing scalar-reduce source child8618 already fixes exact checker inference,
legacy ABI selection and ordered evaluation. I reuse that checked contract and
leave broader source/callable acceptance visible in d099.

## My exact candidate and binding rules

I specialize only the existing builtin map(array, callback) or
reduce(array, initial, callback) after normal builtin/name resolution. I do not
specialize user functions or lexical callables named map/reduce. I do not add a
filter or array_fold expansion in this checkpoint.

The callback must be an ordinary identifier naming a same-module, defined,
non-extern, non-generic, top-level function with no captures and a known exact
scalar signature. My admitted scalar kinds are int, float and bool, matching
the ordinary scalar function ABI. U8, strings, aggregate/resource types, void and
unknown signatures remain outside this specialization.

I resolve the identifier in the actual current binding environment and connect
it to the emitted function by source body identity. A matching text name or
function-table index is insufficient. I check local/global callable bindings,
parent/captured bindings and anonymous/lifted lambda metadata before considering
a declared target. I do not mutate upvalue state merely to probe eligibility.
Imported declarations and ambiguous source ownership retain the existing path.
Forward references are eligible only when ordinary checked declaration resolution
already establishes the same target; I add no new visibility rule.

For map I require exact fn(E)->R and a checked array<E>; R selects the exact
output element tag. For reduce I require exact fn(A,E)->A, checked array<E> and
an initial A. No enum/int, int/float, unknown or nominal compatibility relation
stands in for exact equality. Typed empty arrays retain their known element
kind; any existing contextual empty-literal acceptance must use checked E.
If the candidate does not establish these facts, I preserve the existing
indirect implementation and its current downstream refusal. I do not reject
otherwise VM-accepted source solely because this optimization does not apply.

## My evaluation and runtime boundary

I retain the existing source snapshot first. Reduce then evaluates and snapshots
the initializer; its named function identifier has no runtime evaluation effect.
Map allocates the same exact result kind. I capture source length at the same
point as before: after initializer/callback evaluation. Each loop iteration
loads its scalar arguments in order and emits ordinary CALL to the resolved
function. Reduce stores its result before advancing. Map appends the result with
the existing ownership/array machinery. An empty source performs no callback.

I can omit the callback value temporary only for this checked named candidate.
Every computed/local/global/captured callback still evaluates once at the
existing point and remains an indirect call. Initializer mutations of a callable
binding cannot be bypassed by selecting a same-spelled declaration. Source alias
growth during initialization retains the established captured-length behavior.
I preserve callback side effects, left-to-right iteration and scalar arithmetic
helpers; no array arithmetic policy, opcode, wire format, verifier or native
callable admission changes belong here.

## My implementation and acceptance order

1. After review, add a bounded candidate resolver and direct-call branch to the
   C-seed map/reduce lowering. Keep existing indirect emission as fallback.
   Send this production checkpoint for independent source review before gates.
2. Freeze source and fresh ordinary fixtures. Inspect verified C-seed modules for
   direct CALL and no unnecessary FUNCREF/CALL_INDIRECT in admitted cases. Exact
   helper signatures, scalar result tags, map element kind and shadow selection
   remain observable. No old failure artifact is an acceptance fixture.
3. Execute finite and exact-bit floating map/reduce fixtures through normal VM
   and sanitized native output. Include signed qNaN/sNaN input payloads, canonical
   binary results, NaN divided by signed zero, separate-operation rounding and
   preserved input/transport bits. Int/bool neighbors exercise candidate tags; U8 keeps its existing fallback.
4. Exercise source/initializer order and counts, captured length after initializer
   growth, empty input, multiple callbacks, lexical/global callable shadowing,
   computed callback fallback and user functions named map/reduce. Wrong exact
   signatures retain checker refusal and previous outputs. Unsupported indirect
   native cases retain their explicit refusal and previous output.
5. Build fresh Cseed/Stage1/Stage2 canonical emitters and qualify paired actual
   VM/native behavior and legacy/interpreter source controls under their existing
   supported contracts. If final source integration changes bootstrap inputs,
   record the actual bootstrap pin and tool hashes. Run focused existing
   functional collection, source-reduce, NanoVirt and relevant native gates;
   preserve first failures and freeze any corrected harness before qualification.

I close only this named scalar producer child after reviewed canonical merge.
Computed callbacks, captures, imports, general callable runtime and full callback
acceptance remain separate. I assess d099's original clauses explicitly before
closing scalar5009; this child alone does not admit binary arithmetic source
reconstruction or complete aggregate3717/full release.

My implementation audit finds selfhost nisa_par_scalar does not admit U8. I
retain U8 as an explicit fallback boundary in this paired child; I do not expand
the selfhost functional ABI solely to enlarge this specialization.
