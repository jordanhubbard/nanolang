# My affine scalar-union boundary

I admit a narrow scalar-union profile beside exact owner transfers. This
contract records the boundary for `task_a18a9f752536469faafc4d3ebec01dfd`
before I publish the current-main implementation. It does not widen ordinary
aggregates, imports, foreign calls, or resource-bearing union payloads.

## My retained failures

My first checked emitter reached verification with `.types 1 0 1` but
published only the resource-record layout. Adding an empty union entry reached
affine analysis, which then refused a union local without an exact retained
identity. I keep both refusals: the verifier and affine analysis correctly
rejected incomplete evidence.

On Darwin, my first three-compiler source control passed C-seed and Stage 1,
then Stage 2 exited in `___chkstk_darwin` while parsing the unchanged
`Box<array<int>>` case. The generated native frames measured `0x180340` bytes
for `parse_primary` and `0x92fc0` for `parse_expression_recursive`. Moving
large parser pools to the heap did not change those frames. Compiling the same
generated C at `-O1` reduced them to 10,432 and 10,064 bytes and the control
passed. I therefore use `-O1` only when `NANO_CFLAGS` is absent; an explicit
setting remains authoritative. This is a bounded compiler policy, not an ABI
redesign.

After the clean current-main transplant, all 17 C-seed source cases passed but
Stage 1 and Stage 2 generated duplicate C `case` labels for two guarded arms
of the same variant. I retain that terminal. A C switch cannot express
first-success guarded ordering, so my self-hosted native route uses an ordered
predicate chain only when guards are present. The established unguarded switch
route remains unchanged.

## My admitted union

I admit only a fully instantiated union whose substituted payload fields have
exact scalar transport and one concrete identity for the declaration. I retain
that identity in parameters, locals, call results, and the nominal layout
table. Every variant contributes one positional payload shape; positions used
by multiple variants must have the same scalar tag.

Construction evaluates fields once from left to right and packs declaration
order. A copy is valid only for an incomplete, non-resource union layout with
the exact retained identity. A generic formal, resource argument,
resource-bearing record, unresolved substitution, inconsistent positional
shape, or second concrete identity for the same declaration remains a checked
refusal.

## My match semantics

I evaluate the scrutinee once. Arms remain in source order. A named payload is
lexically bound before its exact `bool` guard runs, and the first arm whose
pattern and guard succeed supplies the result. `return` inside an arm exits the
enclosing function; a final arm expression supplies a match-expression value.

I require static totality for the admitted statement/value profile. The
terminal runtime assertion is defense in depth and must remain unreachable for
a checked program. Owner state must agree across every continuing arm. An
owner consumed by every terminal arm is consumed after the match; a live-owner
disagreement refuses publication.

## My runtime boundary

The descriptor carries the union tag and exact retained layout. NanoVM affine
analysis may copy only an incomplete non-resource union value. `AGG_PACK`,
`AGG_TAG`, and `AGG_GET` retain the same shape in the VM and generated C. The
native translator carries the variant tag and scalar payload through its
refcounted aggregate carrier without treating the union itself as an owner.

I do not admit resource union payloads, nested ordinary aggregate payloads,
recursive unions, unresolved identities, cross-union calls, or a broad tagged
aggregate fallback.

## My acceptance

I require:

- fresh Darwin bootstrap through C-seed, Stage 1, Stage 2, installed compiler,
  and no-C-seed smoke;
- all scalar-union positive and refusal cases under C-seed, Stage 1, and Stage
  2, with refused output left unchanged;
- the guarded case through both self-hosted `--emit-nvm` producers, NanoVM,
  strict native translation, and execution;
- complete affine state and affine bytecode normal/allocation gates; and
- the adjacent owned-value graph lifecycle under Homebrew LLVM LeakSanitizer.

Full product acceptance, fixed points, unrelated aggregate profiles, PR522,
and release publication remain separate held gates.
