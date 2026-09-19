# I lower exact FLOAT-array fields inside owners

I track source child `task_18731b55c66846f9826290148c967ca5` under
`task_430220ce190946518d404088533531b6`. My static audit starts at canonical
`5f988ed79dd1f343fd9dca3d9b40c928264e668c`, after Samples source828 and retained
STRING source820. This is a preimplementation contract. I change no compiler,
validator, runtime, selector or fixture and execute no pending source here.

## I order runtime authority before source publication

My parent [field contract](NANOISA_OWNED_FLOAT_ARRAY_FIELDS.md) retains complete
Bundle/PREFIX acceptance. Descriptor824, origin829, authority830 and optional
operand832 qualify queries only. The separate
[private runtime checkpoint](NANOISA_OWNED_ARRAY_RUNTIME_OBLIGATIONS.md) is owned
by my runtime lane. Its proposed macro-only VM and native adapters prepare a
fresh complete owner-array plan; they do not change public selectors.

I wait for reviewed, qualified private runtime and separately reviewed public
activation before implementing or qualifying dependent source admission. The
activation must cover general/function/max-stack/zero-link verification,
converter/assembler, VM public entry/call/continuation boundaries, native
selection and error cleanup. Required-service rejection must remain ahead of
all candidate/delegation paths. Nonzero links and unsupported closed/LLVM/Wasm
profiles remain refused. I do not call a private test adapter from a compiler,
forge a plan, remove the final verifier or fall back to ordinary lowering.

The runtime lane's current private candidate is08be51c90, unqualified at this
contract checkpoint. Its interfaces and limits must be rechecked against the
actual qualified public checkpoint before source production. A query's positive
shape/lifetime result alone cannot discharge live runtime checks.

## I retain the original source boundary

I preserve `OwnedRecordPatterns.test_ordinary_array_field_keeps_element_type`
in `tests/test_owned_record_patterns.py` verbatim, including the complete PREFIX,
`close` shadow and main shadow. It constructs an ordinary spelling of Bundle
that transitively owns Handle and has `samples: array<float>`, destructures in
reverse field order, compares `(at samples 0)` with1.5 and consumes Handle.
I do not omit owner shadows or separately compile pieces to obtain acceptance.

This first source slice adds exact flat FLOAT-array fields to explicit or
transitive nongeneric owners, including prior-declared nested owners, array
literals, named immutable array aliases, direct rooted retained projections,
complete patterns and existing consuming owner calls/results. Contextual empty
literals are permitted only at an exact declared array<float> constructor field.
I preserve existing Samples ordinary construction/projection/alias behavior on
its separately qualified route. ARRAY-bearing owners are never ordinary rows.

I retain eight acyclic emitted functions including any synthetic shadow entry,
eight parameters,256 locals/stack cells, existing record/field bounds and
prior-child depth32. Hidden constructor/pattern slots consume the same256-slot
budget; I never recycle a live slot or omit a shadow to fit a limit. Full origin
analysis retains its own4,096-instruction/64-allocation-site and work budgets.
Those lower-layer limits remain checked refusals, not source guarantees.

I add no array mutation builtins or generic collection syntax in this first
source slice. The runtime's qualified mutation/alias behavior remains necessary
for parent430220 but is not a claim that all those operations have source
lowering. My separately ordered [source mutation extension](NANOISA_OWNED_FLOAT_ARRAY_MUTATION_SOURCE.md), task_bba6228369c04b6c900f628d501e5553, supplies the parent's required mutation acceptance after its exact builtin-identity prerequisite; it is not an unspecified future gap. Bare ARRAY parameters/results,
ordinary managed-record parameters/results, arrays of owners, nested arrays,
non-FLOAT arrays, owner FLOAT leaves, generic/union owners, callbacks, imports,
externs, globals and recursion remain outside this slice.

A module using both ordinary Samples operations and owner-ARRAY operations needs
an explicitly composed authority/runtime profile; the private owner-array plan
does not supply ordinary managed-record authority. I do not infer that
intersection by setting the old mixed flag. Existing standalone Samples and
STRING-owner acceptance must remain unchanged. Within the new owner-ARRAY
profile, STRING storage/transport follows its qualified plan; unsupported
STRING comparisons/output remain checked refusals. The current private profile
models INT-only PRINT/PRINTLN, not the source STRING print path. I add no source
print widening here and do not narrow existing STRING-only programs.

## I separate exact categories and metadata

I introduce explicit owner-ARRAY profile facts in both producers, separate from
ordinary Samples and borrowed profiles. A shared predicate may identify where
exact FLOAT arrays are permitted, but it cannot grant ordinary-record authority,
reference authority or runtime selection. Managed ARRAY fields remain
ARRAY/NO_INDEX; original nominal/global/source indices and COMPLETE|RESOURCE03
remain intact. Ordinary records keep01 only on their qualified route. No new
wire schema, fake VOID descriptors, compact-ID substitution or module filtering
is required for the bounded source change.

Source facts carry exact FLOAT element identity through literal elements,
field declarations, ordinary array locals/aliases and unpacked fields. Wire
ARRAY alone is insufficient; the final fresh origin/affine/scalar conjunction
checks every possible origin, complete owner calls/results and every body.
Nominal equality never substitutes for field provenance. An annotation mismatch,
unknown/uninitialized value or incompatible joined alternative remains refusal.

I retain checked lexical binding identity and initializer-before-binding order.
A local or function named `at` wins over builtin spelling. All complete source
checks and mandatory shadow selection occur before lowering; the owner-free
selection proof cannot bypass an owner-containing selected graph. I preserve
advisory intervals and deterministic literal/name ordering for exact producer
byte/dump equality, including hidden slots and synthetic entry names.

## I change the paired lowering mechanisms together

My audit locates the following boundaries in `src/nanovirt/borrow_codegen.inc`
and `src_nano/compiler/nanoisa_borrows.nano`:

- Declaration loops explicitly reject an owner with a direct FLOAT-array field.
  I replace that one refusal only with the distinct reviewed profile and
  propagate managed-leaf presence through prior owner children. Reference leaf
  predicates remain INT/BOOL-only; I do not globally widen field/local/parameter
  tag helpers.
- Constructor helpers currently use scalar field tags and expression emission.
  Exact array fields need their own contextual array emission and field tag.
  Every supplied expression is evaluated once in source order, validated and
  rooted in a hidden slot before the next field. Packing reloads declaration
  order with OWN_PACK. Unique child slots use OWN_MOVE_LOCAL; ordinary ARRAY
  slots use retained LOAD_LOCAL. No implicit owner duplication or drop follows.
- Unpack helpers allocate hidden fields using scalar-or-child tags. They need
  exact ARRAY/NO_INDEX slots for array leaves. Full pattern count/names/types
  validate before OWN_UNPACK_LOCAL consumes the shell once. Reversed binding
  order resolves each exact recorded parent/field hidden slot; a consumed owner
  does not become generally readable. Child owners transfer once; array aliases
  retain managed identity. Parent and child owner-consumption checks stay exact.
- Full scalar-leaf destructuring in every helper of this value graph must use
  the existing direct-unpack path, as qualified for Samples. In particular the
  original close shadow cannot introduce REF_GET/REGION operations into a
  reference-free owner-ARRAY graph merely because Handle itself is scalar-only.
  Existing nonmixed borrowed/disposal-holder behavior remains unchanged.
- Exact mode0 rooted owner field projection may use LOAD_LOCAL plus checked
  AGG_GET, retaining ARRAY/STRING leaves before publication. A nested owner path
  carries only temporary nonescaping observations through each AGG_GET; it does
  not create a copied owner local. Any added scalar observation path is checked
  against the live exact owner and current public plan, not borrowed-root
  authority. Whole managed-root borrowing, including scalar sibling paths,
  remains refused.
- Array literal/local/alias/`at` guards currently depend on the ordinary mixed
  flag. I factor only their exact-array availability while keeping ordinary
  construction and AGG_PACK exclusive to the ordinary category. Owner ARRAY
  literals use ARR_LITERAL FLOAT; `at` uses ARR_GET with an exact INT index.
  No general ARRAY tag becomes an exact FLOAT fact.
- C final publication currently chooses the mixed Samples candidate or old
  owned verifier. The new branch must use the actual qualified public
  owner-ARRAY conjunction and positive profile admission, without trusting
  advisory metadata or a failed prior branch. Selfhost assembly and installed
  canonical routes must reach the same public authority. The concrete exported
  API will be pinned after runtime activation; this contract invents no bypass.

Constructor reloads can retain a separate hidden ARRAY root until normal frame
cleanup; a new retain is not an owner move. Pack consumes only the stack roots
it receives. Pattern hidden ARRAY slots and named aliases each have their own
ordinary runtime root. Scope exit/error/return must release them through the
qualified category-aware paths, while unique children still require explicit
source consumption. I preserve earlier prepared roots when a later constructor
field/call fails; failed compilation preserves the prior artifact and never
executes its rejected output. No source-level cleanup inference replaces runtime
failure qualification.

## I preserve optional access semantics

ARR_GET may produce FLOAT|VOID for missing, negative or large indices. The
compiler may select a typed comparison from the source element type, but the
fresh public plan must retain per-operand obligations for all six F64 comparisons.
Both operands are evaluated once; runtime checks live tags before payload access
and drains roots on type failure. I do not add a default element, duplicate the
array read or relabel a bounds failure as a successful comparison.

The current runtime contract does not permit FLOAT|VOID to flow into exact FLOAT
local stores, arithmetic, negation, casts or signatures. Those selected uses must
refuse publication unless a separate reviewed contract qualifies them. Generic
EQ/NE retains its actual scalar semantics; heap/owner/array comparisons remain
refused. NaN and signed-zero checks use suitable bit observations for retained
representation, not floating equality as a bit-preservation proof.

## I qualify before closing this source child

1. I align this contract with the runtime lane's exact public interface and
   admission matrix, then send the complete paired production checkpoint for
   independent review before new source execution. Every new Nano helper has
   meaningful mandatory shadows; exact C/selfhost metadata choices match.
2. I freeze fresh bootstrap and compiler sources/tools. The unchanged original
   Bundle/PREFIX and all selected shadows pass Cseed, Stage1, Stage2 and NanoVirt
   routes. Both canonical source emitters produce equal verified bytes/dumps and
   VM/native behavior; Linux/Darwin and legacy/canonical route scopes are named.
3. I add accepted controls for reversed source constructor order and reversed
   patterns, two ARRAY fields/two shells sharing a retained alias, empty
   contextual arrays, nested owners, zero-argument factories, relay/consume
   helpers and returned owner origins. Aliases remain usable after shell
   unpack/move/call/return, with exact nominal and element types. Existing runtime
   mutation/growth controls remain a separate prerequisite, not silently omitted.
4. I retain explicit-consumption, repeated-move, wrong nominal/element type,
   incomplete/duplicate/unknown patterns, managed-root borrow, owner FLOAT,
   untyped empty, unsupported optional-consumer and profile-intersection
   refusals with expected phase/diagnostic and prior outputs intact. False main
   and helper shadows must fail compilation without publishing an executable.
   A parse failure is not evidence of an ownership/type refusal.
5. I qualify preexisting Samples/STRING/owner-free routing, scalar/nested owners,
   borrowed helpers and relevant complete pattern controls. Fresh valid
   out-of-bounds comparisons execute only after public admission review and
   must report the existing runtime type error with full cleanup. Existing
   allocation/lifecycle instrumentation checks prepared-array/child prefixes,
   traps and later recovery; I state exactly which allocation paths are measured.
6. I seal source/tool/SDK/commands/status/artifacts and actual merge ancestry.
   This child remains open until its own paired platform acceptance lands.
   Parent430220 retains all runtime/public/managed obligations; full ownership,
   full product/fixed-point acceptance and release remain independently open.

## I select the exact scalar opcodes of the new profile

Before completing source18731 production, I record the static interface mismatch:
my existing specialized producers emit generic integer/Boolean operations and
range-test/increment instructions, while owned_array_origins.inc admits exact
I64/BOOL operations (and only generic EQ/NE comparisons). No rejected module was
executed to establish this; the source and query opcode tables are explicit.

Only inside the distinct owner-ARRAY profile, I select I64_ADD/SUB/MUL/DIV_S/REM_S,
I64_NEG and signed I64 ordering comparisons for exact INT source operands;
BOOL_AND/OR/NOT for exact BOOL operands; and existing generic EQ/NE for INT/BOOL
equality. FLOAT operations keep their existing F64 instructions and complete
FLOAT|VOID obligations. Range comparisons and increments use I64_LT_S/I64_ADD.
I do not change non-owner-ARRAY opcode selection or admit new runtime instructions.

The exact INT handlers retain the existing integer contract: add/subtract/multiply
wrap modulo2^64, zero division/remainder returns zero, minimum/-1 division wraps
to minimum, minimum%-1 is zero, and minimum negation wraps to minimum. I verify
these source operand boundaries with meaningful mandatory mapping shadows and
later paired source controls, including range/while Boolean conditions. Source
annotations cannot discharge unknown/optional values in the final authority.

BOOL ordering has no admitted typed instruction in this profile and remains a
checked refusal. STRING comparisons and STRING printing remain refused only in
this new profile; existing STRING-only source acceptance stays unchanged. No
BOOL-to-INT conversion, generic dispatch fallback, new INT-print surface or
broader scalar policy follows. The original Bundle/PREFIX/shadows remain required.
