# My call-scoped reference contract

I require real references for `&T` and `&mut T`. This document orders the
NanoISA implementation of the already supported native cases; it does not
claim that my current VM executes borrowed calls. My C and self-hosted
NanoISA producers still refuse them. My full affine contract remains in
[AFFINE_TYPES_DESIGN.md](AFFINE_TYPES_DESIGN.md).

## Representation

I identify a place by an owning invocation, a frame-local root, its nominal
layout index, and a sequence of numeric record-field indices. A field name
is resolved before emission. Identical printed names do not equate layouts.
Within one module, a layout index is nominal identity; linking must remap
indices while preserving that identity, including distinct same-shaped
records. A runtime invocation identity is not a serialized pointer or a
function index: recursion creates distinct owners.

An empty path denotes the whole local. Each projection must select a record
field whose retained nested layout agrees with its type. I initially admit
only fixed record referents with int, uint8, float or bool fields, reached
through record-only paths. Tuple, union, collection, generic and unknown
referents remain refused until their independent contracts are implemented.
A descriptor never substitutes for the owning local's authoritative type.

My v2 bridge now preserves complete layouts through owned canonical bytes,
required feature bit 7 and canonical `.layouts` reconstruction. Count-only
legacy modules still produce field-less placeholders. Those placeholders do
not establish authoritative record shapes. My producers must populate actual
layouts and resource/function contracts before reference verification can
use them. My [transport evidence](evidence/nanoisa-retained-layouts.md)
distinguishes retention from producer completeness and ownership verification.

My first implementation slice supplies descriptor validation and pure overlap
queries against v2 layouts. It does not install a new section or opcode.
The later executable representation must carry:

- Function parameter mode (owned value, shared reference, exclusive reference)
  and exact referent layout. Reference results are forbidden.
- Authoritative root-local layouts and resource classification, including
  nominal nested layout edges. Existing tag-only signatures are insufficient.
- Reference creation, checked field access and call-region termination in code.
  Every creation names a checked root/path and mode; it cannot claim an
  unrelated layout to make overlap disappear.
- Versioned required ownership metadata. Old readers must reject the required
  feature. Serializers, linkers, reconstruction and translators must preserve
  it or refuse the module; converting to v1 must not erase it.

I will allocate wire codes only with their codec and verification changes.
The existing extended instruction plane provides room without renumbering
legacy operations. An unimplemented declaration is not an executable feature.

## Verification before execution

I compare places by owning invocation and root, then by field-index prefixes.
Equal paths and ancestor/descendant paths overlap. Different field indices
are disjoint, including fields with similar printed names. Two shared holds
may overlap; an exclusive hold may not overlap another hold. A held whole
owner may not be moved, replaced or consumed. Reads overlapping an exclusive
hold and writes overlapping any hold are rejected unless performed through
the authorized reference itself.

I derive holds from actual code and checked signatures, not producer claims.
Argument evaluation begins a hold when the reference is formed, so a later
argument cannot consume or mutate the held place. A direct call checks every
mode and exact referent identity. Forwarding creates a subordinate reborrow;
it preserves provenance, cannot strengthen shared to exclusive, and suspends
incompatible access through the parent until that nested call returns.

Every reachable join must agree on live references, provenance, ownership
and modes. Unknown type information cannot widen a reference into an ordinary
value. Loop back edges must restore entry state. Returns, tail calls and
unwinding must end the current call region without escaping a caller-local
reference. Tail calls with live local-root references are initially refused.
References cannot be stored in globals, heap aggregates, arrays, captures,
returned values or foreign/callback arguments. Ordinary stack duplication
and local loads must not silently copy exclusive authority. Imported calls
remain refused until their contracts can be checked across modules.

My existing ordinary-value type verifier allows unknown values at joins.
That policy is not sufficient for reference verification. I require a
separate precise reference/provenance state and conservative refusal wherever
that state cannot be established. Whole-program ownership verification also
requires explicit moves and resource obligations; place overlap alone does
not prove cleanup.

## Runtime and translator path

I keep reference identity outside ordinary copyable aggregate values. NanoVM
will resolve an invocation handle, root-local slot and checked field path
against the live owning frame. I do not retain a raw pointer into a locals
array that can move when a nested call grows the stack. Invocation generations
must distinguish a returned frame from a later frame reusing its slot.
Reference field reads/writes access the owner's actual scalar field; passing
a copied record and writing it back is not equivalent.

My native translator must preserve the same checked identity and call-region
lifetime. Stable root storage and checked projections may lower to native
addresses only after verification; it must neither box a detached record
copy nor allow a pointer to survive its frame. VM and translated execution
must demonstrate mutation visible through the caller's original nested place.
Unhandled reference operations must fail before output publication.

## Dependency order and acceptance

1. I establish checked place identity, field-path resolution, scalar referent
   bounds and overlap/access queries (task_83d7ced8a4e34d93b7193fdf2b841137).
2. I retain authoritative resource/layout and function-mode contracts through
   v2 codec, bridge, linking and reconstruction, with exact round-trip tests
   and explicit refusal by consumers not yet supporting them.
3. I add reference instructions and precise lifetime/provenance verification,
   including argument order, nested reborrows, joins and non-escape cases.
4. I implement actual NanoVM and native reference access from those verified
   instructions, with recursion, stack growth and caller-visible mutation.
5. I enable both frontends together and compare decisions and retained facts
   for the existing shared/exclusive/nested corpus. I run one pinned artifact
   through VM and native translation before lifting the current refusals.
6. I retain full move/cleanup verification, real service-handle migration and
   the release equivalence matrix as the broader ed702/d03c/28f tasks.

The first slice is a tested prerequisite, not completion of any later row.
My publication hold remains. The old equivalence task's references to explicit
`discard` or replacing borrows do not override my accepted design: I support
real call-scoped borrows and reject `drop`/`discard` syntax.
