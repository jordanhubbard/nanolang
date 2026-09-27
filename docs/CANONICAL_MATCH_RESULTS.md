# My canonical match results

I carry a declared match result type into every selected arm, including concrete
union constructors, record values and nested arrays. Context supplies generic
arguments for empty variants and element types for empty arrays; it does not
make a mismatched arm valid. Inferred results require a supported common type.
An ambiguous empty generic constructor still requires a declaration.

I evaluate the scrutinee once. I test integer-literal or named-union patterns
and wildcards in written order, bind named payloads before their guards, and
execute only the first successful arm. Guards must have exact `bool` type.
A conditional wildcard may fall through. A bare or literal-true wildcard must
be last. Integer matches require an unconditional wildcard; union matches need
unconditional coverage of every variant or a final unconditional wildcard.
My integer pattern profile currently admits decimal digits supplied by the
parser; this does not add negative or alternate-radix pattern syntax.

An arm block retains its preceding locals until its final expression has
produced the value. A `return` leaves the enclosing function. I hide arm-local
names while retaining allocated runtime slots, and end scalar debug-name
intervals at the same lexical boundary. Nested matches retain private labels
and payload scopes. Statement matches keep the enclosing loop's control
labels. I retain an `ASSERT`/`HALT` backstop for a failed total-match invariant.

I preserve declared element kinds on empty C-frontend array literals even when
match or block results separate the literal from its annotation. In native
join inference an unresolved kind supplies no payload provenance. A resolved
heap or unknown boxed producer keeps its provenance checks. Tagged raw modules
may reach a checked runtime type boundary: a heap payload returned as an integer
must trap, and is not evidence that heap data can be interpreted as an integer.

MAC `task_e87e712d53b64195b205e5d89ea8d7f0` tracks the source work,
`task_5358726f98b44f2d8f3d37b1484fdfef` tracks empty-array context, and
`task_ec3252f81f41470d85947aa8e5d2303b` tracks native join provenance.
I also retain exact nominal identity for nongeneric nested union constructors
in both C AST forms. Native record returns preserve a known nested-array shape
when its flat field storage uses a tagged carrier; the carrier alone does not
make that array optional.

My qualification includes the three unchanged match acceptance cases,
paired production/shadow execution, guard effect traces, block returns,
lexical scope, exact-type refusals, raw heap-boundary controls and the full
translator regression. Fresh native bootstrap, the 72-method compatibility/scope matrix and the
91-method frontend gate pass. The complete scalar-match gate passed before the
final C nominal and record-return corrections; my evidence records that limit. This work alone does not qualify #522 for release or
establish the complete shared match policy across all backends.
