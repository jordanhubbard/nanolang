# My nested resource borrow places

I continue `task_71821d84befc46e198795122c1112a27` with the bounded child
`task_0ae2a64d17b0405b880ef7ad310f002c`. My normative affine contract requires
nested field projections. This checkpoint admits named roots followed by
record-field paths, while the borrowed referent remains a fixed resource
record with numeric/bool fields. I keep the wider borrow parent open.

I parse `&owner.member` and `&mut owner.member` with the field chain inside the
borrow marker. Both frontends resolve the leaf's nominal record identity and
the named root's mutability. Existing native lvalue emitters take the actual
field address. My executable controls observe mutation at the caller after
borrowing `wrapper.pair.left`; I do not substitute a copied resource.

I track each active call hold by root binding and field path. Empty paths mean
the whole root; overlap requires equal paths or a prefix ending at a field
boundary. Thus `.left` overlaps `.left.fd`, but not `.leftover` or `.right`.
Shared/shared overlap is allowed. Any overlap involving an exclusive hold is
refused, and whole-owner reads or moves obey the corresponding held paths.
Nonoverlapping fields can be borrowed independently during one call.

My C checker owns immutable hold nodes for the duration of a call. Branch and
loop snapshots borrow their chains; I restore the entry heads before freeing
that call's nodes. My self-hosted checker copies hold arrays before adding a
path, then restores entry holds without discarding moved-state changes. A
later argument containing a match and a nested borrow exercises restoration.

I still reject moved or temporary roots, partial resource moves, borrowed
reference escape and unsupported nominal types. Direct mutation of an owned
record remains outside this slice: field mutation requires an available
exclusive parameter. Generic/aggregate borrowed referents, borrowed callbacks
and NanoISA reference IR remain unfinished. Existing raw and canonical
NanoISA refusal tests remain required.

## My development evidence

At source checkpoint `04a3679d`, the combined nested/shared/exclusive/annotation/
resource-callback suite passes 44 methods across the C seed, Stage 1 and
Stage 2 in 224.616 seconds. Fresh bootstrap passes at the default shadow
budget. Parser/typechecker gates, 45 environment checks, ten lexical-scope
methods and schema consistency plus 33 schema methods pass.

A focused ASan/UBSan build of `resource_flow.c`, linked with the other ordinary
compiler objects, passes the seven nested and seven shared methods in 12.662
seconds. Leak accounting is disabled for this hold-chain lifetime check. I do
not present it as whole-compiler sanitizer coverage.

During development, an existing negative caught one mutation check still
consulting the obsolete whole-owner counter. I converted that check to field
path overlap and removed the counters before the passing 44-method gate.
The failing evidence is `/tmp/nanolang-nested-borrow-mutation-before.log`;
its concurrent C-seed invocation also records a temporary missing compiler
path during bootstrap installation, which is not the semantic failure.

My retained development logs are `/tmp/nanolang-nested-borrow-paired-r1.log`,
`/tmp/nanolang-nested-borrow-core-r1.log` and
`/tmp/nanolang-nested-borrow-asan.log`. Final integration below records the
canonical VM-shadow cutover separately.

## My integrated acceptance

Source `a6be2451` integrates main `8edf21e8`, including the canonical VM-shadow
cutover. Fresh bootstrap passes at the ordinary deadline. The 44-method paired
suite passes again in 182.638 seconds; parser/typechecker gates, 45 environment
checks, ten lexical-scope methods and 33 schema methods pass. My negative
projection controls require a positive compiler error status, excluding signals.

The focused ownership ASan/UBSan gate passes all 14 methods again in 10.709
seconds. I direct sanitizer diagnostics to separate report files and confirm
that none exist, including reports that a negative-case subprocess might
otherwise hide. The instrumentation scope and leak-accounting limit above
remain unchanged.

An adjacent standalone allocation gate initially fails to link on both my
branch and main: its fixture stubs lack the retained-type resource queries.
I record that harness repair separately as
`task_8b86bcc52b8a48dab3e39215c0042dbe`, restore the explicit fixture stubs,
and account for calloc and strdup in addition to malloc/realloc/free. I retain
300 owners and branch growth, add repeated shared call holds before consumption,
and require an ownership error plus zero outstanding tracked allocations at
each of 1,832 injected failure positions. The real-environment classification
gate passes separately; fixture stubs are not classification evidence.

My final retained logs are `/tmp/nanolang-nested-borrow-bootstrap-integrated.log`,
`/tmp/nanolang-nested-borrow-paired-integrated.log`,
`/tmp/nanolang-nested-borrow-core-integrated.log`,
`/tmp/nanolang-nested-borrow-asan-integrated.log` and
`/tmp/nanolang-nested-borrow-allocation-fixed.log`. The baseline harness failure
is `/tmp/nanolang-resource-flow-baseline-link.log`. These checks do not complete
my full ownership contract or authorize release publication.
