# I retain ordinary authority independently of acyclic declaration order

I execute `task_240f73cd751642beb9d09bba72ace3fe` under required ordinary
authority15f. This contract precedes production. Existing publishers and the
shared ownership validator restrict nested complete records to earlier layout
indices. I preserve all executable record ordinals and global layout indices;
I do not sort declarations or infer identity from shape to remove that limit.

## My bounded validated declaration graph

For a payload containing no RESOURCE flag, I validate every COMPLETE record's
existing scalar/string leaves and exact nested complete ordinary record index.
A reference may point forward or backward within the retained table. I require
a finite acyclic graph and reject missing/incomplete/non-record children,
invalid indices, unsupported fields and cycles. Unknown declarations remain
UNKNOWN. Reserved flags and existing signature/slot/path checks remain intact.
I use bounded iterative graph traversal, not recursive host calls, and preserve
existing metadata failure conventions and output atomicity on allocation failure.
No wire flags, field encodings or opcode meanings change; older validators may
conservatively refuse these newly qualified forward ordinary facts.

If any RESOURCE flag is present, I retain the existing prior-only scalar-tree
validation for the entire payload. This explicit boundary prevents the ordinary
extension from changing transitive affine eligibility, including an ordinary
child reached by a resource parent. Broader resource graphs remain separate.

## My structural transport prerequisite

Independent review found that my shared LAYOUTS decoder rejects forward edges
before authority validation. I explicitly extend that structural contract here;
I do not assume an ownership-only change can transport the new facts.

The default decoder preserves its existing path for every prior-only table.
If a table contains any forward edge, I require every layout to be STRUCT and
every field to be a scalar/string leaf with no nested index or an exact STRUCT
edge to another table entry. I validate the whole graph with an iterative
three-color walk before returning any decoded output. I reject cycles, self
edges, out-of-range indices, non-record kinds and unsupported field shapes.
The walk uses storage bounded by the decoded count, checks allocation failure,
and does not recurse on the host stack. Structural decoding proves an acyclic
record graph only: it never fabricates COMPLETE or ORDINARY authority.
Prior-only mixed-kind tables retain their existing structural verdicts.

I audit all decoder consumers before implementation. Module conversion and
cross-section validation preserve table order and exact per-kind indices;
their comments must state acyclicity rather than lower-number ordering.
Retained transport validates names/counts and round-trips the same bytes.
VM OWN_PACK and native owned emission perform indexed field reads only after
ownership/execution validation. Affine state uses explicit bounded field paths;
its reference-place helper retains its prior-only check unchanged. The shared
ownership validator explicitly rejects every non-prior nested edge anywhere in
a RESOURCE-bearing table, including UNKNOWN declarations, before the existing
scalar-tree rules run. This restores the old structural precondition for every
resource consumer despite the broader structural codec.

The managed record descriptor preflight retains its existing prior-only
boundary. Consequently the new field-origin API also continues to refuse
forward records: this child does not silently widen that independently
qualified analysis. Array/record executable selectors remain unchanged.
My codec/header documentation names this limited all-record DAG extension;
I qualify bare codec, retained transport and whole-module round trips, as well
as the separate authority verdict. Existing cyclic fixtures remain invalid;
only an acyclic exact record graph receives the new structural success.

## My paired optional producers

I keep the existing closed plain local non-generic scalar/string/record subset
and all import/foreign/resource/enum/union/unknown-slot exclusions. Field names
resolve against exactly one declared record across the complete table, keeping
its original ordinal. I check acyclicity before optional publication. Ambiguous,
cyclic or otherwise unsupported declarations omit both optional tables and keep
existing accepted source behavior; I do not convert an optional authority gap
into a new source rejection or fabricate empty layouts. Every actually emitted
function/local/shadow still needs the original exact slot facts.

This does not add array element authority, generic substitution, imported
identity, recursive nominal acceptance or any runtime record admission. Those
remain required distinct extensions under15f and aggregate488. New field-origin
queries may consume the checked graph only under their own bounded contracts.

## My acceptance

I require independent review before fresh execution. Shared validator controls
cover forward chains/diamonds, explicit empty and string leaves, same-shaped
nominal identities, canonical byte transport, invalid/missing/unknown children,
cycles and failure-atomic allocation paths. Existing resource/scalar-tree and
borrow controls must retain their verdicts, including a resource-bearing table
that attempts to introduce a forward ordinary edge.

Fresh C-seed/Stage1/Stage2 canonical producers and C NanoVirt publish matching
forward declaration facts without reordering runtime indices. Actual original
source shadows, verified VM and native behavior cover forward nested reads and
construction, while unsupported/cyclic optional publication stays UNKNOWN.
Existing supported prior-order and foreign UNKNOWN cases remain controls.
LLVM/Wasm record output stays refused and preserves prior files; success here
is declaration authority only. I record exact source/tool pins, preserve first
failures and integrate through review before closing this child, not parent15f.
