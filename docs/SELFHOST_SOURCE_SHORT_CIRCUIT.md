# I preserve conditional boolean operands in my canonical source producer

I execute `task_6e16a089a27a43af8703ff1610399a01` independently of the
NanoVirt source producer correction. SPECIFICATION8.5 requires source `and` and
`or` to skip the right operand when the left determines the result. My current
self-hosted binary emitter maps both source forms to eager BOOL_AND/BOOL_OR;
static inspection confirms that it concatenates both operand programs.

Before implementation, I require two exact BOOL operand facts. I emit the left
once, duplicate its result, and use existing JMP_FALSE for and or JMP_TRUE for
or. Those branches consume the duplicate. On the right path I pop the retained
left and emit the right once. The continuation label has one BOOL on every
continuing path. Unique labels come from the existing per-invocation allocator.
Both source expressions remain compiled and checked, including a skipped RHS;
no truthiness, malformed-module execution or runtime opcode changes enter this
repair. Eager BOOL_AND/BOOL_OR remain valid ISA operations. I keep other binary
operators and all existing expression admission/refusal boundaries unchanged.

My implementation helper and changed emitter receive meaningful shadow checks
for branch directions, one-value stack shape and exact-type refusal. I update
the old source-emission shadow that explicitly expected eager BOOL_AND; this is
a documented source-semantics correction, not removal of semantic assertions.
Fresh ordinary fixtures qualify selected/skipped and nested RHS calls with
observable counters, once-only left evaluation, loop conditions and exact
results through the C-seed, Stage1 and Stage2 canonical producers. Verified VM
and native translation run only newly emitted accepted modules. Unsupported
expression-block/control-exit shapes retain their existing checked boundaries;
this task does not introduce those shapes into the canonical grammar.

I require an actual fresh Stage1/Stage2 bootstrap, explicit isolated module
cache and tool identities, unchanged source during gates, and existing codegen
and adjacent canonical producer tests. Each first failure is retained before a
specific correction. The current product gate pins stay frozen; this repair
requires explicit later integration and new matching acceptance.
