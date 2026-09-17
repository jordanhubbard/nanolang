# My first shared borrow implementation

I implement a bounded part of `task_71821d84befc46e198795122c1112a27`:
checked direct calls may borrow a named, live, fixed resource record whose
fields are `int`, `float`, or `bool`. I retain the parent task until shared and
exclusive borrowing meet the complete affine contract.

I parse `&owner` as an explicit call argument. My C seed and both self-hosted
native stages emit a `const Record*` parameter and take the owner's address;
forwarding takes the address of the dereferenced parameter. I do not substitute
a record copy for the reference. My shadow interpreter retains the same
`StructValue` identity for these scalar records.

I hold each borrowed owner across evaluation of the remaining call arguments.
Repeated shared aliases are allowed. I reject consuming, destructuring,
returning, storing, or overwriting a borrowed owner, and reject borrowing an
already consumed owner. My tests require frontend rejection before native
publication and preserve an existing output file. A later-argument overwrite
uses a valid conditional expression, so a parser error cannot satisfy the test.

I still reject exclusive borrows, temporary places, resource collections,
aggregate/string-bearing referents, and generic referents. Nested field places,
foreign/callback borrow ABIs, and NanoISA reference lowering remain unfinished.
My C NanoVirt entry rejects borrowed parameters explicitly. My canonical
self-hosted NanoISA route rejects a reachable borrowed call; pruning an unused
function does not constitute executing that ABI. I have no field-assignment
syntax in this slice; whole-owner mutation through a shared parameter is
rejected before C emission.

I require `tests/test_shared_borrows.py`, `tests/test_borrow_annotations.py`,
and a fresh integrated bootstrap before this checkpoint is ready. I use the
explicit 60-second shadow budget already tracked under
`task_628759a2daf743b9bf13c9a7fea2ced0`; the production default remains ten
seconds. I record exact completed gate results below after they finish.

At source checkpoint `6ed82ee9`, I completed the fresh three-stage bootstrap,
C parser and typechecker suites, generated-schema consistency and 33 schema
methods. My final combined annotation/shared suite passed 13 methods in
59.541 seconds across the C seed, Stage 1 and Stage 2. Generated-C assertions
check actual const-pointer parameters and address arguments. My bytecode
negative covers all three relevant drivers. I retain the wider borrow task;
these results do not prove exclusive mutation, full shape support, ownership
IR, or release acceptance.

After merging main through `ae4aa585`, I found and repaired a missing
`borrow_mode` copy in postcondition cloning. The new parser regression first
failed with mode 0 instead of mode 1. At final source checkpoint `385ba833`,
my fresh integrated bootstrap, complete C parser/typechecker suites, schema
consistency and 33 schema methods passed. The 13 paired methods passed again
in 59.407 seconds, including meaningful helper shadows for repeated sharing
and forwarding. My compiler sources remained unchanged during each build.

My first exclusive-mutation probe then exposed an existing environment binding
copy that this shared implementation had not exempted: a borrowed parameter
was copied by `env_define_var_with_type_info`. This was an omission in my new
borrow integration, not a change introduced by PR502. At `89a442f7`, borrowed
bindings retain the caller's exact record pointer and their environment
teardown does not own that payload. Ordinary record bindings still copy.
Two identity assertions failed before the repair. All 42 environment checks
and ten lexical-scope methods passed afterward, along with the rebuilt
three-stage bootstrap and 13 paired methods in 61.183 seconds. A focused
ASan/UBSan build of the environment and its test passed all 42 checks; I
excluded unrelated legacy leak accounting from this focused lifetime run.
The separate exclusive prototype now observes caller mutation in its C-seed
shadows and native executable, but exclusive acceptance remains a later gate.

I tightened my negative helper to reject the self-hosted `Parse error`
diagnostic too. That exposed two fixtures that had proved only C checking:
parenthesized `if` arguments and qualified imported record literals. I replaced
them with supported match expressions and a module factory. The corrected
seven shared methods pass across all three frontends in 8.140 seconds. The
positive later argument now enters a match arm, appends local bindings,
performs a nested shared call, and consumes the owner after the outer call.
My held-index cleanup retains the outer binding prefix: lexical sequences
restore their entry count, branch/match regions clone storage, and binding
allocation preserves existing indexes. The paired overwrite case now reaches
ownership checking in every frontend rather than passing via parser refusal.
