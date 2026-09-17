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
