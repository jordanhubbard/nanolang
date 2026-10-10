# I retain indirect call identity through execution

I continue #989 from my distinct [indirect hosted plan](NANOISA_FILE_INDIRECT_HOSTED.md).
My private carrier owns that plan and exposes separate create, enter, view,
finish and destroy functions in `file_indirect_runtime.h`. I do not expose it
through an old acyclic or cyclic plan pointer. Both old destruction APIs refuse
an indirect context, and the indirect API refuses an old context.

## My callable and call boundary

I construct a callable only at the exact current `FUNCREF` instruction after
charging fuel. Its internal value retains the owning plan pointer and original
function index. Copy, local store/load and move preserve both fields; dropping
the value clears both. My copied public value view contains no callable handle
that can be imported into another context. There is no integer-to-callable API.

At `CALL_INDIRECT`, I check the top operand's type and plan identity, its bounded
function index and membership in the complete copied candidate set. The checked
and candidate sets must agree. I validate the selected declaration, parameter
and result types, ownership counts, caller continuation and child storage before
clearing the callable or moving any owner. I retain the selected function on the
waiting caller. Validation and return check that exact identity again against
the child and the original call site.

The callable contributes one consumed operand. The prepared staging bound
retains its conservative extra slot. I clear the nonowning callable only after
preflight, stage every supplied argument before installing any child local,
and preserve the existing distinction between reused VM operand storage and
simultaneously live native frame storage. My carrier operations allocate no
project-heap storage after begin; host stdio allocation is a separate boundary.

## My shared invariants

I copy common facts through private accessors for the actual owning plan kind.
I reuse frame layout, owner witnesses, references, regions, service operations
and terminal cleanup. I do not reinterpret an indirect plan as a cyclic plan.
My old public value, frame and report layouts remain unchanged; callable and
selected-target fields are internal to the opaque carrier.

I validate live owners and frame variants before each instruction and charge
once before effects. The bounded instruction counter spans initializer, entry,
calls and returns. Every failure retains the first execution error while
attempting cleanup of every live owner. Results publish only after successful
completion and cleanup. Allocation failures preserve caller outputs and release
every acquired prefix.

## My execution boundary

The first carrier keeps the hosted query's restrictions: no recursive call
graph, indirect borrowed formals, or callable parameters/results. Direct-call
borrows retain their existing rules. These are implementation boundaries, not
a reduction of my 5.1 scope.

My carrier fixture manually supplies instruction operations. VM/native storage
tests do not establish a shipped VM dispatcher or generated native dispatch.
Those adapters must cover every accepted instruction and fact, and native
indirect dispatch must call actual generated functions after checking the
selected target. Paired source lowering, full shadows, public grants, installed
consumers, richer callable signatures and both-host release gates remain open.
