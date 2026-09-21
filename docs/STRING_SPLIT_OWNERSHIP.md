# My complete split-string result

I prepare this prerequisite after the retained Darwin 2c9e bootstrap timeout.
I do not execute the existing affected split path to demonstrate a source-proven
ownership defect. `dyn_array_push_string` borrows its pointer. My evaluator
currently pushes a malloc segment then frees it, and its empty-delimiter branch
pushes a reused stack buffer. Its final tail borrows the input. Native generated
segments instead use `gc_alloc_string`. Both branches can publish incomplete
results after segment allocation failure. These are correctness prerequisites,
not a measured explanation of the ten-second deadline.

I preserve my existing public two-string `str_split` spelling and string-array
result. I allocate the array as `ELEM_STRING`; every successfully published
segment, including empty segments and the final tail, has independent GC string
storage. The array continues to borrow that GC-owned pointer through the
existing runtime contract. I add no manual free of a published segment, no
new foreign ownership mode and no general string/array lifetime claim.

For a nonempty delimiter I scan forward with `strstr`, copy each exact byte
span, advance by the full delimiter length, and always append the final tail.
Empty input therefore gives one empty string; a trailing delimiter gives a
final empty string. Consecutive delimiters preserve empty intermediate strings.
Newline splitting preserves CR bytes and arbitrary non-NUL bytes, including
partial UTF-8 spans. This matches the merger's existing NUL-terminated string
semantics; it does not claim embedded-NUL source retention. I preserve existing
native null-input behavior and each existing empty-delimiter length bound;
this checkpoint does not expand that separate boundary.

I change split-specific array or segment allocation failure to an explicit
first-person diagnostic and process abort, before returning a result. Existing
array growth already terminates on allocation failure. I do not claim a
recoverable status API, paired allocation counts, or transactional process
cleanup after abort. I never label a partial array successful. Allocation-fault
controls run in supervised children and require the exact failure boundary,
not a later invalid-memory crash. Existing compiler output sentinels remain.

My Nano producer presently lacks the builtin mapping, exact return typing and
runtime helper. Before merger substitution I add the actual independent Nano
emission of the same owned-segment algorithm, precise two-string argument and
`array<string>` result facts, and ordinary lexical/declaration precedence.
The C registry already names `nl_str_split`; I audit its array element facts
and actual inferred/explicit uses rather than assuming the broad ARRAY tag
is complete. I do not add a compiler-private foreign split primitive.

I first present evaluator/generated-runtime ownership source for review. The
paired mapping/type/emission and merger substitution follow as a complete
source checkpoint, with all existing merger properties retained. Focused
controls cover segment contents after unrelated allocation, empty delimiter,
empty/consecutive/trailing/newline/CRLF/non-ASCII inputs, independent copied
input lifetime, initial/segment/tail allocation failure, exact string-array
load typing and same-name declared/local calls. All original merger/JSON
shadows and the ten-second supervisor remain. Source and fixture review
precede fresh both-producer bootstrap and original full SDK acceptance.

A newline delimiter uses a forward native scan rather than per-byte interpreted
substring operations. I do not infer total bootstrap margin, libc complexity
for arbitrary multi-byte delimiters, or a passing SDK gate before measurement.

My paired production checkpoint registers `str_split` with the existing exact
array-name declaration/lexical authority in C and Nano. C element inference
returns STRING only for an actual unbound intrinsic call, and both checkers
require exactly two string arguments. Nano does not infer intrinsic authority
from a qualified member suffix. Its actual runtime emitter produces the owned
algorithm independently, with the same native empty-delimiter bound. Existing
declared-function C-name reservation prevents collision with `nl_str_split`.
The evaluator gates intrinsic dispatch with that same declaration decision.
I have not yet substituted this builtin into the merger or executed these
paths. Focused source, typed/inferred array and ownership fixtures remain next.

My focused C fixture includes the actual evaluator translation unit, or the
split helper selected from the actual `generate_string_operations` output.
The full generated runtime output is retained before selection. Eight cases
check every segment's value, distinct GC-owned address and string-array tag
and pointer width. The evaluator receives explicitly fixture-owned input
buffers through real identifier lookup; after the call the fixture detaches
only those borrowed values from environment cleanup, overwrites/frees the
inputs, performs unrelated allocations and verifies all segment contents.

The fixture intercepts only owning-TU initial array and segment GC allocation
calls, recording each index/site before returning NULL. Each measured index
runs in a separate child and must produce the exact diagnostic and SIGABRT;
core dumps are disabled without converting the signal to success. A separate
normal process after each fault checks the original result again. This is
process-isolated repeatability, not recoverable OOM or in-process recovery.
Fresh GC/dynamic-array providers and the included evaluator/native fixture use
the selected instrumentation. All other compiler providers remain ordinary
with explicit before/after hashes. Interior array-growth allocations are not
fault-injected by this fixture; existing terminal growth semantics stay open
to their own tests. No old faulty split program is executed.

My paired control requires an explicit three-role compiler manifest (`cseed`,
`refresh1`, `refresh2`) and verifies each executable hash before and after.
The outer refresh must retain the original producer/tools/products, compile
with every normal shadow and deadline, then self-compile once. I label this
chain a retained-producer refresh, never fresh C-seed bootstrap. Four positive
programs check exact/inferred string arrays and byte segments, declared/local
same-name functions and qualified declarations. Each route retains exact
selected shadow multisets and actual execution. Five refusals check arity,
source/delimiter type, array result type and element type with prior output
sentinels intact. Actual clean C-seed bootstrap and full SDK acceptance remain
mandatory after a separately reviewed merger substitution.

The first paired execution exposes one missing C lookup boundary after both
retained-producer refreshes pass: `env_get_function` still prefers the registry
except for a same-module, non-extern, body-bearing `array_push` declaration.
I add `str_split` to that exact preference, preserving the original owner and
body requirements. The failing declared-name fixture is unchanged. This is a
source correction, not a fixture escape or an extension of foreign binding
policy. All first terminals and successful prerequisite phases remain distinct.
