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

My next corrected paired run stops at the qualified-module case after its
contents, same-module declaration and local-callable cases pass. Module
registration differs from root registration: it treats the bodyless builtin
registry result as a duplicate and then rejects every builtin spelling. I
permit only a non-extern, body-bearing `str_split` source declaration here.
Only an ownerless, non-extern, bodyless builtin result may be ignored; a real
source definition or extern remains a collision. I inspect exact same-owner
function rows before accepting the bodyless registry result, because ordinary
builtin lookup otherwise hides extern rows too. The same exact source
declaration is exempt from the later module builtin-name check. Qualified
lookup still requires its namespace owner and existing visibility rules.
Duplicate source declarations, private member access and extern collisions
remain refusal controls. I do not extend this policy to unrelated builtins.

I add three independent module refusals for each producer: a private source
member selected through a namespace, duplicate public source declarations,
and an extern/source collision with the same original name. Each preserves
a pre-existing output sentinel and requires a diagnostic identifying
`str_split`. Safe shadows do not invoke the invalid member; the original
public qualified positive still checks its exact selected shadows and result.

The 90b first terminal establishes another missing result boundary: all four
C-seed positives and argument refusals pass, but a split result is accepted
under `array<int>`. I preserve that generated product without running it.
My element-inference helper alone does not give contextual consumers a
complete result annotation.

I propose retaining `array<string>` through the existing owned array-expression
metadata API, only for a direct, two-argument intrinsic call after lexical and
declaration ownership checks. The API deep-copies the temporary array and
STRING child together; allocation failure must set the existing preparation
failure flag, without publishing a borrowed stack pointer or partial row.
Existing expression lookup returns that owned fact, and inferred bindings
copy it through their existing metadata path.

At the existing contextual annotation checker I compare a known immediate
STRING-array result against an explicit array element annotation. STRING
matches STRING; an absent or UNKNOWN expected element remains with existing
inference rules. An explicit different element refuses. This comparison also
covers aliases and declared function results already carrying that exact
STRING-array annotation; it does not change numeric array conversions or
claim unknown producers have STRING results. Existing nested array-literal,
tuple, control-flow and payload traversal supplies the same expected context.
The same checker already receives binding, assignment, direct/indirect call,
return, record field and top-level initializer boundaries. I audit each before
claiming the result contract. Original typing refusals remain; I add positive
STRING and negative INT boundaries for direct results and retained aliases.
I keep all allocation and complete-source acceptance requirements open.

My source checkpoint retains the complete intrinsic result through
`env_bind_array_expression`, which copies both the array and STRING child
before committing its row. A false result sets `opaque_resolution_failed`;
program and module checking already require that flag to remain clear. The
existing lexical/declaration gate precedes this path. I compare known STRING
array results in the shared contextual checker and route the actual borrowed
record-field assignment adapter through that checker. This does not prove
general numeric or computed-array compatibility.

My paired controls retain every original program and add a STRING acceptance
program covering inferred aliases, typed lets, sets, returns, direct and
indirect calls and record initialization. Declared, lexical and qualified
user functions named `str_split` return `array<int>` successfully in separate
positive controls. New INT refusals cover aliases, sets, returns, calls,
record initialization, globals and nested arrays, preserving output sentinels.
Both STRING and INT array-valued borrowed record fields retain the existing
public scalar-borrow refusal; I do not expand borrowed-source admission.

A separate C fixture includes the actual checker translation unit and supplies
an explicit existing record/exclusive-borrow environment. It checks the real
field-assignment adapter's STRING acceptance, INT refusal and repeated stable
result metadata. A binding-call refusal control requires no published row and
the preparation-failure flag. This injects the binding API's false result, not
every internal malloc site. The fixture uses ordinary complete hashed compiler
providers; it is not public borrowed-array execution or a whole-provider
sanitizer claim. Complete allocation qualification remains required.

## I resume the measured merger prerequisite without relabeling a bootstrap

My one copied `4e4a17be0` diagnostic ends at the original ten-second limit with 1,142 completed shadow records, all status zero. The active root shadow is `merge_with_imports`; the preceding `merge_with_imports_mode` shadow completes in 1.040023 seconds wall time and 1.039002 seconds process CPU. Both retain real JSON merging and parsing. I do not infer sufficient future margin or scheduling cause.

Before substitution I run a separately named primitive fixture: the unchanged initial contents/declared/local/qualified programs and three direct argument refusals from my full paired fixture, preserving selected shadow multisets and output sentinels. Its manifest states whether a host runs the current C seed, the two attributed retained Nano products, or all three. The complete original STRING/visibility/SDK corpus remains untouched and unfinished. I independently inventory retained source/tool/product closure and the exact split helper/emission bodies; a retained producer is never described as a fresh integrated bootstrap. My existing actual evaluator/native ownership fixture keeps all eight original inputs and fault/recovery assertions and adds a ninth physical buffer with a NUL followed by more bytes/newlines, requiring only the visible prefix to split.

For valid NUL-terminated strings, a newline delimiter preserves every byte before the first NUL, including CR and non-ASCII bytes, leading/consecutive newlines and the final empty tail. Empty input must still produce one empty string. Neither route claims counted embedded-NUL source semantics: bytes after the first NUL are outside the string. Native split segments own independent GC storage; the containing array retains the existing borrowed-GC pointer policy. Allocation failure must terminate at the already reviewed complete-result boundary, never return a partial array.

The source audit identifies an existing larger-input difference I must preserve: C-generated `str_substring` bounds its scan to 64MiB, while the Nano helper uses `strlen`. My present `split_lines` already keeps its legacy substring loop above 64MiB. I therefore propose an early intrinsic fast path only for lengths at or below 64MiB, and retain the entire current larger-input path. A separate zero-length return `[""]` preserves the existing empty result (including the Nano native helper's null-as-empty behavior); no new foreign null-pointer admission is claimed. The delimiter is the literal nonempty newline, so the unrelated empty-delimiter 64MiB difference is not used by the merger. This is a source proposal, not a substituted implementation or a measured complexity/margin claim.
