# My self-hosted array facts

I track this prerequisite under task_f284f62c42a8405baa281e8d6eea6834, within
the original full record-list acceptance. I preserve f54 successful bootstrap
and scoped instrumentation, plus the later both-host first refusal failure.
Stage1 publishes output for `array<array<LocalItem>> = [[], foreign_items]`.
I do not execute that output or replay known incorrect acceptance for diagnosis.

## What my current source establishes

My literal checker takes the first element's type; `types_equal` deliberately
permits unknown array elements for empty-literal inference. This drops later
known imported identity when the first element is empty. My contextual helper
checks byte-array literals, while ordinary typed literals still rely on the
permissive equality path. My field-only checker fixes that boundary but does
not establish every local/call/return destination.

My direct array_push fast path returns a known receiver without checking the
value against it; array_set's builtin result is VOID without that agreement.
array_slice has no complete result inference. The IF expression path checks
only its then tail, unlike statement IF checking. Iteration reconstructs a few
scalar/record kinds and otherwise falls back to INT. These are static findings,
not additional executed failures. I audit map/filter callback and match/block
paths with this same complete contract before qualification.

## Literal facts and contextual destinations

I keep global types_equal unchanged. I unify literal member facts recursively,
retaining canonical nominal names supplied by nb_rewrite. A missing type is an
empty hole only when the actual AST is a genuine empty literal (or recursively
contains only those holes). A later known member fills compatible holes; two
known incompatible nominal leaves cannot unify. Unknown facts from a call,
identifier, unsupported expression or incompatible nonempty literal are not
empty evidence. I retain the existing legitimate empty-literal policy without
turning every unknown array value into a destination type.

I extend destination checking across let/set/return/direct and indirect calls,
record fields and union payload contexts. A typed array literal visits every
member against its declared element type, recursively including nested empties.
Only accepted literal subtrees receive element annotations. A failure rejects
the containing value; I do not claim rollback of previously valid child
annotations. Existing array variables keep their actual complete type.

I reuse root's scalar/byte contextual conversion policy. I do not report a
permitted explicit byte-literal conversion as an irreversible raw inference
error before its destination is available. Inferred literals with incompatible
nonempty members must instead produce a checker diagnostic. Neither an inferred
local nor apply_return_type_hint may erase a known nominal mismatch.

## Actual producers, consumers and branches

I select intrinsics only from the exact bound builtin name with no visible
ordinary/foreign/function-value declaration. I preserve qualification and local
precedence rather than classifying by suffix. Existing argument visits stay
static checks; this change adds no runtime evaluations or reorders them.

I retain array_new's reviewed checked fill inference. Array push/set check
arity, actual receiver element facts and the value destination even when their
result is unused. Slice returns the complete receiver type after index checks.
Get/at and iteration derive the complete nested element annotation rather than
defaulting another array, union or callable to INT. A real empty constructor
can use an explicit checked context; an existing unknown array cannot.

Map uses its actual receiver and unary callback: parameter agreement is exact
where storage/nominal identity requires it, and output carries the complete
callback result. Filter additionally requires BOOL and retains the receiver's
complete element type. I preserve declared same-named functions and the already
checked reduce contract. Unsupported facts diagnose; they do not invent owners.

I use the existing check_block_value/check_statement_list machinery for lexical
locals and statements in expression blocks. IF/COND and match values inspect
all reachable alternatives using their own bindings and existing enclosing
return/control-flow rules. Agreement retains complete array/nominal facts,
including contextual empty literals, without choosing only the first branch.
I do not replace ownership analysis, source evaluation or backend authority.

## Acceptance and attribution

I preserve all fifteen source methods and add helper shadows for first-empty
then-known inference, wrong later nominal leaves, heterogeneous nonempty arrays,
typed and inferred destinations, every changed intrinsic, ignored mutation
returns, callback/iteration facts and declared-call precedence. Positive cases
include matching nested empty branches and block-local values; negative cases
require checker diagnostics and output sentinels before any native execution.

Root reviews the complete source/fixture checkpoint before corrected gates.
Fresh full bootstrap retains the original ten-second shadow deadline and
outer bounds. The full15 matrix, relevant neighbors and unchanged Make remain
required. Exact C-seed/selfhost/emitter integration and enum list parity remain
separate full5.1 obligations; no scoped pass closes them.

## My source checkpoint

I carry `CheckedBlock.ok`, complete value type, AST-proven empty holes and a
separate terminating-control flag. My statement worker takes an explicit
`value_required` flag: an ordinary statement IF need not produce a common
value, while IF/COND/MATCH used as a value must check every alternative. A
checked terminal branch contributes no value to a join. At a declared
unreachable destination it satisfies that destination without creating an empty
array fact or publishing a literal annotation. I retain warning diagnostics
without counting them as type errors.

My recursive literal facts merge later complete members into actual empty
holes, including arrays inside tuple members. Existing identifiers/calls need
complete facts and cannot provide holes. I publish inferred annotations only
when complete; an unresolved genuinely empty initializer retains its AST-based
inference rather than turning into a declared unknown array on a later visit.
My common destination helper checks every array/tuple member, invokes concrete
union constructor checking where required, and preserves existing scalar/byte
conversion policy. Failed existing arrays retain their actual type plus an
error; permissive global equality does not suppress that error.

I retain exact unshadowed dispatch for array operations and preserve complete
nested results through get, slice, iteration and callback signatures. The
actual `array_new` path still requires complete checked fill facts, including
nested fills; I do not add an unknown-fill constructor admission. Explicit
empty literal syntax receives a checked destination. I also retain the full
List annotation when it is itself an array element, rather than serializing
its empty outer name. Qualified declared callables check all parameters and
arity, including function-valued declarations, without selecting array
intrinsics from a qualified suffix.

I add helper shadows and retain all prior source methods and assertions.
Static delimiter balance, added-helper shadow inventory and diff whitespace
checks are preparation checks only. I have not built or executed this
checkpoint. The complete paired source fixtures and original full bootstrap,
fifteen-method corpus, instrumentation and remaining Make acceptance are still
required after independent source review and capacity preflight.

## My paired fixture checkpoint

I add one source method without changing the prior fifteen. Its positive
program runs nested first-empty inference, direct/indirect arguments, record
and union payloads, lexical conditional blocks, integer match alternatives,
terminal return branches, complete nested iteration, slice and push/set through
the existing C-seed, Stage1, Stage2, evaluator and NanoISA/VM routes. Each helper
has a meaningful shadow. I add thirteen refusal programs for later nominal or
scalar mismatches, typed/inferred destinations, calls, payloads, branches,
iteration, ignored mutations, BOOL filter results and return destinations.
They require actual checker diagnostics and preserve output sentinels.

The runner remains unchanged, including retained products, command terminals,
process-group cleanup and deadlines. Static Python AST inspection finds sixteen
methods. I have not imported the runner or executed any fixture at this pin.

## Review prerequisite: discarded value facts

Root's static review of a0a98/2e9fe8 confirms that my expression-statement path
can discard `fact.ok == false` for an unused nonterminal expression. My parser
accepts bare array literals and match expressions through parse_statement's
parse_expression fallback. A heterogeneous literal or incompatible match can
therefore fail fact construction without adding an error. I must diagnose the
final checked fact even when unused, including unreachable expressions, while
allowing genuine literal holes. For an actual declared tail destination I first
perform contextual checking, so a valid byte conversion is not rejected from
its raw inference facts.

My cond parser lowers expressions to PNODE_IF, currently erasing the distinction
from ordinary statement IF. I add an explicit `ASTIf.is_expression: bool` in
my shared schema. Ordinary parser_store_if constructs false; cond construction
marks true through a checked struct copy. My checker uses this flag to validate
all discarded expression branches without requiring value agreement from an
ordinary statement IF. I preserve ordinary unequal/void branch successes and
terminal control separately. The field is parser intent, not an inferred token
location or body-shape heuristic.

My constructor/copy audit covers parser.nano and the legacy nanoc_integrated
ASTIf declaration/constructor, generated Nano/C declarations, value-based
list_ASTIf storage and getters, module/cache/nominal copies and both parsers.
The C seed keeps cond as AST_COND in its separate ASTNode representation and
already has an IF expression parser; I do not alter that representation here.
The schema generator derives fields from JSON; all affected consumers require
fresh providers before later qualification. No NanoISA opcode or admission
changes follow from this frontend field.

I separately record task_13edc03c03d149bc88bfde0be0e0fbe6 for the missing actual
self-hosted `(if ...)` parser route. Existing corpus cases remain requirements,
not successful parser coverage. I retain the peer parser/type-fact overlap
notice and require review before qualification or integration. No invalid
output is executed to reproduce either static finding.

### My discarded-fact correction checkpoint

I add the explicit schema field to both Nano declarations and all constructors;
my cond lowering marks it by copying every ASTIf field. The shared C declaration
is regenerated from the same JSON. The manual bootstrap FieldMetadata table
also carries the BOOL field. Current parser, legacy integrated parser, generated
schema, runtime list copies and getter consumers have been audited; neither C
ASTNode nor opcode representations change. Runtime list storage uses the full
shared struct size, so later providers are rebuilt rather than reused.

My statement worker diagnoses invalid unused facts, including unreachable
expressions. For a live tail whose value is requested I defer raw inference
rejection to its consuming context. Cond expression origin invokes the full
value-branch check even when discarded; ordinary statement IF keeps its
statement behavior. Shadows check actual parser acceptance, a heterogeneous
unused literal, cond/match, unreachable/nested cases, true empty holes, unequal
ordinary IF branches, VOID IF, and a byte-array destination whose raw branch
contains both INT and U8. That byte control is checker coverage, not a claim
about the pending emitter integration.

I add a seventeenth source method: one actual full-route positive program and
five checker refusals with unchanged sentinels, runner and deadlines. The prior
sixteen methods remain byte-for-byte equivalent at the Python AST level. No
compiler or fixture has run; regeneration is only source generation. The
separate IF-expression parser prerequisite remains open.

### My actual IF expression route plan

I record task_13edc03c03d149bc88bfde0be0e0fbe6 before changing parser code.
My primary-expression dispatcher will recognize IF and call a small wrapper
around parse_if_statement. The wrapper propagates errors before marking the
result as an expression. Existing parenthesized-head parsing already delegates
to parse_expression and requires the closing parenthesis; I reuse that route
rather than duplicate condition, block, else-if or delimiter parsing. Direct
expression positions use the same wrapper. Statement dispatch continues calling
parse_if_statement directly, including optional else and unequal/VOID branches.

An else-if remains a statement-origin IF in its synthetic block. The consuming
value context already requests that block tail's value and checks its branches;
I do not mark unrelated statement nodes by token position. I add parser shadows
for grouped/direct/nested/else-if expressions, missing delimiters/condition/body,
and ordinary statement origin. Source controls exercise valid values and
checker refusals through the existing retained routes, with all prior assertions
kept. Parser-malformed controls require actual parser errors, not arbitrary
compiler exits. No build or execution precedes review of this checkpoint.

### My IF expression source checkpoint

I add one primary-dispatch case and a wrapper that propagates parse errors
before marking the existing IF node. Condition/body/else-if parsing, statement
dispatch, constructors and delimiter handling are unchanged. Parser shadows
cover five direct/grouped/nested/else-if/optional-else expressions, five malformed
inputs and statement-origin preservation. The seventeenth source method gains
a complete value program plus discarded heterogeneous IF, non-BOOL condition
and nested incompatible array refusals; all its old cases remain. The other
sixteen methods are unchanged by Python AST comparison.

Source-only Python parsing, method inventory, Nano delimiter balance and
git diff --check pass. No Nano compiler, shadow, build or fixture has executed.
I retain the schema ABI change's fresh-provider requirement and the complete
seventeen-method qualification, original deadlines and all earlier first
terminals. This checkpoint awaits source review; it does not establish parser
or backend acceptance.

### My fresh c923 checked-list shadow refusal

I record task_7760776c77a04b40bd5ece6c7d1aacf4 before diagnosis. Fresh Linux
and Darwin Make builds pass parser compilation, then stop in typecheck shadows
with my checked list_ASTFunction_get refusal. Linux exits2 after54.928s and
Darwin exits2 after57.312s. I retain actual reports under
docs/evidence/record-lists-c923-first-build and all original products/maps/CAS
in each preparation directory. Sources/tools stay equal, process groups are
gone, and neither deadline nor capacity stopped these commands. Bootstrap,
focused configurations and the complete seventeen-method corpus are unreached.

The refusal combines live-handle, argument, bounds and snapshot-storage checks;
these terminals alone do not identify the failing shadow or cause. I inspect
static access/count assumptions first. Any bounded marker diagnostic receives a
separate source checkpoint and keeps the full graph and original deadlines. I
do not weaken this check or execute an invalid published program.

### My actual lexical block expression prerequisite

I record task_c6959732bd934c429b60e42731168c71 before source changes.
I record the actual block-expression parser prerequisite before code. The single approved c923 verbose diagnostic stops in check_if_value; full original graph, unchanged compiler/ten-second child/120-second outer, status1 at20.728991s, no timeout and exact source/tool/product input equality. Static primary dispatch lacks LBRACE, while the original cond/local vector requires a block value. I will reuse parse_block, propagate errors, and normalize only successful expression results to PNODE_BLOCK. I preserve original body/statement grammar and lexical value checking, keep original cond/local assertions, and guard function lookup after failed parse/count. Actual parser controls cover empty/grouped/nested blocks, local declarations, full consumption and missing/invalid braces. Full source controls cover lexical block values and refusals without weakening prior seventeen methods. No gate before source review; all parent acceptance remains open.


### My block-expression source checkpoint

I add a primary LBRACE route through parse_block_expression. The wrapper uses
parse_block unchanged and normalizes only its successful result to PNODE_BLOCK;
function/IF/loop bodies keep their existing block-ID consumers. A block used as
a statement still follows lexical statement checking, whereas a requested tail
or destination invokes value checking. I do not mark every block as requiring
a value or change passive flags.

Four parser positives check complete consumption, empty/nested/grouped blocks,
local statements and non-passive flags; four malformed inputs require parse
errors. The original cond/local shadow source and assertions remain, with an
additional function-count assertion and a guard before the dependent getter.
The seventeenth source method gains nested lexical block/cond values and two
refusals (escaped local and incompatible block destination); its existing
cases remain and the other sixteen methods are AST-identical. Static Python
parsing and diff checks pass. No corrected compiler or fixture has executed.

I retain the approved single verbose diagnostic under
docs/evidence/record-lists-c923-verbose-diagnostic: status1,20.729s, unchanged
inputs and no timeout, exact active check_if_value shadow. Full source/product
maps and generated products remain in the external diagnostic directory.
This evidence identifies the reached shadow; the missing block route follows
from static source inspection. Corrected full acceptance remains required.
