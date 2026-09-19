# My declared array_push identity prerequisite

I track this bounded correction as `task_286bcd3cbf28b8ad009677ed6b9f8c73`,
prerequisite to mutation `task_bba6228369c04b6c900f628d501e5553`.
I retain frozen mutation961a on both hosts. Linux ran four methods in330.031s;
Darwin ran four in325.611s. Each passed the first three and the fourth method's
initializer/scope program, then refused its unchanged declared `array_push`
program in `nano_virt` checking: declared INT body/call versus builtin ARRAY.
These are first terminals, not passing twelve-method qualifications.

My Linux logs remain `/tmp/nanolang-owner-array-mutation-961a-gates.log` and
`/tmp/nanolang-owner-array-mutation-961a-gates/`; Darwin retains the matching
`/private/tmp/nanolang-owner-array-mutation-961a-darwin-gates.log` and directory.
No failed output is executed. I make this correction in a separate tree.

## My static path findings

- `src/env.c:env_get_function` resolves a namespace first, then the builtin
  registry, then current-module declarations. `array_push` is not in the C
  reserved declaration list; `array_set` and `array_length` are. Registration
  accepts the declaration, but body checking, call checking and shadow attachment
  subsequently receive the registry's ARRAY-return function instead.
- `src/typechecker.c` additionally recognizes push by spelling in empty receiver
  inference, array element inference and record-element inference. These facts
  must come from the selected declaration when it wins lookup.
- `src/eval.c` evaluates arguments once in order, then dispatches push by spelling
  before its ordinary user-function path. I must bypass this builtin dispatch
  for the selected declaration and reuse the existing invocation path.
- `src/transpiler_iterative_v3_twopass.c` has spelling-based push emission and
  element inference. The earlier user-function check is for an identifier value,
  not this direct-call branch. A lookup-only correction cannot fix native calls.
- `src_nano/transpiler.nano` resolves the identifier with `mb_resolve` but its
  push special case still uses only the resulting spelling. Its non-identifier call
  path captures arguments in order, but its direct identifier call path does not.
  I add ordered snapshots only for declared or lexically bound array_push calls,
  retaining resolved signature argument types, without treating an empty type
  string as proof that a lexical binding is absent.
- My specialized C/Nano ownership producers already test declared/local identity
  before push lowering. My selfhost checker has the qualified and lexical guard
  from PR835. I preserve those boundaries and inspect their controls unchanged.

## My bounded correction

I permit an explicit same-module, non-extern function body named `array_push`
to win the C registry lookup. I preserve qualified namespace resolution and do
not reorder every builtin or expose foreign-module fallback as new authority.
Duplicate declaration checks and declaration shadow attachment must select the
real row after registration. `array_set` and `array_length` remain reserved by
my existing C policy; I add no declaration permission for them.

At each push-only inference or execution special case, I distinguish my builtin
from that exact declaration and applicable existing lexical binding. I use the
ordinary declared signature, return/element facts and call path for the latter.
I do not infer ARRAY merely from its spelling. Unbound push keeps its existing
array mutation/result semantics, empty-array inference and ordered evaluation.
I retain the existing checks for argument types/counts, named owner transfer,
mandatory shadows and output publication. I add no ownership/runtime admission.
I do not claim to repair generic non-callable or qualified argument checking,
all builtin names, imported declaration ambiguity, or unrelated legacy effects.

## My acceptance before mutation resumes

1. I review the complete production checkpoint before execution. Any Nano helper
   has a meaningful mandatory shadow for declared, builtin and lexical identity.
2. I exercise an ordinary declared `array_push(int,int)->int` with observable
   argument order and a nontrivial shadow through C interpretation/native and
   both selfhost native stages, plus canonical routes. A declared array-return
   control has a return element type different from its first array argument,
   so spelling-derived element facts cannot accidentally pass.
3. I keep ordinary unbound push controls for empty typed receiver, returned alias,
   growth and value order. Exact lexical/initializer controls remain; reserved
   set/length declaration refusals preserve existing output. False declared
   shadows and wrong declared argument type/count must refuse publication.
4. I build fresh stages and run the unchanged full twelve-method mutation/source
   suite on Linux and Darwin with explicit native compiler selection and hashes.
   I preserve every PREFIX/shadow, final assembly refusal and runtime cleanup
   assertion. I do not replace the declared owner helper with a renamed fixture.
5. I retain old961a first outcomes separately and seal the corrected source/tools,
   artifacts and terminal statuses. Mutation, full owner ARRAY and product/release
   parents stay open until their own acceptance and actual merge reconciliation.

## My prepared production checkpoint

I now retain the exact same-module body before registry lookup, share a push-only
lexical/declaration predicate for C inference and lowering, and bypass interpreter
builtin dispatch when the selected function has a body. Native C name mapping
also preserves the selected declaration rather than returning the registry C name.
My push-only lexical checker path reuses checked function signatures and refuses
non-function bindings; it does not repair generic calls under other names.

My Nano native predicate checks membership in `env.names`, including empty type
entries, and the resolved declaration identity. Bound/declared push captures its
callee and arguments once in order, with existing declared/function-value argument
types; lexical result inference uses the selected signature. Both new helpers have
mandatory shadows. I retain ordinary unbound push emission and all other builtin
branches. Prepared source controls exercise native C and both selfhost stages,
canonical scalar/array routes, exact declaration return-element facts, lexical
restoration, meaningful shadows, refusals and prior-output preservation. Indirect
native controls do not claim new canonical indirect-call admission.

Only static inspection, Python fixture parsing and whitespace checks have run.
Production/fixture review precedes fresh bootstrap or program execution. Both
frozen961a terminal logs and trees remain untouched. After identity/mutation
acceptance, my next product prerequisite is the entire unchanged twelve-method
`tests/test_owned_record_patterns.py` corpus, with all four compilers and PREFIX
shadows; I do not start that run before these dependencies qualify.

## My retained naming failure and correction boundary

I record naming prerequisite `task_c6d34913ce079519f958caaeef7817fb` before repair. Frozenbb52
Linux bootstrap280.601s/tools27.079s pass, then the first identity case fails
generated C compilation (Ran1/1.297s): `src/stdlib_runtime.c` retains a builtin
`nl_array_push`, while my accepted same-module declaration emits the same name.
No failed output executes. Darwin bootstrap325.180s/tools59.639s pass; its source
gates remain unrun after this common naming defect was established. I preserve
both bootstrap trees and the Linux first identity log/artifacts.

I retain the builtin helper and every other scope's ability to use it. I name
the unqualified nonextern declared function `__nl_declared_array_push`, outside
my ordinary `nl_` source-name prefix. C definition/prototype/call/function-value
references must agree. Existing valid module-qualified mangling remains intact.
Nano definitions and prototypes share `c_func_name_for_definition`; selected
direct push calls and unbound declaration identifier values must use that same
identity. Lexical variables keep their existing names and signature precedence;
extern references and builtin emission remain unchanged. I add exact emitted-name
shadows and a source function-value control retaining the actual declared push.
I do not rename the program's source function, remove the builtin helper, or
weaken the unchanged mutation/source suite. Complete diff review precedes fresh
bootstrap and gates on both platforms.

## My ordinary canonical identity correction

I record `task_3615e58e593058d66ea4cdd2915df4d9` before code. Frozen925c Linux bootstrap
275.394s/setup27.429s pass. Its first declared array-return identity case passes
all three native compilers/executions and C canonical verification/execution,
then Stage1 canonical lowering refuses the exact float array element type
(Ran1/8.193s). Stage2 canonical and remaining groups are unrun. The initial
message naming nano_virt was corrected from the retained subtest header; the
first failing producer is nanoc_stage1. No refused artifact executes.

My ordinary `nisa_expr_type` uses push receiver inference before declared result
lookup, and `nisa_emit_call` selects ARR_PUSH by spelling. I guard only these push
builtin paths with an exact resolved nonextern declaration predicate using
`nisa_call_index` and its source-owner mapping. The selected declaration retains
its full result and parameter context through existing CALL emission; builtin
push keeps receiver/empty-literal specialization. Local/global indirect calls
remain refused; extern behavior and unrelated builtins remain unchanged.
`[value]` return lowering already carries `nisa_return_type` and exact element
checks to ARR_LITERAL3; I do not change it or invent inferred ARRAY fallback.
Mandatory shadows cover declaration/builtin/owner resolution and resulting
CALL-versus-ARR_PUSH/type behavior. Full review precedes fresh qualification;
the five identity and twelve mutation/source groups remain unchanged.

My prepared canonical checkpoint adds one exact predicate plus mandatory builtin,
declared and source-owner shadows. Declared push CALL emission queues the exact
resolved declaration index, checks its arity, and retains existing ordered
signature-directed argument lowering. Type inference bypasses only builtin
receiver specialization for that declaration. Extern and indirect classifications
stay unchanged. The source acceptance fixture is byte-for-byte925c. Darwin925c
corrected bootstrap296.851s/setup51.843s passed with stable source/head; identity
and mutation remain unrun there after the shared Stage1 defect was localized.
Only static inspection and whitespace checks apply to this new checkpoint.

## My canonical comparison boundary

I record fixture child `task_1b9beb8884f241d34478c3275178f43b` before correction. Frozen0c54
Linux bootstrap275.445s/setup27.526s pass. The first identity case passes all three
native routes and all three canonical compile/verify/execute routes, then fails
Cseed-versus-selfhost dump equality (Ran1/7.896s). Stage1/Stage2 raw580-byte modules
are already identical (SHA256621095cb31d0646dd575b606780eb72087187687f6721bc80736989822c26a81).
I retain the first assertion failure separately from these successful operations.

My Cseed dump retains declaration-order functions, source/debug tables and exact
ARRAY/FLOAT parameter hints. Ordinary selfhost output uses selected entry-first
functions, remapped CALL/entry/lexical indices, no source/debug table here and
existing unknown VOID parameter hints. I do not normalize away these differences
or claim cross-frontend metadata equivalence. `NANOISA_ONLY.md`63-75 explicitly
excludes seed-versus-first-generation comparison; `test_vm_bytecode_bootstrap.py`
requires raw Stage1/Stage2 bytes with no normalization, while
`test_canonical_nvm_output.py` requires same-producer deterministic repeat bytes.

My corrected fixture requires raw Stage1==Stage2 and each producer's repeated
bytes equal its original, all three canonical verify/execute/shadow results and
explicit declared CALL/no-ARR_PUSH/full result-element evidence. The original
twelve owner-profile methods retain their stricter exact C/selfhost equality.
The complete compiler-bytecode fixedpoint remains a separate hard release gate;
these small programs do not complete it. Production/stages remain frozen0c54;
reviewed fixture-only runs use fresh artifact directories and explicit fixture
hashes without rebuilding unchanged compiler sources.

## My interpreted function-value initializer boundary

I record `task_aabb6d691adff0b1f98a3aae572b16ce` before repair. Corrected7e76
fixture runs against unchanged0c54 producers pass their first two groups and
original lexical native program on both hosts, then the added function-value
program fails my C seed's mandatory main shadow: `Undefined function 'selected'`.
Linux retains Ran3/24.523s; Darwin retains Ran3/45.187s. Remaining identity groups
and all twelve mutation groups are unrun at this pin.

My checker creates local `VAL_VOID` placeholders with definition locations
(typechecker.c5199) and retains function-local rows for native type metadata
(8078). My evaluator's identifier lookup reads the latest row without separating
these facts from runtime values (eval.c4837). The later nested local named
`array_push` therefore masks its declaration during the earlier initializer
`let selected = array_push`. The initializer returns VOID; the alias call then
has no function value to capture at eval.c2941. This is an initializer defect,
not evidence that the alias dispatcher selected the builtin.

I scope the correction to the exact unqualified `array_push` identifier when
`env_get_function` positively resolves its same-module nonextern body. For that
case I inspect matching symbols newest first, skipping only non-global VOID
rows with positive definition locations. My checker assigns those locations;
my evaluator's actual local and parameter rows retain definition location zero
(env.c418), including an actual VOID value. Globals and every actual runtime
value retain precedence. If no such binding remains, my existing function-value
construction uses the resolved declaration. I neither delete checker rows nor
change the shared symbol index or source-visibility lookup, which deliberately
prefers located metadata over runtime rows for native lowering.

My initializer still runs before its new binding is appended. Alias invocation
still snapshots its function name, resolves the existing declaration, and uses
existing parameter/result copies and cleanup. I do not change function-value
storage ownership, generic builtin lookup, reserved names, extern/module
resolution, or canonical indirect-call admission. Builtin-only scopes retain
the old path. Meaningful controls retain the failing program and ordinary
function values, add an actual local/formal override with a distinguishable
result, and retain the unbound builtin and output-preservation groups.

I require review of the complete production/fixture delta before execution.
Only C evaluator production changes are planned; I rebuild affected C providers
in fresh trees and preserve frozen0c54 producers. Reuse of unchanged selfhost
stages requires exact source/binary evidence and an approved gate plan. The
first two passed groups remain attributed to the old pin; affected and unrun
identity groups precede the unchanged twelve mutation groups and then the
entire unchanged owned-record-pattern suite.

My production checkpoint changes only the AST_IDENTIFIER declared-push branch
in eval.c (twenty lines). The original failing function-value source remains
unchanged. A sixth identity method observes ordinary function values, distinct
local/formal overrides and declaration restoration through mandatory shadows.
The low-level evaluator control models a located checker VOID row, then global
VOID/INT and actual local VOID/INT values, verifying that only the checker row
permits declaration fallback. These injected values test lookup boundaries;
they do not claim source admission for mismatched return types. My gate runner
must now require all six identity methods, not the previous five, before the
unchanged twelve mutation methods. The existing evaluator suite retains its
other assertions. No execution has occurred at this checkpoint.

## My first evaluator gate terminal

Fresh9f7 Linux C providers pass52.500s with source/head unchanged. My new
low-level initializer control and six existing file-write/handler controls pass.
The existing `eval_handler_return_expression_order` then refuses initialization
at test_eval.c2694 (make2/3.101s). Its helper suppresses checker diagnostics and
its loop does not report the case index. I retain the entire log and binary;
no source identity/mutation/pattern gate has started. Diagnostic child
`task_6ba90e4583d504291f90cf3f11bd2cc9` first records the case and initialization
phase, without changing sources/assertions or replaying the failed binary.
Any demonstrated correction remains subject to review before qualification.

My fresh parse/typecheck-only diagnostic isolates zero-based body13 and reports
E035: I require an unconditional wildcard in an integer match. No evaluator
execution occurs; the exact source, new binary, compiler log and unchanged
provider inventory are retained at `/tmp/nanolang-eval-order-typecheck-9f7`.
My fixture correction adds `_ => (mark)` after the guarded wildcard. The handler
must still return7 nonlocally before either branch runs; any mark sets trace4
and breaks the unchanged exact result7 assertion. I retain all seventeen bodies
and every assertion, preserve totality, and change no production. The corrected
complete evaluator suite is rebuilt as a new binary against qualified9f7
providers; the original failed binary and log remain untouched.

My first corrected-overlay launch stops before compilation at its clean-tree
assertion: the original failed make target retained untracked `tests/test_eval`.
I preserve this binary in place (SHA256
f2afa578e828691d6563445d27f717285526f2536fed3dd6d7de2eb4d79aa47f) and retain the
pre-build traceback separately. The corrected runner permits only that exact
known path/hash on Linux, records it, and still rejects every other dirty path.
A separate fresh evidence directory avoids overwriting the preparation terminal.
