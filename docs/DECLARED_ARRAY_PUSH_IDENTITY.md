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
