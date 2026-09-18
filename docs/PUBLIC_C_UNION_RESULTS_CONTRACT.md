# I preserve exact declared union results in public C

I execute existing `task_81069872995f49a4a8d8a38a386b2c27` under required public C parent6ade from canonical `5e18a453`. This contract precedes production. My existing function definitions and forward declarations special-case STRUCT results but pass UNION results to `c_type`, whose fallback is `int64_t`. My constructors already produce `NanoUnion_Name`. I repair this representation boundary before statement-aware expression lifting; I do not replace expression blocks with comma expressions or claim full C99 coverage.

## My exact result identity

I resolve a union result only to one exact local, nongeneric, nonextern union declaration. I use the same resolver and emitted `NanoUnion_Name` spelling for prototypes, definitions and direct-call result metadata. A missing name, unknown owner, duplicate declaration, generic/unresolved instance or unsupported foreign result produces a checked diagnostic before output publication; I never substitute an integer, first union or similarly spelled declaration. Existing struct/scalar results and the private language-main/hosted-main distinction retain their behavior.

My bounded payload representation is the already qualified scalar INT/BOOL/FLOAT/STRING union layout, including empty variants. I preserve declared variant order, field types, string borrowing/stable snapshot ownership and C aggregate-by-value transport. I add no tagged heap representation, destructor, function-value calling convention or external aggregate ABI.

I retain exact nominal identity through a constructor of the declared owner, an in-scope union parameter/local and an unbound direct declared function call. Direct-call spelling is usable only when no lexical binding and no `func_expr` supersedes it; checked metadata cannot authorize a different callee by spelling. I check return values and new union local initializers against their expected owner. An actual constructor must name a real variant of that owner and retain checked field identity/type constraints; I do not coerce a structurally similar union. Conditional/block/match value expressions, bound callbacks and other unresolved result forms remain explicit required follow-ups, rather than guessed nominal values. Existing unsupported-route behavior is not new admission.

## My evaluation and publication boundaries

I evaluate a returned/call-produced union once. A relay, local binding or match scrutinee transports the complete value without reconstructing fields or duplicating a call. Existing ordered scalar helpers and match scrutinee snapshots remain intact. This result repair does not establish previously unqualified C argument ordering across multiple effectful call arguments. Function bodies retain enclosing return/control semantics and ordinary recursion; I introduce no wrapper loop or extra startup invocation.

I use the established staged FILE and sibling-file publication paths. Exact API refusals retain prior output, first diagnostics and successful later compilation in the same process. Namespace selection and dry emission must reset any new result context without retaining pointers beyond the existing AST lifetime.

## My fresh qualification

I freeze production and harness before each gate. I require strict GCC/Clang C99/C11 O0/O2 ASan/UBSan controls for scalar-payload and empty union returns, forward declarations, direct relay calls, parameter/local returns, local storage and guarded/unguarded matching. Integer bit observers check FLOAT payload transport; string content checks do not use pointer equality. A side-effect counter proves constructor/call/match evaluation occurs once. Distinct same-layout declarations and private-name collisions retain identity and hygiene.

I pair ordinary supported source fixtures with interpreter and newly built NanoVirt/verified VM observations. Public API controls explicitly cover wrong nominal return/local owner, missing/unknown/generic/foreign result identity and bound-callee precedence without executing refused output. I preserve output on refusal and verify later valid same-process recovery. I run the existing public C programs and adjacent scalar/string/union/guarded tests, retaining every first failure and exact source/tool pin. No historical failed artifact is replayed.

A bounded merged result closes only task810698's exact declared union-result contract. Expression-form blocks/matches, wider generic/foreign/function-value results, options/captures, shared wildcard/no-success policies and full public C release remain required and open. If source metadata exposes an additional prerequisite, I record it before expanding production.
