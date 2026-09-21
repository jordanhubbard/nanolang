# My self-hosted record field destinations

I track this prerequisite under task_90077955d5c64a42bf91358769802aba.
My unchanged 6cf first source method stops on both hosts: C seed rejects a local
`Holder.values: array<Item>` initialized from another module's `array<Item>`,
but Stage1 returns zero and publishes native output. I do not execute that
invalid output. The original reports stay under
`/tmp/nanolang-record-lists-6cf-linux-matrix` and
`/private/tmp/nanolang-record-lists-6cf-puck-matrix`. I preserve their source,
provider, tool and product maps; bootstrap is attributed to a448.

## Exact declaration and field checking

I replace the first unchecked STRUCT_LITERAL arm with one checker and remove
the later unreachable duplicate. I find exactly one ASTStruct whose canonical
name equals the literal's name. I do not select a globally unique raw spelling.
The existing `nb_register`/`nb_import`/`nb_rewrite` pipeline rewrites declarations,
field annotations and literals before checking; its canonical names preserve
owner collisions. An absent or ambiguous declaration is a diagnostic.

I visit source field values in source order in the ordinary record helper.
Existing constructor/context helpers can revisit nodes for static checking; this
is not runtime evaluation.
I require the exact declared field count, reject unknown and repeated field
names, and check every present value even when another field is invalid. I use
the declared annotation through `type_from_string_with_parser`, recursively
check the value, then call `check_constructor_argument` for its checked union
or shared scalar/byte-array literal context. I do not relabel an existing array.
I diagnose unknown value facts and mismatched types; the AST keeps its original
record identity so later diagnostic traversal does not invent another type.
I preserve the established scalar numeric destination policy where the existing
checker defines it, rather than widening nominal or array equality.

I also use the same contextual helper for existing record field assignments.
I do not alter global `types_equal`, module visibility, resource authority,
record layouts, mutation order or runtime ownership. Contextual byte-array
literal checking is root's a59e2b1c3 shared helper plus cfa24ac13 scalar INT-to-byte guard; I import that exact reviewed
delta rather than implementing another byte policy.

## Scope and required checks

My current self-hosted ASTStruct has named concrete fields and no generic record
parameters; ASTStructLiteral has named values and no anonymous/spread source.
The C parser separately supports anonymous/spread forms. This checkpoint does
not claim those absent self-hosted schema forms work, nor remove their full
5.1 parity requirement. Concrete generated record definitions use this same
field checker. Generic union field context uses the existing union helper.

I add meaningful shadows for declaration/field completeness, wrong scalar and
array identity, nested records and accepted empty contextual arrays; paired
C-seed/Stage1/Stage2/evaluator/VM source controls retain the original nominal
array refusal. A negative compiler exit counts only with the expected checker
diagnostic and unchanged output sentinel. Fresh full bootstrap keeps the
original ten-second shadow deadline. Full fourteen methods, selected ownership
sanitizers, neighbors and unchanged Make remain required before closure.

I distinguish a real empty array literal from UNKNOWN element facts: I recursively
check every nonempty literal member against its actual destination, and only
annotate a literal after all members pass. Existing array values must retain
known element facts through nested array levels. I do not let the global empty
array compatibility rule accept a heterogeneous nonempty field.

My static C-seed review additionally finds missing duplicate-name checks in
ordinary record literals (the union loop already has them). A count match alone
does not prove completeness. I retain this as a separate required parity repair
before claiming paired equal-count duplicate refusal. No invalid program was run.

## Source checkpoint

I implement the ordinary C-seed duplicate-name check in the existing field loop,
leaving value checking active so diagnostics still cover supplied expressions.
The self-hosted helper checks all literal array members recursively and rejects
unknown existing-array facts; it does not call the permissive return hint. I
reuse the exact a59/cfa shared contextual helper and its shadows; I do not copy
the separate U8 emitter changes or claim their qualification here.

My added source method keeps all fourteen original methods and adds positive
reordered/nested/empty record fields plus seven checker-refusal vectors. Each
new negative compiler case requires an actual checker diagnostic and untouched
output sentinel. My same-count duplicate shadow supplies two `left` fields for
`Pair { left, right }`, so field count cannot satisfy that assertion by itself.
I have not built or executed this source checkpoint.

I preserve 66 first-failure/continuation report files in
`docs/evidence/record-lists-6cf-first-field/`, with hashes and endpoint equality
summaries. The full maps and CAS remain in the original external report roots.
These small committed reports do not replace the later complete qualification
seal. Published invalid native outputs are hashed and remain unexecuted.
