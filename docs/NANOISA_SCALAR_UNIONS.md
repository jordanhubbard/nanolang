# My scalar union source contract

I lower uniquely resolved, nongeneric union declarations whose variant fields
are `int`, `bool`, `float`, or `string`. I retain the parser's union and variant
indices; I do not infer nominal identity from an ordinal or payload shape.
I reject ambiguous names, generic declarations, resource-bearing or nested
payloads, and unknown or incomplete field metadata.

I validate every constructor field exactly once, including missing, duplicate,
and unknown names. I evaluate fields once in source order, then pack them in
declaration order. Union locals, direct calls and returns retain the exact
union name; a different union with the same fields is not interchangeable.

My first match path is an exhaustive statement match with one named arm per
variant. Each arm binds that exact variant's scalar fields in its own lexical
scope. I evaluate the scrutinee once. Returning arms leave their enclosing
function. I retain normal fallthrough in other arms and refuse unsupported
pattern or expression forms explicitly. I do not add resource ownership rules,
implicit disposal, wildcard/or-patterns, generic substitution or nested unions.

I execute selected shadows through the same lowering. Acceptance includes the
unchanged `tests/nl_control_flow.nano` and `tests/nl_control_match.nano`, both
canonical compiler stages, VM and native execution, constructor refusal and
nominal mismatch controls, lexical payload bindings and matched return paths.
I retain the broader product acceptance and unsupported union forms separately.

MAC: `task_9a12d05fafaf4f92bea5919683f9ba32`.
