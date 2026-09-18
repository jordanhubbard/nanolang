# My expression-arm lexical contract

I retain the exclusive source endpoint of each parsed non-block match arm.
The endpoint is the next token after its complete expression, before consuming
an optional separator. Blocks retain their existing closing-brace endpoint.
My existing checker bounds newly retained payload symbols to that endpoint;
I keep exact declaration and nominal metadata available for emission.

An arm binding shadows an outer name only within its arm. A later expression
or sibling arm resolves its own binding or the outer declaration. Nested
matches retain their inner bounds. I do not alter global lookup, resource
moves, generic substitution or accepted pattern forms.

I test ordinary same-line/multiline and nested expressions, sibling bindings,
outer restoration and refused escaped bindings. C-native and NanoVirt paths
must agree with the native selfhost stages. This parser prerequisite does not
claim selfhost NanoISA expression-match admission, which follows separately.

MAC: `task_2faf762553284557ad77a9f7f541801b`.

## My native emission companion

My native emitter registers outer locals while retaining checked arm metadata.
Before emitting a named payload arm, I re-establish its exact checked symbol
using the match source location, source file and declared union.variant name.
I copy annotations through the existing environment ownership API and preserve
its lexical endpoint and declared resource classification. I do not copy
transient checker move state or change lookup rules. Guards and arm expressions
then see the same nominal binding as checking. Wildcard/integer/or patterns
without a payload do not create one.

MAC: `task_109daee0b08147bc89d813211034d89f`.
