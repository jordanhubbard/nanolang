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
