# My explicit generic union constructors

MAC `task_952ac994c2684adc8944cb2373670456`.

I parse `Box<int>.Some { value: 7 }` and `Box<string>.None {}` through
my existing recursive type parser and union field parser. As in my C seed,
only an uppercase simple identifier followed by `<` begins this expression
form. Lowercase comparisons keep their existing expression grammar. I require
the closing type arguments, dot, variant identifier and brace; generic type
names alone are not values. I retain the complete concrete spelling in
`ASTUnionConstruct.union_name` and the type name's original line and column.
Nested type arguments and multiple arguments retain their existing annotation
grammar. I do not add qualified-constructor or generic-function syntax.

My existing source checker and exact-instance lowering remain authoritative.
An explicit instance cannot be replaced by an expected local, argument or
return type, including a variant with no payload. This parser change does not
expand accepted payload shapes, wire identity or ownership authority.

I require parser shadows for nested and multiple arguments, locations,
lowercase comparisons and incomplete syntax. Paired C/Stage1/Stage2 acceptance
covers constructors in locals, calls, returns and selected shadows. Wrong
zero-payload instances must fail with a type/lowering diagnostic and preserve
old output; a parse failure is not that evidence. I run fresh bootstrap before
the paired acceptance and retain unsupported shape boundaries.
