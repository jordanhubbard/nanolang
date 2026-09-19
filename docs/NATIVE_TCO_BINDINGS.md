# My remaining native tail-call contract

I already lower checked scalar self-tail returns into simultaneous argument
updates and one native loop. That evidence does not cover aggregate parameter
metadata, indirect callable expressions, nested loops or lexical bindings that
reuse a parameter name. I keep `task_ab36fda5e6846c45beaf42b3ba819c13`
open until these cases execute with the same results as unoptimized native code.

## Binding identity

I rename a parameter reference to its hidden TCO state only while that source
reference still resolves to the parameter. A `let`, loop variable, match
binding or nested function parameter with the same spelling starts a nearer
binding for its lexical scope. Its initializer and expressions before the
declaration continue to see the outer parameter. Leaving the scope restores the
outer binding. I do not infer identity from spelling alone.

Named direct self calls remain the only calls converted into loop iterations.
I traverse indirect callable expressions and their arguments so captured or
function-valued parameter references retain their checked binding, but I do not
mistake an indirect call or a nested function return for a self-tail call.

## Values and evaluation

Each hidden parameter state and each argument temporary retains the complete
checked annotation from the original parameter: nominal name, array element
type, function signature and recursive `TypeInfo`. I evaluate every next-call
argument exactly once in source order into distinct temporaries, then update all
state slots. This preserves swaps and dependencies between arguments for scalar
and aggregate values.

The ordinary native representation remains authoritative. I do not introduce a
new aggregate ABI or reinterpret ownership. If a checked construct cannot be
lowered with its existing representation, preflight must leave the whole
function unchanged before mutating any node.

## Nested control flow

The generated outer loop is the only target of a self-tail iteration. A tail
return inside an original `while` or `for` first publishes its argument state,
sets a collision-free pending-iteration flag and breaks the nearest original
loop. Each enclosing original loop propagates that flag outward. Only after the
outermost original loop has exited does the generated code continue the TCO
loop. Ordinary source `break` and `continue` retain their original nearest-loop
meaning. A path that falls through the original function still leaves the
generated loop once.

## Acceptance

I require optimized and unoptimized native output parity for:

- array, record, tuple and function-valued parameters;
- left-to-right aggregate argument updates and returned typed values;
- indirect callable expressions that read a parameter;
- parameter-shadowing `let` initializers and nested blocks;
- tail returns in `while`, `for` and two nested loops, alongside ordinary
  source `break` and `continue` controls;
- generated-name collisions and the existing million-step scalar fixture.

The C pass unit gate must prove unsupported constructs remain byte-for-byte
unmodified when preflight refuses them. I retain the existing scalar gate in
`test-opt-passes`; I do not use its success as aggregate or loop evidence.
