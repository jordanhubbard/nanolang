# My scalar reduce source prerequisite

I begin on main `85fd6a527009c029b44a597cc7f57e6a9b7987d6`, after PR734.
Task `task_d0997d4a11184689ac99a91b430340de` retains the original source
acceptance failures. I do not execute its failed legacy callback source or
artifacts before repair. Global-initializer child74b6 is complete by merged
ancestry; scalar-policy parent5009 and public C-target070db remain open.

## My static boundary

`src_nano/transpiler.nano` already emits `nl_reduce_float` with double
accumulator/element/callback types. Its typed selection near array_fold handles
only that spelling and infers element type only from an identifier, otherwise
falling back to int. Ordinary reduce falls through to the integer helper. This
is a selection/type-observation defect, not permission to reinterpret a callback.
`infer_expr_type` also falls back to int for undeclared builtin calls, so nested
reduce observations do not retain the accumulator type.

`src_nano/typecheck.nano` has a dedicated exact map signature check but no
corresponding reduce check. My current canonical selfhost lowering already
requires `(accumulator, element) -> accumulator`, evaluates source before
initializer and captures length after initializer. I preserve that contract.
The C-seed canonical producer's FUNCREF transport is a separate native admission
boundary; this child does not implement closures or function-reference opcodes.

## My implementation contract for review

1. I recognize builtin reduce only after normal lexical/module binding
   resolution. A user function or local callable with the same spelling retains
   ordinary call handling. I preserve existing array_fold behavior where it is
   already recognized; I do not invent a new language alias silently.
2. For builtin reduce I require exactly three operands, an array element type E,
   a known non-void initial accumulator type A, and an actual function signature
   `fn(A,E)->A`. I check both parameter positions and the exact result, without
   int/float conversion, same-base nominal coercion or unknown-type success.
   I return A to enclosing expressions, including float_to_bits and direct
   returns. Typed empty arrays retain their declared E. Any contextual empty
   literal support must use the exact validated callback E, never default int.
3. Legacy emission selects the existing homogeneous int, float or string helper
   only when its complete runtime ABI matches E and A. I replace identifier-only
   guesses with checked expression/callback information. I do not route bool,
   cross-kind, aggregate or unresolved callbacks through an integer function
   pointer. Such unsupported legacy signatures receive an explicit emission
   refusal before output publication; canonical admission is not broadened.
4. I snapshot source, initializer and callback expressions once in source order,
   then invoke the selected helper. Existing helpers capture array length after
   initializer evaluation and traverse left to right. Empty input returns the
   initializer without invoking the callback. I preserve aliases and ownership;
   I add no array runtime allocation or callback ABI. The legacy emitter already
   permits GNU statement expressions; this is not the separate C99/C11 public
   C-source backend contract.
5. I preserve exact scalar binary64 policy in each callback operation, raw input
   bits and the once-only initializer. I change neither array operators nor
   foreign math. No general function-value, heap, resource or nominal expansion
   is part of this patch.

## My qualification boundary

I commit production for independent review, then freeze source and harness before
fresh bootstrap and ordinary post-repair execution. I create new small fixtures
rather than replay retained failing artifacts. I qualify inline float reduce
bit observation, direct return, typed local, nested expression and helper call;
int/string neighboring helpers; typed empty input; source/initializer/callback
order and counts; length captured after an initializer grows an aliased array;
and operand/input bit preservation with integer observers.

I require wrong arity, non-array input, callback arity, accumulator/element/result
mismatch, bound-name shadowing and unsupported legacy shape controls. Failures
must preserve previous output and the first useful diagnostic. Ordinary shadows
cover every new named Nano function. GCC/Clang generated-code checks and actual
interpreter/C-seed legacy/selfhost legacy/canonical VM/native execution retain
producer identity. Any route lacking the required representation is an explicit
refusal or separately retained incomplete acceptance, not paired success.

I remove PR734's inline-reduce refusal expectation only after corrected source
passes its positive replacement. I retain the C-seed canonical native FUNCREF
refusal and previous-output control until its own reviewed implementation; d099
cannot close as an all-route callback task while that boundary remains. I record
any narrower completed source child separately if needed. Public C-target070db,
full scalar-policy5009, arrays3717 and full reconstruction/release remain open.

## My output refusal prerequisite

Static implementation audit finds no structured error state in the legacy
expression emitter: existing unsupported cases can embed C #error text, and the
driver publishes --target c output directly. For this child I add a narrow
first-error emission string, reset for each public transpilation, set only by
checked reduce failures. The emitter returns no source on failure and exposes
the diagnostic; the driver checks both ordinary and shadow emission before
publication or native compilation. I preserve the first diagnostic and do not
change unrelated emitters to use this state. This is necessary for the approved
unsupported-signature/output-preservation contract, not a general error overhaul.

Source child `task_8618bac3cb6a42448c0066867deaed69` can close after its bounded source gates and canonical
merge; d099 retains the separate canonical native FUNCREF obligation.

Static namespace audit also finds the old private runtime name nl_reduce collides
with an ordinary source function named reduce (legacy user names use nl_). I
rename only the three existing static reduce helpers into the established
nano_rt_ runtime namespace, preserving their signatures/bodies and all callers.
This is necessary to test the approved user-binding distinction without silently
reserving an ordinary source name. I do not rename unrelated map/filter helpers.

Independent review of checkpoint780622 finds global types_equal intentionally
accepts UNKNOWN and enum/int compatibility. I do not reuse that relation for
reduce. Before executing these new controls, I add a scoped recursive known-type
predicate and exact kind plus full normalized identity comparison. Unknown
callback parameters/results and nested unknown components fail; nominal generic
arguments, tuple/function components and array element identity remain exact.
I leave global compatibility unchanged. I qualify these boundaries directly in
checker shadows, including enum/int and distinct generic argument controls.
