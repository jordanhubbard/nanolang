# My checked array arithmetic result views

I retain the first full evaluator terminal at source
`6108067d4161746bc9b20508b9f2fce8a82e433e`. Linux and Darwin fresh
stage1 builds pass; both original evaluator methods stop at
`eval_unary_minus_int_array`, `tests/test_eval.c:2421`, when `run_ctx_init`
returns false. I have not executed either sanitizer lane. The source audit
identifies a missing complete result view; the original fixture suppresses
checker diagnostics, so its assertion alone is not a diagnostic transcript.

Task: `task_398942bdc317a4a778f70f1245ddc047`, under
`task_992713bde1494772b0cb0b58bc9ee3c3`.

I derive result facts from complete operand facts after the existing arithmetic
checker accepts the expression as ARRAY. I do not infer them from a destination.
I preserve arity, operator, declaration-owner and recursion-depth checks.

| Operand family | Result contract |
| --- | --- |
| Unary int or enum array | array<int>; arithmetic removes enum identity |
| Unary float array | array<float> |
| Two int/enum arrays, or int/enum array with int/enum/u8 scalar | array<int>; + - * / % |
| Two float arrays or float array with float scalar | array<float>; + - * / |
| String arrays and string scalar/array | array<string>; + only |
| Mixed int/float, bool, records, callables, unknown leaves | No new result admission |
| Actual array<u8> | No new result admission: dynamic evaluator has no ELEM_U8 arithmetic dispatch |
| Nested arrays | No new result admission: known nested operands are refused by the coarse checker; unary evaluator has no nested dispatch |

The scalar u8 case uses the existing VAL_INT carrier. I do not relabel an
ELEM_U8 buffer as int. Nested dynamic binary helpers recurse at runtime, but
that implementation does not establish a source checking contract. These
byte/nested boundaries remain explicit follow-up work, without affected
execution or a runtime-support claim.

My adapter owns temporary operand/child views, constructs a complete primitive
result annotation and wraps it in one array layer. Enum input identities must
already be validated by the operand view. I discard every temporary on every
return. I retain the existing complete STRING addition behavior.

I preserve every original evaluator assertion. Additive checker controls cover
all admitted operators and broadcast directions, enum-result identity removal,
byte scalar promotion, nested arithmetic expressions and refusal boundaries.
Source review precedes focused checks, then the whole original C checker and
whole original evaluator on both hosts with ordinary and ASan/UBSan/LSan
providers. Full Nano compiler deadlines and release acceptance remain separate.
