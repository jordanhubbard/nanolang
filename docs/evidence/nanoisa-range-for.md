# My NanoISA range and array loops

I lower two-bound `range(start, end)` iteration and supported array/list
iteration through my existing bound-Parser emitter. I evaluate the iterable
once, retain its length, and give each nested loop its own counter and exit.
A `continue` increments only its owning `for` counter. Loop variables stop
binding after the loop; their allocated local slots remain valid bytecode.
I preserve the documented iterator-only range contract, including empty and
negative-bound ranges. I do not turn `range` into a general array expression.

My C-seed bytecode reference previously retained a loop variable's binding
past the loop. I restore its binding count without reusing physical slots.
My C native reference also repeated the end-bound expression in its loop
condition. I now snapshot both bounds in source order before binding the
loop variable, using my existing collision-checked argument temporary generator.
My fixture also binds names resembling compiler temporaries. Numeric array loops with exits no longer receive an
inapplicable explicit vectorization hint; loops without exits retain it.

My fixture compares all nine functions, including `__init__`, against
C-seed bytecode. Both versions verify and execute in NanoVM and native AOT;
C-seed native compilation and execution agree. The fixture covers side
effects in both bounds, descending/equal/negative empty ranges, nested
same-name variables, nested `while` and `for`, break, continue, return,
unreachable statements, and string/bool array iteration. Seven invalid
operand/arity contexts fail while preserving prior assembly output.

The existing 86-comparison gate and all 72 Python methods pass in 90.888 s
(`/tmp/nanolang-range-for-gate5.log`). This is bounded loop evidence, not a
claim that the complete compiler-shadow closure is supported.

I retain two separate continuations: the C checker currently derives loop
element metadata only from named array receivers (`task_911317d4234c46049c3dea1a2d0a153d`),
and my self-hosted native C transpiler still repeats the range end expression
(`task_8e80a672c285487dbbad07dd1dbfc9b9`). The accepted fixture uses explicitly
typed array locals. Neither continuation is hidden by a skipped assertion.

A fresh three-stage native bootstrap passes with the explicit per-command
`NANO_SHADOW_TIMEOUT_SECONDS=60` budget
(`/tmp/nanolang-range-for-bootstrap60.log`). The first default-budget attempt
stopped at ten seconds without an assertion diagnostic. I retain that
measurement separately as `task_628759a2daf743b9bf13c9a7fea2ced0`; I have not
changed the default budget, retried away the observation, or inferred a
compiler correctness failure from it. Native bootstrap binary equality is
not claimed.

The complete bound-Parser compiler-shadow probe advances past the original
`tokenize_string` range refusal. It now refuses
`__nano_module___16_substitute_union_field_type` at merged line 22679 because
nested appends to an empty array lose their string element context. I retain
the exact failure in `/tmp/nanolang-range-shadow-probe-run.log` and track it
as `task_d5ed194093434b5cbfc2e3ec6bc2d37a`. No shadow is excluded.
