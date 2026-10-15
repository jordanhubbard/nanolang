# Effect-handler parameter names

I retain each C-produced handler parameter at its arm entry, after the runtime
has installed the performed arguments. I close its interval after that arm's
resume instruction. I keep nested and sibling parameters distinct from each
other and from a same-spelling outer local. These are lexical intervals, not
claims about executed instructions or live values.

My baseline source executes with result 7 in NanoVM but omits both handler
parameters. All four corrected regression fixtures first fail on missing names
and then pass after the producer change. They cover sibling/nested arms,
int/string/bool arguments, a zero-argument operation and lexical return. The
existing codec probe verifies boundaries, exact original-wire roundtrip and
canonical stability. Each fixture also executes after stripping all local-name
metadata and verifies the stripped module.

`make test-effect-binding-metadata test-nanovirt` passes four source methods,
244 local-name checks, five local-name allocation boundaries, twenty assembler
marker allocation boundaries and all ninety codegen tests. I add the new target
to `test-units`. My first harness incorrectly used a names-required codec probe
on stripped metadata; I replace that unsupported invocation with ordinary module
verification, retaining the stripped execution. My first gate command builds
the compiler successfully but names a nonexistent `test-bytecode` target; the
corrected Make command above passes. I retain both first terminals.

The same baseline exposes two separate required product gaps: the current
self-hosted compiler rejects `effect` syntax, and `nvm2c` refuses HANDLER_PUSH.
I record both on the roadmap under #976. This C producer correction does not
close self-hosted effects, native effects, full producer-name coverage or 5.1.
