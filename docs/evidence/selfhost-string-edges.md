# My string prefix and suffix lowering

I lower `str_starts_with(string,string) -> bool` and
`str_ends_with(string,string) -> bool` to their existing NanoISA instructions.
They share a bounded two-string operation selector with `str_concat`.
I preserve explicit raw-parser declarations and reject malformed call shapes.
Direct returns emit the operation and `RET`.

My emitter gate passes 86 comparison checks and all 59 regression methods
(`/tmp/nanolang-string-edges-gate.log`). The new fixture contributes 16 exact
C-seed checks across seven function bodies, including initialization. Both
emitted modules execute in my VM and generated native code. Operand traces
`12` and `13` establish left-to-right, once-only evaluation for prefix and
suffix calls. Empty strings, empty patterns, equal strings, longer patterns
and case-sensitive mismatches are checked. Ten malformed argument shapes
refuse output.

A fresh canonical driver builds. Actual whole-compiler emission advances to
`unsupported local type HashMap<string,string>`; I retain
`/tmp/nanolang-edges-fullcompiler-probe.log` and track that next boundary as
`task_d32f896911e1447da9c6b69059f6d6ea`. I have not produced a complete compiler module or a bytecode fixed
point. Restacking onto `ab155c41` leaves the tested source tree unchanged.
