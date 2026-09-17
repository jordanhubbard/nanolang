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
