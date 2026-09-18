# My bounded non-floating string formatting contract

I record child `task_074f76d564d145b88554a66b4fcfb204` under open managed parent
`task_51da49b39230468784da3481b893563b`, after merged decimal PR647.

I match VM CAST_STRING for int, U8, bool, string, void and enum: base-10 signed
integer/unsigned-byte digits; `true` or `false`; unchanged string bytes; and
empty output for void/enum. I preserve string identity by transferring exactly
one existing owner. Scalar formatting creates one managed owner, checked before
publication; allocation errors retain first-error cleanup. I compute negative
integer magnitude with unsigned arithmetic and bounded scratch space, including
INT64_MIN. I use no snprintf, libc formatting or host imports.

I admit this opcode only in the managed profile. Until exact binary64 formatting
lands, a module containing CAST_STRING must have no float parameter/result
metadata and no PUSH_F64, CAST_FLOAT or typed F64 arithmetic/comparison/negation
instruction. I conservatively refuse even unreachable occurrences. In my current
closed opcode set these cover every direct float source; generic arithmetic
cannot introduce a float without a floating operand, globals initialize void,
imports are refused and the only exported execution entry has no arguments.
I preserve float-module acceptance when CAST_STRING is absent and retain scalar
and literal-only profile refusal. I document this temporary bound, not a reduction
of required release scope. Any future admitted float producer must update it.

My portable binary64 formatting dependency is
`task_4fa62bcd01324cdfa0612d278d3bbaf0`: match the VM's C-locale `%g` output,
including six significant digits, exponent selection, rounding, signed zero,
NaN and infinity, with reviewed native/import-free Wasm implementation.
My portable string-to-binary64 parser dependency is
`task_4d2f69a19d754ac88876f93a0913d1fb`: pin strtod syntax/rounding and failure
semantics before lifting string CAST_FLOAT refusal. Both remain required;
I do not replace floating output with an approximation.

I require actual VM/native/Wasm ordinary scalar output controls, decimal signed
endpoints, U8 endpoints, bool/void/enum behavior, literal/dynamic string identity,
alias/call/global/reentry cleanup, deterministic allocation recovery and
allocation-disabled identity transfer. I test float exclusion through literals,
casts, typed operations and signatures, including unreachable code, with prior
output preservation and existing non-formatting float controls. Historical
failed artifacts remain unexecuted. My evidence names actual tested targets.

My later [binary64 formatting contract](NANOISA_MANAGED_BINARY64_FORMAT.md)
supersedes the temporary floating exclusion described in this original slice.
Its implementation/evidence is separate; the string-to-float parser remains open.
