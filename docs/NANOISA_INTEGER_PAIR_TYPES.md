# My integer-pair verifier rules

MAC `task_6729384a65fa4c8688db86b6bfc91442`.

I give `I64_ADD_CARRY` and `I64_SUB_BORROW` the VM's exact three-INT
operand rule and two-INT result rule. I give `I64_MUL_WIDE_S` and
`I64_MUL_WIDE_U` the corresponding two-INT operand and two-INT result rules.
The VM pushes the low result first and the high/carry/borrow result last;
both have the integer tag. Carry input values retain the VM's low-bit
semantics; I add no boolean-value restriction.

My type pass already refuses a known tag contradiction and preserves unknown
values. I retain that lattice, its bounded analysis, allocation behavior,
control-flow joins and the separate height verifier. This change supplies
missing instruction rules and retains both homogeneous result tags; it does
not turn unknown values into a static type guarantee or change execution.

I inspect rule shapes directly and exercise ordinary valid integer programs
through assembly, verification and VM execution. I include both
result positions, composition and loops. I do not execute malformed modules,
replay historical failures or construct crash inputs. Existing verifier and
ordinary arithmetic suites remain adjacent acceptance.

My first gate exposed a separate native coverage boundary: `nvm2c` has no
classifier for these four opcodes. I retained the positive programs and
recorded `task_ebf9bb417d9e4d6aa3b007c3fe868c92`; native execution is not a
completed gate. This verifier slice requires explicit unsupported-opcode
refusal with prior native output preserved. The separate native companion
must later execute these same ordinary arithmetic cases.
