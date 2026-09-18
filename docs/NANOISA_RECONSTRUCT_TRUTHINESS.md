# My exact integer and boolean reconstruction

I record task `task_6426a39bec8f48db8dd2649fd979d687` under reconstruction
parent `task_4bd034f6029b7458201db74e2c3aeb32` before implementation.

My portable ISA names casts and generic boolean operations independently of
source syntax. I follow the existing `src/nanovm/value.c::val_truthy` and
`src/nanovm/vm.c` dispatch contracts for this bounded domain:

| Instruction | Admitted operands | Result |
| --- | --- | --- |
| CAST_INT | int or bool | unchanged int, or integer 0/1 |
| CAST_BOOL | int or bool | integer nonzero, or unchanged bool |
| AND / OR | two int/bool operands | bool conjunction/disjunction of truth values |
| NOT | int or bool | bool negation of truth value |

I retain exact typed BOOL operation requirements. I do not extend entry or
function signatures beyond my existing int/bool profile. Float, void, byte,
enum, string and heap reconstruction remain refused; their VM behavior is
not redefined by this profile.

Both binary operands have already been evaluated on the ISA stack. My
ordinary region analysis emits immutable temporaries for calls, loads and
other evaluated instructions before combining their truth values. A target
language's boolean short circuit therefore cannot skip an operand call or
change an earlier local snapshot. Pure loop conditions continue to refuse
calls and stores; their admitted expressions are total and have no effects.

I require independent small truth tables, zero/negative/signed-endpoint
casts, calls, local mutation snapshots, and structured loops. I compare the
same module in VM, reconstructed sanitizer C, and NanoLang compiled by
three explicitly pinned producers. I retain byte roundtrip and previous
output on refusal. I never replay or minimize the preserved carry679
compiler failure; that task and full reconstruction remain open.
