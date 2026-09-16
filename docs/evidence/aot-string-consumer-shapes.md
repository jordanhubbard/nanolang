# My string consumers and projected storage

In `generate_enum_definitions_from_tokens` (394), `STORE_LOCAL 16` at offset
496 expected a string but received an emitted integer read. The preceding
record-field projection was unresolved. A later string operation marked the
local's flat representation as string without supplying the corresponding
shape constraint, leaving the producer and destination inconsistent.

My shared string-operand helper now propagates an inferred string storage flow
to an unresolved operand's shape while retaining the existing local marking.
It covers length, both concatenation operands, substring, prefix/suffix/contains
predicates, character access, map keys and string returns. Observed tagged
values retain their runtime unboxing. I use the existing directed field-flow
mechanism, not a new exact string constraint on every operand.

Sixteen regression variants failed translation before this change. They now
execute the string operations in both function orders through a projected local.
Sixteen companion runs trap on a non-string field tag before the consumer can
use the string pointer. The native harness provides no caller-derived field
facts to the classifier.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; I do not infer leak freedom.

Fresh compiler acceptance passes twelve focused test methods. Full compiler
emission passes the enum-generation store, then fails in `genenv_get_mut`
(410): `ARR_GET` expects an array. I track that as MAC
`task_3b10cd3d807f44d8bf04a5128b697932`. Full compiler and release acceptance
remain incomplete.

MAC refuses my claim for `task_34d2d9345d584ff78fab59261eb112d6` with
`agent_status_unavailable`. I retain evidence without forcing closure.
