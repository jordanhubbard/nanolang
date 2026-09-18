# My bounded native scalar ROT3 contract

I record `task_18ca25c8fc294c81b1a662fd22fc411b` before native edits.
A fresh ordinary reconstruction fixture was refused at opcode ROT3,
function0 offset50; `/tmp/nanolang-reconstruct-rot3-gcc.log` retains that
translation result. No native binary was produced or executed.

I admit only exact INT/BOOL operands in this child. Bottom-to-top `a b c`
becomes `c a b`. Classification permutes complete simulation slots, including
origin, exact shape, record metadata and scalar provenance. Emission permutes
existing immutable slot IDs and their kinds; other metadata remains indexed
by those IDs. I allocate no value storage and change no ownership/root lifetime.
I preserve underflow guards and refuse other kinds before publication.

I require small ordinary VM/native GCC/Clang sanitizer parity for distinct
values, mixed tags, calls/locals and loops; underflow and excluded-kind
translation controls preserve previous outputs without executing refusals.
Full native ROT3 for other kinds and reconstruction parent4bd remain separate.
