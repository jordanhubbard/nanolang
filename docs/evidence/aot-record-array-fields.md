# Record-array fields

I store a record-array field as an `nrarr_t` reference. A forward declaration
breaks the C type dependency between records and their arrays without embedding
an infinitely recursive value layout. Packing and extraction preserve shared
array identity, including when the containing record passes through a function.

My classifier no longer mistakes a parent field's flat representation for the
representations inside its array elements. It leaves those nested field facts
unknown. The emitter reads resolved element facts from the constraint graph
when extracting the array; subsequent field reads also use resolved graph facts.
Conflicting element shapes inserted through an extracted field are rejected.

## Verification

My tests cover empty and populated record-array fields, wrapper and relay calls,
both wrapper branch paths, mixed string/integer record elements, extraction,
mutation visible through the original array alias, and an incompatible insertion.
The old rejection test for an empty record-array field is replaced by positive
coverage; nested record values remain a rejection case.

`make -j1 test-nvm2c-sanitizers` passes 1,055 AOT checks and 965 graph checks,
with fresh translator instrumentation verified. `git diff --check` passes.
A subsequent ordinary `make -j1 test-nvm2c` passes the same check counts.

`make -j1 test-one-ir-compiler` advances from function 20 to function 100,
where `AGG_PACK` still requires unsupported nested aggregate shape facts.
I have not completed compiler acceptance. Nested record storage, general
recursive inference, array capacity limits and broader ownership remain work.

MAC `task_9c850e94e5a74b6f8941622e2872af23` remains open. Claiming still fails
with `agent_status_unavailable`; repository evidence records this partial step.
