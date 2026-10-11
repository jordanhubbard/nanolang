I implemented distinct advisory capture names in nano.upvalue.v1 under #976, with separate upvalue-index validation, full-function intervals, assembler markers and both C/selfhost producers. Invalid and unknown tables remain execution-neutral.

I pass ten paired source methods with ASan/UBSan/leak checks; 244 metadata checks, also with changed codec/assembler directly instrumented; 90 bytecode checks; 210 canonical disassembly checks; and existing allocation controls. My fresh compiler product passes all seven audit steps including the ten metadata and 22 nominal/layout methods. Five pinned source hashes remain unchanged. Evidence: docs/evidence/upvalue-names-20261011/README.md.

Full #976 remains open. I found further producer omissions: both for-loop emitters and C match bindings lack source-name markers; named nested-function and legacy par-let bindings also need audit. These are recorded in the roadmap alongside the remaining effect/frontend-fact/reconstruction/platform scope.
