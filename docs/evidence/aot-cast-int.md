# Integer casts and inference diagnostics

My classifier previously skipped `CAST_INT`, leaving its string operand on the
simulated stack. This caused a false call-site conflict: function 150 at offset
479 passed an integer tuple index to parameter 3 of function 167, but a previous
caller's cast had incorrectly established that parameter as a string.

I now pop the cast operand and push an integer fact. Emission preserves integer
values, converts strings using the same decimal `strtoll` behavior as NanoVM,
and returns zero for supported aggregate representations, also matching NanoVM.
This is not strict numeric parsing: whitespace, a leading sign and a numeric
prefix are accepted, invalid/empty text yields zero, and host `strtoll` handles
overflow. Float representation support remains outside this AOT subset.

Conflict diagnostics now report the caller's function and instruction offset,
the conflicting C representations, and the destination parameter or field.
Tests check that caller and destination information is present.

## Verification

`make -j1 test-nvm2c-sanitizers` passes 1,060 AOT checks and 965 graph checks,
with fresh ASan/UBSan instrumentation verified. Conversion tests cover signed
decimal prefixes, invalid/empty text, both int64 bounds, over/underflow, integer
and boolean identity, aggregate-to-zero behavior, and a shared callee receiving
both converted text and ordinary integer values. `git diff --check` passes.
A subsequent ordinary `make -j1 test-nvm2c` passes the same check counts.

Compiler acceptance clears the tuple-index conflict. It now fails at function
170 offset 925: parameter 1 of function 141 has conflicting `nrarr_t`/`narr_t`
facts. The retained compiler disassembly shows `parse_function_definition`
constructing a record array with `ARR_NEW 8`, copying elements from a nested
array field, then calling `parser_store_block`. I must preserve known array
representations while nested element facts remain unknown.

I filed a broader classifier opcode-coverage audit as
`task_f90db79b0f464637a18486c44262c4d3`. Missing transfer functions must not
silently act as no-ops. The parent shape task
`task_9c850e94e5a74b6f8941622e2872af23` and compiler acceptance remain open.
