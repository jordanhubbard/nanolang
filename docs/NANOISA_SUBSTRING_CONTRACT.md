# My checked substring prerequisite

I record this repair under `task_ce840367841a4bdb94ab69fd2446b635`, before
admitting managed substring in LLVM/Wasm. My current shared translators keep
STR_SUBSTR outside their profiles until this VM prerequisite is merged and a
separate matched lowering gate passes.

I preserve byte slicing: start at or beyond the stored length produces an
empty string; otherwise I clip the requested length to the remaining bytes.
I compute remaining bytes by subtraction only after checking start, then
compare the requested length with that remainder. I do not form an unchecked
end position. Embedded NUL bytes retain their stored length and content.

The existing opcode converts integer start/length values to uint32 and treats
other index tags as zero. This bounded repair preserves that policy. Every
popped operand is released, including index values, on success and error. A
failed result allocation reports VM_ERR_MEMORY before publishing a string
value. Ordinary VM error unwinding retains borrowed caller arguments and
cleans its frames; I check recovery on the same VM after allocator failure.

My implementation order is checked clipping, opcode allocation/operand cleanup,
then fresh ordinary controls. I use small valid byte strings, empty/end/clipped
ranges and deterministic allocation failure after the defensive checks exist.
I preserve historical failure artifacts without executing them. These tests
exercise the corrected implementation; I do not reproduce malformed memory
failures or infer an exploit claim from the earlier static review.

Acceptance requires direct heap and real opcode results, input alias/reference
counts, unchanged output on allocation error, zero transient objects after
unwind, same-instance recovery, and native sanitizer checks. This prerequisite
does not close the managed-runtime parent or admit substring in any translator.
