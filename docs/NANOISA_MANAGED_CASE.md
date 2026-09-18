# My managed ASCII case-conversion contract

I implement `task_31acd26441144d52a82a533929669763` after checked VM result
allocation543/PR674. I admit STR_TO_LOWER/UPPER to the managed LLVM/Wasm
profile only; I retain existing scalar/literal-only decisions.

I map ASCII A..Z to a..z for lower and a..z to A..Z for upper across the
complete stored byte length. NUL, high bytes and every other byte remain
unchanged. I do not decode Unicode, consult a locale or mutate locale state.

My consuming helper first validates the source view, output pointer and mode,
then uses my existing checked allocator to copy into a fresh private dynamic
slot. Only after allocation succeeds do I access the current slot table and
transform that new slot's bytes. The new handle has one unshared owner and is
not published until conversion and source release succeed. Empty or unchanged
managed text still has a fresh handle. My VM can intern equal strings; matched
byte/ownership semantics do not imply matching allocation events or physical
identity. I preserve that distinction in tests and evidence.

Every helper path releases exactly the transferred source owner. Allocation
failure leaves the output untouched and unrelated aliases valid. No view or
slot-table pointer is cached across allocator growth. LLVM marks the source
operand transferred to its adapter, which releases it on source-type refusal;
other frame roots keep ordinary cleanup responsibilities. Success publishes
one new string owner, errors follow the existing first-error/status protocol.

I require all256-byte mapping, empty/unchanged and long ordinary inputs on
actual VM/native LLVM/import-free Node/Wasmtime. Core native/Wasm checks cover
fresh handles, retained aliases, invalid mode/null output and deterministic
allocation failure, including descriptor-table growth. Calls/globals/reentry,
emitted failure recovery, source-type refusal and complete disposal must pass.
Shared profile and remaining unsupported-operation output preservation remain
required. This does not complete runtime51da, Darwin7ba, evaluator791a,
aggregate/cycle488 or host-linkage2d2 acceptance.
