# My managed split-result arrays

I build on the qualified shared string-array core and checked VM array creation
boundary. I admit STR_SPLIT, ARR_GET and ARR_LEN only in my managed closed
profile. All arrays originate inside this closed module from STR_SPLIT; I have
no heap-bearing host inputs/imports. Function parameters/results may carry the
array tag, but entry remains zero-arity with integer/bool result. ARR_NEW,
PUSH, SET, POP, SLICE and arbitrary element shapes remain refused. In particular,
I do not add a string-child rejection to VM-accepted heterogeneous pushes.

I split stored bytes, including NUL/high bytes. A nonempty delimiter matches
left-to-right without overlap and retains leading, repeated and trailing empty
segments. An absent delimiter yields one whole-source segment. Empty delimiter
splits into single bytes; empty source then yields an empty array. A nonempty
delimiter on empty source yields one empty string. Every produced child owns
copied bytes. I consume one owner for each input, including equal handles, and
publish the array only after every child succeeds. Failure releases the partial
array and both inputs without changing the output or unrelated aliases. Source
views remain alive across table relocation and child allocation.

ARR_GET requires exact array/int tags. Signed negative or out-of-range indices
return TAG_VOID, without narrowing the original64-bit index; valid indices
return a retained string. ARR_LEN requires an array and returns its uint32 length
as int64. Frame cleanup releases consumed operands after retained results exist.
Wrong tags follow checked type-error cleanup, never an internal trap while roots
remain live. Missing values are not fabricated empty strings.

Array handles remain context-local stable identities. DUP/local/global loads
retain them; stores replace/release prior roots; calls transfer argument owners,
results transfer back, and all returns/error paths clear frame roots. Global
writes persist across repeated entry on one instance, including writes before
later failure; a fresh instance starts empty. Terminal disposal releases roots
and then storage. Read-only accounting includes arrays and their capacity.
I preserve default array casts (integer/float zero, string empty), nonnull
truthiness including empty arrays, identity equality, and VM's default same-tag
three-way ordering0. I never treat a handle as numeric payload during conversion.

I require ordinary paired VM/native LLVM/import-free Wasm results for byte
segments, boundaries, empty/missing values, aliases, calls/returns, globals,
reentry, identity/casts and errors. Core and emitted-module allocator controls
exercise partial child/table/buffer failure and recovery under native sanitizers;
Wasm exercises the actual allocator and finite-memory cleanup. Existing package
hash/ABI, scalar/literal and managed gates must pass. Publication refusal retains
prior output for unsupported mutation/shape instructions. These are bounded
collection prerequisites; parent488, parent51da, general cycles/nominal shapes,
Darwin sanitizer7ba and evaluator791a remain open.

My first focused run rejected a declared array-return helper in the native
status harness. Static inspection found my new private lowering used tag6,
while the authoritative ISA declares TAG_ARRAY=7. The VM controls passed;
three other focused groups passed, but this mixed-target result was not
acceptance. I correct the private tag to7, add an explicit compiler-side ABI
assertion and exact TYPE_CHECK7 controls before the next fresh gate. I retain
`/tmp/nanolang-managed-split-focused.log` as the original failure evidence.
