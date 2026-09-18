# My shared-table string-array foundation

I implement `task_83a671a98d724de2b219e75dcad39c8b` as the first bounded child of aggregate/collection
parent488. This is a runtime foundation before ARR/STR_SPLIT lowering, not
executable profile or FrameOutput admission.

I add private descriptor kinds STRING and STRING_ARRAY to my existing
context-local stable handle table. String descriptors keep their current byte
length and storage ownership. Array descriptors store uint32 length/capacity
and aligned handle storage; their kind prevents string-view interpretation.
Literal handles remain strings. These layouts are private per-target ABI,
regenerated into native64/wasm32 runtime IR, not serialized wire facts.

An array owns one reference per string element. Borrowed append retains the
new child only after storage preparation succeeds; failure leaves the array,
its aliases and child references unchanged. Buffer growth uses widened size
arithmetic, copies existing handles without changing their owner counts, then
commits pointer/capacity/length together. The shared array handle does not
change, so mutation is observable through every retained alias. Retained get
returns one child owner, or handle0 for a missing unsigned index; length is a
read-only query. Public outputs change only on success. Helpers report TYPE
for a valid object of the wrong kind and preserve existing state/lifetime errors.

Only string children are eligible. Appending an array, including itself, is
refused before mutation. This subset cannot form heap cycles. It does not
establish a general cycle policy, nested/variant array support or authoritative
nominal aggregate layouts; those remain required under parent488.

I reuse the existing reclaiming native/Wasm allocator and descriptor table.
String byte buffers remain independent of table relocation. Array creation
and table growth are transactional; failed allocation publishes no handle.
Array capacity storage contributes capacity*sizeof(handle) to live_bytes;
string logical byte accounting is unchanged. live_objects counts both kinds.
Final array release drops each child owner exactly once and reclaims capacity.
Terminal disposal frees every live slot buffer once and resets all counters;
it does not recursively double-release edges while walking the whole table.

I require actual native LLVM and import-free Wasm core execution for alias
mutation, retained get, missing indices, duplicate children, buffer/table growth,
allocation rollback, child teardown, independent contexts and terminal disposal.
Native sanitizer and Wasm bounded-memory/reuse gates exercise ordinary finite
objects. Packaging must hash current source/header/generator, validate each
ABI and preserve every existing string target test. Wrong-kind refusals use
valid ordinary objects, not fabricated pointers or failed historical artifacts.
No opcode/profile/FrameOutput eligibility changes occur in this child. Later
integration must wire array retain/release across frames/globals/calls and
prove actual ARR/STR_SPLIT behavior before publishing those operations.
