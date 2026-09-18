# My consuming managed replacement contract

I implement `task_4930450b7e4f483f91d47f2e24707f3e` after checked VM sizing and
allocation6b3/PR680. STR_REPLACE becomes managed-profile-only; scalar/literal
selectors and aggregate operations retain their current admission boundaries.

I match exact stored bytes, including NUL/high bytes, with left-to-right
nonoverlapping matches. An empty needle returns unchanged bytes. My managed
result is a fresh owned string; VM interning may reuse equal objects, so I do
not claim physical identity or allocation-event equivalence.

The helper consumes three owners, even when handles alias. It validates all
views while their owners are live, counts bounded matches, and checks removed
spans and uint32 result growth before multiplication/addition. For a nonempty
needle it allocates one checked scratch output, copies segments/replacements,
then calls the existing checked string constructor to copy that output into
its final owner. It frees scratch before return on every path. Empty needle
uses the constructor directly. The temporary buffer and second copy are an
explicit cost of this bounded implementation; no arena or new allocator is
introduced.

Input views refer to separately allocated immutable byte storage, not movable
slot-table entries. I keep all three input owners until scratch/final copying
finishes, and retain no slot-table pointer across allocation. Every success or
failure releases each transferred input once, preserving unrelated aliases.
I publish output only after construction and release succeed. The LLVM adapter
transfers all three operands; wrong tags release all three through the same
status protocol, while remaining frame roots use ordinary cleanup.

I require ordinary VM/native LLVM/import-free Node/Wasmtime byte equivalence,
empty/no-match/deletion/overlap, long scratch output, alias permutations and
all-three-same input owners. Core native/Wasm tests cover scratch/result/table
allocation failure, table growth, unchanged output and surviving aliases.
Calls/globals/reentry, emitted failure/recovery, three wrong-tag positions and
complete disposal must pass. Profile and previous-output controls remain.
Full runtime51da, Darwin7ba, evaluator791a and aggregate-dependent split stay
open; no historical failed artifact is replayed.
