# My tagged leaf-value boxed-array foundation

I implement a runtime prerequisite under parent488, not new executable admission.
VM boxed arrays retain an entire NanoValue regardless of declared element kind;
packed int/float/bool/U8 arrays instead coerce into their declared storage kind.
I must not substitute a new runtime child-type rejection for VM-accepted pushes.
This child supplies owned tagged storage before a later eligibility/lowering step.

My private NmsValue has uint64 payload and uint32 tag, with explicit native64 and
wasm32 layout assertions. It preserves every payload bit and tag for void, int,
U8, float, bool, enum and string. Scalars own no storage; a string value owns one
context-local string handle. I validate string kind/lifetime before retaining or
releasing, and refuse other tags in this private leaf API. I do not interpret
numeric payloads as pointers or normalize their bits. Actual producers remain
responsible for their canonical scalar values; matched opcode admission is later.

I add BOXED_LEAF_ARRAY descriptors to my existing stable handle table. Generic
leaf-array create/append/set/get/pop/length accepts that kind and the existing
STRING_ARRAY kind. A borrowed append/set retains its new string child only after
all required storage preparation succeeds. Replacement publishes the new edge
before releasing the old one. Get returns a retained value or exact zero/VOID
for a missing unsigned index; pop removes and transfers one owned child, or
returns zero/VOID when empty. An out-of-range set returns NMS_STATE without
mutation in this private API; a later opcode adapter must establish the VM's
separate out-of-bounds status rather than claim this is already ARR_SET lowering.

Appending or setting a generic value into STRING_ARRAY prepares a boxed buffer,
converts existing string handles into tagged edges without retaining/releasing
them, retains the new child, and commits buffer/kind/capacity/length atomically.
The array handle and all aliases stay identical. Any allocation/retain failure
keeps the old kind, buffer, contents, refcounts and outputs unchanged. Promotion
uses checked widened capacity/byte arithmetic and does not expose slot pointers
across allocation. Existing string-only append/get/length accessors return TYPE
on a promoted descriptor without changing output or references. Generic accessors
continue to see its values. Existing admitted split modules never invoke this
private promotion and retain their current string-only GET behavior.

Final release drops each string child exactly once; scalar bits are ignored by
ownership traversal. Pop clears its removed slot before transferring the owner.
Terminal context disposal frees each live buffer once without recursive duplicate
release. Array backing capacity contributes its actual storage bytes to accounting.
Only scalar/string leaves are admitted here, so this core cannot form cycles.
I keep current STRING/STRING_ARRAY behavior and all executable profiles unchanged.

I require native verified LLVM/sanitizer and import-free Wasm core acceptance for
mixed exact tags/bits (including NaN/signed-zero), duplicate string children,
alias mutation, promotion rollback/success, legacy accessor refusal, same-child
replacement, retained gets, transferring pops, missing values and independent
contexts. Deterministic allocator controls, table relocation, finite Wasm pressure,
repeated reclamation and terminal disposal must preserve ownership and outputs.
Production package hashes/ABIs and existing scalar/string/split gates must pass.

Parent488 still requires packed scalar coercion, static element-shape eligibility,
array/nominal child traversal and cycle collection, then matched mutable opcode
semantics. I do not allow nested arrays in this private leaf API or claim that
restriction can replace general language support. Parent51da/platform/evaluator
obligations remain separate and open.
