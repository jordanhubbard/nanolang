# My closed literal-string contract

I extend required LLVM/Wasm coverage with immutable module-owned literal bytes.
This child does not complete strings, managed storage or the full translator.

I retain `%V` payload/tag transport. A string payload is its constant-pool index
plus one, never a host pointer. Read-only descriptors map it to target-native
byte storage and a length. Empty strings have a nonzero handle and are true.
Storage lasts for the entire module instance; copies, returns and global writes
need no allocator, reference counting or dynamic reclamation for these literals.
No reference escapes through a host ABI in this closed profile.

I admit PUSH_STR, STR_LEN, STR_EQ, content-based generic equality/order, string
parameters/results, and existing local/global/call/branch operations. Length and
comparison use all bytes, including UTF-8 and embedded NUL; unsigned byte order
matches my VM. Duplicate bytes at distinct indices compare equal. Executable
entry remains zero-argument int/bool. Scalar-only eligibility is unchanged.

My new closed-literal-string profile first runs ordinary verification and retains
all existing closed-module limits. If any function pushes a string or declares a
string parameter/result, I refuse the entire module if it contains generic ADD,
CAST_INT or CAST_FLOAT. This conservative rule includes unreachable code and
numeric uses in such a module; I do not claim a provenance proof. Computed string
opcodes and heap/import/nominal features remain refused. Other arithmetic retains
its checked numeric-tag errors. Typed numeric instructions never interpret a
string handle as numeric data. CAST_BOOL uses the VM's truthiness.

I test VM/LLVM/Wasm execution, optimized and sanitized native LLVM output,
nonempty/empty/NUL/multibyte literals, equal bytes at distinct indices, lexical
ordering, mixed tags, call/return/global aliases and repeated/fresh instances.
Unsupported modules preserve prior output before publication. These are child
tests of my existing target-equivalence requirement.

My next recorded requirements are managed string allocation/lifetime, then
aggregate/collection identity and declared host/module linkage. Those require
explicit native/Wasm allocation, release, failure and teardown semantics; a
grow-only arena does not satisfy them. Full applicable-language coverage remains
required, including capabilities a Wasm host supplies explicitly.
