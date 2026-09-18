# My scalar globals in LLVM and Wasm

I extend the existing closed-scalar profile with `LOAD_GLOBAL`, `STORE_GLOBAL`
and ordered scalar module initialization. MAC
`task_2baf4ba2b74b4d5aa94fb31cc9e81356`. Full target coverage remains open.

I audited `vm_decoded_module_global_slots`, `vm_ensure_globals`, the global
instruction handlers and `vm_execute` at main `d41ee3ae`.

- I size one internal tagged-value array to the highest referenced global slot
  plus one across all functions, not just reachable functions. Ordinary
  verification bounds every index below `NVM_MAX_GLOBALS` (4096). No unchecked
  module integer controls an allocation. A module without globals needs no table.
- Every slot initially contains void. Stores replace the whole scalar value;
  loads preserve exact int/U8/float/bool/void tags and bits. All module functions
  share the table. Branches and calls preserve execution order and last writes.
- The first function named `__init__` runs before entry. I require zero arguments
  for that selected initializer and retain the existing scalar/void result rules.
  Its result is discarded. Failure prevents entry. If initializer and entry are
  the same function, I invoke it twice, matching `vm_execute`.
- Storage belongs to the module instance, not an invocation. Repeated exported
  entry calls do not clear it; each repeats initialization before entry. A fresh
  process or Wasm instance begins with void slots.
- Existing import, heap, nominal, retained-layout, ownership/passive, capture,
  signature and instruction guards remain. I do not add concurrent access or
  linked-module global lifetime semantics. My ordinary verifier is unchanged.

My gates compare VM, LLVM, optimized LLVM, sanitized native LLVM and Wasm on
initial reads, exact tags, highest valid slot, overwrites, branches, helper calls,
initializer ordering/result discard/failure and persistent repeated invocation.
A test-only VM host observes repeated `vm_execute`; repeated Wasm exports observe
one instantiated module. Refused profiles preserve existing output. These tests
extend the scalar contract, not the full language acceptance matrix.
