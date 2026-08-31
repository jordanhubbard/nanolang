---
name: nanovm-opcode-debugging
description: >-
  Diagnose NanoVM opcode, stack, value, and FFI failures with the optional
  NANO_VM_TRACE instrumentation.
---

# NanoVM Opcode Debugging

I keep opcode tracing disabled by default. Enable it only while diagnosing a
specific VM failure; tracing writes one record per retired instruction and can
produce substantial output.

## Enable tracing

Build the normal VM, then set the environment variable before starting it:

```bash
NANO_VM_TRACE=1 ./bin/nano_vm failing.nvm
```

The VM reads `NANO_VM_TRACE` once during `vm_init`. The instruction dispatch
path checks only the cached boolean. Any non-empty value other than `0` enables
tracing. `NANO_VM_TRACE=0` disables it.

To inspect a NanoLang source program through NanoVM:

```bash
NANO_VM_TRACE=1 ./bin/nano_virt program.nano --emit-nvm -o /tmp/program.nvm
NANO_VM_TRACE=1 ./bin/nano_vm /tmp/program.nvm
```

## Read the trace

Each record includes the function index, function name, bytecode offset,
opcode, stack depth before and after execution, and the top stack values. Heap
values include their tag, address, length where applicable, and printable
contents. FFI records include the import, C symbol, argument count, return tag,
and marshaled result.

Use the first divergent record, not the last error message, as the starting
point. For string failures compare the `PUSH_STR`, FFI return, and equality
records. For stack failures compare the stack depth around the first operation
whose declared effect differs from the observed effect.

Tracing is diagnostic evidence, not a correctness proof. Reproduce the failure
with tracing disabled after the fix and run the relevant NanoVM, NanoVirt, and
benchmark quality gates.
