# My native record-local frames

I track the parser stack boundary in MAC `task_4c5aeade9fa944dd80eebfbd1ae91072`.
My retained native compiler gives every record a 75-field representation,
4360 bytes on this AArch64 host. `parse_primary` declares 87 record locals;
`parse_expression_recursive` declares 24. Record temporaries already have
invocation-owned heap storage, but these locals previously occupied the C stack.

I now allocate one zeroed, dense record-local slab per invocation. Parameter
records and local assignments still copy values. Arrays, maps and nested record
snapshots retain their existing handle/ownership semantics. I publish slab
addresses to the same root frame at the same collection points. The result is
copied before the common return path unregisters roots and frees both record
slabs. Self-tail restarts stage all arguments before assignment, reset nonparameter
locals, and reuse the invocation's allocations. Allocation failure aborts before
an absent slab can be read, matching existing temporary-allocation behavior.

I keep the private by-value argument/result ABI. This repair does not eliminate
all stack use or establish an unlimited recursion depth. I do not change
collector scheduling, array ownership or system stack limits.

## Static measurement

I translate the retained `/tmp/nanolang-bootstrap-8d769-seed.nvm` with the new
translator and compile its C with GCC 13.3 on AArch64 using
`-std=c11 -O0 -fstack-usage -c`. This measurement does not run the generated
compiler. I compare the resulting reports with the retained growth compiler's
report, compiled with the same flags:

| Function | Retained frame bytes | Heap-local frame bytes |
| --- | ---: | ---: |
| `parse_primary` | 397216 | 17904 |
| `parse_expression_recursive` | 123664 | 19024 |
| `parse_expression` | 13264 | 8912 |
| `parser_has_error` | 4544 | 192 |

Reports: `/tmp/nanolang-native-stack-inspection.su` and
`/tmp/nanolang-native-record-locals-seed.su`. Historical parser failure logs and
executables remain intact; I do not rerun those failing inputs for this repair.

## Regression contract

`tests/test_native_record_locals.py` checks 96 distinct record locals across
32 recursive calls and verifies every retained value during unwind. It also
requires the unoptimized generated function's static frame to stay below
64 KiB. A compile-only comparison using the retained translator reports
364224 bytes for that fixture and fails this bound before any old native binary
is executed. A second fixture swaps two record parameters through 1001 self-tail
restarts while tracing owned strings, copying a record local and mutating an
aliased array. Both fixtures execute in my VM and under native ASan, UBSan and
leak detection.

My existing temporary-storage allocation counter now expects two allocations
per record-bearing invocation: the local slab and temporary slab. It retains
zero-live-allocation checks, ordinary/cross-tail recursion, constant self-tail
allocation bounds and injected allocation failure.

The first fixture draft exposed an independent native `JMP_TRUE` classifier
refusal after successful assembly and VM execution. I record it as
`task_211f22859e164287a07a63cba74ace5b`; the storage fixtures use the supported
`BOOL_NOT`/`JMP_FALSE` equivalent. I do not narrow the VM truthiness contract.

The native semantic/shape suites, adjacent ownership checks and both ordinary
compiler-product gates are the acceptance checks for this storage change.
Their measured outcomes are recorded below when complete. Full native
self-compilation and release-wide bootstrap acceptance remain separate results.

My first full product run passes 56 of 57 methods in 341.462 seconds, including
the seeded compiler's native and NanoISA products. The canonical product stops
at its first C-seed command, before native translation, with `I stopped shadow
execution after 10 seconds.` I retain this failure in
`/tmp/nanolang-record-local-product-gate.log`; it is not evidence of a
record-local failure and I do not relabel it as infrastructure without proof.
The native suite passes 2390 checks and shape constraints pass 1092 checks.
