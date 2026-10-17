# I distinguish field lookup from module initialization

My full OneIR main corpus passes 108 of 109 methods in 1696.659 seconds; its sole failure is `lookup_maps=1` in the field-metadata allocation harness (`one-ir-baseline.log`). The isolated method reproduces it (`isolated-baseline.log`). Both product invariants, all 28 callable-array controls and seven of eight callable-union methods pass; the remaining union method requires a separately built checker component and is explicitly skipped.

My imported `module_bindings.nano` initializes a global map before guest entry. The harness counted all allocations in the process instead of checking whether field lookup allocates an index. I now snapshot the initialization count and reset the allocation counter immediately before guest main. I retain all VM/native lookup semantics, including repeated reads, replacement precedence and missing fields, and require zero guest map allocations.

I also inject a map allocation after the reset and require that native control to fail with exactly one counted allocation. Both executions retain AddressSanitizer and UndefinedBehaviorSanitizer. The corrected method, including its negative control, passes in 64.597 seconds (`corrected.log`). I add a dedicated hosted quick target for this regression.

I rerun the demonstrated failing method, not the complete 28-minute corpus after this harness-only change. Hosted platform and exact-release qualification remain separate.
