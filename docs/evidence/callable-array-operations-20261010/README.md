# Callable array operations

I execute callable array elements in map, filter and reduce through evaluator
shadows and native C emission. Map can return callable elements. Filter keeps
the value tested by its predicate even when that predicate replaces its source
slot. Reduce can carry a callable accumulator, specializes a generic initializer
from that accumulator's complete signature, and evaluates array, initializer,
and callback in source order before capturing the array length.

I copy callable parameters and results in my evaluator builtin callback path.
My map/filter checker compares complete element signatures, rejects mismatches
in unused bodies, and retains full result-array annotations. My native reads
use typed function pointers in sized array slots and check missing slots.

My 20 focused callable storage/operation methods pass (`corrected.log`). My
broader native suite passes 80 methods (`regression.log`), and the same 80 pass
with nine compiler components instrumented by ASan/UBSan (`sanitized.log`,
leak detection disabled, 65520 KiB stack). The suite contains my eight modules:
`test_native_callback_fields`, `test_native_global_types`,
`test_cseed_generic_functions`, `test_contextual_generic_functions`,
`test_higher_order_generic_functions`, `test_generic_callback_storage`,
`test_native_callable_arrays`, and `test_native_callable_operations`.

My current C-seed NanoISA regression suite passes 84 methods (`nanoisa.log`),
and evaluator/transpiler unit checks pass (`units.log`). Four existing core
map/filter/array/operator programs produce matching native and NanoVM output
(`adjacent.log`). Initial failures remain in `first.log` and `expanded.log`.

I separately run all nine positive operation fixtures through the C NanoISA
frontend and NanoVM. Six also execute through C AOT (`nanoisa-probe.log`).
C AOT still refuses empty functional operations for unresolved callback targets
and rejects callback-returning map/reduce with conflicting tagged parameter
kinds. I retain all source fixtures here. A generic map transform also remains
unsupported (`generic-map-callable.nano` and its failure log). I keep those
implementation gaps open under #976; this checkpoint is not complete functional
parity, a refreshed self-hosted compiler, or release qualification.

I separately retain the successful isolated component-shadow trace in
[its evidence](../component-shadow-trace-20261010/README.md). I did not reproduce
or establish the cause of the earlier component timeout.
