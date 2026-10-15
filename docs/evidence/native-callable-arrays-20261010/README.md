# Native callable array storage

I retain complete callback signatures in array literals, repeated initializers,
reads, writes, pushes, pops, slices, returned arrays, record fields and parameters.
My evaluator owns each callable slot independently. My native representation copies
typed function pointer bytes through sized array storage. I reject incompatible
writes even in unused bodies and preserve prior output on rejection.

I ran 66 native methods (`regression.log`), the same 66 with an ASan/UBSan
compiler (`sanitized.log`, leak detection disabled), 84 C-seed NanoISA methods
(`nanoisa.log`), and evaluator/transpiler unit checks (`units.log`). All passed.
`corrected.log` retains 13 focused passing methods. `first.log` retains the
initial generic-write and incompatible-push failures corrected by this change.

I used `NANO_CONTEXTUAL_NATIVE_COMPILER=bin/nanoc_c` for the native suite:
`tests.test_native_callback_fields`, `tests.test_native_global_types`,
`tests.test_cseed_generic_functions`, `tests.test_contextual_generic_functions`,
`tests.test_higher_order_generic_functions`, `tests.test_generic_callback_storage`,
and `tests.test_native_callable_arrays`. My sanitizer run used the same suite
with the instrumented compiler and a 65520 KiB stack limit.

I still fail callable iteration and map (`iteration.nano`, `map.nano`, and
matching logs). This checkpoint does not establish full callable-array parity,
captured closure coverage, a refreshed bootstrap, or release qualification.
I track the remaining implementation under #976 in my roadmap.
