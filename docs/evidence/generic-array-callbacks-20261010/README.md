# Generic functional-array callbacks

I infer declared generic map, filter and reduce callbacks from complete input
identities. I bind reducer accumulator and element variables independently,
and give a generic initializer its inferred accumulator context. My selected
signature reaches evaluator function values, native callable slot reads and
C-frontend NanoISA specialization. I retain complete reduce result metadata
for inferred callable locals and reject incompatible map/filter result-array
annotations in unused bodies before replacing an existing output.

My eight-method `tests.test_generic_array_callbacks` suite covers scalar
specializations, callable identity and mutation, predicates, callable reducers,
independent generic variables, typed empty arrays, inline callable arrays,
inferred callable results and four incompatible controls. It joins
`make -f Makefile.gnu test-native-callable-arrays`.

My native regression passes 88 methods in 101.451 seconds (`regression.log`).
The same 88 pass in 114.354 seconds with parser, checker, environment,
transpiler, nominal types, module, evaluator, resource flow and module metadata
instrumented by ASan/UBSan (`sanitized.log`). I disable leak detection for this
compiler harness and retain a 65520 KiB stack; I do not claim complete compiler
leak qualification. My C-seed NanoISA regression passes 84 methods in 21.164
seconds (`nanoisa.log`), and evaluator/transpiler unit checks pass (`units.log`).

All seven positive new fixtures pass C-frontend NanoISA, NanoVM and C AOT with
ASan/UBSan on generated C (`nanoisa-probe.log`). I retain the sources and
`probe.py`, which runs from my repository root. Initial inference failures and
the accepted incompatible result are retained in `first.log` and
`result-refusal.log`.

My unchanged earlier callable-operation corpus still has three C AOT failures
(`prior-aot-recheck.log`): unresolved targets in empty functional operations,
and conflicting tagged parameter identities in callable-returning map/reduce.
These remain open under #976. These checks do not refresh my self-hosted stages
or establish exact-candidate Linux/Darwin release qualification.
