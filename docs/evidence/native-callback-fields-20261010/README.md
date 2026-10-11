# My native callback field checkpoint

I emit complete callable and tuple field types. I discover global and function-body type dependencies before emitting layouts, forward-declare aggregate names, order nested callable typedefs, and place tuple layouts in the same dependency graph as records and unions. I emit each typedef once for strict C99 compatibility. Function signatures may refer to their containing record without introducing a by-value layout cycle.

I retain complete tuple-field initializer contexts after ordinary expression checking and register their exact tuple identities during buffered body emission. I also preserve a parsed function-valued result's nested signature in its full result annotation, allowing generic specialization to compare both parameter and result identities.

## My validation

I pass six new `tests.test_native_callback_fields` methods: forward-record signatures, self-record signatures, record-bearing tuple signatures, nested callable signatures, tuple field layouts and incompatible nested callable/tuple field refusals that preserve an existing executable.

I pass 55 native methods, including those six, six global methods, the existing generic/contextual/higher-order suites and the original local/global callback-record storage controls. I pass the same 55 methods with nine compiler components instrumented with ASan/UBSan. I use the previously documented 65,520 KiB stack for the instrumented recursive interpreter shadow.

I pass 84 C-producer generic/tuple/contextual/higher-order/storage regression methods, including NanoVM and sanitized C AOT execution. `make -f Makefile.gnu test-transpiler` passes the StringBuilder boundary controls and two assert-literal methods.

My broader 41-method record-name/generic-record/numeric-union/union-padding/effect run passes 40 methods. Its remaining failure is `NativeUnionPadding.test_conflicting_heap_and_scalar_fields_still_refuse`: the unchanged C AOT translator accepts a mixed int/array variant used only for its tag, while the old test expects refusal. I retain the failing log and track contract reconciliation separately. The translator executable SHA-256 is `b79ed86c887ae4f9056f117ab3e0b93c180595e67538b293390e2fd78e95312b`; this checkpoint changes neither its source nor that refusal test. I do not claim that broader suite is green or remove its assertion.

## My remaining scope

Callable array storage in the legacy evaluator/native path remains open. I have not refreshed installed Stage 1/Stage 2 or completed exact-candidate platform qualification. These checks do not close full generic acceptance, issue #976 or my complete 5.1 scope. GitHub API access failed; my issue update remains local until it can be posted.
