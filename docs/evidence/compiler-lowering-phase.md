# My compiler lowering phase

I rename my compiler phase from `PHASE_TRANSPILER` to `PHASE_LOWERING` and my diagnostic constructor to `diag_lowering_error`. The generated C and NanoLang enums retain the same ordering: lowering remains numeric phase 3. My seed diagnostic JSON now reports `phase_name: "lowering"`. Canonical helpers and retained legacy compiler consumers use the same generated name.

My schema generator produced the checked-in headers/contracts. My C seed built successfully, `tests/test_error_messages.nano` compiled and executed with its mandatory shadows, and the complete canonical compiler source compiled with the renamed diagnostic dependencies. `make test-diagnostics` passed. An ordinary valid hello with an intentionally failing host compiler produced exit 1 and diagnostic CCC01, phase 3, phase name lowering; this verifies the observable JSON name without changing error handling.

This draft is stacked on PR536/534/522. It implements the phase-name requirement under the product cutover task; it does not claim a native bootstrap pass. The legacy C seed still generates C during bootstrap, so the broader lowering name describes that retained implementation honestly.
