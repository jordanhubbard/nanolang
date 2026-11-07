# I retain the borrowed caller-local FLOAT baseline

I reproduce exactly test_source_borrow_emission.py:test_float_local_in_borrowed_control_flow: replace the two `set total (+ total 2)` occurrences in source_borrow_control_flow.nano with `let value: float = 2.0 assert (== value 2.0) set total (+ total 2)`. No source assertion or control edge is removed.

Current seed and both fresh3446 stages produce ordinary main/step with24/3locals; the independently emitted original-shadow input from the pinned C-hosted driver produces shadowentry/step with24/3locals. All four verify and VM execute0 with empty stdout. Each caller contains two PUSH_F64 operands with raw bits4000000000000000 and one F64_EQ. Helper remains one borrowed resource formal, three locals and no FLOAT. This is the existing nested v2 single-exclusive control route, not FLOAT fields or signatures.

Current facts refuses all four. C/Nano reconstruction attempts refuse and preserve prior output. Ten selected provider/source pins remain unchanged. I retain commands, cleanup statuses, source, dumps and exact errors. No regenerated source, local feasibility, helper-FLOAT admission or new runtime support is claimed. Binaries remain under /private/tmp/borrowed-float-local-baseline-20261019.
