# Iteration/match names and order-independent scalar inference

I retain iteration variable and union match-arm source names in both emitters.
Intervals begin after initialization, include a match guard, and end with the
lexical scope. Generated loop temporaries and wildcard patterns stay unnamed.
My fixtures check same-spelling outer/inner bindings, sequential loops, record
array elements, break/continue, false guards and both function declaration orders.

This coverage exposed an independent C-AOT defect. The C module verified and
executed, but native translation refused a scalar match result when the callee
preceded its caller. Stripping names retained the refusal; the previous selfhost
module placed the caller first and translated successfully. The attributed
failure showed local 4 retaining 0x8003 (unresolved plus integer/void) before
receiving the finite 0x4002 variant-integer fact. I keep unresolved inference
separate from actual opaque payload evidence so later finite joins can settle.
I retain the refusal boundary for real unproved/heap provenance, with a more
specific function/local diagnostic.

I pass all 2,443 native-translator checks and prerequisites, 90 bytecode checks,
and twelve source methods through both the C producer and fresh selfhost seed
with ASan/UBSan/leak checks. All three existing match-block regression methods
pass through the C-native, fresh selfhost and C-NanoISA routes, covering block
values/effects, early returns and refusal-before-publication for invalid types. I retain the
initial malformed test fixture and slot-number shadow assumptions, then correct
them using actual syntax and direct emitter checks. One concurrent check saw
nano_vm terminate with -9; the log is retained. The complete suite passed after
the build gate finished. That historical termination has no established cause.

My remaining producer audit accepts named nested functions and legacy par-let
through C, while the selfhost parser rejects both. Those reproductions remain
open in the roadmap; they require frontend parity as well as source names.
The parent metadata, reconstruction and complete 5.1 gates remain open.

My fresh native compiler product passes all seven audit steps, including all
twelve sanitized metadata methods and 22 nominal/layout regressions. All three
pinned production source hashes remain unchanged during the audit.
