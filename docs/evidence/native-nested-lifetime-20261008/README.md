# Native nested-array and byte-root checkpoint

I preserve the initial byte-root ASan heap-use-after-free and all three initial
nested lifetime failures. My byte root walker omitted storage kind 15. I now
trace it in direct and record roots. Nested storage kind 16 keeps each child's
storage tag and actual pointer in the existing managed sidecar, and the root
worklist follows those edges with cycle deduplication.

I preserve successive failures: optional shape at tagged slice, unsupported
identity comparison, tagged string comparison, and unused generated literal
helper. An attempted optional array-read solver change regressed both expanded
source producers; I withdrew that solver change and retained the failed log.
The corrected run passes all 17 methods in 11.275 seconds: six byte methods,
three nested lifetime methods, and all eight source slice methods. Both source
producers run through NanoVM and sanitized native C. Lifetime controls count
actual collections and require at least two, with ASan/UBSan and leak checks.

I retain fixture sources and command logs, without native binaries. Final
native regression, integration and clean installed-stage qualification are
recorded separately; these focused tests do not establish full 5.1 readiness.

My first full native gate passes 2,430 checks and fails one historical refusal
for empty `ARR_NEW 7`. I move that exact input into execution coverage and
retain wrong-element refusal for scalar-to-nested and nested-to-record arrays.
My dedicated nested Make target passes all three lifetime methods.

My final native gate passes all 2,434 execution checks, 3,098 shape checks
and 379 callable checks. Tested sources and tools match `tested-inputs.json`
before and after the final run. I preserve the sole first-gate failure above;
no wrong-element refusal was removed. Full installed-stage and platform
qualification remains required.
