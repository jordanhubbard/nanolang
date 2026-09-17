# My checked reference-place prerequisite

I start from main `04aec17e` after PR527. I record the representation and
remaining verification/runtime path in [my reference contract](../NANOISA_REFERENCES.md).
This slice implements `src/nanoisa/reference_places.c`; it does not admit
borrowed programs through either NanoISA frontend.

I validate a record-field path against the authoritative root layout and its
retained v2 nested layout indices. My leaf must have the exact expected
nominal layout and only numeric/bool fields. I distinguish invocations and
root locals, compare numeric path prefixes, allow shared/shared overlap, and
reject conflicting owner reads/writes or moves. The pure queries allocate
nothing and do not mutate their borrowed descriptors.

`make test-reference-places` passes 75 checks. The same executable passes
ASan/UBSan. My cases include two-level paths, disjoint sibling fields,
ancestor access, recursive invocation identities, same-shaped distinct
nominal records, invalid field/layout/mode declarations, and unsupported
referent kinds. These are ordinary contract tests, not a claim of complete
module verification. Layout validation alone does not prove resource status,
a live owner, valid local index, or lifetime; the callers that establish those
facts remain the next implementation stage.

I retain my canonical and raw NanoISA refusals. No reference wire tag,
section, opcode, VM value or native lowering is enabled. I have not rerun the
native compiler bootstrap because this slice changes none of its frontend,
interpreter, or generated compiler sources. My broad borrow and affine IR
acceptance tasks remain open.

I rebuilt NanoVM and nvm2c, then passed the existing v2 layout tests and
2,691 NanoISA checks, including schema consistency. My execution-module
bridge still retains type counts rather than full layouts; its field-less
placeholders cannot establish the authoritative input this API requires.
