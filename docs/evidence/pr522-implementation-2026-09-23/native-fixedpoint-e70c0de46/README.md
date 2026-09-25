# Native fixed point at e70c0de46

I pass both full standalone native compiler generations from clean source
`e70c0de4666139a01a77edd0e6dfa7fb9abe6f26` in Linux ARM64, using GCC 13
in the isolated 16 GiB VM. Stage 1 takes 483.081 seconds and Stage 2 takes
479.697 seconds. Both verified modules contain 526,188 bytes with SHA-256
`2c107f991005a04259a9978e2609ad97be5c45299712f9ca0c223d63e4d70ca5`.
I compare raw bytes, preserve the exact host-library and helper hashes, and
verify the clean source pin again at completion. Each command retains its
1,800-second bound and default shadow deadline.

I also translate and compile the final compared module with strict GCC,
then use that standalone compiler to compile a hello product. The product
verifies and executes. `manifest.json` records all terminal exits.
`native_gate.py` is the exact external runner; this record does not add a
permanent native gate to the test suite. `artifacts.json` hashes every retained
artifact. Full products remain in the local qualification evidence directory
`/tmp/pr522-fixedpoint-e70c0de46/native-final`.

This pin predates the C-seed enum-allocation and generic-call cleanup fixes.
It is checkpoint evidence, not final acceptance of later source revisions.
The VM route has a different host closure; I do not compare its raw bytes
against this native route or infer compiler correctness from a fixed point.
