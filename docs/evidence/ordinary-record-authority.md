# My checked ordinary declaration transport checkpoint

I qualify the first transport/query stage of15f at source/test pin
`ce865addbe30ff9f316806ad9368dac5d3e1bd33` on canonical main460db419 (through738).
Production remains identical to independently reviewedb0584676. COMPLETE
ordinary declarations may describe scalar/string/prior ordinary-record fields.
My transitive scalar-tree summary preserves the previous RESOURCE boundary:
direct and indirect string-bearing resource declarations remain invalid, as do
ordinary declarations containing a resource or incomplete child. No opcode,
profile or source producer changes in this checkpoint.

My read-only query validates the complete existing declaration payload and
publishes UNKNOWN, ORDINARY or RESOURCE only on success. No payload means UNKNOWN;
its caller must separately resolve a retained layout index. The new scalar-summary
allocation follows the existing TRUNCATED allocation-error convention. Query
output remains unchanged when that allocation fails. My descriptor plan still
refuses ownership-bearing input pending its checked adapter; I do not claim it
already consumes the new ordinary classification.

I passed the GCC and Clang ASan/UBSan authority fixture, exact canonical layout/
ownership byte roundtrip, fresh normal VM nested-string record execution, and
unchanged LLVM/Wasm nominal refusal with prior-output preservation. Controls
include explicit empties, same-shaped separate records, incomplete facts, both
existing metadata versions and direct/transitive resource classification.
Existing ownership transport and descriptive-record tests also pass unchanged.

I then passed owned-runtime/same-frame/nested/caller/multi-caller reference,
assertion and value-graph controls, including unchanged allocation ceilings.
The first adjacent run stopped at a pre-existing diagnostic assertion in
owned-transfers: f922 source refused correctly but the test expected an older
borrowed-parameter phrase. I recorded395cfd7 before fixing only the expected
string. After the frozen corrected run passed, I fetched/rebased main460db419;
738 had further generalized that diagnostic. I updated the exact expectation
to its current scalar-entry/value-result wording while preserving every
admission/refusal boolean and allocation budget. Both historical logs remain.

The final integrated command passed:

```
NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 \
  make -j6 test-ordinary-record-authority test-ownership-contracts \
  test-managed-record-plan test-verifier-profiles test-owned-transfers \
  test-owned-value-graphs test-nanovm
```

This includes274541 VM checks, heap/stack recovery,184 ordinary and275 allocation
owned-transfer checks,338 graph preflight,529 invocation-proof,69 verification
reuse and1847 owned-value graph checks. No existing fault ceiling was raised.
The [manifest](ordinary-record-authority-artifacts.json) hashes all source,
tools and retained logs. These are Linux gates, not a new compiler bootstrap or
Darwin/full release qualification.

15f stays open for the descriptor adapter, paired ordinary Cseed/selfhost layout
and authority producers, array-element authority, generic/import/forward-order
and remaining nominal coverage. Field-origin analysis and generated nominal
ownership/lowering remain required before managed admission.
