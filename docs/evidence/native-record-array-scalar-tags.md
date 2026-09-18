# I preserve present scalar field representations in record arrays

I recorded task `task_a94dd8267d374549b2d578f191b5049f` before changing code.
Fresh product `85d2e294` passed bootstrap and 27 of 28 ordinary product methods.
Export-shadow compilation named an invariant in `parser_set_let_var_type`.
Source-only emission located raw field-kind equality during record-array
replacement. The failed artifact and exact log remain preserved; I did not
replay it or inspect its runtime field contents.

I compare equivalent present scalar representations when storage kinds differ.
Plain int/bool/string and boxed values must have the same actual payload tag;
string payloads must be non-null. Width, record kind, bounds, static shape
constraints and existing same-kind behavior are unchanged. This does not
admit absent or heap values through the new bridge. Whole-record assignment
preserves the stored metadata and existing array alias behavior.

My first six direct mixed-storage assembly controls were rejected by existing
flat field classification before emission. I retain that result in
`/tmp/nanolang-record-array-tags-focused.log`; I did not widen those static
constraints. My corrected gate tests the generated representation helper's
bidirectional equivalence and refusal, plus already admitted ordinary VM/native
replacement and alias observation. I also reference the emitted helper in
primitive-array-only programs so strict Clang accepts that existing route.

At `c16f16ac`, with production unchanged from `052a0956` except integrated typed
enum lowering, all 2,422 native and 1,269 shape checks pass. Two focused GCC
methods pass in 1.113 seconds at the preceding production checkpoint; eight
integrated Clang methods (including six optional-storage methods) pass in
7.958 seconds with generated ASan/UBSan/leak checks. Independent static review
confirmed the bridge boundaries. Logs: `/tmp/nanolang-record-array-tags-*`.

Fresh integrated product `aab9d30d` passes both bootstrap stages, hello and
installed execution without the C seed. All 28 ordinary product acceptance
methods pass in 13.533 seconds, including export shadows. Logs:
`/tmp/nanolang-product-record-array-bootstrap.log` and
`/tmp/nanolang-product-aab9-acceptance.log`. This establishes the corrected
product gate at that source; broader release and current full-suite/fixed-point
acceptance remain separate.
