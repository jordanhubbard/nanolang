# My native C layout-cycle refusal evidence

I continue #976 from 5c1bf97ea. My native C composite dependency graph excluded direct self-edges and emitted remaining declarations in source order when topological sorting stalled. Invalid layouts consequently reached host C compilation instead of a compiler diagnostic.

I retain direct self-edges for records, unions and registered generic union instances. If no remaining declaration has zero incoming dependencies, I refuse with `I cannot emit cyclic by-value type layouts.` before publication. Tuple dependencies retain their existing graph edges. Pointer-backed arrays/lists and callable signatures do not create by-value edges.

The unchanged original suite plus four new negative methods records six baseline diagnostic failures. After correction all 14 methods pass: direct record/union, mutual record, mixed record/union, tuple/record and instantiated generic-union/record cycles fail with the required diagnostic and preserve prior output. Forward order, alternating chains, union/record payloads and primitive-list runtime layouts remain accepted. Two further positive methods retain self-referential callback signatures and generic union array payloads; both pass.

All 16 methods pass with ASan/UBSan instrumentation of the changed transpiler and the metadata object retained in this audit recipe. I disable interpreter leak detection; this does not establish leak freedom. The complete transpiler unit gate and its two assert-literal methods also pass.

This correction is specific to native C layout emission. The self-hosted compiler's declaration-validation failures remain open, as do aggregate union payload lowering and full 5.1 candidate/platform qualification. I do not infer that a native emitter refusal supplies a language-wide frontend check.
