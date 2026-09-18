# My indexed scalar stack reconstruction contract

I record task_70b0aa8374904271b66caa86bdf77d7f before implementation under full reconstruction parent4bd034. PICK depth duplicates the scalar item that many slots below the top. ROLL depth moves that item to the top while preserving every other item's order. Depth zero is valid; a depth outside the current expression stack is refused before mutation.

I reuse the existing exact INT/BOOL expression snapshots, so duplication and movement do not reexecute an effectful producer or reread a changed local. Pure loop expressions remain side-effect-free under existing admission. I add no heap, reference, signature or multiresult ABI behavior. Both structured source surfaces consume the same checked stack model.

I require fresh small mixed-tag/depth/order, direct-call/snapshot and loop cases through VM, reconstructed C and three pinned NanoLang compiler tools, canonical byte roundtrip, strict output-preserving refusal and GCC/Clang sanitizers. This slice does not use the failed carry endpoint fixtures or depend on draft679; task100952 and full reconstruction remain open. Compiler-source pins and tool hashes remain separate from this generator source.

Before these fresh gates I include the isolated reporting companion from draft679: exact quoted argv is retained for subprocess failures, tested with mocked ordinary exit7 for all three compiler paths. No carry implementation or failed fixture comes into this branch. Historical carry-stage identity remains unknown.
