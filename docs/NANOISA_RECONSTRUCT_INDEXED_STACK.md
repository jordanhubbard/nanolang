# My indexed scalar stack reconstruction contract

I record task_70b0aa8374904271b66caa86bdf77d7f before implementation under full reconstruction parent4bd034. PICK depth duplicates the scalar item that many slots below the top. ROLL depth moves that item to the top while preserving every other item's order. Depth zero is valid; a depth outside the current expression stack is refused before mutation.

I reuse the existing exact INT/BOOL expression snapshots, so duplication and movement do not reexecute an effectful producer or reread a changed local. Pure loop expressions remain side-effect-free under existing admission. I add no heap, reference, signature or multiresult ABI behavior. Both structured source surfaces consume the same checked stack model.

I require fresh small mixed-tag/depth/order, direct-call/snapshot and loop cases through VM, reconstructed C and three pinned NanoLang compiler tools, canonical byte roundtrip, strict output-preserving refusal and GCC/Clang sanitizers. This slice does not use the failed carry endpoint fixtures or depend on draft679; task100952 and full reconstruction remain open. Compiler-source pins and tool hashes remain separate from this generator source.

Before these fresh gates I include the isolated reporting companion from draft679: exact quoted argv is retained for subprocess failures, tested with mocked ordinary exit7 for all three compiler paths. No carry implementation or failed fixture comes into this branch. Historical carry-stage identity remains unknown.

## My measured acceptance

I test production `2624739b` plus reporting companion `6530469c`, on main base `17ed33cc`. The fresh focused four-method gate passes in 17.145s (`/tmp/nanolang-reconstruct-indexed-focused.log`). The full 33-method reconstruction inventory passes with GCC in 307.212s and Clang in 292.075s (`/tmp/nanolang-reconstruct-indexed-{gcc,clang}.log`). This includes all existing arithmetic, corrected unsigned refusal, byte-roundtrip and structured-region controls plus the new indexed operations. Draft679 carry fixtures are absent.

I compile reconstructed C with strict O1 ASan/UBSan and execute reconstructed NanoLang through three reused compiler tools from source checkpoint `4a75f984`; the Cseed-generated C path also runs with UBSan enabled. I do not claim a current-source compiler bootstrap or fixed point. Before/after manifests match for the generator/runtime tools and reused compiler tools:

```text
b1f89939bc5b2e475fce2e3580f715a6b45f05dccf6f9db3a5119ee43cd35b6d  bin/nanoisa
6cb25bea5a8f4d9522d46e3d5c17e22ec217466d1c662293cd81749332eb1d87  bin/nano_vm
c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4  bin/nvm2hl
b26bc026fa159b864764a7069ba31dabb2bc8a6e2e892d8abfe4232fcddb1150  scripts/nanoisa_reconstruction.py
442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041  /home/jkh/Src/nanolang-functional-array-builtins/bin/nanoc_c
d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa  /home/jkh/Src/nanolang-functional-array-builtins/bin/nanoc_stage1
b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40  /home/jkh/Src/nanolang-functional-array-builtins/bin/nanoc_stage2
```

I retain full reconstruction and the independent pinned-compiler failure as open obligations. The small indexed cases establish their own tested scope.
