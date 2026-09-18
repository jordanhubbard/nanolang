# My typed unsigned division reconstruction contract

I record task_d0b4b473329745cd848694ec835965ad before implementation, under full reconstruction parent task_4bd034f6029b7458201db74e2c3aeb32. I admit only exact INT operands/results for I64_DIV_U and I64_REM_U. I interpret their signed carriers as unsigned 64-bit patterns, matching my VM dispatcher: division and remainder by zero return zero. All other quotients and remainders return their exact bits in signed carriers.

My C helpers use uint64_t arithmetic and my existing representable signed-bit conversion. My NanoLang helpers use at most 64 binary long-division steps, exact unsigned comparison and total add/subtract helpers; signed high-bit tests consume dividend bits in order through wrapped doubling. A saved high-bit carry distinguishes an overflowing doubled remainder before subtraction. I never take abs(MIN), rely on signed overflow, or assume unsigned source syntax. This finite helper cost is not a performance-parity claim.

I require endpoint/high-bit/zero pairs, calls/loops/evaluation snapshots, strict tag/arity and previous-output refusals, byte roundtrip, and the same module through VM, reconstructed C and three pinned NanoLang compiler tools. I retain generator source hashes separately from reused compiler tools, run GCC/Clang sanitizers, and preserve initial failures. Full reconstruction, generic numeric operations and heap/metadata expansion remain separate.

My first focused three-method gate passes in 29.061s. During the combined gate I identify one stale foundation refusal fixture that still expects I64_DIV_U rejection; its negative control must move to still-unsupported generic ADD. I retain the initial combined logs before correcting that fixture. I also add explicit missing-operand refusal controls, without executing rejected modules.

## My measured acceptance

I test generator production `a7e530ab`, with test-only correction `111f2bb0`. My first focused three-method gate passes in 29.061s (`/tmp/nanolang-reconstruct-udiv-focused.log`): 338 endpoint/high-bit/zero pairs, a 64-iteration unsigned division loop, saved operand semantics, canonical byte roundtrip and wrong-tag refusal. Both executable surfaces come from the same module, and I compare them with VM execution.

My combined 28-method runs retain one obsolete expectation each: GCC 291.949s (`/tmp/nanolang-reconstruct-udiv-gcc.log`) and Clang 278.033s (`/tmp/nanolang-reconstruct-udiv-clang.log`) pass the other 27 methods, including all new executable arithmetic cases. The old foundation refusal still used now-supported I64_DIV_U. I replace that control with still-unsupported generic ADD; the corrected method and new missing-operand refusal method pass separately in 0.906s and 0.887s (`/tmp/nanolang-reconstruct-udiv-corrected-refusal{,-clang}.log`). I do not describe the initial combined runs as green or claim a complete combined rerun. No production change follows those runs.

Reconstructed C uses strict O1 GCC/Clang with ASan/UBSan. Reconstructed NanoLang executes through three reused compiler tools from source checkpoint `4a75f984`; the C-seed-generated C path also completes with UBSan enabled. This is not a current-main compiler bootstrap or a same-source compiler fixed point. Before/after manifests `/tmp/nanolang-reconstruct-udiv-tools.sha256` and `/tmp/nanolang-reconstruct-udiv-compilers.sha256` match:

| Tool | SHA-256 |
| --- | --- |
| Generator | `08db5a376f6877b1b7b866c37cd7ec8c086f07ac06c2114d5546f55e298b9b9e` |
| nanoisa | `8451ed2b785b82fa3aa536c5415dd6ce1b0301fa1de1d151fe0a9d897ba6d2ad` |
| nano_vm | `ab88577fb9e4199f42568bbf6d32803142a0f7fb9d45a98d2eb6f164b178d384` |
| nvm2hl | `c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4` |
| C seed | `442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041` |
| Stage1 | `d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa` |
| Stage2 | `b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40` |

My bounded helpers take at most 64 division steps; I make no native multiplication/division performance claim. Full reconstruction remains open.
