# My selfhost-emitted native compiler product

I resolved MAC `task_250092bed54749ad988f06af5b88c228` without relaxing a native shape guard. My canonical emitter omitted nominal declaration counts. My native scalar-field inference deliberately uses only declared nominal identities; with zero declared records, it could not reconstruct `CompilerDiagnostic` in `remap_diagnostics`.

PR #471 supplies my parser's full counts, including unreachable declarations so that IDs remain stable. My actual compiler declares `.types 69 5 0`. Removing only that directive from its canonical dump reproduces the original failure:

```text
function 10 at offset 94: I cannot resolve AGG_PACK field 0
```

I require a separate regression, `OneIrCompiler.test_selfhost_emitted_compiler_to_native_nanoisa_product`, in `make test-one-ir-compiler`:

1. I build my canonical frontend from `src_nano/nanoc_v06.nano` using my C seed.
2. I run that frontend with `--emit-nvm` on the same compiler source, retaining dependency and root shadow checks.
3. I translate its compiler bytecode with `nvm2c` and compile the resulting C with strict warnings. Its source contains no VM execution bridge.
4. I run the native compiler's help, then ask it to emit hello bytecode.
5. I execute that same hello module in my VM and through native translation, requiring the exact greeting from both.

The focused regression passed on Linux ARM64 in 286.861 seconds. I separately retained the generated artifacts and repeated the native help and both hello executions. My actual compiler bytecode was 350652 bytes, SHA-256 `cda38d373c5aa1e1b1095df0ede170aba38514e773119a09b8a0dc4060e4bf4b`; its immutable import paths belong to this checkout, so this hash records the observed artifact rather than a portable reproducibility claim.

I retain the existing `test_compiler_bytecode_to_native_to_program`, which begins with `nano_virt` emission. These paths prove different compiler products. Neither test proves byte-identical self-compilation stages or removes my C seed; those bootstrap acceptance criteria remain open.

Local evidence uses `/tmp/nanolang-diagnostic-canonical-*` for preserved artifacts and logs, and `/tmp/nanolang-diagnostic-reconstruction-product-test.log` for the focused regression. These paths are supplementary local evidence; the committed regression is my durable acceptance check.
