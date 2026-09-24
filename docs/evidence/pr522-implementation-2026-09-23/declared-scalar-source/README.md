# My declaration-driven scalar source path

I resolve uncatalogued scalar externs through their declaration owner and its
immutable module artifact. I preserve the full bounded parameter/result
signature, emit kind 4, and add `-lffi` to native products that contain that
contract. I keep named runtime/artifact adapters on their existing paths.
Extern returns use `CALL_EXTERN` followed by `RET`; they cannot become native
NanoISA function tail calls.

My original three `tests.test_native_module_linking` cases pass unchanged
through fresh Stage 1 and Stage 2 on Darwin. Three added source methods cover
mixed int/float/bool/u8/string arguments, borrowed strings, native/VM products,
missing manifests, unsupported shapes and prior-output preservation. All six
methods also pass through both stages with generated-product ASan/UBSan,
leak detection and use-after-return checks. Compiler/VM/provider instrumentation
is not established by those product flags.

I retain the initial diagnostic shadow failures, the invalid extern-tail-call
failure, and a Darwin ARM64 LLDB trace. Declaration lookup returned the correct
index; return lowering bypassed my call emitter. The probe directory contains
the source and debugger scripts; its main source refers to
`/tmp/pr522-scalar-owner-probe/api.nano` as used during the trace.

The broad emitter run passes 86 bytecode comparisons and 90 of 91 Python
methods. Its sole failure requires the original unsupported-extern diagnostic
text. I restore that text without changing the assertion; final-source
bootstrap passes and all six affected shadow methods pass. I then repeat all
six linker/source methods through both final-source stages, ordinarily and
with instrumented products; all pass. Final compiler and log hashes are in
`inputs.json` and the per-stage JSON reports. I have not rerun the entire
91-method gate after the diagnostic-only correction.

I do not close task_b8838417bbc54fb98a4c49eea1b0885a. Installed publication,
Linux/full instrumentation, unused source-declaration validation in pruned
program mode, and full acceptance remain required. The installed runtime gap
is tracked separately as task_5e91db0e212e427186b67a4dbcabe7fe.

My instrumented product environment is `NANO_CC=/opt/homebrew/opt/llvm/bin/clang`,
`NANO_CFLAGS=-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`,
`NANO_LDFLAGS=-fsanitize=address,undefined`,
`ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1`, and
`UBSAN_OPTIONS=halt_on_error=1`. Test deadlines remain unchanged.
