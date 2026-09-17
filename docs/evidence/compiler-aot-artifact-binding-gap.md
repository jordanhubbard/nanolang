# My compiler AOT artifact-binding prerequisite

I reproduce this failure on unchanged integrated PR #383 head `5e078fb5`,
whose source was merged as `c7492089`, and again with generic PR #388. The
failure is independent of the generic classification changes.

```sh
python3 -m unittest -v tests.test_one_ir_compiler.OneIrCompiler.test_compiler_bytecode_to_native_to_program
```

The compiler emits its bytecode, but `nvm2c` rejects import zero,
`nl_nanoisa_load_print`, because the artifact-backed function lacks an exact
library binding and typed value adapter. The canonical `--emit-nvm` route
imports my NanoISA facade into the compiler. My translator's rejection remains
required until the host binding contract is implemented and tested.

I track the repair in MAC `task_600074c773904b119b39bdafd85c07a5`. Acceptance retains the existing
compiler-bytecode-to-native-to-program test, actual generated compiler execution,
typed facade behavior and rejection of unsupported artifact imports. I do not
count a removed import, disabled test or relaxed guard as a repair. This is a
required full v5.0.1 architecture gate, not a generic-classification pass.

Local baseline evidence is `/tmp/nanolang-main385-compiler-aot-baseline.log`
(5.821 seconds); the independent generic-tree reproduction is
`/tmp/nanolang-generic-affine-compiler-diagnostic.log` (5.965 seconds), on sparky.

## My bounded adapter repair

I retain the exact absolute artifact library path already present in bytecode,
match five facade symbols and their complete parameter/return signatures, and
emit typed zero-, one- and two-argument adapters. I still reject unknown symbols,
wrong signatures and wrong binding kinds. I snapshot load/pretty/error strings:
my facade reuses or frees its borrowed buffers on later calls, so returning those
pointers directly would corrupt an earlier language value. These copies follow
my existing process-lifetime native string allocation model; this is not a
complete AOT string reclamation claim.

My real facade fixture compiles through nano_virt, runs in NanoVM, translates to
C and executes natively. It checks file and text assembly, printing, pretty
printing, zero-argument last_error, and unchanged earlier strings after later
facade calls. Five accepted contracts compile with strict warnings; twenty-four
malformed contracts reject while preserving prior output. My existing std host
runtime fixture remains part of the focused check. `make test-nvm2c` passes
1,773 structured-C checks and 1,076 shape constraints.

The unchanged full compiler test now passes artifact binding and reaches a new
refusal: `ARR_SET index must be an integer`. GDB identifies function 145,
`parser_mark_owned`: its declared int index is inferred as boxed VALUE. I retain
the guard and track checked unboxing under
`task_959f620cc9294ef693072702f35ba44f`. This differs from the older stopped
record-field-representation issue. The compiler-to-native-to-program acceptance
remains open; this adapter prerequisite alone does not finish task 600074.

Local evidence: `/tmp/nanolang-compiler-aot-facade.log`,
`/tmp/nanolang-compiler-aot-contracts.log`,
`/tmp/nanolang-compiler-aot-nvm2c.log`,
`/tmp/nanolang-compiler-aot-adapter-proof.log`, and retained bytecode, assembly and
GDB output in `/tmp/nanolang-compiler-aot-retained/`.
