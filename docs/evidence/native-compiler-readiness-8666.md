# My refreshed native compiler product acceptance

I track this evidence in MAC `task_f1bcd093f86b433a8f3d59139939187b`.
I test exact main `8666cb095c9da23a115f41c6d83aa0dafe5be579` on Linux
AArch64 after my typed F64 and `JMP_TRUE` native companions. I leave the source
unchanged during both compiler-product gates.

I first build the real tools and host runtime:

```sh
make -j4 bin/nanoc_c nano_virt nvm2c nanoisa_dump nano_vm nvm2c-runtime
```

I run these existing methods, preserving every assertion and their normal
per-command budgets:

```sh
env -u NANO_SHADOW_TIMEOUT_SECONDS python3 -m unittest -v \
  tests.test_one_ir_compiler.OneIrCompiler.test_compiler_bytecode_to_native_to_program \
  tests.test_one_ir_compiler.OneIrCompiler.test_selfhost_emitted_compiler_to_native_nanoisa_product
```

My observed run uses an external wrapper around those exact methods solely to
retain their temporary directories and record command timings. It leaves shadow
selection, commands, assertions and timeouts unchanged. Both methods pass in
405.053 seconds total.

| Compiler module producer | Commands' elapsed time | Compiler module bytes | Result |
| --- | ---: | ---: | --- |
| C-seed `nano_virt` | 106.549 s | 382888 | Native compiler produces both a native hello and a hello NanoISA product |
| Canonical frontend built by my C seed | 298.451 s | 361672 | Native compiler produces a hello NanoISA product |

For both paths I translate the full compiler module with `nvm2c`, compile its
structured C using `-std=c11 -Wall -Wextra -Werror -O0` and the real artifact
host runtime, and run its help. The generated source contains no VM execution
bridge. I execute each hello NanoISA product in my VM and through standalone
native translation; both print exactly `Hello from NanoLang!` followed by a
newline. The seeded route also compiles and executes the ordinary native hello.

My canonical frontend retains normal dependency and root shadow checks while
emitting its compiler module. That emission passes in 142.855 seconds; native
translation then passes in 4.111 seconds. Neither path exposes an unsupported
shape, opcode or artifact ABI within this tested workload.

## Retained evidence

I preserve artifacts under `/tmp/nanolang-native-readiness-8666/`:

- `nano-one-ir-compiler-rot42m17/` contains the seeded module, generated C,
  native compiler and both hello products.
- `nano-selfhost-native-product-it_bqyi8/` contains the canonical frontend,
  emitted compiler module, generated C, native compiler and hello products.
- `commands.jsonl` records each exact command, budget, working directory,
  explicit helper environment, elapsed time and outcome.
- `manifest.json` records the tested source pin and artifact sizes/hashes.

The seeded compiler module SHA-256 is
`84ff89213a6e18cccb7f9f74b17e26992d803eb953502e2c02bf772042607f16`.
The canonical compiler module SHA-256 is
`4935fb3c3f4c01795a38ebe2efa65431e529f64a9e39046de40d8542a9ce9113`.
Their immutable import paths belong to the retained checkout; these hashes
identify observed artifacts, not portable reproducibility. Build and test logs
are `/tmp/nanolang-native-readiness-8666-build.log` and
`/tmp/nanolang-native-readiness-8666-products.log`.

## Boundary

I tested that two native compiler products can compile a real hello NanoISA program.
I do not run the historical native parser-crash workload or claim full native
self-compilation convergence. I do not replace the canonical VM bootstrap gate.
The pending VM-shadow cutover changes what the generated compiler invokes for
shadows and needs fresh integration acceptance. Removing the C product route
and completing the NanoISA-only architecture remain separate roadmap work.
