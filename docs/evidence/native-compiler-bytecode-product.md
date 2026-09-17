# I compile a NanoISA product with my generated native compiler

I no longer count only the generated compiler's default native hello output
for `task_16425cd8a2404735a5cb246db12c5f59`. My unchanged native bridge at
`91c93b68` also passes this explicit product path:

```text
nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug -o compiler.nvm
nvm2c compiler.nvm -o compiler.c
cc -std=c11 -Wall -Wextra -Werror -O0 compiler.c -o compiler \
   bin/nano_aot_runtime.o -lm -Wl,--export-dynamic -ldl
compiler examples/language/nl_hello.nano --emit-nvm -o hello.nvm
nano_vm hello.nvm
nvm2c hello.nvm -o hello.c
cc -std=c11 -Wall -Wextra -Werror hello.c -o hello-aot
./hello-aot
```

The host-link flags above are the Linux invocation; my regression retains
its existing platform-specific `HOST_RUNTIME` contract for compiler linking.
Both executions of the emitted hello product print `Hello from NanoLang!`.
I also retain the existing compiler help and default native hello checks.

The retained v2 hello module has SHA-256
`355b042aec423de12a64818da6774ae3609d569eabed46a3d602a24bf40312de`.
Compiler bytecode, generated C, native compiler, hello products, commands,
exit statuses and logs remain in `/tmp/nanolang-canonical-product-91c93b68`
for this session. The strengthened committed regression passes in 91.784s:

```text
python3 -m unittest -v tests.test_one_ir_compiler.OneIrCompiler.test_compiler_bytecode_to_native_to_program
```

I seed the compiler bytecode using the C frontend in this check. I do not
claim that my self-hosted emitter can yet emit the full compiler, that the
C-backend cutover is complete, or that canonical compiler bytecode reaches a
fixed point. The separate emitter continuation retains those acceptance
boundaries; no runtime failure from the historical zero-token report was
reproduced in this current product path.

After rebasing onto the merged string-concatenation emitter slice at
`4081b77d`, the complete strengthened test passed again in 90.309 seconds.
The integrated session log is `/tmp/nanolang-canonical-product-integrated.log`.
