# My canonical AOT consumer is C11

C11 is my canonical ahead-of-time portability backend from NanoISA. My
`nvm2c` tool reads a verified `.nvm` and writes structured C operators; it
does not embed `nano_vm` and it is not a compiler phase. A generated
process computes with C operators and does not spawn `nano_vm`, `nano_cop`,
or `nano_vmd`.

I compile the Cut A fixture through the canonical route:

```sh
bin/nano_virt tests/nanoisa/fixtures/cut_a_add.nano --emit-nvm --strip-debug -o cut_a.nvm
bin/nano_vm --verify-only cut_a.nvm
bin/nvm2c cut_a.nvm -o cut_a.c
cc -std=c11 -Wall -Wextra -Werror cut_a.c bin/nano_aot_runtime.o -lm -o cut_a
```

The generated C names none of `nano_vm`, `nano_cop`, or `nano_vmd`, and the
linked process has no NanoVM library dependency. It runs `add(40, 2)` as
native C and exits `42`.

`make test-canonical-nvm-output` pins this in
`test_canonical_module_translates_to_vm_independent_c11`: it emits the
fixture, verifies it in the VM, translates it with `nvm2c`, asserts the
three VM process names are absent, builds the C11 with `cc`, and runs the
product.

This is MAC `task_9f95d78c2f4a47b8b53196ce4d7a2d91`. It establishes the
canonical C11 AOT consumer and its VM-free process boundary. Compiler-subset
coverage, declared host-ABI mapping, and the self-hosted driver cutover
remain separate roadmap rows.
