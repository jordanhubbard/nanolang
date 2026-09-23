# I link my native artifact host runtime explicitly

My `nvm2c` output contains native C operations, not an embedded VM. Pure programs
use the helpers emitted in that C file. Foreign artifacts can additionally
require exported host functions from my native array and reference-counting
runtime. My standard-library artifact has this requirement even when the called
entry point is scalar: `RTLD_NOW` resolves the whole library.

I build that host ABI from the same implementation used by my native compiler:

```sh
make nvm2c nvm2c-runtime
bin/nvm2c program.nvm -o program.c
```

On macOS:

```sh
cc -std=c11 program.c bin/nano_aot_runtime.o -lm -o program
```

On Linux:

```sh
cc -std=c11 program.c bin/nano_aot_runtime.o -Wl,--export-dynamic -ldl -lm -o program
```

I use a relocatable object so the linker retains functions referenced only by
loaded artifacts. A normally linked static archive can discard those functions.
Linux also needs dynamic symbol export so `dlopen` can resolve them. I do not
link a separate collector into each module; ownership stays with the host.

This object provides my existing `DynArray`, GC and GC-struct implementation.
It also provides the shared JSON and UTF-8 support used by declared native
module providers. Those support objects do not grant file or compiler authority;
the source module manifests still select and link the owning providers. This is
not the complete NanoVM runtime, a new ABI version, or permission to cast foreign
arrays into AOT arrays. My filesystem adapter still checks its foreign array ABI,
copies the result and invokes the foreign release function. I retain the exact
library path recorded in the module.

These commands describe repository builds. Relocatable packaging, broader
foreign APIs, Windows linking and full compiler acceptance remain separate gates.
