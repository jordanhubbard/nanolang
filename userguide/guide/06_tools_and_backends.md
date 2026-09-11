# Tools and Backends

I have several execution paths. They share syntax but not complete feature parity.

## Tools

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | Compile source and expose analysis or backend options |
| `bin/nano` | Interpret a source file |
| `bin/nanolang-repl` | Interactive evaluation |
| `nano-fmt` | Format source |
| `nano-docs` | Search local documentation |
| `bin/nanolang-lsp` | Language Server Protocol support |
| `bin/nanolang-dap` | Debug Adapter Protocol support |
| `bin/nano_virt` | Lower source to NanoISA bytecode |
| `bin/nano_vm` | Execute NanoISA bytecode |
| `bin/nano_vmd` | Run the NanoVM daemon |
| `bin/nano_cop` | Isolate supported foreign calls in a co-process |
| `bin/nvm2c` | Translate a closed NanoISA subset to structured C11 |
| `bin/nanoisa_emit` | Lower a pinned integer subset to NanoISA assembly (`make nanoisa_emit`) |

Run each tool with `--help` where provided. The generated [Compiler CLI](../generated/cli.md) page records the compiler's current help text.

## Backends

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | Production path through generated C |
| C source | `nanoc source.nano --target c -o program.c` | Standalone generated C |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | Shared typed VM representation |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | GPU kernel subset |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | GPU kernel subset |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | Experimental subset |
| NanoISA | `nano_virt source.nano -o program.nvm` | Virtual-machine path with isolated FFI support |

Future LLVM and WebAssembly targets translate from NanoISA rather than branching from my source AST. C11 from NanoISA is `bin/nvm2c` on a closed subset (`make test-nvm2c`), not `nano_virt`'s default native wrapper that still embeds the VM.

## Diagnostics

Machine-facing diagnostics include JSON and TOON forms. Useful compiler options include `--llm-diags-json`, `--llm-diags-toon`, `--json-errors`, `--emit-typed-ast-json`, and `--reflect`. I select a process locale with `--locale <tag>` (then `NANO_LOCALE`, POSIX `LC_ALL`/`LANG`, else `en`) and print the axes with `--print-locale`. JSON/TOON diagnostics stay English. Human stderr looks up UTF-8 catalogs under `catalogs/messages/` or `NANO_CATALOG_DIR`, with English fallback. Pipeline compiler diagnostics use stable IDs (`CIO01`, `CSRC01`, `L0003`, `P0001`, `E001`–`E035`, `LOG01`, and the rest of `src/diag_id.c`). Typechecker titles that go through `emit_context_error` use unique `E001`–`E034`; unlabeled JSON fallback is `E035`. Identifiers are ASCII; a non-ASCII byte fails closed (`L0003`). I reject invalid UTF-8 in `.nano` source (`CSRC01`) and at JSON/TOON/`module.json`/docgen/log boundaries. I do not call the system internationalized. Consult the generated CLI page because flags change more often than prose should pretend they do not.

`-pg` and `--profile-output` wrap a native binary with the host profiler and emit JSON on stdout. That path is not `--profile-runtime` and is not `--pgo`. I document it in [Performance Profiling](07_performance_profiling.md).

NSI, capabilities, the POSIX fabric, and the trap journal are the secure-runtime tools. They are libraries and tests, not extra CLIs. See [Secure Runtime](08_secure_runtime.md).
