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

Future LLVM and WebAssembly targets translate from NanoISA rather than branching from my source AST.

## Diagnostics

Machine-facing diagnostics include JSON and TOON forms. Useful compiler options include `--llm-diags-json`, `--llm-diags-toon`, `--json-errors`, `--emit-typed-ast-json`, and `--reflect`. Consult the generated CLI page because flags change more often than prose should pretend they do not.

`-pg` and `--profile-output` wrap a native binary with the host profiler and emit JSON on stdout. That path is not `--profile-runtime` and is not `--pgo`. I document it in [Performance Profiling](07_performance_profiling.md).
