---
title: 工具与后端
machine_generated: true
reviewed: false
lang: zh
---

> 本页是机器生成的草稿，尚未经人工审校。代码示例、命令和标识符保持英文。

# 工具与后端

我有多条执行路径。它们共享语法，但没有完整的功能对等。

## 工具

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | 编译源码并暴露分析或后端选项 |
| `bin/nano` | 解释源文件 |
| `bin/nanolang-repl` | 交互求值 |
| `nano-fmt` | 格式化源码 |
| `nano-docs` | 搜索本地文档 |
| `bin/nanolang-lsp` | Language Server Protocol 支持 |
| `bin/nanolang-dap` | Debug Adapter Protocol 支持 |
| `bin/nano_virt` | 把源码降到 NanoISA 字节码 |
| `bin/nano_vm` | 执行 NanoISA 字节码 |
| `bin/nano_vmd` | 运行 NanoVM 守护进程 |
| `bin/nano_cop` | 在协同进程中隔离受支持的外部调用 |

在提供 `--help` 的地方，用它运行每个工具。生成的 [Compiler CLI](../generated/cli.md) 页记录编译器当前的帮助文本。

## 后端

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | 经生成 C 的生产路径 |
| C source | `nanoc source.nano --target c -o program.c` | 独立生成的 C |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | 共享的带类型 VM 表示 |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | GPU 内核子集 |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | GPU 内核子集 |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | 实验性子集 |
| NanoISA | `nano_virt source.nano -o program.nvm` | 带隔离 FFI 支持的虚拟机路径 |

未来的 LLVM 和 WebAssembly 目标从 NanoISA 翻译，而不是从我的源 AST 分叉。来自 NanoISA 的 C11 是封闭子集试验（`nvm2c`、`make test-nvm2c`），不是 `nano_virt` 默认的、仍嵌入 VM 的原生包装。

## 诊断

面向机器的诊断包括 JSON 和 TOON 形式。有用的编译器选项包括 `--llm-diags-json`、`--llm-diags-toon`、`--json-errors`、`--emit-typed-ast-json` 和 `--reflect`。我用 `--locale <tag>` 选择进程区域设置（随后是 `NANO_LOCALE`、`LC_ALL`、`LANG`，否则为 `en`），并用 `--print-locale` 打印各轴。这不会改变机器可读的诊断输出。目录已经存在：面向人的 stderr 会查找 UTF-8 目录；JSON/TOON 保持英文；`--locale` / `NANO_CATALOG_DIR`。流水线编译器诊断使用稳定 ID（`CIO01`、`CSRC01`、`L0003` 以及 `src/diag_id.c` 中的其余项）。经过 `emit_context_error` 的类型检查器标题使用唯一的 `E001`–`E035`。标识符是 ASCII；非 ASCII 以失败闭合（`CLEX01`）。我拒绝 `.nano` 源中的无效 UTF-8（`CSRC01`），以及 JSON/TOON/`module.json`/文档生成边界上的无效 UTF-8。我不把这套系统称为已国际化。请查阅生成的 CLI 页面，因为标志变化比散文愿意承认的更频繁。

`-pg` 和 `--profile-output` 用宿主分析器包装原生二进制，并在 stdout 上发出 JSON。那条路径不是 `--profile-runtime`，也不是 `--pgo`。我在 [性能分析](07_performance_profiling.md) 中记载它。

