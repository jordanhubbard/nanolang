---
title: Herramientas y backends
machine_generated: true
reviewed: false
lang: es
---

> Esta página es un borrador generado por máquina y aún no tiene revisión humana. Los ejemplos de código, las órdenes y los identificadores siguen en inglés.

# Herramientas y backends

Tengo varios caminos de ejecución. Comparten sintaxis pero no paridad completa de funciones.

## Herramientas

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | Compilar fuente y exponer opciones de análisis o de backend |
| `bin/nano` | Interpretar un archivo fuente |
| `bin/nanolang-repl` | Evaluación interactiva |
| `nano-fmt` | Formatear fuente |
| `nano-docs` | Buscar documentación local |
| `bin/nanolang-lsp` | Soporte de Language Server Protocol |
| `bin/nanolang-dap` | Soporte de Debug Adapter Protocol |
| `bin/nano_virt` | Bajar fuente a bytecode NanoISA |
| `bin/nano_vm` | Ejecutar bytecode NanoISA |
| `bin/nano_vmd` | Ejecutar el demonio NanoVM |
| `bin/nano_cop` | Aislar llamadas extranjeras admitidas en un co-proceso |

Ejecuta cada herramienta con `--help` donde se ofrezca. La página generada [Compiler CLI](../generated/cli.md) registra el texto de ayuda actual del compilador.

## Backends

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | Ruta de producción a través de C generado |
| C source | `nanoc source.nano --target c -o program.c` | C generado independiente |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | Representación de VM tipada compartida |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | Subconjunto de núcleos GPU |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | Subconjunto de núcleos GPU |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | Subconjunto experimental |
| NanoISA | `nano_virt source.nano -o program.nvm` | Camino de máquina virtual con FFI aislado |

Los futuros destinos LLVM y WebAssembly traducen desde NanoISA en lugar de ramificar desde mi AST fuente. C11 desde NanoISA es un spike de subconjunto cerrado (`nvm2c`, `make test-nvm2c`), no el envoltorio nativo por defecto de `nano_virt` que aún incrusta la VM.

## Diagnósticos

Los diagnósticos orientados a máquinas incluyen formas JSON y TOON. Opciones útiles del compilador incluyen `--llm-diags-json`, `--llm-diags-toon`, `--json-errors`, `--emit-typed-ast-json` y `--reflect`. Selecciono un locale de proceso con `--locale <tag>` (luego `NANO_LOCALE`, `LC_ALL`, `LANG`, si no `en`) e imprimo los ejes con `--print-locale`. Eso no cambia la salida de diagnóstico legible por máquina. Los catálogos existen: el stderr humano consulta catálogos UTF-8; JSON/TOON siguen en inglés; `--locale` / `NANO_CATALOG_DIR`. Los diagnósticos del compilador de tubería usan IDs estables (`CIO01`, `CSRC01`, `L0003` y el resto de `src/diag_id.c`). Los títulos del verificador de tipos que pasan por `emit_context_error` usan `E001`–`E035` únicos. Los identificadores son ASCII; lo no ASCII falla cerrado (`CLEX01`). Rechazo UTF-8 inválido en fuente `.nano` (`CSRC01`) y en fronteras JSON/TOON/`module.json`/docgen. No llamo al sistema internacionalizado. Consulta la página CLI generada porque los flags cambian más a menudo de lo que la prosa debería fingir.

`-pg` y `--profile-output` envuelven un binario nativo con el perfilador anfitrión y emiten JSON en stdout. Ese camino no es `--profile-runtime` y no es `--pgo`. Lo documento en [Perfilado de rendimiento](07_performance_profiling.md).

