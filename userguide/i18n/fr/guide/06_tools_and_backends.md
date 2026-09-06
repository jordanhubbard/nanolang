---
title: Outils et backends
machine_generated: true
reviewed: false
lang: fr
---

> Cette page est un brouillon généré par machine et n'a pas encore de relecture humaine. Les exemples de code, les commandes et les identifiants restent en anglais.

# Outils et backends

J'ai plusieurs chemins d'exécution. Ils partagent la syntaxe mais pas une parité de fonctions complète.

## Outils

| Tool | Purpose |
| --- | --- |
| `bin/nanoc` | Compiler la source et exposer des options d'analyse ou de backend |
| `bin/nano` | Interpréter un fichier source |
| `bin/nanolang-repl` | Évaluation interactive |
| `nano-fmt` | Formater la source |
| `nano-docs` | Chercher la documentation locale |
| `bin/nanolang-lsp` | Prise en charge de Language Server Protocol |
| `bin/nanolang-dap` | Prise en charge de Debug Adapter Protocol |
| `bin/nano_virt` | Abaisser la source en bytecode NanoISA |
| `bin/nano_vm` | Exécuter le bytecode NanoISA |
| `bin/nano_vmd` | Lancer le démon NanoVM |
| `bin/nano_cop` | Isoler les appels étrangers pris en charge dans un co-processus |

Lance chaque outil avec `--help` là où c'est fourni. La page générée [Compiler CLI](../generated/cli.md) consigne le texte d'aide actuel du compilateur.

## Backends

| Output | Command | Boundary |
| --- | --- | --- |
| Native executable | `nanoc source.nano -o program` | Chemin de production via du C généré |
| C source | `nanoc source.nano --target c -o program.c` | C généré autonome |
| NanoISA | `nano_virt source.nano --emit-nvm -o program.nvm` | Représentation de VM typée partagée |
| PTX | `nanoc source.nano --target ptx -o program.ptx` | Sous-ensemble de noyaux GPU |
| OpenCL C | `nanoc source.nano --target opencl -o program.cl` | Sous-ensemble de noyaux GPU |
| RISC-V assembly | `nanoc source.nano --target riscv -o program.s` | Sous-ensemble expérimental |
| NanoISA | `nano_virt source.nano -o program.nvm` | Chemin machine virtuelle avec FFI isolé |

Les cibles LLVM et WebAssembly futures traduisent depuis NanoISA plutôt que de ramifier depuis mon AST source. Le C11 depuis NanoISA est un spike en sous-ensemble fermé (`nvm2c`, `make test-nvm2c`), pas l'enrobage natif par défaut de `nano_virt` qui embarque encore la VM.

## Diagnostics

Les diagnostics destinés aux machines incluent des formes JSON et TOON. Les options utiles du compilateur incluent `--llm-diags-json`, `--llm-diags-toon`, `--json-errors`, `--emit-typed-ast-json` et `--reflect`. Je choisis une locale de processus avec `--locale <tag>` (puis `NANO_LOCALE`, `LC_ALL`, `LANG`, sinon `en`) et j'imprime les axes avec `--print-locale`. Cela ne change pas la sortie de diagnostic lisible par machine. Les catalogues existent : le stderr humain consulte des catalogues UTF-8 ; JSON/TOON restent en anglais ; `--locale` / `NANO_CATALOG_DIR`. Les diagnostics du compilateur de pipeline utilisent des ID stables (`CIO01`, `CSRC01`, `L0003` et le reste de `src/diag_id.c`). Les titres du vérificateur de types qui passent par `emit_context_error` utilisent des `E001`–`E035` uniques. Les identifiants sont ASCII ; le non-ASCII échoue fermé (`CLEX01`). Je refuse l'UTF-8 invalide dans la source `.nano` (`CSRC01`) et aux frontières JSON/TOON/`module.json`/docgen. Je n'appelle pas le système internationalisé. Consulte la page CLI générée parce que les drapeaux changent plus souvent que la prose ne devrait le prétendre.

`-pg` et `--profile-output` enrobent un binaire natif avec le profileur hôte et émettent du JSON sur stdout. Ce chemin n'est pas `--profile-runtime` et n'est pas `--pgo`. Je le documente dans [Profilage de performance](07_performance_profiling.md).

