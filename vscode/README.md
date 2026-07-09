# Nanolang VS Code Extension

VS Code extension for the [Nanolang](https://nanolang.dev) programming language, providing:

- **Syntax highlighting** — TextMate grammar for `.nano` files
- **LSP integration** — diagnostics, completions, hover, go-to-definition via `nanolang-lsp`
- **Semantic tokens** — powered by the LSP server
- **Format on save** — calls `nano-fmt --write` via the LSP `textDocument/formatting` handler
- **Build task** — `nanoc` compile command template

## Requirements

- `nanolang-lsp` on `$PATH` (or configure `nanolang.lspPath`)
- `nano-fmt` on `$PATH` (or configure `nanolang.fmtPath`) for format-on-save
- `nanoc` on `$PATH` for the build task

## Installation

### From source

```bash
cd vscode
npm install
npm run compile
npx vsce package          # produces nanolang-0.1.0.vsix
code --install-extension nanolang-0.1.0.vsix
```

### Development

```bash
cd vscode
npm install
# Open in VS Code and press F5 to launch Extension Development Host
code .
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `nanolang.lspPath` | `"nanolang-lsp"` | Path to the nanolang-lsp binary |
| `nanolang.fmtPath` | `"nano-fmt"` | Path to the nano-fmt binary |
| `nanolang.trace.server` | `"off"` | LSP message tracing (`off`/`messages`/`verbose`) |

## Building

From the repo root:

```bash
make vscode-ext
```

This runs `npm install && npm run compile` in `vscode/`, and packages with `vsce` if available.

## Features

### Syntax Highlighting

Covers all Nanolang constructs:
- Keywords: `fn`, `let`, `if`, `else`, `while`, `return`, `match`, `import`, `export`, `from`, `type`, `effect`, `handle`, `perform`
- Built-in types: `Int`, `Float`, `Bool`, `String`, `Unit`
- Annotations: `@property`, `@bench`, `@gpu`
- Operators: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `&&`, `||`, `->`
- String interpolation (`${expr}`)

### LSP Features

All LSP features are provided by `nanolang-lsp`:
- **Diagnostics** — type errors, syntax errors
- **Completions** — identifiers, keywords, types
- **Hover** — type information
- **Go to definition** — jump to declarations
- **Semantic tokens** — additional token classification
- **Format on save** — via `textDocument/formatting`
