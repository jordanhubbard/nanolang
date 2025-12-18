# Nanolang Editor Support

Syntax highlighting and editor integration for Nanolang across major editors.

## Available Editors

### Visual Studio Code

Full-featured extension with syntax highlighting, bracket matching, and indentation.

- **Location**: `editors/vscode/`
- **Installation**: See [VSCode README](vscode/README.md)
- **Features**:
  - TextMate grammar for precise highlighting
  - Auto-closing pairs
  - Bracket matching
  - Indentation rules
  - Comment toggling

### Vim / Neovim

Comprehensive syntax file with folding support.

- **Location**: `editors/vim/`
- **Installation**: See [Vim README](vim/README.md)
- **Features**:
  - Keyword and type highlighting
  - S-expression folding
  - Filetype detection
  - Compatible with Vim 7.4+ and Neovim

### Emacs

Major mode with indentation and navigation support.

- **Location**: `editors/emacs/`
- **Installation**: See [Emacs README](emacs/README.md)
- **Features**:
  - Font-lock mode integration
  - Automatic indentation
  - S-expression navigation
  - Comment commands
  - Compatible with Emacs 24.3+

## Quick Start

### VSCode

```bash
cp -r editors/vscode/nanolang ~/.vscode/extensions/
```

### Vim

```bash
mkdir -p ~/.vim/syntax ~/.vim/ftdetect
cp editors/vim/nanolang.vim ~/.vim/syntax/
cp editors/vim/ftdetect/nanolang.vim ~/.vim/ftdetect/
```

### Emacs

```elisp
(add-to-list 'load-path "path/to/nanolang/editors/emacs")
(require 'nanolang-mode)
```

## Syntax Highlighting Features

All editor plugins support:

- **Keywords**: `fn`, `let`, `if`, `while`, `match`, `return`, etc.
- **Types**: `int`, `float`, `bool`, `string`, `array`, custom types
- **Built-ins**: `println`, `map`, `reduce`, `filter`, `len`
- **Literals**: Numbers (int, float, hex), strings with escapes
- **Comments**: Line (`//`) and block (`/* */`)
- **Operators**: Arithmetic, comparison, logical
- **Constants**: `true`, `false`, `null`

## Language Server Protocol (LSP)

An LSP implementation is planned (see `nanolang-kvz`). Once available, it will provide:

- Code completion
- Go-to-definition
- Hover documentation
- Error diagnostics
- Refactoring support

## Contributing

To add support for additional editors:

1. Create a directory under `editors/` for your editor
2. Implement syntax highlighting using editor's native format
3. Add installation instructions in a README
4. Test with nanolang examples
5. Submit a PR

### Highlighting Reference

Use these patterns for consistency:

- **Keywords**: Control flow, declarations, modifiers
- **Types**: Primitives and user-defined types (PascalCase)
- **Functions**: Built-ins and user functions (snake_case)
- **Constants**: Booleans, null, uppercase identifiers
- **Literals**: Numbers, strings with escape sequences
- **Comments**: Single-line and multi-line

## Testing

Test your syntax highlighting with these examples:

```nano
// Test file for syntax highlighting
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    return (* n (factorial (- n 1)))
}

shadow factorial {
    assert (== (factorial 5) 120)
}

let array_test: array<int> = [1, 2, 3, 4, 5]
let mapped: array<int> = (map array_test (fn (x: int) -> int { return (* x 2) }))
```

## Support

For editor-specific issues, see the README in each editor's directory.
For general nanolang questions, see the main project README.

