# NanoLang for Visual Studio Code

Language support for [NanoLang](https://github.com/jordanhubbard/nanolang) - a minimal, LLM-friendly programming language with mandatory shadow-tests.

## Features

### Syntax Highlighting

- **Keywords**: `fn`, `let`, `mut`, `if`, `while`, `for`, `match`, `return`, etc.
- **Types**: `int`, `float`, `string`, `bool`, `void`, `array<T>`, `List<T>`, etc.
- **Operators**: Prefix notation operators: `+`, `-`, `*`, `/`, `==`, `<`, `>`, etc.
- **Comments**: `# comment` style
- **Strings**: Double-quoted strings with escape sequences
- **Numbers**: Integers, floats, hex literals
- **Functions**: Function definitions and calls in prefix notation

### Language Configuration

- **Bracket Matching**: Automatic matching for `()`, `{}`, `[]`
- **Auto-Closing Pairs**: Automatic closing of brackets and quotes
- **Comment Toggling**: Quick comment/uncomment with `Cmd+/` or `Ctrl+/`
- **Smart Indentation**: Automatic indentation inside blocks

### Code Snippets

Type the prefix and press `Tab` to expand:

**Functions & Control Flow:**
- `fn` - Function with shadow test
- `main` - Main function template
- `if` - If-else statement
- `while` - While loop
- `for` - For loop with range
- `match` - Pattern matching

**Data Structures:**
- `struct` - Struct definition
- `enum` - Enum definition
- `union` - Union (tagged union) definition
- `array` - Array declaration

**Testing:**
- `shadow` - Shadow test block
- `assert` - Assertion

**Variables:**
- `let` - Immutable variable
- `letmut` - Mutable variable

**Imports & FFI:**
- `import` - Import statement
- `extern` - External C function

**Result & Option:**
- `ok` - Result::Ok variant
- `err` - Result::Err variant
- `some` - Option::Some variant
- `none` - Option::None variant

**Common Operations:**
- `+`, `-`, `*`, `/` - Arithmetic in prefix notation
- `==` - Equality comparison
- `print` - Print statement

## Installation

### From Marketplace (when published)

1. Open VS Code
2. Go to Extensions (`Cmd+Shift+X` or `Ctrl+Shift+X`)
3. Search for "NanoLang"
4. Click Install

### From Source (Development)

1. Clone the NanoLang repository:
   ```bash
   git clone https://github.com/jordanhubbard/nanolang.git
   cd nanolang/tools/vscode-nanolang
   ```

2. Install extension:
   ```bash
   # Copy to VS Code extensions directory
   # macOS/Linux:
   cp -r . ~/.vscode/extensions/nanolang-0.1.0

   # Windows:
   # copy to %USERPROFILE%\.vscode\extensions\nanolang-0.1.0
   ```

3. Reload VS Code

### Package and Install (VSIX)

```bash
# Install vsce (VS Code Extension Manager)
npm install -g @vscode/vsce

# Package extension
cd tools/vscode-nanolang
vsce package

# Install the .vsix file
code --install-extension nanolang-0.1.0.vsix
```

## Usage

1. Open any `.nano` file
2. VS Code will automatically detect it as NanoLang
3. Enjoy syntax highlighting and code snippets!

## Example

```nano
# Calculate factorial recursively
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}

fn main() -> int {
    let result: int = (factorial 6)
    (println (int_to_string result))  # Prints: 720
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## NanoLang Resources

- [GitHub Repository](https://github.com/jordanhubbard/nanolang)
- [Documentation](https://github.com/jordanhubbard/nanolang/tree/main/docs)
- [Learning Path](https://github.com/jordanhubbard/nanolang/blob/main/docs/LEARNING_PATH.md)
- [Examples](https://github.com/jordanhubbard/nanolang/tree/main/examples)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](https://github.com/jordanhubbard/nanolang/blob/main/CONTRIBUTING.md) file in the main repository.

## Feedback

Found a bug or have a feature request? Please open an issue on the [GitHub repository](https://github.com/jordanhubbard/nanolang/issues).

## Release Notes

### 0.1.0

Initial release of NanoLang for VS Code:

- Syntax highlighting for NanoLang
- Bracket matching and auto-closing pairs
- Comment toggling support
- 30+ code snippets
- Smart indentation rules

## License

MIT License - See [LICENSE](https://github.com/jordanhubbard/nanolang/blob/main/LICENSE) file for details.

---

**Enjoy coding in NanoLang!** 🚀
