# Nanolang VSCode Extension

Syntax highlighting for Nanolang in Visual Studio Code.

## Installation

### From Source

1. Copy the `nanolang` directory to your VSCode extensions folder:
   - **Windows**: `%USERPROFILE%\.vscode\extensions`
   - **macOS/Linux**: `~/.vscode/extensions`

2. Restart VSCode

3. Open any `.nano` file to see syntax highlighting

### Manual Install via VSIX

```bash
cd editors/vscode/nanolang
vsce package
code --install-extension nanolang-1.0.0.vsix
```

## Features

- Syntax highlighting for Nanolang keywords, types, and operators
- Comment support (line and block)
- Auto-closing pairs for brackets and quotes
- Proper indentation rules
- Function and type recognition

## Development

To modify the syntax highlighting:

1. Edit `syntaxes/nanolang.tmLanguage.json`
2. Reload VSCode window (Cmd/Ctrl + R)
3. Test with `.nano` files

## Publishing

To publish to VSCode Marketplace:

```bash
vsce publish
```

Requires a Personal Access Token from Visual Studio Marketplace.

