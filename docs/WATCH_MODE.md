# Nanolang Watch Mode

Automatic file monitoring and rebuilding for rapid development iteration.

## Overview

Watch mode monitors your nanolang source files for changes and automatically triggers recompilation or testing. This eliminates the manual compile-run-debug cycle and speeds up development.

## Installation

### macOS

```bash
brew install fswatch
```

### Ubuntu/Debian

```bash
sudo apt-get install inotify-tools
```

### Fedora/RHEL

```bash
sudo dnf install inotify-tools
```

## Usage

### Watch and Compile a Specific File

```bash
./scripts/watch.sh myapp.nano
```

Monitors `myapp.nano` and the standard library. Automatically recompiles when changes are detected.

### Watch and Run Tests

```bash
./scripts/watch.sh --test
```

Monitors `tests/`, `src/`, and `src_nano/`. Runs full test suite on changes.

### Watch and Rebuild Compiler

```bash
./scripts/watch.sh --bootstrap
```

Monitors compiler source files. Performs full 3-stage bootstrap on changes.

### Watch and Build Examples

```bash
./scripts/watch.sh --examples
```

Monitors `examples/`, `src/`, and `std/`. Rebuilds all examples on changes.

### Watch and Run in Interpreter

```bash
./scripts/watch.sh --interpreter myapp.nano
```

Runs the file in the interpreter automatically when it changes.

## Features

### Intelligent Monitoring

- Watches relevant directories based on mode
- Filters out build artifacts (`.o`, `.a`, `.so`, `.dylib`)
- Ignores editor temporary files (`~`, `.swp`)
- Debounces rapid changes (500ms delay)

### Cross-Platform Support

- Uses `fswatch` on macOS (recommended)
- Uses `inotifywait` on Linux
- Automatic detection of available watcher

### Clear Feedback

- Color-coded output (green = success, red = error, yellow = working)
- Timestamps for each run
- Clear separation between runs

### Error Handling

- Continues watching even after compilation failures
- Shows compilation errors inline
- Non-zero exit codes captured and displayed

## Examples

### Development Workflow

```bash
# Terminal 1: Watch and compile
./scripts/watch.sh src_nano/parser_mvp.nano

# Terminal 2: Edit file
vim src_nano/parser_mvp.nano
# Save file → watch mode automatically recompiles
```

### TDD Workflow

```bash
# Watch tests
./scripts/watch.sh --test

# Edit test file or source
vim tests/test_arrays.nano
# Save → tests run automatically
```

### Example Development

```bash
# Watch examples
./scripts/watch.sh --examples

# Edit example
vim examples/nl_game_demo.nano
# Save → example rebuilds automatically
```

## Advanced Usage

### Custom Watch Paths

Modify `WATCH_PATHS` array in the script:

```bash
WATCH_PATHS=("myproject/" "libs/" "tests/")
```

### Adjust Debounce Time

Change the sleep value:

```bash
sleep 0.5  # Debounce → sleep 1.0 for longer delay
```

### Add Custom Actions

Extend the `run_action()` function:

```bash
case $MODE in
    mycustom)
        echo "Running custom action..."
        ./my_custom_script.sh
        ;;
esac
```

## Troubleshooting

### "No file watcher found"

Install `fswatch` (macOS) or `inotify-tools` (Linux).

### Watch Mode Not Triggering

1. Check that files are actually changing (try `touch file.nano`)
2. Verify watcher is running (`ps aux | grep fswatch` or `inotifywait`)
3. Check file permissions

### Too Many Triggers

Increase debounce time or add more exclusions:

```bash
--exclude "mypattern"
```

### High CPU Usage

Reduce watched directories to only what's necessary:

```bash
./scripts/watch.sh my_specific_file.nano
```

## Performance

- Minimal overhead when idle
- Fast detection (typically <100ms)
- Efficient file filtering
- Safe for large projects

## Integration with IDEs

### VSCode

Add to `.vscode/tasks.json`:

```json
{
  "label": "Watch Nanolang",
  "type": "shell",
  "command": "./scripts/watch.sh --test",
  "isBackground": true
}
```

### Vim/Neovim

Run in a tmux pane or terminal split:

```vim
:terminal ./scripts/watch.sh %
```

### Emacs

Run in a compilation buffer:

```elisp
(compile "./scripts/watch.sh test.nano")
```

## Tips

1. **Use specific files** when possible for faster recompilation
2. **Run tests in watch mode** for immediate feedback during TDD
3. **Combine with terminal multiplexer** (tmux, screen) for best experience
4. **Use interpreter mode** for rapid prototyping
5. **Watch mode doesn't replace CI** - always run full tests before committing

## See Also

- [Benchmarking Suite](../scripts/benchmark.sh)
- [Testing Guide](TESTING.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

