# Nanolang REPL Design

## Overview

Interactive Read-Eval-Print Loop for nanolang, enabling exploratory programming, rapid prototyping, and learning.

## Goals

1. **Interactive Development**: Test code snippets without creating files
2. **Learning Tool**: Help new users explore nanolang interactively
3. **Debugging**: Quickly test functions and expressions
4. **Prototyping**: Rapid iteration on algorithms

## Architecture

### Core Components

```
┌─────────────┐
│   Readline  │ ← User input with history/editing
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Parser   │ ← Parse partial/complete expressions
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Evaluator  │ ← Execute in persistent environment
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Printer   │ ← Pretty-print results
└─────────────┘
```

### Implementation Strategy

#### Phase 1: Basic REPL (MVP)
- Single-line expression evaluation
- Use existing interpreter (`src/eval.c`)
- Persistent environment across evaluations
- Basic readline integration

#### Phase 2: Multi-line Support
- Detect incomplete expressions (unclosed parens)
- Continue reading until complete
- Handle multi-line function definitions

#### Phase 3: Advanced Features
- Command history (persistent across sessions)
- Tab completion (keywords, functions, variables)
- Syntax highlighting
- Error recovery

## Technical Design

### File Structure

```
src/
  repl/
    repl.c           # Main REPL loop
    repl.h           # REPL API
    repl_commands.c  # Special commands (:help, :quit, etc.)
    repl_completion.c # Tab completion
    repl_history.c   # History management

bin/
  nanorepl         # REPL executable
```

### REPL Loop

```c
Environment* repl_env = env_new(NULL);

while (true) {
    // Read
    char* input = readline("nano> ");
    if (!input) break;
    
    // Check for special commands
    if (input[0] == ':') {
        handle_command(input);
        continue;
    }
    
    // Parse
    Program* prog = parse_expression(input);
    if (!prog) {
        if (is_incomplete(input)) {
            input = read_multiline(input);
            prog = parse_expression(input);
        } else {
            print_error();
            continue;
        }
    }
    
    // Eval
    Value result = eval(prog, repl_env);
    
    // Print
    print_value(result);
    
    // Add to history
    add_history(input);
}
```

### Special Commands

```
:help           - Show help
:quit / :q      - Exit REPL
:clear          - Clear screen
:env            - Show environment variables
:type <expr>    - Show type of expression
:load <file>    - Load and execute file
:reset          - Reset environment
:history        - Show command history
:debug <expr>   - Show debug info for expression
```

### Readline Integration

```c
#include <readline/readline.h>
#include <readline/history.h>

void repl_init(void) {
    // Set up completion
    rl_attempted_completion_function = nano_completion;
    
    // Load history
    read_history("~/.nanolang_history");
    
    // Set up keybindings
    rl_bind_key('\t', rl_complete);
}

void repl_cleanup(void) {
    // Save history
    write_history("~/.nanolang_history");
}
```

### Tab Completion

```c
char** nano_completion(const char* text, int start, int end) {
    rl_attempted_completion_over = 1;
    
    if (start == 0) {
        // Complete commands or keywords
        return rl_completion_matches(text, command_generator);
    } else {
        // Complete function/variable names
        return rl_completion_matches(text, symbol_generator);
    }
}

char* command_generator(const char* text, int state) {
    static const char* keywords[] = {
        "fn", "let", "if", "while", "return", "import", NULL
    };
    static int idx;
    
    if (state == 0) idx = 0;
    
    while (keywords[idx]) {
        if (strncmp(keywords[idx], text, strlen(text)) == 0) {
            return strdup(keywords[idx++]);
        }
        idx++;
    }
    
    return NULL;
}
```

### Expression Completeness Detection

```c
bool is_complete_expression(const char* input) {
    int paren_depth = 0;
    int brace_depth = 0;
    bool in_string = false;
    
    for (const char* p = input; *p; p++) {
        if (*p == '"' && *(p-1) != '\\') {
            in_string = !in_string;
        } else if (!in_string) {
            if (*p == '(') paren_depth++;
            else if (*p == ')') paren_depth--;
            else if (*p == '{') brace_depth++;
            else if (*p == '}') brace_depth--;
        }
    }
    
    return !in_string && paren_depth == 0 && brace_depth == 0;
}
```

### Pretty Printing

```c
void repl_print_value(Value v) {
    switch (v.type) {
        case VAL_INT:
            printf("%lld\n", v.as.integer);
            break;
        case VAL_FLOAT:
            printf("%g\n", v.as.floating);
            break;
        case VAL_STRING:
            printf("\"%s\"\n", v.as.string);
            break;
        case VAL_BOOL:
            printf("%s\n", v.as.boolean ? "true" : "false");
            break;
        case VAL_ARRAY:
            print_array(v.as.array);
            break;
        case VAL_VOID:
            // Don't print anything for void
            break;
        default:
            printf("<value>\n");
    }
}
```

## Features

### Basic Features (Phase 1)
- [x] Single-line expression evaluation
- [x] Persistent environment
- [x] Variable definitions
- [x] Function definitions
- [x] Error display
- [x] Readline support (history, editing)

### Advanced Features (Phase 2)
- [ ] Multi-line input
- [ ] Special commands (`:help`, `:quit`, etc.)
- [ ] Tab completion
- [ ] Syntax highlighting
- [ ] Pretty-printing of complex values

### Power User Features (Phase 3)
- [ ] Import/load modules in REPL
- [ ] Debug mode (step through evaluation)
- [ ] Performance profiling
- [ ] Save/restore REPL sessions
- [ ] Scripting REPL commands

## Examples

### Basic Usage

```
$ nanorepl
Nanolang REPL v1.0.0
Type :help for help, :quit to exit

nano> (+ 2 3)
5

nano> let x: int = 42
nano> (* x 2)
84

nano> fn double(n: int) -> int { return (* n 2) }
nano> (double 21)
42
```

### Multi-line Functions

```
nano> fn factorial(n: int) -> int {
...     if (<= n 1) {
...         return 1
...     }
...     return (* n (factorial (- n 1)))
... }
nano> (factorial 5)
120
```

### Debugging

```
nano> :type (+ 2 3)
int

nano> :env
x: int = 42
double: (int) -> int
factorial: (int) -> int

nano> :load examples/nl_factorial.nano
Loaded: factorial

nano> :debug (factorial 5)
Call: factorial(5)
  Call: factorial(4)
    Call: factorial(3)
      Call: factorial(2)
        Call: factorial(1)
        Return: 1
      Return: 2
    Return: 6
  Return: 24
Return: 120
```

## Dependencies

- **GNU Readline**: For input, history, and completion
  - Ubuntu: `sudo apt-get install libreadline-dev`
  - macOS: `brew install readline`
  - Fedora: `sudo dnf install readline-devel`

- **Alternative**: linenoise (embedded, lighter weight)

## Challenges

### 1. Incomplete Expression Detection
**Solution**: Track paren/brace depth and string state

### 2. Error Recovery
**Solution**: Preserve environment on errors, show clear error messages

### 3. Multi-line Editing
**Solution**: Use readline's multi-line support or custom prompt

### 4. Import Resolution
**Solution**: Maintain module search paths in REPL environment

### 5. Performance
**Solution**: Keep environment persistent, only re-parse/eval new input

## Testing

### Unit Tests
- Expression completeness detection
- Command parsing
- Value printing
- History management

### Integration Tests
- Run REPL with scripted input
- Verify output matches expected
- Test special commands
- Test multi-line input

### Manual Testing
- Try various expressions
- Test edge cases (syntax errors, etc.)
- Verify readline functionality
- Check completion and history

## Implementation Roadmap

### Milestone 1: Basic REPL (1 week)
- Readline integration
- Single-line evaluation
- Persistent environment
- Basic error handling

### Milestone 2: Multi-line & Commands (1 week)
- Multi-line input detection
- Special commands (`:help`, `:quit`, etc.)
- Improved error messages
- History persistence

### Milestone 3: Completion & Polish (1 week)
- Tab completion
- Syntax highlighting (if terminal supports)
- Pretty-printing
- Documentation

### Milestone 4: Advanced Features (2 weeks)
- Module loading
- Debug mode
- Performance profiling
- Session save/restore

## Future Enhancements

1. **Jupyter Kernel**: Enable nanolang in Jupyter notebooks
2. **Web REPL**: Browser-based REPL via WASM
3. **Remote REPL**: Connect to running nanolang processes
4. **Plugin System**: Extend REPL with custom commands
5. **Visualization**: Graphical representation of data structures

## References

- [GNU Readline Manual](https://tiswww.cwru.edu/php/chet/readline/rltop.html)
- [Python REPL Implementation](https://github.com/python/cpython/tree/main/Modules/_io)
- [Racket REPL Design](https://docs.racket-lang.org/reference/repl.html)
- [Lisp REPL History](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)

## Related Issues

- `nanolang-kvz`: LSP implementation (shares completion logic)
- `nanolang-3yg`: Web playground (web-based REPL)
- Debugging infrastructure (can leverage REPL eval)

