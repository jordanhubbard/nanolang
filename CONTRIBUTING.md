# Contributing to Nanolang

Thank you for your interest in contributing to nanolang! This document provides guidelines and best practices for contributors.

## Quick Start

1. **Fork and clone** the repository
2. **Build the project**: `make`
3. **Run tests**: `make test`
4. **Build examples**: `make examples`

## Development Workflow

### 1. Interpreter-First Development

**Always develop with the interpreter before compiling:**

```bash
# Debug with interpreter (immediate feedback)
./bin/nano your_program.nano --trace-all

# Compile for production only after interpreter passes
./bin/nanoc your_program.nano -o program
```

### 2. Shadow Tests Required

**Every function MUST have a shadow test:**

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

### 3. Code Style

- **Prefix notation**: All operations use prefix: `(+ a b)`, not `a + b`
- **Explicit types**: Always annotate variable types: `let x: int = 42`
- **Mutable variables**: Use `mut` keyword: `let mut counter: int = 0`
- **Both branches required**: `if/else` must have both branches

See [.cursorrules](.cursorrules) for complete style guidelines.

## Project Organization

### File Structure

```
nanolang/
├── src/           # Compiler source (C)
├── src_nano/      # Self-hosted components (nanolang)
├── modules/       # Standard library modules
├── examples/      # Example programs
├── tests/         # Test suite
├── docs/          # User-facing documentation
└── planning/      # Future work and design docs
```

### Documentation Rules

Per [.cursorrules](.cursorrules):

- **Obsolete files**: Delete completed implementation summaries
- **Planning docs**: Keep in `planning/` directory
- **User docs**: Consolidate in `docs/` directory
- **Update index**: Always update `docs/DOCS_INDEX.md` when adding docs

## Making Changes

### Adding a Language Feature

1. **Update parser** (`src/parser.c`) to recognize new syntax
2. **Update typechecker** (`src/typechecker.c`) to validate semantics
3. **Update interpreter** (`src/eval.c`) for interpretation
4. **Update transpiler** (`src/transpiler.c`) for C code generation
5. **Add unit tests** in `tests/`
6. **Add examples** in `examples/`
7. **Update documentation** in `docs/`
8. **Update SPECIFICATION.md** with new feature

### Fixing a Bug

1. **Write a failing test** that demonstrates the bug
2. **Fix the bug** in the appropriate source file
3. **Verify test passes**: `make test`
4. **Add regression test** to prevent recurrence
5. **Document fix** in commit message

### Adding an Example

1. **Create example** in `examples/` directory
2. **Add shadow tests** to demonstrate correctness
3. **Test with interpreter**: `./bin/nano examples/your_example.nano`
4. **Test compilation**: `make examples`
5. **Verify zero warnings**: Examples must compile cleanly
6. **Update examples/Makefile** if needed
7. **Add comments** explaining complex algorithms

## Build System

### Makefile Targets

```bash
make              # Build compiler and interpreter
make examples     # Build all examples
make test         # Run test suite
make check        # Build + test
make clean        # Remove all build artifacts

make sanitize     # Build with memory sanitizers
make coverage     # Build with coverage instrumentation
make valgrind     # Run valgrind memory checks
```

### Zero Warnings Policy

**All code must compile without warnings:**

- Compiler: `-Wall -Wextra` enforced
- Examples: Must build cleanly
- No exceptions

## Testing

### Shadow Tests

- Run automatically during compilation
- Test every function at compile time
- Zero runtime overhead (stripped from builds)

### Unit Tests

```bash
./test.sh         # Run all unit tests
./bin/nano tests/unit/test_name.nano
```

### Manual Testing

```bash
# Test with tracing
./bin/nano examples/program.nano --trace-function=function_name

# Test compilation
./bin/nanoc examples/program.nano -o test_output
./test_output
```

## Commit Guidelines

### Commit Message Format

```
<type>: <subject>

<body>

Co-authored-by: Your Name <your.email@example.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build system or tooling changes
- `perf`: Performance improvements

### Before Committing

```bash
# Check status
git status

# Review changes
git diff

# Verify builds cleanly
make clean && make

# Run tests
make test

# Build examples
make examples
```

## Pull Request Process

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** following guidelines above
3. **Test thoroughly**: `make check`
4. **Commit with descriptive message**
5. **Push to your fork**: `git push origin feature/your-feature`
6. **Create pull request** against `main` branch
7. **Address review feedback**

### PR Checklist

- [ ] Code builds without warnings
- [ ] All tests pass
- [ ] New features have shadow tests
- [ ] Examples updated if needed
- [ ] Documentation updated
- [ ] `docs/DOCS_INDEX.md` updated if docs changed
- [ ] Commit messages follow format

## Getting Help

- **Documentation**: See `docs/` directory
- **Specification**: See `docs/SPECIFICATION.md`
- **Examples**: See `examples/` directory
- **Issues**: Check GitHub issues for known problems

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Questions?

Open an issue or discussion on GitHub.

---

**Thank you for contributing to nanolang!** 🚀
