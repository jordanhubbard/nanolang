# Contributing to nanolang

Thank you for your interest in contributing to nanolang! This document provides guidelines for contributing to the project.

## Philosophy

nanolang is designed to be:
- **Minimal**: Keep the feature set small and focused
- **Unambiguous**: Every construct should have exactly one meaning
- **LLM-friendly**: Optimize for AI understanding and generation
- **Test-driven**: All code must have shadow-tests
- **Self-documenting**: Code should be clear without excessive comments

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - nanolang version (once releases exist)

### Proposing Features

Before proposing new features, consider:

1. **Does it fit the philosophy?** - Is it minimal and unambiguous?
2. **Is it necessary?** - Can existing features accomplish this?
3. **Is it LLM-friendly?** - Will LLMs understand and use it correctly?

Feature proposals should include:
- Use cases
- Example syntax
- How it maintains language simplicity
- Shadow-test examples

### Code Contributions

#### Writing nanolang Code

When writing example programs:

1. **Follow the spec** - Adhere to `SPECIFICATION.md`
2. **Include shadow-tests** - Every function needs tests
3. **Test edge cases** - Cover 0, negatives, boundaries
4. **Comment sparingly** - Code should be self-explanatory
5. **Use consistent notation** - Both prefix `(+ a b)` and infix `a + b` are valid; prefer whichever is clearest. Note: all infix operators have equal precedence (left-to-right, no PEMDAS), so use parentheses to group: `a * (b + c)`

Example:
```nano
fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    } else {
        return (gcd b (% a b))
    }
}

shadow gcd {
    assert (== (gcd 48 18) 6)
    assert (== (gcd 100 10) 10)
    assert (== (gcd 7 13) 1)
}
```

#### Writing Implementation Code

When implementing the compiler/interpreter:

1. **Keep it simple** - Favor clarity over cleverness
2. **Test thoroughly** - Write tests for all components
3. **Document decisions** - Explain non-obvious choices
4. **Follow conventions** - Match existing code style

#### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch** - `git checkout -b feature/my-feature`
3. **Make your changes** - Keep commits focused and atomic
4. **Write tests** - Test your changes thoroughly
5. **Update documentation** - Keep docs in sync with code
6. **Submit PR** - Provide clear description of changes

### Commit Messages

Use clear, descriptive commit messages:

```
Add shadow-test validation to parser

- Check that every function has a shadow block
- Verify shadow block is defined after function
- Add error messages for missing tests
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed explanation (if needed)

## Development Setup

### Prerequisites

(To be added when implementation begins)

### Building

(To be added when implementation begins)

### Testing

(To be added when implementation begins)

## Code Style

### nanolang Code

- Use 4 spaces for indentation
- Function names: `snake_case`
- Types: `lowercase`
- Constants: `UPPER_CASE` (when added to language)
- Keep lines under 80 characters when reasonable

### Implementation Code

(To be defined based on implementation language)

## Testing Guidelines

### Writing Shadow-Tests

Good shadow-tests:

```nano
shadow my_function {
    # Test normal cases
    assert (== (my_function 10) 20)
    
    # Test edge cases
    assert (== (my_function 0) 0)
    assert (== (my_function -5) -10)
    
    # Test boundaries
    assert (== (my_function 1) 2)
}
```

Shadow-tests should:
- Cover normal operation
- Test edge cases (0, negative, empty)
- Test boundary conditions
- Be clear and self-documenting
- Not be overly exhaustive (balance coverage vs. clarity)

## Documentation

### What to Document

- **Language features** - In `SPECIFICATION.md`
- **Examples** - In `examples/` with README
- **Getting started** - In `GETTING_STARTED.md`
- **Implementation** - Code comments for non-obvious logic

### Documentation Style

- Be concise but complete
- Use examples liberally
- Keep it LLM-friendly (clear structure, explicit)
- Update when code changes

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Look at example code
3. Open an issue for discussion

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Assume good intentions
- Focus on what's best for the project

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Publishing private information

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

Contributors will be recognized in:
- GitHub contributor list
- Release notes (for significant contributions)
- Project documentation (for major features)

Thank you for helping make nanolang better! 🚀
