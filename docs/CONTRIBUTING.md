# Contributing to me

I appreciate your interest in contributing. I am a living project, and this document defines how you can interact with me.

## Philosophy

I am designed with these convictions:
- **Minimal**: I keep my feature set small and focused.
- **Unambiguous**: Every construct I offer has exactly one meaning.
- **LLM-friendly**: I optimize for machine understanding and generation.
- **Test-driven**: I require shadow tests for all code.
- **Self-documenting**: I expect code to be clear without excessive comments.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists.
2. Create a new issue with:
   - A clear description.
   - Steps to reproduce, for bugs.
   - Expected vs actual behavior.
   - My version.

### Proposing Features

Before you propose new features, consider:

1. **Does it fit my philosophy?** Is it minimal and unambiguous?
2. **Is it necessary?** Can my existing features accomplish this?
3. **Is it LLM-friendly?** Will machines understand and use it correctly?

Feature proposals should include:
- Use cases.
- Example syntax.
- How it maintains my simplicity.
- Shadow test examples.

### Code Contributions

#### Writing my Code

When you write example programs:

1. **Follow my spec** - Adhere to `SPECIFICATION.md`.
2. **Include shadow tests** - Every function you write needs tests.
3. **Test edge cases** - Cover 0, negatives, and boundaries.
4. **Comment sparingly** - Your code should be self-explanatory.
5. **Use consistent notation** - Both prefix `(+ a b)` and infix `a + b` are valid. Prefer whichever is clearest. All my infix operators have equal precedence (left-to-right). Use parentheses to group: `a * (b + c)`.

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

When you implement my compiler or interpreter:

1. **Keep it simple** - I favor clarity over cleverness.
2. **Test thoroughly** - Write tests for all components.
3. **Document decisions** - Explain non-obvious choices.
4. **Follow conventions** - Match my existing code style.

#### Pull Request Process

1. **Fork my repository.**
2. **Create a feature branch** - `git checkout -b feature/my-feature`.
3. **Make your changes** - Keep commits focused and atomic.
4. **Write tests** - Test your changes thoroughly.
5. **Update documentation** - Keep my docs in sync with my code.
6. **Submit PR** - Provide a clear description of changes.

### Commit Messages

Use clear, descriptive commit messages:

```
Add shadow-test validation to parser

- Check that every function has a shadow block
- Verify shadow block is defined after function
- Add error messages for missing tests
```

Format:
- First line: Brief summary (50 characters or less).
- Blank line.
- Detailed explanation, if needed.

## Development Setup

### Prerequisites

(I will add this when implementation begins)

### Building

(I will add this when implementation begins)

### Testing

(I will add this when implementation begins)

## Code Style

### my Code

- Use 4 spaces for indentation.
- Function names: `snake_case`.
- Types: `lowercase`.
- Constants: `UPPER_CASE`.
- Keep lines under 80 characters when reasonable.

### Implementation Code

(I will define this based on my implementation language)

## Testing Guidelines

### Writing Shadow Tests

Good shadow tests:

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

Shadow tests should:
- Cover normal operation.
- Test edge cases (0, negative, empty).
- Test boundary conditions.
- Be clear and self-documenting.
- Not be overly exhaustive. I value the balance between coverage and clarity.

## Documentation

### What to Document

- **Language features** - In `SPECIFICATION.md`.
- **Examples** - In `examples/` with README.
- **Getting started** - In `GETTING_STARTED.md`.
- **Implementation** - Code comments for non-obvious logic.

### Documentation Style

- Be concise but complete.
- Use examples.
- Keep it LLM-friendly.
- Update when code changes.

## Questions?

If you have questions about contributing:

1. Check my existing documentation.
2. Look at my example code.
3. Open an issue for discussion.

## Code of Conduct

### My Standards

- Be respectful and inclusive.
- Welcome newcomers.
- Assume good intentions.
- Focus on what is best for me.

### Unacceptable Behavior

- Harassment or discrimination.
- Trolling or inflammatory comments.
- Personal attacks.
- Publishing private information.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

I recognize my contributors in:
- My GitHub contributor list.
- My release notes, for significant contributions.
- My documentation, for major features.

I value your time. If you follow my rules, your contributions will become part of me.

