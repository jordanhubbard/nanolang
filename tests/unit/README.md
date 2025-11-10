# Unit Tests

Unit tests for individual compiler components.

## Structure

- `lexer/` - Lexer tests
- `parser/` - Parser tests  
- `typechecker/` - Type checker tests
- `transpiler/` - C transpiler tests
- `eval/` - Interpreter/evaluator tests

## Running

These tests verify that each component works correctly in isolation.

```bash
# Run all unit tests
make test-unit

# Run specific component tests
make test-lexer
make test-parser
```

