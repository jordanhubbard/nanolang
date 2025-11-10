# Negative Tests

Tests that should fail compilation with appropriate error messages.

## Categories

- `type_errors/` - Type mismatches and violations
- `syntax_errors/` - Parse errors
- `missing_shadows/` - Functions without shadow tests
- `undefined_vars/` - Undeclared variable usage
- `return_errors/` - Missing or incorrect returns

## Format

Each test file should:
1. Have a descriptive name: `type_mismatch_add_string_int.nano`
2. Include a comment describing expected error
3. Be verifiable by the test runner

Example:
```nano
# Expected error: Type mismatch in addition
# Cannot add int and string

fn bad_add(x: int, y: string) -> int {
    return (+ x y)
}

shadow bad_add {
    assert (== (bad_add 5 "hello") 5)
}
```

## Running

```bash
make test-negative
```

The test runner verifies these files fail compilation with the correct error.

