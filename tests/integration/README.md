# Integration Tests

End-to-end tests that verify the entire compilation pipeline.

## Structure

Tests are organized by feature:
- `basic/` - Basic language features
- `control_flow/` - If/while/for loops
- `functions/` - Function definitions and calls
- `operators/` - Arithmetic, comparison, logical operators
- `types/` - Type system features
- `shadows/` - Shadow-test features

## Running

```bash
make test-integration
```

Each test should include:
1. A `.nano` source file
2. Expected output or behavior
3. Any edge cases

