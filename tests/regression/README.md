# Regression Tests

Tests for previously fixed bugs to prevent regressions.

## Naming Convention

Use format: `issue_NNN_description.nano` or `bug_YYYY_MM_DD_description.nano`

Example: `bug_2025_09_30_for_loop_segfault.nano`

## Documentation

Each test should include:
1. Description of the original bug
2. When it was fixed
3. Minimal reproduction case
4. Expected behavior

## Running

```bash
make test-regression
```

These tests are run on every commit via CI to ensure bugs don't resurface.

