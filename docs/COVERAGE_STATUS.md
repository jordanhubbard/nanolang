# Code Coverage Status

**Last Measured:** Not yet measured  
**Project Goal:** 80% line coverage by v1.0

## How to Measure Coverage

```bash
# Install lcov (if not already installed)
brew install lcov  # macOS
# or: sudo apt-get install lcov  # Ubuntu/Debian

# Generate coverage report
make coverage-report

# View report
open coverage/index.html
```

## Coverage Targets

| Component | Line Target | Status | Notes |
|-----------|-------------|--------|-------|
| **Lexer** | 90% | ⏳ Not measured | - |
| **Parser** | 85% | ⏳ Not measured | - |
| **Type Checker** | 85% | ⏳ Not measured | - |
| **Transpiler** | 80% | ⏳ Not measured | - |
| **Eval (Interpreter)** | 75% | ⏳ Not measured | - |
| **Runtime (GC)** | 90% | ⏳ Not measured | - |
| **Standard Library** | 95% | ⏳ Not measured | - |

## Next Steps

1. ✅ Install lcov (`brew install lcov`)
2. ⏳ Run initial coverage measurement
3. ⏳ Document baseline coverage
4. ⏳ Identify untested code paths
5. ⏳ Add tests to improve coverage
6. ⏳ Set up CI coverage tracking

## Documentation

See [COVERAGE.md](COVERAGE.md) for complete coverage guide.

---

**Updated:** January 25, 2026  
**Infrastructure:** ✅ Complete  
**Measurement:** ⏳ Pending first run
