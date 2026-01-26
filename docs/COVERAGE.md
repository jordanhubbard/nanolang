# Code Coverage Guide

Guide to generating and analyzing code coverage reports for NanoLang.

## Overview

NanoLang uses **gcov** and **lcov** for code coverage analysis. This helps identify:
- Which lines of code are executed by tests
- Which branches are taken
- Which functions are called
- Untested code paths

**Coverage Metrics:**
- **Line Coverage** - Percentage of lines executed
- **Function Coverage** - Percentage of functions called
- **Branch Coverage** - Percentage of branches taken

---

## Prerequisites

### Install Coverage Tools

**macOS (Homebrew):**
```bash
brew install lcov
```

**Ubuntu/Debian:**
```bash
sudo apt-get install lcov
```

**Fedora/RHEL:**
```bash
sudo dnf install lcov
```

**FreeBSD:**
```bash
sudo pkg install lcov
```

### Verify Installation

```bash
lcov --version
genhtml --version
```

---

## Generating Coverage Reports

### Quick Coverage Report

Generate and view coverage in one command:

```bash
make coverage-report
```

This will:
1. Clean previous build artifacts
2. Rebuild with coverage instrumentation (`-fprofile-arcs -ftest-coverage`)
3. Run the full test suite
4. Generate coverage data (`.gcda` files)
5. Process coverage with lcov
6. Generate HTML report in `coverage/` directory

### View the Report

```bash
open coverage/index.html        # macOS
xdg-open coverage/index.html    # Linux
firefox coverage/index.html     # Manual
```

---

## Step-by-Step Process

### 1. Build with Coverage

Build the compiler with coverage instrumentation:

```bash
make coverage
```

This adds these flags:
- `-fprofile-arcs` - Instrument for arc profiling
- `-ftest-coverage` - Instrument for coverage testing

### 2. Run Tests

Execute the test suite to generate coverage data:

```bash
make test
```

This creates `.gcda` files (coverage counters) next to each `.o` object file.

### 3. Generate Coverage Info

Process coverage data with lcov:

```bash
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
```

### 4. Generate HTML Report

Create browsable HTML report:

```bash
genhtml coverage.info --output-directory coverage
```

---

## Interpreting Results

### Coverage Summary

The main page (`coverage/index.html`) shows:

```
Directory         Line Coverage    Function Coverage    Branch Coverage
src/              85.3% (2134/2503)  92.1% (147/160)      73.4% (892/1215)
src/runtime/      91.2% (456/500)    95.0% (38/40)        82.1% (123/150)
```

**Color Coding:**
- 🟢 **Green (≥ 90%)** - Excellent coverage
- 🟡 **Yellow (75-89%)** - Good coverage, room for improvement
- 🔴 **Red (< 75%)** - Poor coverage, needs attention

### Line-by-Line View

Click on a file to see line-by-line coverage:

```c
  1:   void my_function(int x) {
  2:       if (x > 0) {            // Executed 15 times
  3:           do_something();     // Executed 15 times
  4:       } else {
  5:   #####   do_other();         // NEVER executed!
  6:       }
  7:   }
```

**Line Annotations:**
- `42:` - Line executed 42 times
- `#####` - Line never executed (uncovered)
- `=====` - Line not instrumentable (comments, etc.)

### Branch Coverage

Branches show which paths were taken:

```c
if (condition)     Branch 0: taken 10 times
                   Branch 1: taken 5 times
```

---

## Coverage Targets

### Project Coverage Goals

**Current Status:** Unknown (coverage not yet measured)

**Target for v1.0:**
- ✅ **Line Coverage:** ≥ 80%
- ✅ **Function Coverage:** ≥ 85%
- ✅ **Branch Coverage:** ≥ 70%

### Per-Component Goals

| Component | Line Target | Function Target | Branch Target |
|-----------|-------------|-----------------|---------------|
| Lexer | 90% | 95% | 80% |
| Parser | 85% | 90% | 75% |
| Type Checker | 85% | 90% | 75% |
| Transpiler | 80% | 85% | 70% |
| Eval (Interpreter) | 75% | 80% | 65% |
| Runtime (GC, arrays) | 90% | 95% | 80% |
| Standard Library | 95% | 100% | 85% |

---

## Improving Coverage

### Finding Untested Code

1. **Generate report:** `make coverage-report`
2. **Sort by coverage:** Click "Line Coverage" column in report
3. **Focus on red files:** Files with < 75% coverage
4. **Review uncovered lines:** Look for `#####` lines

### Adding Tests

For uncovered lines in `src/parser.c:450`:

```c
450: #####   if (tok->type == TOKEN_UNEXPECTED) {
451: #####       error("Unexpected token");
452: #####   }
```

Add a negative test:

```nano
# tests/negative/syntax_errors/unexpected_token.nano
fn main() -> int {
    let x: int = @@@  # Unexpected token!
    return 0
}
```

Then re-run coverage:
```bash
make coverage-report
```

---

## Filtering Coverage

### Exclude System Headers

Already done automatically:

```bash
lcov --remove coverage.info '/usr/*' --output-file coverage.info
```

### Exclude Specific Files

To exclude generated code or test files:

```bash
lcov --remove coverage.info \
    '*/cJSON.c' \
    '*/test_*.c' \
    --output-file coverage.info
```

### Include Only Specific Directories

Focus on core compiler:

```bash
lcov --extract coverage.info \
    '*/src/lexer.c' \
    '*/src/parser.c' \
    '*/src/typechecker.c' \
    '*/src/transpiler.c' \
    --output-file coverage_core.info

genhtml coverage_core.info --output-directory coverage_core
```

---

## CI Integration

### GitHub Actions (Planned)

```yaml
# .github/workflows/coverage.yml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: sudo apt-get install -y lcov
      
      - name: Generate coverage
        run: make coverage-report
      
      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.info
          fail_ci_if_error: true
      
      - name: Check coverage thresholds
        run: |
          COVERAGE=$(lcov --summary coverage.info | grep lines | awk '{print $2}' | sed 's/%//')
          if [ $(echo "$COVERAGE < 80" | bc) -eq 1 ]; then
            echo "Coverage $COVERAGE% is below 80% threshold"
            exit 1
          fi
```

### Coverage Badges

Once CI is set up, add badge to README.md:

```markdown
[![Coverage](https://codecov.io/gh/USERNAME/nanolang/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/nanolang)
```

---

## Troubleshooting

### "lcov: command not found"

**Solution:** Install lcov (see [Prerequisites](#prerequisites))

### "No coverage data found"

**Cause:** Tests not run after coverage build

**Solution:**
```bash
make coverage  # Build with instrumentation
make test      # Run tests to generate .gcda files
make coverage-report  # Process coverage
```

### "Permission denied: coverage.info"

**Solution:**
```bash
chmod +w coverage.info
```

Or clean and rebuild:
```bash
make clean
make coverage-report
```

### Old Coverage Data

**Symptom:** Coverage report shows old data

**Solution:** Clean before rebuilding
```bash
make clean
make coverage-report
```

### Coverage Files in Git

Add to `.gitignore`:
```
*.gcda
*.gcno
*.gcov
coverage.info
coverage/
```

---

## Best Practices

### 1. Measure Regularly

Run coverage weekly or after major changes:

```bash
make coverage-report
```

### 2. Set Coverage Gates

Don't let coverage decrease:

```bash
# Record baseline
BASELINE=$(lcov --summary coverage.info | grep lines | awk '{print $2}')
echo "Baseline: $BASELINE"

# After changes, check
NEW=$(lcov --summary coverage.info | grep lines | awk '{print $2}')
if [ "$NEW" -lt "$BASELINE" ]; then
  echo "Coverage decreased from $BASELINE to $NEW!"
  exit 1
fi
```

### 3. Test Edge Cases

Focus on:
- Error paths (parser errors, type errors)
- Boundary conditions (empty arrays, NULL pointers)
- Unusual input (negative numbers, very large values)

### 4. Don't Chase 100%

**Realistic goals:**
- Core compiler: 85%
- Runtime: 90%
- Examples: Not required

**Unrealistic:**
- 100% line coverage (some code is unreachable)
- 100% branch coverage (defensive checks may never trigger)

### 5. Review Uncovered Code

Ask for each uncovered line:
1. **Is it dead code?** → Delete it
2. **Is it an error path?** → Add negative test
3. **Is it unreachable?** → Add assertion or comment
4. **Is it hard to test?** → Refactor

---

## Advanced Topics

### Differential Coverage

Show coverage only for changed lines:

```bash
# Get changed files
git diff --name-only HEAD^ HEAD > changed_files.txt

# Filter coverage
lcov --extract coverage.info $(cat changed_files.txt) \
  --output-file coverage_diff.info

genhtml coverage_diff.info --output-directory coverage_diff
```

### Merging Coverage

Combine coverage from multiple test runs:

```bash
# Run unit tests
make coverage
make test-unit
lcov --capture --directory . --output-file coverage_unit.info

# Run integration tests
make test-integration
lcov --capture --directory . --output-file coverage_integration.info

# Merge
lcov --add-tracefile coverage_unit.info \
     --add-tracefile coverage_integration.info \
     --output-file coverage_total.info

genhtml coverage_total.info --output-directory coverage_total
```

### Coverage per Feature

Track coverage by feature:

```bash
# Test only lexer
make coverage
./bin/nanoc --test-lexer
lcov --capture --directory . --output-file coverage_lexer.info

# Test only parser
make clean && make coverage
./bin/nanoc --test-parser
lcov --capture --directory . --output-file coverage_parser.info
```

---

## Related Documentation

- [Testing Guide](SHADOW_TESTS.md) - How to write tests
- [Debugging Guide](DEBUGGING_GUIDE.md) - Debugging techniques
- [Contributing Guide](../CONTRIBUTING.md) - Contribution guidelines

---

**Last Updated:** January 25, 2026
**Status:** Infrastructure complete, measurement pending
**Version:** 0.2.0+
