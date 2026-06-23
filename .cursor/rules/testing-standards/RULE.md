---
description: "Always Green testing philosophy - 100% passing tests required at all times"
alwaysApply: true
---

# Testing Standards - Always Green 🟢

## Core Principle: Always Green

This project maintains 100% passing tests. **Any commit that breaks tests is unacceptable.**

## Self-Hosted Parser Requirements

The self-hosted parser (`src_nano/parser_mvp.nano`) must:

1. ✅ **Compile successfully** with zero errors
2. ✅ **Pass all shadow tests** (internal validation)
3. ✅ **Parse itself** (self-hosting validation)
4. ✅ **Support all features** it uses in its own code

## Test Requirements

### All Code Changes Must:
- ✅ Pass compilation (zero errors)
- ✅ Pass all existing tests
- ✅ Not break self-hosting
- ✅ Include tests for new features
- ✅ Update documentation

### Running Tests

```bash
# Quick check
./bin/nanoc src_nano/parser_mvp.nano

# Full test suite
./tests/run_all_tests.sh

# Examples
make examples
```

### Self-Hosting Check

Before committing parser changes:

```bash
# Must succeed with zero errors
./bin/nanoc src_nano/parser_mvp.nano

# Must show PASSED
./tests/run_all_tests.sh | grep "All runnable tests passed"
```

### Test Coverage Standards

- **Unit tests:** Must pass 100%
- **Integration tests:** Must pass 100%
- **Shadow tests:** Must pass 100%
- **Self-hosting:** Must validate 100%

## Emergency Procedures

### If Tests Break

1. **DO NOT** commit broken code
2. **DO NOT** merge to main
3. Fix immediately or revert
4. Investigate root cause
5. Add tests to prevent recurrence

### If Self-Hosting Breaks

**CRITICAL:** This is a blocker. Must fix before any other work.

1. Identify which feature broke it
2. Check if parser uses that feature
3. Fix or revert immediately
4. Add test to catch this pattern

## Success Metrics

### Green Status Indicators

- ✅ All tests passing
- ✅ Self-hosting works
- ✅ Zero compilation errors
- ✅ Zero warnings
- ✅ Examples build
- ✅ Documentation current

### Quality Metrics

- Test pass rate: 100%
- Coverage: >95%
- Self-hosting: Validated
- Performance: < 10s compile time

## Remember

**"Always Green" means ALWAYS GREEN.**

No exceptions. No "temporary breaks". No "will fix tomorrow".

If it breaks tests, it doesn't get committed.  
If it breaks self-hosting, it's a critical bug.  
If examples fail, it's blocking.

Keep nanolang green! 🟢
