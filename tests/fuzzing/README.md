# NanoLang Fuzzing Infrastructure

This directory contains fuzzing targets for testing NanoLang compiler components with random/malformed inputs to find crashes, hangs, and memory errors.

## Overview

Fuzzing is an automated testing technique that generates random or mutated inputs to find bugs. We provide fuzzer targets for:

- **Lexer** (`fuzz_lexer.c`) - Tests tokenization of arbitrary source code
- **Parser** (`fuzz_parser.c`) - Tests parsing of tokenized input

## Prerequisites

Choose one fuzzing engine:

### Option 1: libFuzzer (Fast, modern)

```bash
# macOS with Homebrew
brew install llvm
# Note: Use full path to clang, e.g. /opt/homebrew/opt/llvm/bin/clang

# Ubuntu/Debian
sudo apt-get install clang-15
# Note: libFuzzer support may require building clang from source
```

**Note**: libFuzzer requires a clang build with fuzzer support. If you encounter linker errors about `libclang_rt.fuzzer_osx.a`, either use AFL++ (Option 2) or build clang from source with fuzzer support enabled.

### Option 2: AFL++ (Industry standard)

```bash
# macOS with Homebrew
brew install afl++

# Ubuntu/Debian
sudo apt-get install afl++
```

## Building Fuzzers

### With libFuzzer

```bash
# Build lexer fuzzer
clang -g -O1 -fsanitize=fuzzer,address \
  -I../../src \
  fuzz_lexer.c ../../src/lexer.c \
  -o fuzz_lexer

# Build parser fuzzer (requires more source files)
clang -g -O1 -fsanitize=fuzzer,address \
  -I../../src \
  fuzz_parser.c ../../src/lexer.c ../../src/parser.c ../../src/nanolang.c \
  -o fuzz_parser
```

### With AFL++

```bash
# Build lexer fuzzer
afl-clang-fast -g -O1 -fsanitize=address \
  -I../../src \
  fuzz_lexer.c ../../src/lexer.c \
  -o fuzz_lexer

# Build parser fuzzer
afl-clang-fast -g -O1 -fsanitize=address \
  -I../../src \
  fuzz_parser.c ../../src/lexer.c ../../src/parser.c ../../src/nanolang.c \
  -o fuzz_parser
```

## Running Fuzzers

### With libFuzzer

```bash
# Run lexer fuzzer (Ctrl+C to stop)
./fuzz_lexer corpus_lexer/ -max_len=10000 -timeout=1

# Run parser fuzzer
./fuzz_parser corpus_parser/ -max_len=10000 -timeout=1

# Run for a specific duration (e.g., 60 seconds)
./fuzz_lexer corpus_lexer/ -max_total_time=60

# Parallel fuzzing (use all CPU cores)
./fuzz_lexer corpus_lexer/ -jobs=$(nproc) -workers=$(nproc)
```

### With AFL++

```bash
# Run lexer fuzzer
afl-fuzz -i corpus_lexer -o findings_lexer ./fuzz_lexer

# Run parser fuzzer
afl-fuzz -i corpus_parser -o findings_parser ./fuzz_parser

# Parallel fuzzing (use multiple cores)
# Terminal 1 (main fuzzer)
afl-fuzz -i corpus_lexer -o findings_lexer -M fuzzer1 ./fuzz_lexer

# Terminal 2 (secondary fuzzer)
afl-fuzz -i corpus_lexer -o findings_lexer -S fuzzer2 ./fuzz_lexer
```

## Seed Corpus

The `corpus_lexer/` and `corpus_parser/` directories contain seed inputs:

- `seed1_function.nano` - Simple function definition
- `seed2_struct.nano` - Struct declaration
- `seed3_string.nano` - String literal handling

These seeds help the fuzzer start with valid input patterns and mutate from there.

## Interpreting Results

### libFuzzer

- Saves crashing inputs to files like `crash-<hash>`
- Shows statistics: executions/sec, coverage, corpus size
- Crashes include stack trace and input details

### AFL++

- Saves findings in `findings_<target>/crashes/` and `findings_<target>/hangs/`
- Shows fuzzing statistics in terminal UI
- Check `findings_<target>/crashes/` for crashing inputs

## Reproducing Crashes

```bash
# Test a crash with lexer
./fuzz_lexer crash-a1b2c3d4

# Test a crash with parser
./fuzz_parser findings_parser/crashes/id:000000,sig:06,src:000000
```

## Adding New Fuzzer Targets

To fuzz additional components (type checker, evaluator, etc.):

1. Create `fuzz_<component>.c` following the pattern in `fuzz_lexer.c`
2. Include necessary headers and link required source files
3. Implement `LLVMFuzzerTestOneInput()` for libFuzzer
4. Add AFL++ support with `__AFL_FUZZ_TESTCASE_LEN` macro
5. Create seed corpus in `corpus_<component>/`
6. Document build and run instructions

## Continuous Fuzzing

For long-term fuzzing campaigns:

```bash
# Run fuzzer overnight/weekend
nohup ./fuzz_lexer corpus_lexer/ -max_total_time=86400 &> fuzz_lexer.log &

# Monitor progress
tail -f fuzz_lexer.log
```

## CI Integration

To integrate fuzzing into CI (GitHub Actions):

```yaml
- name: Fuzz lexer (short run)
  run: |
    cd tests/fuzzing
    clang -g -O1 -fsanitize=fuzzer,address -I../../src \
      fuzz_lexer.c ../../src/lexer.c -o fuzz_lexer
    ./fuzz_lexer corpus_lexer/ -max_total_time=60 -max_len=10000
```

## Best Practices

1. **Start with short runs** - Run for 1-5 minutes to catch low-hanging fruit
2. **Use sanitizers** - Always compile with `-fsanitize=address` to catch memory errors
3. **Minimize crashing inputs** - Use `afl-tmin` or libFuzzer's `-minimize_crash=1`
4. **Expand corpus** - Add interesting test cases to corpus directories
5. **Run regularly** - Integrate into CI or run periodic fuzzing campaigns
6. **Fix bugs promptly** - Don't let fuzzer findings accumulate

## Troubleshooting

**"ERROR: libFuzzer requires -fsanitize=fuzzer"**
- Use clang (not gcc) and add `-fsanitize=fuzzer`

**"AFL++ not found"**
- Install AFL++: `brew install afl++` or `apt-get install afl++`

**Slow fuzzing speed (<100 exec/sec)**
- Reduce input size with `-max_len=1000`
- Check for slow operations in fuzzed code
- Use release build flags `-O2` or `-O3`

**Out of memory**
- Limit corpus size or use `-rss_limit_mb=2048`
- Reduce `-max_len` parameter

## References

- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [AFL++ Documentation](https://github.com/AFLplusplus/AFLplusplus)
- [Fuzzing Book](https://www.fuzzingbook.org/)
- [OWASP Fuzzing](https://owasp.org/www-community/Fuzzing)

## Contributing

Found a bug via fuzzing? Please:

1. Minimize the crashing input
2. Create a negative test in `tests/negative/`
3. Open an issue with the stack trace and input
4. Submit a PR with the fix and test

Happy fuzzing! 🐛🔨
