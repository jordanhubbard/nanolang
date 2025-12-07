# Known Issues

## Critical: Parser Failure on ARM64 (aarch64) Fresh Clone

**Status:** 🔴 **CRITICAL BUG** - Blocks all usage on ARM64 systems  
**Affects:** ARM64/aarch64 Linux systems (tested on Ubuntu 24.04)  
**First Reported:** December 6, 2025  
**System:** sparky.local - Ubuntu 24.04, GCC 13.3.0, aarch64

### Symptoms

After a completely fresh `git clone`, the build completes successfully but the resulting compiler **cannot parse any programs**, even trivial ones:

```bash
$ git clone git@github.com:jordanhubbard/nanolang.git
$ cd nanolang
$ make
# Build succeeds - Stage 1 complete
$ ./bin/nanoc examples/factorial.nano -o test
Error: Parser reached invalid state
Error: Program must define a 'main' function
Type checking failed
```

### Test Results

| Test | Result |
|------|--------|
| Fresh clone + build | ✅ Succeeds |
| `./bin/nanoc --version` | ✅ Works |
| Simple lexer test | ✅ Returns 11 tokens correctly |
| Parse `fn main() -> int { return 0 }` | ❌ "Parser reached invalid state" |
| Any .nano file | ❌ All fail with same error |
| Interpreter mode | ❌ Also fails (uses same parser) |
| AddressSanitizer build | ℹ️ No memory errors detected |

### What Works

- ✅ Compilation succeeds (no compiler errors)
- ✅ Binary runs and can print version
- ✅ Lexer tokenizes input correctly
- ✅ **Works perfectly on x86_64 macOS and Linux**

### What's Broken

- ❌ Parser immediately returns NULL
- ❌ Error message: "Parser reached invalid state"
- ❌ Affects ALL programs (even single-function examples)
- ❌ Both compiler AND interpreter broken

### Investigation Findings

**Not caused by:**
- Memory corruption (AddressSanitizer clean)
- File encoding issues (files are valid ASCII)
- Lexer bugs (tokenization works correctly)
- Module/dependency issues
- File reading problems

**Likely causes:**
1. **Uninitialized global/static variable** that behaves differently on ARM64 vs x86_64
2. **Struct alignment/padding differences** between architectures
3. **Undefined behavior** that manifests only on ARM (compiler-specific)
4. **Pointer size assumption** (int vs long vs pointer)

### Architecture Comparison

| Platform | Arch | Result |
|----------|------|--------|
| macOS (dev machine) | x86_64 | ✅ Works perfectly |
| Ubuntu 22.04 (ubuntu.local) | x86_64 | ✅ Works perfectly |
| **Ubuntu 24.04 (sparky.local)** | **aarch64** | ❌ **Completely broken** |

### Debugging Steps Taken

1. ✅ Tested lexer directly - works fine (returns 11 tokens for test input)
2. ✅ Checked file contents - valid ASCII, proper format
3. ✅ Built with AddressSanitizer - no memory errors
4. ✅ Tested with simplest possible program - still fails
5. ✅ Checked both compiler and interpreter - both broken
6. ✅ Verified binary can execute (--version works)

### Code Locations to Investigate

**Parser (src/parser.c):**
- Line 134, 149, 170, 209, 225, 240, etc. - "Parser reached invalid state (NULL token)"
- The parser is receiving NULL tokens from somewhere
- Global parser state initialization?

**Lexer (src/lexer.c):**
- Lexer works in isolation but may have issues when called from parser
- Check token array allocation and initialization

**Potential Issues:**
```c
// Look for uninitialized statics:
static ParserState state;  // Might not be zero-initialized on ARM?

// Look for int/pointer confusion:
int index = (int)pointer;  // Undefined on 64-bit ARM

// Look for struct padding assumptions:
struct Token {
    int type;      // 4 bytes
    char *value;   // 8 bytes on 64-bit
    // Padding differences on ARM?
};
```

### Temporary Workaround

**None available.** The compiler is completely non-functional on ARM64.

Users with ARM64 systems cannot currently use nanolang at all.

### Priority & Impact

**Priority:** 🔴 **P0 - Critical**  
**Impact:** Blocks all ARM64 users (Raspberry Pi, ARM servers, Apple Silicon via Rosetta workarounds)  
**Urgency:** High - Growing ARM64 adoption makes this a major blocker

### Next Steps for Fix

1. **Add extensive debug logging** to parser to see where NULL tokens originate
2. **Check all static/global variable initialization** in parser.c and lexer.c
3. **Review Token struct definition** for ARM64 alignment issues
4. **Search for pointer-to-int casts** that break on ARM64
5. **Compare objdump** of working (x86_64) vs broken (aarch64) binaries
6. **Test with -O0** (no optimization) to rule out compiler optimization bugs
7. **Add ARM64 CI testing** to prevent future regressions

### How to Help

If you have access to ARM64 Linux:
1. Add printf debugging to src/parser.c around lines 134-150
2. Print token count, first token type, parser state
3. Check if tokens array is NULL or if individual tokens are NULL
4. Report findings in GitHub issue

### Related Issues

- Bootstrap sentinel file bug (FIXED) - was masking this issue
- Self-hosted components fail (EXPECTED) - unrelated to this bug

---

**Last Updated:** December 6, 2025  
**Test System:** sparky.local (Ubuntu 24.04, GCC 13.3.0, aarch64)
