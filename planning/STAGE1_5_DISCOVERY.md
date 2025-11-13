# Stage 1.5 Discovery: Transpiler Wrapper Issue

**Date:** November 13, 2025  
**Status:** Issue Identified

---

## Problem

Both Stage 0 and Stage 1.5 are failing to compile programs:

### Stage 0 (C Compiler)
- Shadow tests pass ✓
- C compilation fails ✗
- Binary not generated

### Stage 1.5 (Hybrid Compiler)
- Parsing fails with truncated function names
- Error: `Function 'add(a:' is missing a shadow test`
- Suggests tokenization or parsing issues

---

## Root Cause Analysis

The transpiler change that makes `main()` → `nl_main()` has broken compilation:

### Before (Working):
```c
// Generated C code
int main() {
    // user code
}
```

### After (Broken):
```c
// Generated C code
int64_t nl_main() {
    // user code  
}

// Wrapper (added by transpiler)
int main() {
    return nl_main();
}
```

**Problem:** The `gcc` compilation command expects a `main` function with `int` return type, but the generated wrapper may have issues, or the C runtime linking is failing.

---

## Evidence

1. **Shadow tests pass:** Interpreter works correctly
2. **"C compilation failed":** The gcc step is failing
3. **No error messages:** gcc is being silenced (`2>/dev/null`)

---

## Solution Strategy

1. **Remove `2>/dev/null`** from gcc command to see actual errors
2. **Check generated C code** to verify wrapper is correct
3. **Test Stage 0 first** - if it's broken, fix it before Stage 1.5
4. **Consider:** Revert the `main()` → `nl_main()` change for now

---

## Next Steps

1. Check generated C code for simple example
2. Run gcc with visible errors
3. Fix C generation or revert changes
4. Validate Stage 0 works again
5. Then proceed with Stage 1.5 testing

---

**Status:** Debugging in progress

