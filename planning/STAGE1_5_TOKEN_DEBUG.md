# Stage 1.5 Token Bridge Debugging

**Date:** November 13, 2025  
**Status:** 🔍 Investigating Token Conversion

---

## Problem

Stage 1.5 builds successfully, but parser fails with errors at every token position:
```
Error at line 4, column 1: Expected struct, enum, extern, function or shadow-test definition
Error at line 4, column 4: Expected struct, enum, extern, function or shadow-test definition
...
```

This suggests the parser is receiving malformed tokens from the nanolang lexer.

---

## Hypothesis

**Token Bridge Issue:** The conversion from `List_token*` (nanolang) to `Token*` (C parser) may be incorrect.

Possible causes:
1. Token types don't match between nanolang and C
2. Token values are corrupted during conversion
3. Token list is empty or NULL
4. Memory layout issues with struct Token

---

## Investigation Steps

1. ✅ Verify `nl_tokenize` is exported from `lexer_nano.o`
2. Call `nl_tokenize` directly to check if it produces tokens
3. Compare token output: C lexer vs nanolang lexer
4. Check token struct layout matches between C and nanolang

---

## Findings

### Build Status
- ✅ Stage 1.5 builds without errors
- ✅ No duplicate `main` symbol (fixed with sed)
- ✅ All object files link successfully

### Token Bridge Code
```c
extern List_token* nl_tokenize(const char* source);

Token *tokenize_nano(const char *source, int *token_count) {
    List_token *token_list = nl_tokenize(source);
    // Convert list to array...
}
```

---

## Next Steps

1. Test `nl_tokenize` directly
2. Print first few tokens from nanolang lexer
3. Compare with C lexer output
4. Fix any mismatches in token structure or types

---

**Status:** Debugging in progress

