# Lexer Self-Hosting Status

## Overview
We've begun the self-hosting journey by implementing the nanolang lexer in nanolang itself. This is a major milestone towards a fully self-hosted compiler.

## Progress Summary

### ✅ Completed
1. **Enum definition for token types** - Clean TokenType enum with 50+ variants
2. **Character classification helpers** - Functions for identifying digits, letters, identifiers
3. **String utilities** - String comparison and substring extraction
4. **Keyword classification** - Maps string to token type for all keywords
5. **Token struct** - Proper struct for storing token data (type, value, line, column)
6. **Shadow tests** - All helper functions have shadow tests
7. **No builtin conflicts** - Renamed functions to avoid shadowing built-ins

### 🚧 Partially Complete
**Main `lex()` function** - The tokenization logic is structured but incomplete:
- ✅ Whitespace skipping
- ✅ Comment handling (# and /* */)
- ✅ Number recognition (int and float)
- ✅ String literal parsing
- ✅ Identifier/keyword recognition
- ❌ Token storage (blocked by lack of dynamic arrays)
- ❌ Operator tokenization (partially implemented)

### 🔴 Blocked
**Dynamic Array Support** - The main blocker is the lack of generic data structures:
- Need: `List<Token>` or `array_push()` equivalent
- Current: Only `List_int` and `List_string` available
- Options:
  1. Add `List_token` type to runtime
  2. Implement generic `List<T>` support
  3. Use fixed-size pre-allocated arrays (limiting but workable)

## File Structure

### Current Files
- `src_nano/lexer_v2.nano` - Latest working version with all fixes
- `src_nano/lexer.nano` - Original version (has issues)
- Other lexer_*.nano files - Various experimental versions

### Key Components
```nano
/* Clean enum without TOKEN_ prefix */
enum TokenType {
    EOF = 0,
    NUMBER = 1,
    IDENTIFIER = 4,
    FN = 19,
    ...
}

/* Token struct */
struct Token {
    token_type: int,
    value: string,
    line: int,
    column: int
}

/* Helper functions (renamed to avoid builtins) */
fn char_is_digit(c: int) -> bool { ... }
fn char_is_letter(c: int) -> bool { ... }
fn classify_keyword(word: string) -> int { ... }

/* Main lexer (incomplete) */
fn lex(source: string) -> array<Token> { ... }
```

## Next Steps

### Short Term
1. **Add `List_token` support to runtime**
   - Implement `src/runtime/list_token.c`
   - Add list_token_new(), list_token_push(), etc.
   - Update builtin function registry

2. **Complete lexer implementation**
   - Use `List_token` for dynamic token storage
   - Finish operator tokenization
   - Add comprehensive testing

### Medium Term
3. **Test against C lexer**
   - Compare token streams
   - Verify correctness
   - Performance benchmarking

4. **Integration**
   - Use nano lexer in compiler pipeline
   - Bootstrap: compile lexer with itself

### Long Term
5. **Parser in nanolang**
6. **Type checker in nanolang**
7. **Code generator in nanolang**
8. **Fully self-hosted compiler**

## Technical Details

### Why This Approach?
- **Enums**: Perfect for token types, cleanly maps to C enum
- **Structs**: Natural representation for tokens
- **String operations**: Existing stdlib functions work well
- **Shadow tests**: Ensures correctness at compile time

### Lessons Learned
1. **Builtin conflicts**: Can't redefine `is_whitespace`, `is_digit`, `is_alpha`, `str_equals`
2. **Enum access**: Use `TokenType.EOF` syntax (field access notation)
3. **Dynamic arrays**: Major missing feature for self-hosting
4. **Shadow tests**: Required for all functions, good discipline

### Performance Considerations
- String concatenation for substring extraction (not optimal)
- Character-by-character processing (expected for lexer)
- Once complete, can compare performance vs C implementation

## Dependencies

### Required Language Features (✅ Available)
- Enums with explicit values
- Structs
- Arrays (fixed-size)
- String operations (str_length, char_at, str_concat, etc.)
- Integer comparisons and arithmetic
- While loops and conditionals
- Functions with shadow tests

### Required Language Features (❌ Missing)
- Generic data structures (`List<T>`)
- OR: Specific `List<Token>` type
- OR: `array_push()` / `array_pop()` for dynamic arrays

## Testing

### Current Status
```bash
$ ./bin/nano src_nano/lexer_v2.nano
# ✅ Compiles successfully
# ⚠️  Some unused variable warnings (expected for incomplete implementation)
```

### Test Cases Needed
1. Simple tokens: `fn add() {}`
2. Numbers: `42`, `-10`, `3.14`
3. Strings: `"hello"`, `"with \"quotes\""`
4. Keywords: All 30+ keywords
5. Operators: All operators (single and double char)
6. Comments: `#` and `/* */`
7. Edge cases: Empty file, only whitespace, unterminated strings

## Recommendations

### Priority 1: Enable Dynamic Arrays
**Option A**: Add `List_token` type (fastest)
- Copy `src/runtime/list_int.c` to `src/runtime/list_token.c`
- Replace `int` with `Token` struct
- Update build system
- Add builtin functions

**Option B**: Generic Lists (better long-term)
- Design generic list interface
- Implement List<T> template system
- More complex but enables List<anything>

### Priority 2: Complete Lexer
Once we have dynamic arrays:
- Finish operator tokenization
- Add error handling (unterminated strings, unknown chars)
- Add position tracking for better error messages
- Create comprehensive test suite

### Priority 3: Integration Testing
- Create test comparing nano lexer vs C lexer output
- Verify token-by-token equivalence
- Test on all example programs

## Conclusion
We've made excellent progress on the self-hosted lexer. The foundation is solid with proper enums, structs, helper functions, and shadow tests. The main blocker is dynamic array support, which is a reasonable next step for the language itself. Once that's added, we can complete the lexer and move forward with self-hosting the parser.

**Status**: 🟡 Partially complete, blocked on language features
**Confidence**: High - the approach is sound and the implementation quality is good
**Next Action**: Implement `List_token` or generic `List<T>` support

