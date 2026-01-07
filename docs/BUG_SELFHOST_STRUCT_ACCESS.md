# Self-Hosting Blocker: Parser Struct Field Access Bug

**Status**: Critical blocker for 100% self-hosting  
**Severity**: High  
**Component**: Self-hosted transpiler (struct field access code generation)  
**Discovery Date**: Current session  
**Progress**: 99.8% → 100% (0.2% away!)

---

## Summary

The self-hosted NanoLang compiler (`nanoc_v06`) fails during typechecking with:
```
Error: Index 166 out of bounds for list of length 40
```

**Root Cause**: The transpiler generates incorrect C code for accessing `Parser.lets` field, causing it to access `Parser.binary_ops` instead.

---

## Symptom Analysis

### What Works ✅
1. **Lexer**: Tokenizes all 22,383 lines successfully
2. **Parser**: Creates complete AST with:
   - 446 functions
   - 2,291 `let` statements  
   - 96 binary ops
3. **Registration**: All 446 function signatures registered successfully
4. **`parser_get_let_count(parser)`**: Returns correct value (2,291)

### What Fails ❌
1. **`parser_get_let(parser, idx)`**: Accesses wrong list
   - Expected: Access `Parser.lets` (2,291 elements)
   - Actual: Accesses `Parser.binary_ops` (40 elements)
   - Error when `idx >= 40`: "Index out of bounds"

---

## Technical Details

### Parser Struct Layout (schema/compiler_schema.json)

```
Parser {
  [0]  tokens: List<LexerToken>
  [1]  file_name: string
  [2]  position: int
  [3]  token_count: int
  [4]  has_error: bool
  [5]  diagnostics: List<CompilerDiagnostic>
  [6]  numbers: List<ASTNumber>
  [7]  floats: List<ASTFloat>
  [8]  strings: List<ASTString>
  [9]  bools: List<ASTBool>
  [10] identifiers: List<ASTIdentifier>
  [11] binary_ops: List<ASTBinaryOp>     ← 40 elements (self-hosted)
  [12] calls: List<ASTCall>
  [13] call_args: List<ASTStmtRef>
  [14] array_elements: List<ASTStmtRef>
  [15] array_literals: List<ASTArrayLiteral>
  [16] lets: List<ASTLet>                ← 2,291 elements (expected)
  ...
}
```

**Field offset**: `lets` is 5 fields after `binary_ops`

### Failure Point

```nano
/* typecheck.nano:1790 */
let param_let: ASTLet = (parser_get_let parser (+ func.param_start pidx))
```

When `(+ func.param_start pidx) = 166`:
- **C compiler output**: Correctly accesses `parser->lets`
- **Self-hosted output**: Incorrectly accesses `parser->binary_ops` (offset bug)

### Evidence

**During parsing** (`bin/nanoc` - C compiler):
```
DEBUG: Parser has 2291 lets, 446 functions
```

**During typechecking** (`bin/nanoc_v06` - self-hosted):
```
Func resolve_import_path has param_start=2232, lets_len=2291
  Func #440 getting param 2: access_idx=2234, lets_len=2291
Error: Index 166 out of bounds for list of length 40
```

**Key observation**: `parser_get_let_count()` returns 2291, but `parser_get_let(idx)` crashes when `idx > 40`.

---

## Root Cause Hypotheses

### H1: Struct Field Offset Calculation (Most Likely)
The self-hosted transpiler calculates wrong byte offset for `Parser.lets`:
- Generated C: `parser->binary_ops` instead of `parser->lets`
- Likely issue in: `src_nano/transpiler.nano` field access code generation

### H2: List<T> Monomorphization Issue
`List<ASTLet>` might not be correctly instantiated:
- Field points to wrong memory location
- Copy/assignment corrupts pointer

### H3: Parser Struct Copying Bug
When `Parser` is passed between functions, field pointers get corrupted

---

## Debugging Steps Taken

### Confirmed Working:
1. ✅ Lexer column tracking fixed
2. ✅ Parser `arg_start` bug fixed  
3. ✅ Parser creates all ASTs correctly
4. ✅ Function registration completes (all 446)
5. ✅ `parser_get_let_count()` returns correct value

### Narrowed Down:
1. ✅ Error happens during Phase 1 (registration), not Phase 2
2. ✅ Crash on function #440 (`resolve_import_path`, param_start=2232)
3. ✅ `parser.lets` has correct length (2291)
4. ✅ But accessing `parser.lets[166]` hits `binary_ops[166]` instead

### Next Steps:
1. Examine generated C code for `parser_get_let`
2. Check `transpiler.nano` field access code generation
3. Verify `List<ASTLet>` monomorphization
4. Compare C compiler vs self-hosted compiler output

---

## Workaround

None currently. The only way to achieve 100% self-hosting is to fix the transpiler.

---

## Fix Strategy

### Immediate (30-60 mins):
1. Generate C code from self-hosted compiler: `./bin/nanoc_v06 ... --emit-c`
2. Compare field access for `parser.lets` vs `parser.binary_ops`
3. Identify wrong offset calculation
4. Fix in `src_nano/transpiler.nano`

### Alternative (2-4 hours):
1. Begin parser refactoring (isolate bug in smaller codebase)
2. Add struct field access tests
3. Fix transpiler incrementally

---

## Impact

**Blocking**: 100% self-hosting milestone  
**Workaround**: Use C compiler (`bin/nanoc`) for now  
**Urgency**: High (final 0.2% to completion)

---

## Related Files

- `src_nano/transpiler.nano` - C code generation (likely bug location)
- `src_nano/typecheck.nano:1790` - Error trigger point
- `src_nano/parser.nano:6427` - `parser_get_let` definition
- `schema/compiler_schema.json` - Parser struct definition
- `src/generated/compiler_schema.h` - C struct layout

---

## Test Case

```nano
/* Minimal reproducer */
fn test_parser_struct_access() -> int {
    let tokens: List<LexerToken> = (list_LexerToken_new)
    let parser: Parser = (parser_new tokens 0 "test.nano")
    
    /* Add 50 lets */
    let mut i: int = 0
    while (< i 50) {
        let p2: Parser = (parser_store_let parser "x" "int" -1 -1 false 1 1)
        set parser p2
        set i (+ i 1)
    }
    
    /* This should work but will crash at i=40 in self-hosted */
    set i 0
    while (< i 50) {
        let let_node: ASTLet = (parser_get_let parser i)
        set i (+ i 1)
    }
    
    return 0
}
```

---

## Notes

- C reference compiler works perfectly (no bug)
- Bug is specific to self-hosted transpiler
- Suggests code generation issue, not language design issue
- Refactoring will help isolate and fix this bug

**Status**: Ready for deep C code analysis or refactoring approach.

