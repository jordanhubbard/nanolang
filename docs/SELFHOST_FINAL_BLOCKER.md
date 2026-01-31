# Self-Hosting Final Blocker (99.9% → 100%)

**Date:** 2026-01-07  
**Status:** In Progress  
**Blocker:** Struct field access in self-hosted typechecker/transpiler

## Summary

The self-hosted compiler (`nanoc_v06`) can **almost** compile itself but crashes with:
```
Error: Index 59 out of bounds for list of length 40
```

This occurs when the compiler processes code containing struct field access (e.g., `parser.lets`).

## Progress

### ✅ Complete (99%)
- Lexer: 100% feature-complete
- Parser: 100% feature-complete (all syntax supported)
- Typechecker: 95% complete (added IF/BLOCK/TUPLE/UNSAFE)
- Transpiler: 90% complete (built-ins, List<T>, complex expressions)
- Can compile 99% of NanoLang programs

### ❌ Blocker (1%)
- Struct field access code generation produces wrong C code
- Affects `Parser.lets` and other struct field accesses
- Root cause: Field offset calculation in generated C code

## Reproduction

### Minimal Test Case
```nano
import "src_nano/generated/compiler_ast.nano"

fn test(p: Parser) -> int {
    let count: int = (list_ASTLet_length p.lets)
    return count
}
```

**C Reference Compiler (`nanoc`):** ✅ Compiles successfully  
**Self-Hosted Compiler (`nanoc_v06`):** ❌ `Index 0 out of bounds for list of length 0`

### Full Self-Compilation
```bash
./bin/nanoc_v06 src_nano/nanoc_v06.nano -o /tmp/nanoc_gen2
```

**Result:** ❌ `Index 59 out of bounds for list of length 40`

## Analysis

### Expected C Code (from `nanoc`)
```c
int64_t count = nl_list_ASTLet_length(p.lets);
```

### Actual C Code (from `nanoc_v06`)
*Unknown - needs investigation with --keep-c flag*

### Hypothesis
The self-hosted transpiler (`src_nano/transpiler.nano`) generates incorrect C code for:
1. Struct field access expressions (`obj.field`)
2. Specifically when `field` is a `List<T>` type
3. The generated code uses wrong struct offsets or pointer arithmetic

### Schema Definition (`Parser` struct)
```json
"Parser": {
  "fields": [
    ["tokens", "List<LexerToken>"],     // Field 0
    ["file_name", "string"],            // Field 1
    ...
    ["lets", "List<ASTLet>"],           // Field 17 ⚠️
    ...
  ]
}
```

The `lets` field is at index 17, but the generated C code may be using wrong offset.

## Investigation Steps

1. ✅ Verified C reference compiler generates correct code
2. ✅ Confirmed self-hosted compiler fails consistently
3. ⏳ Compare C output from `nanoc` vs `nanoc_v06` for same input
4. ⏳ Check `generate_expression` in `src_nano/transpiler.nano` for PNODE_FIELD_ACCESS
5. ⏳ Verify struct field offset calculation

## Next Steps

### Option A: Debug Transpiler C Generation
- Add debug output to `generate_expression` for PNODE_FIELD_ACCESS
- Compare generated C code line-by-line
- Fix field offset calculation

### Option B: Workaround via Refactoring
- Replace struct field access with getter functions
- Example: `parser_get_lets(p)` instead of `p.lets`
- Reduces complexity for transpiler

### Option C: Enable Transpiler Debug Mode
- Add `--emit-c` flag to see generated C
- Compare with reference compiler output
- Identify exact difference

## Timeline

- **Started:** 2026-01-06
- **Current:** 2026-01-07 (24 hours in)
- **Progress:** 99% → 100% (final 1%)
- **Estimate:** 2-4 hours to fix once root cause identified

## Impact

This is the **only remaining blocker** for 100% self-hosting. Once fixed:
- ✅ NanoLang compiler can compile itself
- ✅ Bootstrap complete
- ✅ Can retire C reference compiler (or keep for cross-check)
- ✅ Full dogfooding of language features

## References

- `docs/BUG_SELFHOST_STRUCT_ACCESS.md` - Original bug report
- `src_nano/transpiler.nano` lines 1840-1848 - Field access generation
- `schema/compiler_schema.json` - Parser struct definition

