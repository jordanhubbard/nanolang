# Phase 1 Implementation: Module-Level Safety

**Status:** 🔄 In Progress  
**Issue:** nanolang-dc8u  
**Date Started:** 2025-01-08

---

## Implementation Checklist

### **Step 1: AST Changes** ✅
- [x] Add `is_unsafe` field to `import_stmt` in `src/nanolang.h`

### **Step 2: Parser Changes** (In Progress)
- [ ] Update `parse_import()` in `src/parser.c`
  - [ ] Replace `import` keyword with `module`
  - [ ] Add `unsafe` prefix support
  - [ ] Set `is_unsafe` flag when parsing `unsafe module`
- [ ] Add `parse_module_declaration()` for `unsafe module name { ... }`
- [ ] Update main parse loop to recognize `module` keyword
- [ ] Remove `import` keyword support (clean break)

### **Step 3: Lexer Changes**
- [ ] Check if `module` is already a keyword (it is - TOKEN_MODULE exists)
- [ ] Ensure `unsafe` is a keyword (need to verify)

### **Step 4: Environment Changes**
- [ ] Add module safety tracking to Environment
- [ ] Add `current_module_is_unsafe` flag
- [ ] Track which modules are unsafe

### **Step 5: Typechecker Changes**
- [ ] Allow FFI calls in unsafe modules without `unsafe {}` blocks
- [ ] Validate import safety annotations
- [ ] Warn if `unsafe module` imports a safe module

### **Step 6: Update All Examples**
- [ ] Replace all `import` with `module` in examples/
- [ ] Add `unsafe` prefix where needed (SDL, bullet, etc.)
- [ ] Test each example compiles

### **Step 7: Update Tests**
- [ ] Replace `import` with `module` in tests/
- [ ] Add new tests for unsafe module syntax
- [ ] Test FFI calls work without unsafe blocks in unsafe modules

### **Step 8: Update Documentation**
- [ ] Update MEMORY.md
- [ ] Update docs/MODULE_SYSTEM.md
- [ ] Add migration notes

---

## Files to Modify

### Core Compiler
1. `src/nanolang.h` - AST structures ✅
2. `src/parser.c` - Parse `module` keyword
3. `src/lexer.c` - Verify `module` and `unsafe` tokens
4. `src/typechecker.c` - Module safety checking
5. `src/module.c` - Module resolution (may need updates)
6. `src/transpiler.c` - May need safety metadata

### Examples (bulk update)
- `examples/*.nano` - Replace `import` → `module`
- Add `unsafe` prefix to FFI modules

### Tests (bulk update)
- `tests/*.nano` - Replace `import` → `module`
- Add safety tests

---

## Current Progress

### ✅ Completed
1. AST structure updated with `is_unsafe` field
2. Architecture documented
3. libc module created as proof of concept

### 🔄 In Progress
1. Parser modifications

### ⏳ Pending
1. Environment tracking
2. Typechecker updates
3. Example/test updates

---

## Testing Strategy

### Unit Tests
1. Parse `module "path"` successfully
2. Parse `unsafe module "path"` successfully  
3. Parse `unsafe module name { ... }` successfully
4. Reject `import` keyword (clean break)

### Integration Tests
1. Compile libc_demo.nano successfully
2. FFI calls work in unsafe modules without `unsafe {}`
3. Safe modules reject FFI calls

### Regression Tests
1. All existing tests pass with `import` → `module` replacement

---

## Next Steps (Right Now)

1. Finish parser modifications
2. Test parsing with simple example
3. Move to typechecker
4. Bulk update examples

---

## Estimated Time Remaining
- Parser: 2-3 hours
- Typechecker: 1-2 hours
- Example updates: 1 hour (bulk find/replace)
- Testing: 2 hours

**Total: ~6-8 hours of focused work**

---

## Notes

- No backward compatibility needed (all code in this repo)
- Clean break from `import` to `module`
- `unsafe module` becomes the standard for FFI
- This sets foundation for Phases 2-4

---

**Last Updated:** 2025-01-08 (starting parser implementation)
