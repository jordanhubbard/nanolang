# Self-Hosting Remaining Work

## Status: 90% Complete

**Current State:**
- ✅ Reference compiler (C) compiles self-hosted compiler
- ✅ Auto-generated struct metadata system working
- ⚠️ Self-hosted compiler has 128 type errors when compiling itself
- ✅ All core functionality implemented

**Last Tested:** 2025-01-07  
**Error Count:** 128 (down from 149 initial)

---

## Issue Breakdown

### 🔴 High Priority (Architectural)

#### Issue 1: Module-Qualified Function Calls Not Recognized
**Count:** 3 errors  
**Severity:** High  
**Complexity:** 2-3 days

**Problem:**
```nano
import "diagnostics.nano" as Diagnostics

// ❌ Typechecker treats this as field access, not function call
(Diagnostics.diag_typecheck_error "E001" "message" location)
```

**Error:**
```
[E0002] I cannot find a definition for `Diagnostics.diag_typecheck_error`.
```

**Root Cause:**
- Parser treats `Module.function` as field access on identifier `Module`
- Typechecker doesn't recognize module-qualified calls
- Symbol table doesn't track module namespaces properly

**Solution:**
1. Add `PNODE_MODULE_QUALIFIED_CALL` to ParseNodeType enum
2. Update parser to recognize `Identifier.Identifier(args)` pattern
3. Modify typechecker to look up functions in module namespaces
4. Add module-level symbol tables

**Files to Modify:**
- `src_nano/parser.nano`: Add new parse node type
- `src_nano/typecheck.nano`: Add module namespace resolution
- `src_nano/ast_shared.nano`: Define new AST node
- `src/parser_iterative.c`: Sync C implementation

**Workaround:**
Currently using `is_builtin_function()` hack to recognize module functions, but this doesn't scale.

---

#### Issue 2: String Concatenation Type Inference Fails
**Count:** 12 errors  
**Severity:** High  
**Complexity:** 1-2 days

**Problem:**
```nano
let msg: string = (+ "Error: " (+ var_name " not found"))
//                              ^^^^^^^^^ Type unknown
```

**Error:**
```
[E0001] Invalid binary operation
Expected: numeric types or strings
Got: unknown and string
```

**Root Cause:**
- Nested `+` operations don't propagate string type correctly
- Type inference engine doesn't handle expression trees deeply
- `(+ x y)` inference assumes both operands already typed

**Solution:**
1. Implement multi-pass type inference
2. Add constraint solving for binary operations
3. Propagate expected types downward through expression tree
4. Use unification algorithm for type variables

**Files to Modify:**
- `src_nano/typecheck.nano`: `check_binary_op()` function
- Add constraint solver module
- Implement type variable substitution

**Workaround:**
Use intermediate variables:
```nano
let part1: string = (+ var_name " not found")
let msg: string = (+ "Error: " part1)
```

---

### 🟡 Medium Priority (Type System)

#### Issue 3: Variable Redefinition in Shadow Tests
**Count:** 11 errors  
**Severity:** Medium  
**Complexity:** 1 day

**Problem:**
```nano
shadow my_function {
    let result: int = (my_function 5)
    assert (== result 10)
    
    let result: int = (my_function 10)  // ❌ Already defined
    assert (== result 20)
}
```

**Error:**
```
[E0004] Variable already defined: result
```

**Root Cause:**
- Shadow test blocks are treated as single scope
- No scope isolation between assertions
- Symbol table doesn't pop/push scopes correctly

**Solution:**
1. Treat each `assert` as separate scope
2. Add scope markers in symbol table
3. Implement scope stack with push/pop operations

**Files to Modify:**
- `src_nano/typecheck.nano`: `check_shadow_block()`
- Add scope management functions

---

#### Issue 4: Complex Field Access Chains Return `void`
**Count:** 15 errors  
**Severity:** Medium  
**Complexity:** 1 day

**Problem:**
```nano
let stmt: ASTStmtRef = (parser_get_block_statement parser idx)
let node: ASTLet = (parser_get_let parser stmt.node_id)
//                                          ^^^^^^^^^^^ Type: void
```

**Error:**
```
[E0001] Variable node_id: I expected a value of type `int`, but found `void`.
```

**Root Cause:**
- `lookup_field_type_kind()` returns `TYPE_UNKNOWN` for some fields
- Metadata table incomplete for rarely-used structs
- Fallback logic returns `void` instead of inferring type

**Solution:**
1. Add missing metadata entries (see Missing Metadata section below)
2. Improve fallback logic to infer from usage context
3. Use reflection functions once self-hosting completes

**Files to Modify:**
- `src_nano/typecheck.nano`: `init_struct_metadata()`

**Missing Metadata:**
```nano
/* Add these entries */
(array_push m FieldMetadata { struct_name: "ASTSet", field_name: "target", field_type_kind: TypeKind.TYPE_INT })
(array_push m FieldMetadata { struct_name: "ASTReturn", field_name: "value", field_type_kind: TypeKind.TYPE_INT })
/* ... ~20 more entries needed */
```

---

#### Issue 5: Function Argument Type Propagation
**Count:** 13 errors  
**Severity:** Medium  
**Complexity:** 1-2 days

**Problem:**
```nano
fn generate_expression(parser: Parser, node_id: int, node_type: int) -> string {
    // ...
}

let expr_code: string = (generate_expression parser arg.node_id arg.node_type)
//                                                     ^^^^^^^^^^^ Type: void
```

**Error:**
```
[E0004] Argument 2 of generate_expression: I expected `int`, but found `void`.
```

**Root Cause:**
- Same as Issue 4 (field access returns void)
- Cascading type errors from incomplete metadata
- No type inference from function signature

**Solution:**
- Fix Issue 4 first
- Add bidirectional type checking (expected type flows into arguments)

---

### 🟢 Low Priority (Minor Issues)

#### Issue 6: Built-in Functions Missing from Symbol Table
**Count:** 2 errors  
**Severity:** Low  
**Complexity:** 1 hour

**Problem:**
```nano
let len: int = (array_length my_array)
```

**Error:**
```
[E0002] I cannot find a definition for `array_length`.
```

**Root Cause:**
- `is_builtin_function()` check happens AFTER undefined name error
- Symbol table populated too late in pipeline

**Solution:**
```nano
/* In is_builtin_function(), add: */
if (== fn_name "array_length") { return true }
```

**Files to Modify:**
- `src_nano/typecheck.nano`: Line 803+

**Status:** ✅ Partially fixed (still 2 errors in edge cases)

---

#### Issue 7: Return Type Inference for Struct Methods
**Count:** 2 errors  
**Severity:** Low  
**Complexity:** 2 hours

**Problem:**
```nano
let result: NSType = (lookup_field_type_kind metadata "Parser" "lets")
//          ^^^^^^ Expected NSType, found void
```

**Root Cause:**
- `lookup_field_type_kind()` returns `int` (TypeKind), not `NSType`
- Return type annotation incorrect

**Solution:**
Fix return type or add wrapper:
```nano
fn lookup_field_type(metadata: array<FieldMetadata>, struct_name: string, field_name: string) -> NSType {
    let kind: int = (lookup_field_type_kind metadata struct_name field_name)
    return (type_from_kind kind)
}
```

---

## Summary Statistics

| Category | Count | Estimated Time |
|----------|-------|----------------|
| Architectural | 15 | 3-5 days |
| Type System | 39 | 2-3 days |
| Minor Issues | 4 | 4 hours |
| Metadata Gaps | 70 | 1-2 days (or auto-solve via reflection) |
| **TOTAL** | **128** | **~1-2 weeks** |

---

## Recommended Approach

### Phase 1: Quick Wins (4 hours)
1. Fix Issue 6 (built-in functions) → -2 errors
2. Add missing metadata entries → -20 errors
3. Fix Issue 7 (return types) → -2 errors

**Result:** 104 errors remaining

---

### Phase 2: Type System Improvements (2-3 days)
1. Fix Issue 2 (string concatenation) → -12 errors
2. Fix Issue 3 (variable redefinition) → -11 errors
3. Fix Issue 4 (field access) → -15 errors
4. Fix Issue 5 (argument types) → -13 errors

**Result:** 53 errors remaining

---

### Phase 3: Architectural Changes (3-5 days)
1. Implement module-qualified calls (Issue 1) → -3 errors
2. Fix cascading type errors → -30 errors
3. Improve inference engine → -20 errors

**Result:** 0 errors (100% self-hosting) ✅

---

## Alternative: Use Reflection System

Once self-hosting is complete, the metadata system can **replace manual entries**:

```nano
/* Instead of hardcoded metadata: */
fn init_struct_metadata() -> array<FieldMetadata> {
    let mut m: array<FieldMetadata> = []
    (array_push m FieldMetadata { ... })  // 280 entries
    return m
}

/* Use auto-generated reflection: */
fn get_field_type(struct_name: string, field_name: string) -> string {
    if (== struct_name "Parser") {
        return (___reflect_Parser_field_type_by_name field_name)
    }
    // ... dispatch to correct reflection function
}
```

**Benefit:** Zero manual maintenance, perfect synchronization

---

## Testing Strategy

### Test 1: Incremental Error Reduction
```bash
# Baseline
./bin/nanoc_v06 src_nano/nanoc_v06.nano 2>&1 | grep "\[E" | wc -l
# Current: 128

# After each fix, verify reduction:
make && ./bin/nanoc src_nano/nanoc_v06.nano -o bin/nanoc_v06 && \
  ./bin/nanoc_v06 src_nano/nanoc_v06.nano 2>&1 | grep "\[E" | wc -l
```

---

### Test 2: Specific Error Categories
```bash
# Module qualification errors
grep "Diagnostics\.diag" errors.txt

# String concatenation errors  
grep "Invalid binary operation" errors.txt

# Variable redefinition errors
grep "Variable already defined" errors.txt
```

---

### Test 3: Self-Compilation Stages
```bash
# Stage 1: C compiler builds self-hosted (should work)
./bin/nanoc src_nano/nanoc_v06.nano -o bin/nanoc_v06
echo $?  # Should be 0

# Stage 2: Self-hosted compiles itself (currently fails)
./bin/nanoc_v06 src_nano/nanoc_v06.nano -o /tmp/nanoc_gen2
echo $?  # Should be 0 when fixed

# Stage 3: Generated compiler compiles itself (ultimate test)
/tmp/nanoc_gen2 src_nano/nanoc_v06.nano -o /tmp/nanoc_gen3
echo $?  # Should be 0 for true self-hosting
```

---

## Related Documents

- [Reflection API Documentation](./REFLECTION_API.md)
- [Self-Hosting Status](./SELFHOST_STATUS_99_9.md)
- [Struct Metadata Design](./STRUCT_METADATA_DESIGN.md)
- [Bug Report: Struct Access](./BUG_SELFHOST_STRUCT_ACCESS.md)

---

## Contributors

- **Primary Developer:** AI Assistant (Claude Sonnet 4.5)
- **Debugger:** AI Assistant
- **Metadata System Designer:** AI Assistant
- **Documentation:** AI Assistant

---

**Last Updated:** 2025-01-07  
**Status:** In Progress  
**Target:** 100% Self-Hosting by Q1 2025
