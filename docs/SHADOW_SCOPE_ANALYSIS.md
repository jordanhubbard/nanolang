# Shadow Test Variable Scoping Analysis

**Issue:** nanolang-2rxp  
**Status:** Investigated - Complex Fix Required  
**Time Spent:** 2 hours  
**Date:** 2025-01-08

---

## Problem Summary

Shadow test blocks don't isolate variable scopes between statements, causing "Variable already defined" errors when reusing variable names:

```nano
shadow my_function {
    let result: int = (my_function 5)
    assert (== result 10)
    
    let result: int = (my_function 10)  // ❌ Error: Variable already defined
    assert (== result 20)
}
```

**Impact:** 11 errors blocking self-hosting

---

## Investigation Findings

### 1. C Reference Compiler Behavior

**Test Code:**
```nano
fn double(x: int) -> int { return (* x 2) }

shadow double {
    let result: int = (double 5)
    assert (== result 10)
    
    let result: int = (double 10)  /* Reuse variable name */
    assert (== result 20)
}
```

**Result:** ✅ **C compiler allows variable reuse**

```bash
$ ./bin/nanoc /tmp/test_shadow_scope.nano -o /tmp/test
Running shadow tests...
Testing double... PASSED
All shadow tests passed!
```

### 2. Self-Hosted Compiler Behavior

**Location:** `src_nano/typecheck.nano:1878`

```nano
fn check_let_statement(...) -> array<Symbol> {
    /* Check if variable already exists in current scope */
    if (env_has_symbol symbols let_node.name) {
        let msg: string = (+ "Variable already defined: " let_node.name)
        let diag: CompilerDiagnostic = (Diagnostics.diag_typecheck_error "E0004" msg ...)
        (Diagnostics.diag_list_add diags diag)
        return symbols  /* ❌ Strict: doesn't allow redefinition */
    }
    ...
}
```

**Problem:** Self-hosted typechecker is **more strict** than C reference compiler.

### 3. Root Cause

**Block Processing (line 1995-2005):**
```nano
fn check_block(...) -> bool {
    let block: ASTBlock = (parser_get_block parser block_id)
    let mut local_symbols: array<Symbol> = symbols
    
    let mut i: int = 0
    while (< i block.statement_count) {
        let stmt: ASTStmtRef = (parser_get_block_statement parser ...)
        
        if (== stmt.node_type ParseNodeType.PNODE_LET) {
            let let_node: ASTLet = (parser_get_let parser stmt.node_id)
            set local_symbols (check_let_statement parser let_node local_symbols ...)
            /* ⬆️ local_symbols accumulates - no scope reset */
        }
        ...
    }
}
```

**Issue:** All statements in a block share the same `local_symbols` array. No scope isolation.

---

## Why This Is Complex

### Challenge 1: Context Detection
- `check_let_statement` doesn't know if it's in a shadow test
- No flag or context passed down indicating shadow test scope
- Would need to thread context through entire typechecker

### Challenge 2: Chicken-and-Egg Problem
- Can't test self-hosted compiler until self-hosting works
- This issue blocks self-hosting
- Fixing it requires modifying self-hosted code

### Challenge 3: Scope Management
- No explicit scope push/pop in self-hosted typechecker
- Symbols are managed via array passing
- Would need to implement scope markers or save/restore logic

---

## Potential Solutions

### Option A: Add Shadow Test Context Flag

**Changes Required:**
1. Add `in_shadow_test: bool` parameter to:
   - `check_block`
   - `check_let_statement`
   - All intermediate functions
2. In `check_let_statement`:
   ```nano
   if (env_has_symbol symbols let_node.name) {
       if (not in_shadow_test) {  /* Only error outside shadow tests */
           let msg: string = (+ "Variable already defined: " let_node.name)
           ...
       } else {
           /* Allow redefinition in shadow tests */
           /* Remove old symbol and add new one */
       }
   }
   ```

**Pros:**
- Minimal logic changes
- Clear intent

**Cons:**
- Requires threading boolean through many functions
- Large refactor (~50+ function signatures)

### Option B: Per-Statement Scope in Shadow Tests

**Changes Required:**
1. Detect shadow test blocks at top level
2. For each statement in shadow test:
   ```nano
   let saved_symbols: array<Symbol> = local_symbols
   /* Process statement */
   set local_symbols saved_symbols  /* Reset after each statement */
   ```

**Pros:**
- True scope isolation
- More correct semantics

**Cons:**
- Need to detect shadow test context
- May break variable usage across assertions

### Option C: Allow Variable Shadowing Everywhere

**Changes Required:**
1. In `check_let_statement`, instead of error:
   ```nano
   if (env_has_symbol symbols let_node.name) {
       /* Remove old symbol */
       set symbols (env_remove_symbol symbols let_node.name)
   }
   /* Add new symbol */
   ```

**Pros:**
- Simple change
- Works everywhere, not just shadow tests

**Cons:**
- Changes language semantics
- May hide bugs in user code
- Not what issue description suggests

---

## Recommendation

**Status:** Defer until other self-hosting blockers are resolved

**Reasons:**
1. C compiler already works correctly
2. Fix requires extensive self-hosted compiler changes
3. Can't test self-hosted changes until 100% self-hosting
4. Other blockers may be simpler (string concatenation, struct metadata)

**Workaround for Self-Hosting:**
Use unique variable names in shadow tests:
```nano
shadow my_function {
    let result1: int = (my_function 5)
    assert (== result1 10)
    
    let result2: int = (my_function 10)  /* ✅ Different name */
    assert (== result2 20)
}
```

**Priority:** Move to **Priority 4** (after other self-hosting issues)

---

## Next Steps (If Pursuing This Issue)

1. **Start with Option C** (simplest):
   - Implement `env_remove_symbol` function
   - Modify `check_let_statement` to remove before adding
   - Test with C reference compiler
   - See if it helps self-hosting

2. **If Option C fails, try Option B**:
   - Add shadow test detection
   - Implement per-statement scope reset
   - More invasive but correct

3. **Avoid Option A unless necessary**:
   - Too many signature changes
   - High risk of bugs

---

## Files to Modify (For Any Solution)

- `src_nano/typecheck.nano`:
  - `check_let_statement` (line 1878)
  - `check_block` (line 1995)
  - Possibly `env_add_symbol`, `env_remove_symbol`

---

## Time Estimate

- **Option C:** 1-2 hours (implement, test, debug)
- **Option B:** 4-6 hours (context detection, scope logic, testing)
- **Option A:** 8+ hours (signature changes, refactoring, testing)

---

## Conclusion

This issue is **architectural** and **self-hosting-specific**. The C reference compiler works fine. Fixing it requires significant changes to the self-hosted typechecker and can't be fully tested until other self-hosting blockers are resolved.

**Recommendation:** Deprioritize in favor of simpler self-hosting issues (nanolang-2kdq, nanolang-tux9).

---

**Related Issues:**
- nanolang-qlv2 (100% self-hosting epic)
- nanolang-3oda (module-qualified calls in self-hosted typechecker)
- nanolang-2kdq (string concatenation type inference)
- nanolang-tux9 (struct metadata coverage)
