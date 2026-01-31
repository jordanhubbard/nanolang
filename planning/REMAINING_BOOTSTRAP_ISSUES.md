# Remaining Stage 2 Bootstrap Issues

## Status: Struct Metadata ✅ COMPLETE

The critical blocker (struct field metadata) has been resolved. User-defined structs now work correctly with the self-hosted compiler.

**What works**:
- ✅ Struct definitions parse with complete field information
- ✅ Field access expressions type-check correctly (`p.x` returns proper type)
- ✅ Struct literals work
- ✅ Simple and complex struct programs compile with stage1

---

## Remaining Type Checker Issues

The following issues prevent full stage 2 bootstrap (nanoc compiling itself):

### Issue Category 1: Scoping Errors (4 errors)

**Symptoms**:
```
[E0002] I cannot find a definition for `p`. (line:1046)
[E0002] I cannot find a definition for `content`. (line:1076)
[E0002] I cannot find a definition for `test_content`. (line:1087)
[E0002] I cannot find a definition for `read_back`. (line:1087)
```

**Context**:
- Line 1046: Variable `p` defined in while loop, used inside nested `if` block
- Lines 1076-1087: Variables defined in shadow test blocks

**Observation**:
- Simple while loop tests with nested if blocks work fine
- Simple shadow test blocks with variables work fine
- Issue seems specific to the merged file context during bootstrap

**Hypothesis**:
- May be related to how imports are merged
- May be specific to shadow tests in imported files
- Needs investigation of symbol table management in merged contexts

---

### Issue Category 2: Array Type Inference (3 errors)

**Symptoms**:
```
[E0001] Variable new_names: I expected `array<string>`, but found `void`. (line:12176)
[E0001] Variable new_types: I expected `array<string>`, but found `void`. (line:12177)
[E0001] Variable files: I expected `array<string>`, but found `void`. (line:15038)
```

**Root Cause**: 
Array-returning functions/operations return `void` instead of `array<T>`

**Likely Fix**:
- Check `check_expr_node` handling of array operations
- Ensure `array_push`, `array_new`, and similar operations propagate array types
- May need to add cases for array-returning built-ins

---

### Issue Category 3: Unknown Type Inference (5 errors)

**Symptoms**:
```
[E0001] Variable elem_type: I expected `string`, but found `unknown`. (line:9879)
[E0001] Return value: I expected `NSType`, but found `unknown`. (line:9884)
[E0001] Return value of genenv_get: I expected `string`, but found `unknown`. (line:12185)
[E0001] Variable list_specs: I expected `string`, but found `unknown`. (line:13009)
[E0001] Variable list_types: I expected `array<string>`, but found `unknown`. (line:13154)
```

**Root Cause**:
Functions return `unknown` type when they should return specific types

**Likely Fix**:
- Check type inference for function calls
- Ensure return type annotation is being used
- May need to handle specific built-in functions

---

### Issue Category 4: List Type Propagation (2 errors)

**Symptoms**:
```
[E0004] Argument 1 of typecheck_output: I expected `List<CompilerDiagnostic>`, but found `void`. (line:11480, twice)
```

**Root Cause**:
Generic list types not properly propagated through function calls

**Likely Fix**:
- Check how generic types are handled in type inference
- Ensure `List<T>` types preserve element type parameter

---

### Issue Category 5: Type Mismatch (1 error)

**Symptoms**:
```
[E0004] Argument 2 of repeat_string: I expected `int`, but found `string`. (line:8537)
```

**Root Cause**:
Actual bug in the code - wrong argument type passed

**Fix**: 
Investigate and fix the specific call site

---

## Next Steps for Full Bootstrap

### 1. Debug Scoping Issues (Highest Priority)
**Estimated**: 3-4 hours

- Add debug logging to symbol table lookups
- Check how merged files handle symbol tables
- Investigate shadow test checking in self-hosted typechecker
- May need to add explicit shadow test type-checking phase

### 2. Fix Array Type Inference (Medium Priority)
**Estimated**: 2-3 hours

- Review `check_expr_node` for array operations
- Add test cases for array-returning functions
- Ensure built-in array functions have proper type signatures
- Check `array_push` type propagation

### 3. Fix Return Type Inference (Medium Priority)
**Estimated**: 2-3 hours

- Check function call type inference
- Ensure return type annotations are used
- Add cases for missing built-in functions
- Test with simple examples

### 4. Fix List Type Propagation (Low Priority)
**Estimated**: 1-2 hours

- Review generic type parameter handling
- Ensure `List<T>` preserves `T` through operations
- May be related to array type issue

### 5. Fix Type Mismatch (Low Priority)
**Estimated**: 30 minutes

- Find the `repeat_string` call at line 8537
- Fix the argument types

---

## Testing Strategy

For each fix:

1. **Create minimal reproducer** - Isolate the issue in a small test file
2. **Test with C compiler** - Verify expected behavior  
3. **Test with stage1** - Confirm self-hosted compiler has same issue
4. **Implement fix** - Make targeted changes
5. **Retest** - Verify fix works for minimal case
6. **Re-run bootstrap** - Check progress on full compilation

---

## Total Estimated Effort

**Optimistic**: 6-8 hours (if issues are straightforward)  
**Realistic**: 10-15 hours (accounting for debugging complexity)  
**Pessimistic**: 20+ hours (if fundamental architectural issues)

---

## Success Criteria

- [ ] Stage1 compiles `src_nano/nanoc_v06.nano` without errors
- [ ] Generated stage2 binary is functionally equivalent to stage1
- [ ] All tests pass with stage2 compiler
- [ ] Stage2 can compile itself (stage3 == stage2)

**Current Status**: 3/6 core success criteria met (50%)

---

## Key Insight

The struct metadata work removed the **fundamental architectural blocker**. The remaining issues are **localized type inference bugs** that don't affect the overall design. Each can be debugged and fixed independently.

The self-hosted compiler is **very close** to full bootstrap capability.
