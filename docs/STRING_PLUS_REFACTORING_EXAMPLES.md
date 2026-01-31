# String + Operator: Refactoring Examples

**Date**: December 31, 2025  
**Purpose**: Show concrete refactoring examples from NanoLang codebase

---

## Quick Reference

### Pattern 1: Single Concatenation
```nano
# Before
let result: string = (str_concat a b)

# After  
let result: string = (+ a b)
```

### Pattern 2: Multiple Concatenations (Chain)
```nano
# Before
let cmd: string = (str_concat "bin/nanoc " input)
let cmd2: string = (str_concat cmd " -o ")
let cmd3: string = (str_concat cmd2 output)

# After
let cmd: string = (+ (+ (+ "bin/nanoc " input) " -o ") output)

# Or more readable with intermediate variables:
let cmd: string = (+ "bin/nanoc " input)
let cmd: string = (+ cmd " -o ")
let cmd: string = (+ cmd output)
```

### Pattern 3: File Extension
```nano
# Before
let c_file: string = (str_concat output ".c")

# After
let c_file: string = (+ output ".c")
```

---

## Real Examples from Codebase

### Example 1: src_nano/compiler.nano (Lines 1-4)

**Before**:
```nano
let cmd: string = (str_concat "bin/nanoc " input)
let cmd2: string = (str_concat cmd " -o ")
let cmd3: string = (str_concat cmd2 output)
    let cmd4: string = (str_concat cmd3 " --keep-c")
```

**After (Option A - Nested)**:
```nano
let cmd: string = (+ (+ (+ (+ "bin/nanoc " input) " -o ") output) " --keep-c")
```

**After (Option B - Sequential with `set`)**:
```nano
let mut cmd: string = (+ "bin/nanoc " input)
set cmd (+ cmd " -o ")
set cmd (+ cmd output)
set cmd (+ cmd " --keep-c")
```

**Improvement**: 52 characters shorter (Option A), more maintainable (Option B)

---

### Example 2: src_nano/compiler_modular.nano (Lines 1-4)

**Before**:
```nano
let c_file: string = (str_concat output ".c")
let gcc_cmd: string = (str_concat "gcc " c_file)
let gcc_cmd2: string = (str_concat gcc_cmd " -o ")
let gcc_cmd3: string = (str_concat gcc_cmd2 output)
```

**After**:
```nano
let c_file: string = (+ output ".c")
let mut gcc_cmd: string = (+ "gcc " c_file)
set gcc_cmd (+ gcc_cmd " -o ")
set gcc_cmd (+ gcc_cmd output)
```

**Improvement**: 36 characters shorter, clearer intent

---

### Example 3: src_nano/driver.nano (Command Building)

**Before**:
```nano
set cc_cmd (str_concat cc_cmd output_file)
set cc_cmd (str_concat cc_cmd " ")
set cc_cmd (str_concat cc_cmd c_file)
set cc_cmd (str_concat cc_cmd " -Isrc ")
set cc_cmd (str_concat cc_cmd runtime_files)
```

**After**:
```nano
set cc_cmd (+ cc_cmd output_file)
set cc_cmd (+ cc_cmd " ")
set cc_cmd (+ cc_cmd c_file)
set cc_cmd (+ cc_cmd " -Isrc ")
set cc_cmd (+ cc_cmd runtime_files)
```

**Improvement**: 50 characters shorter (10 chars per line × 5 lines)

---

### Example 4: src_nano/driver.nano (Error Messages)

**Before**:
```nano
set error_msg (str_concat error_msg (int_to_string exit_code))
```

**After**:
```nano
set error_msg (+ error_msg (int_to_string exit_code))
```

**Improvement**: 10 characters shorter

---

### Example 5: src_nano/nanoc.nano (Nested Concatenation)

**Before**:
```nano
let cmd1: string = (str_concat "bin/nanoc_c " input)
let cmd2: string = (str_concat cmd1 " -o ")
let cmd: string = (str_concat cmd2 output)
```

**After (Nested)**:
```nano
let cmd: string = (+ (+ (+ "bin/nanoc_c " input) " -o ") output)
```

**After (Intermediate - More Readable)**:
```nano
let cmd: string = (+ "bin/nanoc_c " input)
let cmd: string = (+ cmd " -o ")
let cmd: string = (+ cmd output)
```

**Improvement**: 30 characters shorter (nested), same readability (intermediate)

---

## Statistics

### Current Usage
```bash
$ grep -r "str_concat" src_nano/*.nano | wc -l
20+
```

### Potential Savings
- **Average savings per replacement**: ~10 characters
- **Total potential savings**: ~200+ characters
- **Readability improvement**: Significant (familiar operator)
- **Maintenance benefit**: Consistency with numeric operations

---

## Refactoring Strategy

### Phase 1: Low-Hanging Fruit
Start with simple 2-argument cases:
```nano
# Find all
grep -n "str_concat.*)" src_nano/*.nano

# Replace
(str_concat a b)  →  (+ a b)
```

### Phase 2: Chained Concatenations
Refactor chains of concatenations:
```nano
# Pattern
let x1 = (str_concat a b)
let x2 = (str_concat x1 c)
let x3 = (str_concat x2 d)

# Replace with
let mut x = (+ a b)
set x (+ x c)
set x (+ x d)
```

### Phase 3: Nested Concatenations
For deep nesting, consider intermediate variables:
```nano
# Before (hard to read)
let result = (+ (+ (+ (+ a b) c) d) e)

# After (more readable)
let mut result = (+ a b)
set result (+ result c)
set result (+ result d)
set result (+ result e)
```

---

## Automated Refactoring Script

### Simple sed Replacement (Be Careful!)
```bash
#!/bin/bash
# Replace str_concat with +

for file in src_nano/*.nano; do
    # Backup
    cp "$file" "$file.bak"
    
    # Replace
    sed -i '' 's/(str_concat/(+/g' "$file"
    
    # Review changes
    git diff "$file"
done
```

**⚠️ Warning**: This is a naive replacement! Manual review required for:
- Complex nested cases
- Comments containing "str_concat"
- String literals containing "str_concat"

### Better Approach: Manual with IDE
1. Search for `str_concat`
2. Replace each occurrence with `+`
3. Test after each file
4. Commit with clear message

---

## Testing After Refactoring

### Run All Tests
```bash
make test
```

### Specific Test for String Operations
```bash
./bin/nanoc tests/test_string_plus.nano -o tests/test_string_plus
./tests/test_string_plus
```

### Self-Hosted Compiler Tests
```bash
make stage2
make stage3
```

---

## Commit Message Template

```
refactor: replace str_concat with + operator in [file]

Converts explicit str_concat calls to idiomatic + operator:
- Before: (str_concat a b)
- After: (+ a b)

Benefits:
- 10 characters shorter per replacement
- More idiomatic (consistent with numeric +)
- Same generated C code (zero overhead)

Files modified:
- [list files]

Tested with:
- make test (all pass)
- manual inspection of generated C code
```

---

## Conclusion

**Estimated time to refactor all str_concat calls**: 1-2 hours  
**Benefit**: Cleaner, more idiomatic codebase  
**Risk**: Low (same generated C code)  
**Recommendation**: Refactor incrementally, test frequently

**Next**: Start with `src_nano/driver.nano` (5 simple replacements)

