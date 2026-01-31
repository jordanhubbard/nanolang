# Stage 2 Bootstrap Progress Report

**Date**: 2026-01-12  
**Status**: In Progress - Schema Extended, Field Collection Remaining

## Summary

Working to achieve full stage 2 bootstrap where `bin/nanoc_stage1` (self-hosted compiler) can successfully compile itself to create `bin/nanoc_stage2`.

## Completed Work

### 1. ✅ Diagnostic Printing Fixed
**Issue**: Self-hosted compiler said "NSType checking failed" without showing errors.  
**Fix**: Already present in `src_nano/nanoc_v06.nano` line 771-790 - diagnostics are printed to console.  
**Status**: Working correctly.

### 2. ✅ File I/O Functions Added
**Issue**: Self-hosted compiler declared `extern fn file_read/file_write/file_exists` but stdlib didn't provide them.  
**Fix**: Added aliases in `src/stdlib_runtime.c` lines 1265-1277:
```c
static char* file_read(const char* path) {
    return nl_os_file_read(path);
}
static int64_t file_write(const char* path, const char* content) {
    return nl_os_file_write(path, content);
}
static bool file_exists(const char* path) {
    return nl_os_file_exists(path);
}
```
**Status**: Committed in previous session.

### 3. ✅ System Execution Available
**Issue**: Self-hosted compiler needs `nl_exec_shell()` to invoke C compiler.  
**Status**: Already existed in stdlib as `nl_exec_shell()` wrapper around `system()`.

### 4. ✅ Root Cause Identified
**Problem**: Self-hosted typechecker cannot type-check user-defined structs (like `Point` from test programs).

**Analysis**:
- C compiler stores struct field info: `field_names[]` and `field_types[]` arrays
- Self-hosted parser `ASTStruct` only stored: `name` and `field_count`  
- Field names and types were **parsed but discarded**
- Typechecker's `init_struct_metadata()` only knew about compiler-internal structs
- User structs like `Point { x: int, y: int }` → field access `p.x` returns `void`

**Example Error**:
```
Error: Return value of get_x: I expected type `int`, but found `void`
```

### 5. ✅ Schema Extended (Commit 8cdeae4)
**Changes**:
- Modified `schema/compiler_schema.json`:
  - Added `field_start: int` to track field storage index
  - Added `field_names: array<string>` to store field names
  - Added `field_types: array<string>` to store field type names
  
- Regenerated type definitions:
  - `src/generated/compiler_schema.h` (C types)
  - `src_nano/generated/compiler_ast.nano` (NanoLang types)

- Updated `src_nano/parser.nano`:
  - Modified `parser_store_struct()` to accept new fields
  - Currently uses placeholder empty arrays: `(array_new 0 "")`
  - Added TODO comment for field collection logic

**Git Diff**:
```diff
struct ASTStruct {
    name: string,
+   field_start: int,
    field_count: int,
+   field_names: array<string>,
+   field_types: array<string>
}
```

## Current Status

### ✅ Builds Successfully
- C compiler builds with new schema
- Self-hosted components compile
- Runtime List<ASTStruct> works with new fields

### ⚠️ Functionality Incomplete
**Struct definitions still store empty field arrays**:
```nano
let empty_strings: array<string> = (array_new 0 "")
let node: ASTStruct = ASTStruct {
    name: name,
    field_start: 0,
    field_count: field_count,
    field_names: empty_strings,  /* ← TODO: Collect actual fields */
    field_types: empty_strings
}
```

## Remaining Work

### Task 1: Implement Field Collection in Parser (High Priority)

**File**: `src_nano/parser.nano`  
**Function**: `parse_struct_fields()` (line 5738)

**Current Behavior**:
```nano
fn parse_struct_fields(p: Parser, field_count: int, ...) -> Parser {
    /* Recursively parses fields but only increments count */
    if (== tok.token_type (token_rbrace)) {
        return (parser_store_struct p1 name field_count ...)  /* No field info! */
    } else {
        /* Parse field name (discarded) */
        /* Parse field type (discarded) */
        return (parse_struct_fields p3 (+ field_count 1) ...)
    }
}
```

**Needed**:
- Accumulate field names and types as arrays during recursive parsing
- Pass collected arrays to `parser_store_struct()`

**Challenge**:
Parser uses functional/immutable style - can't easily mutate arrays during recursion.

**Possible Approaches**:

**Option A: Add Parser State Fields**
- Add `current_struct_field_names: array<string>` to Parser struct
- Add `current_struct_field_types: array<string>` to Parser struct  
- Populate during `parse_struct_fields()` recursion
- Extract when calling `parser_store_struct()`

**Option B: Change to Iterative Loop**
```nano
fn parse_struct_fields(p: Parser, name: string, ...) -> Parser {
    let mut field_names: array<string> = (array_new 8 "")
    let mut field_types: array<string> = (array_new 8 "")
    let mut count: int = 0
    
    while (not (== tok.token_type (token_rbrace))) {
        /* Parse field name */
        set field_names (array_push field_names field_name)
        /* Parse field type */
        set field_types (array_push field_types field_type)
        set count (+ count 1)
    }
    
    return (parser_store_struct p name count field_names field_types ...)
}
```

**Option C: Use Dynamic Lists**
- Store fields in `Parser.struct_field_names: List<string>`
- Store types in `Parser.struct_field_types: List<string>`
- Convert to arrays when creating ASTStruct

**Recommendation**: Option B (iterative loop) is clearest and most NanoLang-idiomatic.

### Task 2: Update Typechecker to Use Struct Metadata

**File**: `src_nano/typecheck.nano`  
**Function**: `init_struct_metadata()` (line 157)

**Current**: Hardcoded metadata for ~40 compiler-internal structs only.

**Needed**: 
1. Add function `extract_user_struct_metadata(parser: Parser) -> array<FieldMetadata>`
2. Iterate through `parser.structs: List<ASTStruct>`
3. For each struct, add entries to metadata for all its fields:
   ```nano
   for each struct in parser.structs:
       for each field in struct.field_names:
           add FieldMetadata {
               struct_name: struct.name,
               field_name: field_name,
               field_type_kind: (type_kind_from_string field_type),
               field_type_is_list: false
           }
   ```
4. Call this before `typecheck_phase()` in `nanoc_v06.nano`

**Location to Call**:
```nano
/* In nanoc_v06.nano, around line 770 */
let parser: Parser = (parse_result.parser)

/* NEW: Extract user-defined struct metadata */
let user_metadata: array<FieldMetadata> = (extract_user_struct_metadata parser)

let tc: TypecheckPhaseOutput = (typecheck_phase parser source_to_compile)
```

Then modify `typecheck_phase()` to merge user metadata with built-in metadata.

### Task 3: Test and Iterate

**Test Cases**:
1. ✅ Simple hello world (already works)
2. ❌ Struct example: `examples/language/nl_struct.nano`
3. ❌ Self-compilation: `nanoc_stage1` compiles `nanoc_v06.nano`

**Expected Errors to Fix**:
- Field access type mismatches
- Variable scoping issues  
- Array operation type inference bugs

## Estimated Effort

- **Task 1** (Field Collection): 1-2 hours
- **Task 2** (Metadata Extraction): 1 hour  
- **Task 3** (Testing): 1-2 hours

**Total**: 3-5 hours of focused work

## Success Criteria

1. ✅ `bin/nanoc_c examples/language/nl_struct.nano` works (already true)
2. ✅ `bin/nanoc_stage1 examples/language/nl_hello.nano` works (already true)
3. ⚠️ `bin/nanoc_stage1 examples/language/nl_struct.nano` works (currently fails)
4. ❌ `bin/nanoc_stage1 src_nano/nanoc_v06.nano -o bin/nanoc_stage2` works
5. ❌ `bin/nanoc_stage1` and `bin/nanoc_stage2` produce identical binaries

## Notes

- The infrastructure is now in place (schema, types, runtime support)
- Only the "business logic" remains (field collection and metadata extraction)
- This is a tractable problem with clear solution path
- No language features missing - just implementation work

## Next Steps

1. Implement iterative `parse_struct_fields()` with field collection
2. Test with simple struct example
3. Add `extract_user_struct_metadata()` function
4. Test full self-compilation
5. Debug any remaining typechecker issues
6. Achieve stage 2 bootstrap! 🎉
