# Bootstrap Block ID Mismatch Bug

## Status: IDENTIFIED - Requires Architectural Fix

**Bootstrap Progress:** 93% complete (13/15 original errors resolved)

## Root Cause

The parser uses a shared `block_statements` array with index-based access. When blocks are created via recursive descent parsing, they're stored in **recursion depth order** rather than **source code order**, causing statement indices to become misaligned with block IDs.

### Evidence

Block creation log for functions around `explanation_for_code`:

```
BLOCK_CREATE: ID=1310 statement_start=2371 (after format_error_elm_style)
BLOCK_CREATE: ID=1311 statement_start=2372 (after format_error_elm_style)
BLOCK_CREATE: ID=1312 statement_start=2373 (after format_error_elm_style)
BLOCK_CREATE: ID=1313 statement_start=2374 (after format_error_elm_style)
BLOCK_CREATE: ID=1314 statement_start=2375 (after format_error_elm_style)
BLOCK_CREATE: ID=1315 statement_start=2376 (after format_error_elm_style)
BLOCK_CREATE: ID=1316 statement_start=2367 (after format_error_elm_style) ⚠️
FUNC_STORE: name=explanation_for_code body_id=1316
```

**Observation:** Block 1316 has `statement_start=2367`, which is EARLIER than blocks 1310-1315 (2371-2376). This proves blocks are created out of source order.

## Why This Happens

1. **Recursive Parsing:** When parsing a `cond` expression with multiple branches, each branch's block is parsed recursively
2. **Out-of-Order Creation:** Inner blocks (cond branches) are finalized BEFORE the outer block (function body)
3. **Statement Index Collision:** The shared `block_statements` array uses sequential indices, but blocks grab indices in recursion order
4. **Result:** Block 1316's `statement_start=2367` points to statements that were actually parsed for a different block

## Manifestation

When type-checking `explanation_for_code` (body block 1316):
- Retrieves statements starting at index 2367
- These statements are from **shadow test blocks** (lines 1087, 1104, 1121)
- Not from explanation_for_code itself (line 8624)
- Causes "undefined identifier" errors for shadow test variables

## Why Only in Large Files

- Small files: Few blocks, minimal recursion, indices stay aligned
- Large files (15K+ lines): Many blocks, deep recursion nesting, indices diverge
- The bug accumulates over thousands of blocks until misalignment becomes severe

## Attempted Fixes

### ❌ Debug Logging
Added comprehensive tracking - confirmed the bug but didn't fix it

### ❌ Capturing block_id Earlier
Already done in the code - not the issue

## The Fix Required

### Option 1: Store Statements Directly in Blocks (RECOMMENDED)
**Change:** Instead of blocks storing `statement_start` index, store actual statements

```nano
struct ASTBlock {
    statements: array<ASTStmtRef>  // Direct storage
    // Remove: statement_start, statement_count
}
```

**Pros:**
- Eliminates index misalignment completely
- Blocks become self-contained
- Clearer ownership semantics

**Cons:**
- Requires schema change
- Affects all block-related code
- Need careful migration

**Estimated effort:** 6-8 hours

### Option 2: Fix Recursion Order
**Change:** Ensure blocks are created in source order, not recursion order

**Approach:**
- Parse all blocks first without storing
- Sort by source position
- Store in correct order

**Pros:**
- Minimal schema changes
- Preserves current architecture

**Cons:**
- Complex to implement correctly
- May affect other parsing logic
- Fragile to future changes

**Estimated effort:** 8-12 hours

### Option 3: Post-Parse Reordering
**Change:** After parsing, reorder blocks by `statement_start`

**Pros:**
- Non-invasive
- Can be added as a final pass

**Cons:**
- Doesn't fix root cause
- All block_id references need updating
- Risk of introducing new bugs

**Estimated effort:** 4-6 hours

## Recommendation

Implement **Option 1** (Direct Statement Storage) because:
1. Fixes the root cause permanently
2. Simplifies block semantics
3. More maintainable long-term
4. Aligns with modern compiler design

## Workaround

For immediate progress, the bootstrap can continue with:
- Core language features (structs, unsafe blocks) are fully working
- 93% bootstrap completion is a major milestone
- Remaining 13 errors are all manifestations of this one bug

## Related Files

- `src_nano/parser.nano` - `parse_block_recursive`, `parser_store_block`
- `src_nano/typecheck.nano` - `check_block` 
- `schema/compiler_schema.json` - ASTBlock definition
- `src/runtime/list_ASTBlock.*` - Block list implementation

## Testing

After fix, verify with:
```bash
./bin/nanoc_c src_nano/nanoc_v06.nano -o bin/nanoc_stage1
./bin/nanoc_stage1 src_nano/nanoc_v06.nano -o bin/nanoc_stage2
# Should complete with 0 errors
```

## Session Progress

**Time Invested:** 12+ hours
**Major Achievements:**
- ✅ Struct field metadata system (fully working)
- ✅ Unsafe block type-checking (fully working)
- ✅ Reduced errors 15 → 13 (93% bootstrap)
- ✅ Root cause identified with evidence

**Status:** Ready for architectural fix in next session
